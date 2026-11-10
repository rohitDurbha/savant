"""The data card, a compact description of a table that the model reads in place of rows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import polars as pl

KINDS = ("numeric", "categorical", "datetime", "boolean", "id", "entity", "code", "text")
ID_HINTS = ("id", "uuid", "key", "number", "no")
# Integers with these words in their name are labels, such as a postal code or a census
# tract, so averaging them or correlating them with quantities means nothing.
CODE_HINTS = ("code", "zip", "zipcode", "postal", "district", "geoid", "fips", "tract", "ward")
MAX_LEVELS = 100  # more distinct values than this is too many groups to compare usefully


@dataclass
class ColumnCard:
    name: str
    kind: str
    dtype: str
    missing_share: float
    unique: int
    examples: list[str]
    stats: dict[str, float] = field(default_factory=dict)


@dataclass
class DataCard:
    rows: int
    columns: list[ColumnCard]
    dependencies: list[tuple[str, str]]
    time_column: str | None
    entity_column: str | None = None
    duplicates: list[tuple[str, str]] = field(default_factory=list)  # (duplicate, kept)
    placeholder_zeros: dict[str, str] = field(default_factory=dict)  # column -> evidence

    def column(self, name: str) -> ColumnCard:
        return next(c for c in self.columns if c.name == name)

    def of_kind(self, *kinds: str) -> list[str]:
        return [c.name for c in self.columns if c.kind in kinds]

    def testable(self, *kinds: str) -> list[str]:
        """Columns of these kinds, leaving out any that duplicate an earlier column."""
        dupes = {d for d, _ in self.duplicates}
        return [n for n in self.of_kind(*kinds) if n not in dupes]

    def to_text(self, max_examples: int = 3) -> str:
        """Render the card as short plain text for a prompt, most useful facts first."""
        lines = [f"rows {self.rows}, columns {len(self.columns)}"]
        if self.time_column:
            lines.append(f"time column {self.time_column}")
        if self.entity_column:
            lines.append(f"rows repeat per {self.entity_column}, so findings are per entity")
        for c in self.columns:
            ex = ", ".join(c.examples[:max_examples])
            stats = " ".join(
                f"{k}={v:.4g}" for k, v in c.stats.items() if k != "placeholder_zeros" or v
            )
            lines.append(
                f"- {c.name} [{c.kind}, {c.dtype}] missing={c.missing_share:.1%} "
                f"unique={c.unique} {stats} e.g. {ex}".rstrip()
            )
        for col, why in self.placeholder_zeros.items():
            lines.append(f"zeros in {col} likely mean unknown, {why}")
        if self.duplicates:
            same = "; ".join(f"{d} repeats {k}" for d, k in self.duplicates)
            lines.append(f"duplicate columns, tested once: {same}")
        if self.dependencies:
            deps = "; ".join(f"{a} determines {b}" for a, b in self.dependencies)
            lines.append(f"dependencies: {deps}")
        return "\n".join(lines)


def _kind(s: pl.Series, rows: int) -> str:
    dtype = s.dtype
    unique = s.n_unique()
    name = s.name.lower()
    if dtype == pl.Boolean:
        return "boolean"
    if dtype.is_temporal():
        return "datetime"
    looks_like_id = unique >= 0.95 * rows and rows >= 20
    tokens = re.split(r"[^a-z0-9]+", name)
    named_like_id = any(t in ID_HINTS for t in tokens) or name.endswith("id")
    if dtype.is_integer() and unique > 2 and any(t in CODE_HINTS for t in tokens):
        return "code"
    if dtype.is_integer() and looks_like_id and named_like_id:
        return "id"
    # An id-like name whose values repeat marks rows that belong to the same thing, such
    # as one customer. Such rows are not independent, so the column is never tested.
    repeats = max(20, 0.02 * rows) < unique < 0.95 * rows
    if named_like_id and repeats and (dtype.is_integer() or dtype == pl.String):
        return "entity"
    if (
        dtype.is_integer()
        and unique > 2
        and (any(t in CODE_HINTS for t in tokens) or name.endswith("id"))
    ):
        return "code"
    if dtype.is_numeric():
        return "categorical" if unique <= 2 else "numeric"
    if looks_like_id:
        return "id"
    if unique <= min(MAX_LEVELS, max(20, int(0.05 * rows))):
        return "categorical"
    return "text"


def _stats(s: pl.Series, kind: str) -> dict[str, float]:
    if kind != "numeric":
        return {}
    clean = s.drop_nulls().cast(pl.Float64)
    if clean.is_empty():
        return {}
    median = float(clean.median())
    mad = float((clean - median).abs().median())
    # Iglewicz and Hoaglin: a modified z score above 3.5 marks an unusual value. It is a
    # note for the reader, and no row is dropped because of it.
    unusual = int(((0.6745 * (clean - median) / mad).abs() > 3.5).sum()) if mad else 0
    # A spike at exactly zero with nothing just above it, in a column whose values are
    # usually large, looks like a placeholder for unknown. Small counts with real zeros,
    # where ones and twos are common, never qualify.
    positive = clean.filter(clean > 0)
    zero_share = float((clean == 0).mean())
    placeholder = 0.0
    if zero_share >= 0.05 and positive.len() >= 20:
        pos_median = float(positive.median())
        low = float(positive.quantile(0.01))
        if pos_median >= 10 and low >= 0.1 * pos_median:
            placeholder = zero_share
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std() or 0.0),
        "min": float(clean.min()),
        "median": median,
        "max": float(clean.max()),
        "unusual": float(unusual),
        "placeholder_zeros": placeholder,
    }


def functional_dependencies(df: pl.DataFrame, columns: list[str]) -> list[tuple[str, str]]:
    """Pairs (a, b) where every value of a maps to exactly one value of b.

    Findings between such pairs are easy to infer, so the engine drops them.
    """
    deps = []
    for a in columns:
        for b in columns:
            if a == b or df[b].n_unique() <= 1:
                continue
            per_a = df.select([a, b]).drop_nulls().group_by(a).agg(pl.col(b).n_unique())
            if per_a.height > 1 and per_a[b].max() == 1:
                deps.append((a, b))
    return deps


def duplicate_columns(
    df: pl.DataFrame, cats: list[str], nums: list[str], deps: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Columns that carry the same information as an earlier one.

    Two categories that determine each other, such as a state code and a state name, or
    two numeric columns with identical values, would each produce every finding twice.
    """
    order = {c: i for i, c in enumerate(df.columns)}
    dep = set(deps)
    dupes: dict[str, str] = {}
    for a, b in deps:
        if (b, a) in dep and order[a] < order[b] and b not in dupes:
            dupes[b] = a
    for i, a in enumerate(nums):
        for b in nums[i + 1 :]:
            if b in dupes or a in dupes:
                continue
            x, y = df[a].cast(pl.Float64), df[b].cast(pl.Float64)
            if x.is_null().equals(y.is_null()) and (x == y).fill_null(True).all():
                dupes[b] = a
    return sorted(dupes.items(), key=lambda kv: order[kv[0]])


def placeholder_evidence(df: pl.DataFrame, card: DataCard) -> dict[str, str]:
    """Why each flagged column's zeros look like a stand-in for unknown."""
    out = {}
    for c in card.columns:
        if not c.stats.get("placeholder_zeros"):
            continue
        why = f"{c.stats['placeholder_zeros']:.0%} are exactly zero with nothing just above it"
        zero = df[c.name] == 0
        for other in df.columns:
            if df[other].dtype != pl.String:
                continue
            at_zero = df.filter(zero)[other].drop_nulls()
            if at_zero.is_empty():
                continue
            label = at_zero.mode().sort().first()
            elsewhere = df.filter(~zero)[other]
            if (at_zero == label).mean() >= 0.8 and (elsewhere == label).mean() <= 0.05:
                why += f", and they line up with {other} = {label}"
                break
        out[c.name] = why
    return out


def build_card(df: pl.DataFrame) -> DataCard:
    rows = df.height
    cols = []
    for name in df.columns:
        s = df[name]
        kind = _kind(s, rows)
        examples = [str(v) for v in s.drop_nulls().unique(maintain_order=True).head(5).to_list()]
        cols.append(
            ColumnCard(
                name=name,
                kind=kind,
                dtype=str(s.dtype),
                missing_share=s.null_count() / rows if rows else 0.0,
                unique=s.n_unique(),
                examples=examples,
                stats=_stats(s, kind),
            )
        )
    card = DataCard(rows=rows, columns=cols, dependencies=[], time_column=None)
    times = card.of_kind("datetime")
    card.time_column = times[0] if times else None
    entities = card.of_kind("entity")
    card.entity_column = entities[0] if entities else None
    card.dependencies = functional_dependencies(df, card.of_kind("categorical"))
    card.placeholder_zeros = placeholder_evidence(df, card)
    card.duplicates = duplicate_columns(
        df, card.of_kind("categorical"), card.of_kind("numeric"), card.dependencies
    )
    return card
