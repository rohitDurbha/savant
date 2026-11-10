"""The data card, a compact description of a table that the model reads in place of rows."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import polars as pl

KINDS = ("numeric", "categorical", "datetime", "boolean", "id", "entity", "text")
ID_HINTS = ("id", "uuid", "key", "code", "number", "no")


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

    def column(self, name: str) -> ColumnCard:
        return next(c for c in self.columns if c.name == name)

    def of_kind(self, *kinds: str) -> list[str]:
        return [c.name for c in self.columns if c.kind in kinds]

    def to_text(self, max_examples: int = 3) -> str:
        """Render the card as short plain text for a prompt, most useful facts first."""
        lines = [f"rows {self.rows}, columns {len(self.columns)}"]
        if self.time_column:
            lines.append(f"time column {self.time_column}")
        if self.entity_column:
            lines.append(f"rows repeat per {self.entity_column}, so findings are per entity")
        for c in self.columns:
            ex = ", ".join(c.examples[:max_examples])
            stats = " ".join(f"{k}={v:.4g}" for k, v in c.stats.items())
            lines.append(
                f"- {c.name} [{c.kind}, {c.dtype}] missing={c.missing_share:.1%} "
                f"unique={c.unique} {stats} e.g. {ex}".rstrip()
            )
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
    if dtype.is_integer() and looks_like_id and named_like_id:
        return "id"
    # An id-like name whose values repeat marks rows that belong to the same thing, such
    # as one customer. Such rows are not independent, so the column is never tested.
    repeats = max(20, 0.02 * rows) < unique < 0.95 * rows
    if named_like_id and repeats and (dtype.is_integer() or dtype == pl.String):
        return "entity"
    if dtype.is_numeric():
        return "categorical" if unique <= 2 else "numeric"
    if looks_like_id:
        return "id"
    if unique <= max(20, int(0.05 * rows)):
        return "categorical"
    return "text"


def _stats(s: pl.Series, kind: str) -> dict[str, float]:
    if kind != "numeric":
        return {}
    clean = s.drop_nulls().cast(pl.Float64)
    if clean.is_empty():
        return {}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std() or 0.0),
        "min": float(clean.min()),
        "median": float(clean.median()),
        "max": float(clean.max()),
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
    return card
