"""Long tables, where each row holds one measure for one place and time.

Public health and survey tables often store every indicator in one value column, with a
column that says which indicator a row belongs to. Testing such a table pair by pair
finds mostly structure, such as the indicator predicting the value's scale. Savant
detects the layout and analyzes the most common measures one at a time.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field

import polars as pl

from .engine import Analysis, Settings, analyze

MEASURE_HINTS = ("question", "indicator", "measure", "metric", "variable", "series", "parameter")
VARIANT_HINTS = ("type", "unit", "units", "scale", "format")
IGNORE_HINTS = (
    "confidence", "limit", "lower", "upper", "ci", "footnote", "symbol", "flag", "note",
    "notes", "se", "error", "standard", "alt",
)  # fmt: skip


def _tokens(name: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return [t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t]


@dataclass
class LongLayout:
    measure: str
    value: str
    variants: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [f"long table, one row per {self.measure} with values in {self.value}"]
        if self.variants:
            parts.append(f"one level kept per measure for {', '.join(self.variants)}")
        if self.ignored:
            parts.append(f"set aside {', '.join(self.ignored)}")
        return ", ".join(parts)


def detect(df: pl.DataFrame) -> LongLayout | None:
    """The measure and value columns of a long table, or None for an ordinary table."""
    measures = [
        c
        for hint in MEASURE_HINTS
        for c in df.columns
        if hint in _tokens(c) and df[c].dtype == pl.String and 2 <= df[c].n_unique() <= 5000
    ]
    values = [
        c
        for c in df.columns
        if df[c].dtype.is_numeric() and "value" in c.lower() and not set(_tokens(c)) & {"alt"}
    ]
    if not measures or not values:
        return None
    measure, value = measures[0], values[0]
    # A long table repeats each measure across many places and times. A check on how much
    # the measure explains the value failed on real CDC data, where a few raw counts in the
    # tens of thousands sit next to percentages, so the layout itself is checked.
    sizes = df[measure].drop_nulls().value_counts()["count"]
    if df[value].drop_nulls().len() < 20 or sizes.len() < 2 or sizes.median() < 10:
        return None
    others = [c for c in df.columns if c != measure]
    per = df.group_by(measure).agg([pl.col(c).n_unique().alias(c) for c in others])
    variants = [
        c
        for c in df.columns
        if c not in (measure, value)
        and df[c].dtype == pl.String
        and set(_tokens(c)) & set(VARIANT_HINTS)
        and per[c].max() > 1
    ]
    ignored = [
        c
        for c in df.columns
        if c not in (measure, value, *variants) and set(_tokens(c)) & set(IGNORE_HINTS)
    ]
    return LongLayout(measure, value, variants, ignored)


# Rates compare places of different sizes fairly, and raw counts mostly measure population,
# so a measure's value type is chosen in this order before falling back to the most common.
PREFERRED_TYPES = ("age-adjusted", "crude rate", "rate", "prevalence", "percent", "mean", "median")


def _choose(levels: list[str]) -> str:
    for word in PREFERRED_TYPES:
        hit = sorted(lv for lv in levels if word in lv.lower())
        if hit:
            return hit[0]
    return levels[0]


def split(
    df: pl.DataFrame, layout: LongLayout, top: int = 8, min_rows: int = 100
) -> dict[str, tuple[pl.DataFrame, list[str]]]:
    """The most common measures, each as its own table with one level of each variant."""
    counts = df[layout.measure].value_counts(sort=True)
    out = {}
    for m in counts.filter(pl.col("count") >= min_rows)[layout.measure].head(top).to_list():
        sub = df.filter(pl.col(layout.measure) == m)
        notes = []
        for v in layout.variants:
            counts = sub[v].drop_nulls().cast(pl.String).value_counts(sort=True)
            if counts.height < 2:
                continue
            keep = _choose(counts[v].to_list())  # most common first, so ties keep it
            sub = sub.filter(pl.col(v).cast(pl.String) == keep)
            notes.append(f"{v} kept at {keep}")
        sub = sub.drop([layout.measure, *layout.variants, *layout.ignored])
        out[str(m)] = (sub, notes)
    return out


def analyze_long(
    df: pl.DataFrame, settings: Settings | None = None, top: int = 8
) -> dict[str, Analysis]:
    """Analyze each common measure on its own.

    Each measure gets its own false discovery control at q divided by the number of
    measures, so running several analyses does not loosen the overall guard.
    """
    s = settings or Settings()
    layout = detect(df)
    if layout is None:
        return {}
    parts = split(df, layout, top)
    each = dataclasses.replace(s, q=s.q / max(1, len(parts)))
    return {m: analyze(sub, each) for m, (sub, _) in parts.items()}


def findings_by_measure(results: dict[str, Analysis], per_measure: int = 4) -> str:
    """Guarded findings grouped under each measure, short enough for a prompt."""
    from .agent import findings_text

    blocks = [f"Measure {m}\n{findings_text(a, limit=per_measure)}" for m, a in results.items()]
    return "\n\n".join(blocks) if blocks else "No measure had enough rows to analyze."
