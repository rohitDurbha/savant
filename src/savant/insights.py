"""Turn guarded findings into a short ranked list of insights in plain language.

The score follows the impact times significance idea from QuickInsights and from Tang and
colleagues. Strength is the effect size against the conventional large effect for its
measure, and impact grows with the share of rows a finding covers. Findings that share a
column are grouped under the strongest one, so a report reads as a few insights.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .engine import Finding

LARGE = {"abs_rho": 0.5, "cramers_v": 0.5, "max_group_d": 0.8, "shift_d": 0.8}
SHIFT = re.compile(r"level shift around ([0-9-]+)")
STANDOUT = re.compile(r"^(.*?)=(.*) stands out, standardized difference ([+-][\d.]+)")
NEAR_WEIGHT = 0.5  # definitions are worth knowing, but they are rarely the news


@dataclass
class Insight:
    lead: Finding
    score: float
    headline: str
    related: list[Finding] = field(default_factory=list)

    @property
    def columns(self) -> set[str]:
        return set(self.lead.test.columns)


def score(f: Finding) -> float:
    t = f.test
    strength = min(1.0, t.effect / LARGE.get(t.effect_name, 1.0))
    coverage = t.support if t.support is not None else 1.0
    impact = 0.5 + 0.5 * math.sqrt(max(0.0, coverage))
    weight = NEAR_WEIGHT if f.status == "leakage" else 1.0
    return strength * impact * weight


def headline(f: Finding) -> str:
    """One plain sentence for a finding that never claims a cause."""
    t = f.test
    a, b = t.columns
    if f.status == "leakage":
        return f"{a} nearly determines {b}, likely by definition."
    if t.kind == "correlation":
        return f"{a} and {b} rise together." if t.statistic > 0 else f"As {a} rises, {b} falls."
    if t.kind == "trend":
        return f"{b} {'rises' if t.statistic > 0 else 'falls'} over {a}."
    if t.kind == "shift":
        m = SHIFT.search(t.note)
        when = f" around {m.group(1)}" if m else ""
        return f"{b} shifted {'up' if t.statistic > 0 else 'down'}{when}."
    if t.kind == "segment":
        m = STANDOUT.match(t.note)
        if m:
            way = "higher" if float(m.group(3)) > 0 else "lower"
            return f"{b} is {way} when {m.group(1)} is {m.group(2)} than in other rows."
        return f"{b} differs across {a}."
    if t.kind == "association":
        return f"{a} and {b} are linked, so the mix of {b} differs across {a}."
    return f"{a} and {b} are related."


def rank(findings: list[Finding], limit: int = 8, time_column: str | None = None) -> list[Insight]:
    """Strongest findings first, each grouped with weaker ones that share a column.

    The time column is shared by every trend and shift, so it never makes two findings
    related on its own.
    """
    usable = [f for f in findings if f.status in ("confirmed", "leakage")]
    usable.sort(key=lambda f: (-score(f), f.test.columns))
    insights: list[Insight] = []
    for f in usable:
        cols = set(f.test.columns) - {time_column}
        home = next((i for i in insights if (i.columns - {time_column}) & cols), None)
        if home is not None:
            home.related.append(f)
        elif len(insights) < limit:
            insights.append(Insight(f, score(f), headline(f)))
    return insights


def insights_text(insights: list[Insight]) -> str:
    if not insights:
        return "No insights. Nothing survived the checks."
    lines = []
    for n, i in enumerate(insights, 1):
        lines.append(f"{n}. {i.headline} (score {i.score:.2f})")
        lines += [f"   related: {headline(f)}" for f in i.related[:3]]
    return "\n".join(lines)
