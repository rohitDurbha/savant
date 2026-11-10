"""Compare engine output with planted truth to get recall and false findings."""

from __future__ import annotations

from dataclasses import dataclass, field

from .engine import Analysis
from .planted import Task


def _pair(cols) -> frozenset[str]:
    return frozenset(cols)


@dataclass
class Score:
    task: str
    null: bool
    planted: int = 0
    found: int = 0
    false_findings: int = 0
    leaks: int = 0
    leaks_flagged: int = 0
    derived: int = 0
    by_kind: dict[str, list[int]] = field(default_factory=dict)  # kind -> [found, planted]


def derived_pairs(task: Task) -> set[frozenset[str]]:
    """Pairs linked only through a shared planted column, such as two metrics that both
    trend over time. They are real associations in the data, so they are not counted as
    false findings, and they are reported apart because a better engine should explain
    them through the shared column."""
    others: dict[str, list[str]] = {}
    for p in task.plants:
        a, b = p.columns
        others.setdefault(a, []).append(b)
        others.setdefault(b, []).append(a)
    pairs = set()
    for linked in others.values():
        for i, x in enumerate(linked):
            for y in linked[i + 1 :]:
                pairs.add(_pair((x, y)))
    return pairs


def grade(task: Task, analysis: Analysis) -> Score:
    real = {_pair(p.columns): p for p in task.plants if p.kind != "leak"}
    leaks = {_pair(p.columns) for p in task.plants if p.kind == "leak"}
    reported = {_pair(f.test.columns) for f in analysis.reported}
    flagged = {_pair(f.test.columns) for f in analysis.findings if f.status == "leakage"}
    s = Score(task.id, task.is_null, planted=len(real), leaks=len(leaks))
    for pair, p in real.items():
        hit = pair in reported
        s.found += hit
        found, total = s.by_kind.get(p.kind, [0, 0])
        s.by_kind[p.kind] = [found + hit, total + 1]
    derived = derived_pairs(task) - set(real) - leaks
    s.derived = len(reported & derived)
    s.false_findings = len(reported - set(real) - leaks - derived)
    s.leaks_flagged = len(leaks & flagged)
    return s


def summarize(scores: list[Score]) -> dict:
    hard = [s for s in scores if s.task.startswith("hard-null")]
    nulls = [s for s in scores if s.null and not s.task.startswith("hard-null")]
    planted = [s for s in scores if not s.null]
    by_kind: dict[str, list[int]] = {}
    for s in planted:
        for k, (f, t) in s.by_kind.items():
            cur = by_kind.setdefault(k, [0, 0])
            cur[0] += f
            cur[1] += t
    total_planted = sum(s.planted for s in planted)
    total_leaks = sum(s.leaks for s in planted)
    return {
        "null_files": len(nulls),
        "false_findings_per_null_file": (
            sum(s.false_findings for s in nulls) / len(nulls) if nulls else None
        ),
        "null_files_with_any_false_finding": sum(s.false_findings > 0 for s in nulls),
        "hard_null_files": len(hard),
        "false_findings_per_hard_null_file": (
            sum(s.false_findings for s in hard) / len(hard) if hard else None
        ),
        "hard_null_files_with_any_false_finding": sum(s.false_findings > 0 for s in hard),
        "planted_files": len(planted),
        "recall": sum(s.found for s in planted) / total_planted if total_planted else None,
        "recall_by_kind": {k: round(f / t, 3) for k, (f, t) in sorted(by_kind.items())},
        "false_findings_in_planted_files": sum(s.false_findings for s in planted),
        "derived_findings_in_planted_files": sum(s.derived for s in planted),
        "leaks_flagged": f"{sum(s.leaks_flagged for s in planted)} of {total_leaks}",
    }
