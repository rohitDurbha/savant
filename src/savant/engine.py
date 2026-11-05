"""The guarded discovery path that every finding must pass before anyone reads it.

Gates, in order: the test ledger with false discovery control, a practical effect size,
the dependency check (done when tests are planned), the leakage check, confirmation on
rows the scan never saw, and bootstrap stability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl

from . import stats
from .ledger import Ledger, Test, false_discovery_control
from .profile import DataCard, build_card

LADDER = ("association", "adjusted_association", "consistent_with_cause", "estimated_effect")


@dataclass
class Settings:
    q: float = 0.05
    dependent: bool = False
    min_effect: dict[str, float] = field(
        default_factory=lambda: {
            "abs_rho": 0.1,
            "epsilon_squared": 0.01,
            "cramers_v": 0.1,
        }
    )
    min_support: float = 0.01
    explore_share: float = 0.5
    min_rows_confirm: int = 500
    stability_runs: int = 100
    stability_keep: float = 0.8
    leak_threshold: float = 0.95
    seed: int = 0


@dataclass
class Finding:
    test: Test
    status: str  # confirmed, exploratory, leakage
    claim: str = "association"
    confirm_p: float | None = None
    stability: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["test"] = self.test.to_dict()
        return d


@dataclass
class Analysis:
    card: DataCard
    ledger: Ledger
    findings: list[Finding]
    dropped: dict[str, int]
    confirmed_split: bool

    @property
    def reported(self) -> list[Finding]:
        return [f for f in self.findings if f.status == "confirmed"]


def split(
    df: pl.DataFrame, card: DataCard, s: Settings
) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    """Exploration and confirmation rows, by time when the table has a time column."""
    if df.height < s.min_rows_confirm:
        return df, None
    cut = int(df.height * s.explore_share)
    if card.time_column:
        ordered = df.sort(card.time_column, nulls_last=True)
        return ordered.head(cut), ordered.tail(df.height - cut)
    idx = np.random.default_rng(s.seed).permutation(df.height)
    return df[idx[:cut].tolist()], df[idx[cut:].tolist()]


def _passes(t: Test | None, s: Settings, alpha: float, sign: float | None = None) -> bool:
    if t is None or t.p_value > alpha or t.effect < s.min_effect[t.effect_name]:
        return False
    return sign is None or np.sign(t.statistic) == sign


def scan(df: pl.DataFrame, card: DataCard, s: Settings) -> Ledger:
    ledger = Ledger()
    for i, (kind, a, b) in enumerate(stats.plan(card)):
        t = stats.run(df, kind, a, b, s.min_support)
        if t is not None:
            t.id = f"t{i}"
            ledger.add(t)
    return ledger


def stability(df: pl.DataFrame, t: Test, s: Settings) -> float:
    rng = np.random.default_rng(s.seed + 1)
    half = df.height // 2
    sign = np.sign(t.statistic) if t.method == "spearman" else None
    hits = 0
    for _ in range(s.stability_runs):
        rows = rng.choice(df.height, size=half, replace=False).tolist()
        hits += _passes(stats.run(df[rows], t.kind, *t.columns, s.min_support), s, 0.05, sign)
    return hits / s.stability_runs


def analyze(df: pl.DataFrame, s: Settings | None = None) -> Analysis:
    s = s or Settings()
    card = build_card(df)
    explore, confirm = split(df, card, s)
    ledger = scan(explore, card, s)
    dropped = {"fdr": 0, "effect": 0, "confirmation": 0, "stability": 0}
    candidates: list[Finding] = []
    discovered = {t.id for t in ledger.discoveries(s.q, s.dependent)}
    for t in ledger.tests:
        if t.id not in discovered:
            dropped["fdr"] += 1
        elif t.effect < s.min_effect[t.effect_name]:
            dropped["effect"] += 1
        elif t.effect >= s.leak_threshold:
            candidates.append(Finding(t, "leakage"))
        else:
            candidates.append(Finding(t, "exploratory"))

    live = [f for f in candidates if f.status == "exploratory"]
    if confirm is not None and live:
        reruns = [stats.run(confirm, f.test.kind, *f.test.columns, s.min_support) for f in live]
        p_values = [r.p_value if r is not None else 1.0 for r in reruns]
        ok = false_discovery_control(p_values, s.q, s.dependent)
        for f, r, keep in zip(live, reruns, ok, strict=True):
            f.confirm_p = r.p_value if r is not None else None
            sign = np.sign(f.test.statistic) if f.test.method == "spearman" else None
            if keep and _passes(r, s, 1.0, sign):
                f.status = "confirmed"
            else:
                dropped["confirmation"] += 1

    for f in candidates:
        if f.status == "confirmed":
            f.stability = stability(explore, f.test, s)
            if f.stability < s.stability_keep:
                f.status = "exploratory"
                dropped["stability"] += 1

    return Analysis(card, ledger, candidates, dropped, confirm is not None)
