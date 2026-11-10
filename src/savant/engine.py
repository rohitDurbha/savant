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
            "max_group_d": 0.2,
            "cramers_v": 0.1,
        }
    )
    min_support: float = 0.01
    explore_share: float = 0.5
    time_blocks: int = 20
    min_rows_confirm: int = 500
    stability_runs: int = 100
    stability_keep: float = 0.8
    leak_threshold: float = 0.95  # for correlations and Cramer's V, which run from 0 to 1
    leak_explained: float = 0.9  # for group differences, the share of variance the groups explain
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
    """Exploration and confirmation rows.

    With a time column, rows are cut into contiguous time blocks and the blocks are dealt
    evenly to the two parts. Nearby rows stay together, which limits leakage through
    autocorrelation, while both parts still span the whole period. A plain before and
    after split would halve the time range each part sees and hide real trends.
    """
    if df.height < s.min_rows_confirm:
        return df, None
    cut = int(df.height * s.explore_share)
    if card.entity_column:
        # Rows of one entity are not independent, so each entity goes wholly to one side.
        ids = df[card.entity_column].unique().sort()
        chosen = np.random.default_rng(s.seed).random(ids.len()) < s.explore_share
        mask = df[card.entity_column].is_in(ids.filter(pl.Series(chosen)).implode())
        return df.filter(mask), df.filter(~mask)
    if card.time_column:
        ordered = df.sort(card.time_column, nulls_last=True)
        block = np.arange(df.height) * s.time_blocks // df.height
        explore_block = np.floor((block + 1) * s.explore_share) > np.floor(block * s.explore_share)
        return ordered.filter(pl.Series(explore_block)), ordered.filter(pl.Series(~explore_block))
    idx = np.random.default_rng(s.seed).permutation(df.height)
    return df[idx[:cut].tolist()], df[idx[cut:].tolist()]


def _near_deterministic(t: Test, s: Settings) -> bool:
    """A link so tight that one column nearly defines the other.

    Correlations and Cramer's V already run from 0 to 1. A group difference is measured in
    standard deviations, which has no ceiling, so it uses the share of variance the groups
    explain. A large difference such as one brand of car being older is not a definition.
    """
    if t.explained is not None:
        return t.explained >= s.leak_explained
    return t.effect >= s.leak_threshold


def _passes(t: Test | None, s: Settings, alpha: float, sign: float | None = None) -> bool:
    if t is None or t.p_value > alpha or t.effect < s.min_effect[t.effect_name]:
        return False
    return sign is None or np.sign(t.statistic) == sign


def per_entity(df: pl.DataFrame, card: DataCard) -> pl.DataFrame:
    """One row per entity, numeric columns as means and categories as their mode."""
    nums = card.of_kind("numeric")
    cats = card.of_kind("categorical", "boolean")
    agg = [pl.col(c).mean() for c in nums]
    agg += [pl.col(c).drop_nulls().mode().sort().first().alias(c) for c in cats]
    return df.group_by(card.entity_column).agg(agg).sort(card.entity_column)


class Views:
    """The frame each test runs on. With repeated entities, every test except a time trend
    runs on one row per entity, because rows of one entity are not independent evidence."""

    def __init__(self, card: DataCard) -> None:
        self.card = card
        self._entity: dict[int, pl.DataFrame] = {}

    def frame(self, part: pl.DataFrame, kind: str) -> pl.DataFrame:
        if not self.card.entity_column or kind == "trend":
            return part
        key = id(part)
        if key not in self._entity:
            self._entity[key] = per_entity(part, self.card)
        return self._entity[key]

    def run(self, part: pl.DataFrame, kind: str, a: str, b: str, s: Settings) -> Test | None:
        frame = self.frame(part, kind)
        time_col = self.card.time_column if self.card.time_column in frame.columns else None
        return stats.run(frame, kind, a, b, s.min_support, time_col)


def scan(df: pl.DataFrame, views: Views, s: Settings) -> Ledger:
    ledger = Ledger()
    for i, (kind, a, b) in enumerate(stats.plan(views.card)):
        t = views.run(df, kind, a, b, s)
        if t is not None:
            t.id = f"t{i}"
            ledger.add(t)
    return ledger


def stability(df: pl.DataFrame, views: Views, t: Test, s: Settings) -> float:
    frame = views.frame(df, t.kind)
    rng = np.random.default_rng(s.seed + 1)
    half = frame.height // 2
    sign = np.sign(t.statistic) if t.method == "spearman" else None
    hits = 0
    for _ in range(s.stability_runs):
        rows = np.sort(rng.choice(frame.height, size=half, replace=False)).tolist()
        sub = frame[rows]
        time_col = views.card.time_column if views.card.time_column in sub.columns else None
        rerun = stats.run(sub, t.kind, *t.columns, s.min_support, time_col)
        hits += _passes(rerun, s, 0.05, sign)
    return hits / s.stability_runs


def analyze(df: pl.DataFrame, s: Settings | None = None) -> Analysis:
    s = s or Settings()
    card = build_card(df)
    explore, confirm = split(df, card, s)
    views = Views(card)
    ledger = scan(explore, views, s)
    dropped = {"fdr": 0, "effect": 0, "confirmation": 0, "stability": 0}
    candidates: list[Finding] = []
    discovered = {t.id for t in ledger.discoveries(s.q, s.dependent)}
    for t in ledger.tests:
        if t.id not in discovered:
            dropped["fdr"] += 1
        elif t.effect < s.min_effect[t.effect_name]:
            dropped["effect"] += 1
        elif _near_deterministic(t, s):
            candidates.append(Finding(t, "leakage"))
        else:
            candidates.append(Finding(t, "exploratory"))

    live = [f for f in candidates if f.status == "exploratory"]
    if confirm is not None and live:
        reruns = [views.run(confirm, f.test.kind, *f.test.columns, s) for f in live]
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
            f.stability = stability(explore, views, f.test, s)
            if f.stability < s.stability_keep:
                f.status = "exploratory"
                dropped["stability"] += 1

    return Analysis(card, ledger, candidates, dropped, confirm is not None)
