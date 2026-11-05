"""Statistical tests chosen by a fixed decision table, never by the language model.

Each test checks its own assumptions first and returns None when they fail, so an
inapplicable method can never produce a finding.
"""

from __future__ import annotations

import itertools

import numpy as np
import polars as pl
from scipy import stats as sps

from .ledger import Test
from .profile import DataCard

MIN_GROUP = 5


def _numeric(df: pl.DataFrame, col: str) -> np.ndarray:
    s = df[col]
    if s.dtype.is_temporal():
        s = s.cast(pl.Datetime("ms")).cast(pl.Int64)
    return s.cast(pl.Float64).to_numpy()


def num_num(df: pl.DataFrame, a: str, b: str, kind: str = "correlation") -> Test | None:
    x, y = _numeric(df, a), _numeric(df, b)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if x.size < 10 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rho, p = sps.spearmanr(x, y)
    return Test(
        id="",
        kind=kind,
        columns=(a, b),
        method="spearman",
        statistic=float(rho),
        p_value=float(p),
        effect=abs(float(rho)),
        effect_name="abs_rho",
        n=int(x.size),
    )


def cat_num(df: pl.DataFrame, cat: str, num: str, min_support: float) -> Test | None:
    sub = df.select([cat, num]).drop_nulls()
    n_all = sub.height
    if n_all < 10:
        return None
    sizes = sub.group_by(cat).len()
    keep = sizes.filter((pl.col("len") >= MIN_GROUP) & (pl.col("len") >= min_support * n_all))[
        cat
    ].to_list()
    if len(keep) < 2:
        return None
    sub = sub.filter(pl.col(cat).is_in(keep))
    groups = [g[num].cast(pl.Float64).to_numpy() for _, g in sub.group_by(cat)]
    if all(np.ptp(g) == 0 for g in groups):
        return None
    h, p = sps.kruskal(*groups)
    n, k = sub.height, len(groups)
    eps2 = max(0.0, (h - k + 1) / (n - k)) if n > k else 0.0
    means = sub.group_by(cat).agg(pl.col(num).mean().alias("m")).sort("m", descending=True)
    top = str(means[cat][0])
    return Test(
        id="",
        kind="segment",
        columns=(cat, num),
        method="kruskal",
        statistic=float(h),
        p_value=float(p),
        effect=float(eps2),
        effect_name="epsilon_squared",
        n=n,
        note=f"highest mean in {cat}={top}",
    )


def cat_cat(df: pl.DataFrame, a: str, b: str) -> Test | None:
    sub = df.select([a, b]).drop_nulls()
    if sub.height < 10:
        return None
    table = sub.group_by([a, b]).len().pivot(on=b, index=a, values="len").fill_null(0)
    counts = table.drop(a).to_numpy().astype(float)
    if counts.shape[0] < 2 or counts.shape[1] < 2:
        return None
    chi2, p, _, expected = sps.chi2_contingency(counts, correction=False)
    if (expected < 1).any() or (expected >= 5).mean() < 0.8:
        return None
    n = counts.sum()
    v = float(np.sqrt(chi2 / (n * (min(counts.shape) - 1))))
    return Test(
        id="",
        kind="association",
        columns=(a, b),
        method="chi_square",
        statistic=float(chi2),
        p_value=float(p),
        effect=v,
        effect_name="cramers_v",
        n=int(n),
    )


def run(df: pl.DataFrame, kind: str, a: str, b: str, min_support: float) -> Test | None:
    """Rerun one test of a known kind on another slice of rows."""
    if kind in ("correlation", "trend"):
        return num_num(df, a, b, kind)
    if kind == "segment":
        return cat_num(df, a, b, min_support)
    if kind == "association":
        return cat_cat(df, a, b)
    raise ValueError(kind)


def plan(card: DataCard) -> list[tuple[str, str, str]]:
    """Every (kind, a, b) test the scan will run, from column kinds alone."""
    nums = card.of_kind("numeric")
    cats = card.of_kind("categorical", "boolean")
    inferable = set(card.dependencies)
    tests = [("correlation", a, b) for a, b in itertools.combinations(nums, 2)]
    tests += [("segment", c, n) for c in cats for n in nums]
    tests += [
        ("association", a, b)
        for a, b in itertools.combinations(cats, 2)
        if (a, b) not in inferable and (b, a) not in inferable
    ]
    if card.time_column:
        tests += [("trend", card.time_column, n) for n in nums]
    return tests
