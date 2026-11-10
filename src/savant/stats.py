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
MIN_EFFECTIVE_ROWS = 30  # below this a time-ordered test has too little independent evidence


def _numeric(df: pl.DataFrame, col: str) -> np.ndarray:
    s = df[col]
    if s.dtype.is_temporal():
        s = s.cast(pl.Datetime("ms")).cast(pl.Int64)
    return s.cast(pl.Float64).to_numpy()


def _lag1(x: np.ndarray) -> float:
    x = x - x.mean()
    denom = float((x * x).sum())
    return float((x[:-1] * x[1:]).sum() / denom) if denom and x.size > 2 else 0.0


def _detrended_ranks(v: np.ndarray) -> np.ndarray:
    """Ranks of v with their straight-line drift over time order removed."""
    r = sps.rankdata(v)
    pos = np.arange(v.size, dtype=float)
    slope, intercept = np.polyfit(pos, r, 1)
    return r - (slope * pos + intercept)


def effective_n(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    """Rows worth of independent evidence when both series are in time order.

    Slowly drifting series repeat much of their last value, so n rows carry far less
    information than n independent draws. This uses the lag one autocorrelation of the
    detrended ranks, the Bartlett correction for a correlation between two series and the
    usual first order correction for a trend.
    """
    n = float(x.size)
    ry = _lag1(_detrended_ranks(y))
    rx = 1.0 if kind == "trend" else _lag1(_detrended_ranks(x))
    rho = max(rx * ry, 0.0)
    return max(3.0, n * (1 - rho) / (1 + rho))


def partial_on_time(df: pl.DataFrame, a: str, b: str, time_col: str) -> tuple[float, float, int]:
    """Spearman correlation of a and b after removing what each shares with time.

    Two metrics that both rise over the year move together even when neither affects the
    other. Ranking each column and the time column, then taking out the straight-line fit
    on time, leaves only the movement that time does not explain.
    """
    sub = df.select([time_col, a, b]).drop_nulls()
    t = sps.rankdata(_numeric(sub, time_col))
    x = sps.rankdata(_numeric(sub, a))
    y = sps.rankdata(_numeric(sub, b))
    if t.size < 10 or np.ptp(t) == 0:
        return 0.0, 1.0, int(t.size)
    rx = x - np.polyval(np.polyfit(t, x, 1), t)
    ry = y - np.polyval(np.polyfit(t, y, 1), t)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return 0.0, 1.0, int(t.size)
    r, _ = sps.pearsonr(rx, ry)
    n = t.size
    stat = r * np.sqrt((n - 3) / max(1e-12, 1 - r**2))
    return float(r), float(2 * sps.t.sf(abs(stat), n - 3)), int(n)


def num_num(
    df: pl.DataFrame, a: str, b: str, kind: str = "correlation", time_col: str | None = None
) -> Test | None:
    frame = df.sort(time_col, nulls_last=True) if time_col else df
    x, y = _numeric(frame, a), _numeric(frame, b)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if x.size < 10 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rho, p = sps.spearmanr(x, y)
    note = ""
    if time_col:
        n_eff = effective_n(x, y, kind)
        if n_eff < MIN_EFFECTIVE_ROWS:
            return None
        if n_eff < x.size - 1:
            t = rho * np.sqrt((n_eff - 2) / max(1e-12, 1 - rho**2))
            p = max(p, 2 * sps.t.sf(abs(t), n_eff - 2))
            note = f"autocorrelation leaves about {n_eff:.0f} independent rows"
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
        note=note,
    )


SHIFT_CUTS = np.arange(0.1, 0.91, 0.05)  # candidate cut points, as shares of the time order


def shift(df: pl.DataFrame, time_col: str, num: str) -> Test | None:
    """A level shift, where a metric moves to a new level at some point in time.

    Every candidate cut is tried and the largest standardized difference is kept, so its
    p-value is multiplied by the number of cuts. Autocorrelation left after the step is
    taken into account, and a series with fewer than 30 effective rows is not tested.
    """
    frame = df.select(time_col, num).drop_nulls().sort(time_col)
    y = frame[num].cast(pl.Float64).to_numpy()
    n = y.size
    if n < 40 or np.ptp(y) == 0:
        return None
    best_k, best_d, best_t = 0, 0.0, 0.0
    for share in SHIFT_CUTS:
        k = int(n * share)
        before, after = y[:k], y[k:]
        pooled = np.sqrt(
            ((before.size - 1) * before.var(ddof=1) + (after.size - 1) * after.var(ddof=1))
            / (n - 2)
        )
        if pooled == 0:
            continue
        d = (after.mean() - before.mean()) / pooled
        t = d / np.sqrt(1 / before.size + 1 / after.size)
        if abs(d) > abs(best_d):
            best_k, best_d, best_t = k, float(d), float(t)
    if best_k == 0:
        return None
    step = np.where(np.arange(n) < best_k, y[:best_k].mean(), y[best_k:].mean())
    r1 = max(_lag1(y - step), 0.0)
    n_eff = n * (1 - r1) / (1 + r1)
    if n_eff < MIN_EFFECTIVE_ROWS:
        return None
    t_eff = best_t * np.sqrt(n_eff / n)
    p = min(1.0, 2 * sps.t.sf(abs(t_eff), n_eff - 2) * SHIFT_CUTS.size)
    when = frame[time_col][best_k]
    total = float(((y - y.mean()) ** 2).sum())
    return Test(
        id="",
        kind="shift",
        columns=(time_col, num),
        method="step",
        statistic=best_d,
        p_value=float(p),
        effect=abs(best_d),
        effect_name="shift_d",
        n=n,
        note=f"level shift around {str(when)[:10]}, standardized difference {best_d:+.2f}",
        explained=float(((step - y.mean()) ** 2).sum()) / total if total else 0.0,
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
    parts = sub.group_by(cat, maintain_order=True)
    labels = [str(key[0]) for key, _ in parts]
    groups = [g[num].cast(pl.Float64).to_numpy() for _, g in sub.group_by(cat, maintain_order=True)]
    if all(np.ptp(g) == 0 for g in groups):
        return None
    h, p = sps.kruskal(*groups)
    every = np.concatenate(groups)
    total = float(((every - every.mean()) ** 2).sum())
    between = float(sum(g.size * (g.mean() - every.mean()) ** 2 for g in groups))
    top, d = _standout(groups, labels)
    return Test(
        id="",
        kind="segment",
        columns=(cat, num),
        method="kruskal",
        statistic=float(h),
        p_value=float(p),
        effect=abs(d),
        effect_name="max_group_d",
        n=sub.height,
        note=f"{cat}={top} stands out, standardized difference {d:+.2f} against the rest",
        explained=between / total if total else 0.0,
        support=float(next(g.size for g, lab in zip(groups, labels, strict=True) if lab == top))
        / every.size,
    )


def _standout(groups: list[np.ndarray], labels: list[str]) -> tuple[str, float]:
    """The group whose mean differs most from all other rows, in pooled standard deviations.

    An omnibus effect such as epsilon squared dilutes a shift that sits in one group, so the
    effect bar is applied to this group against the rest. The p-value still comes from the
    single omnibus test, so picking the largest group adds no extra test to the ledger.
    """
    best, best_d = labels[0], 0.0
    for i, (g, label) in enumerate(zip(groups, labels, strict=True)):
        others = np.concatenate([x for j, x in enumerate(groups) if j != i])
        if g.size < 2 or others.size < 2:
            continue
        pooled = np.sqrt(
            ((g.size - 1) * g.var(ddof=1) + (others.size - 1) * others.var(ddof=1))
            / (g.size + others.size - 2)
        )
        if pooled == 0:
            continue
        d = (g.mean() - others.mean()) / pooled
        if abs(d) > abs(best_d):
            best, best_d = label, float(d)
    return best, best_d


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


def run(
    df: pl.DataFrame, kind: str, a: str, b: str, min_support: float, time_col: str | None = None
) -> Test | None:
    """Run one test of a known kind on a slice of rows, in time order when one is given."""
    if kind in ("correlation", "trend"):
        return num_num(df, a, b, kind, time_col)
    if kind == "segment":
        return cat_num(df, a, b, min_support)
    if kind == "shift":
        return shift(df, a, b)
    if kind == "association":
        return cat_cat(df, a, b)
    raise ValueError(kind)


def plan(card: DataCard) -> list[tuple[str, str, str]]:
    """Every (kind, a, b) test the scan will run, from column kinds alone."""
    nums = card.testable("numeric")
    cats = card.testable("categorical", "boolean")
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
        tests += [("shift", card.time_column, n) for n in nums]
    return tests
