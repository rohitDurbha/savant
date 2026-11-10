import numpy as np
import polars as pl

from savant.engine import Settings, analyze
from savant.planted import make_task, plant

FAST = Settings(stability_runs=20)


def _two_trends(seed: int) -> pl.DataFrame:
    base = make_task("t", 3000, 0, seed=seed).table
    rng = np.random.default_rng(seed)
    df, _ = plant(base, "trend", 0, 0.5, rng)
    df, _ = plant(df, "trend", 1, 0.5, rng)
    return df


def test_two_metrics_that_only_share_a_time_trend_are_explained_by_time():
    res = analyze(_two_trends(3), FAST)
    status = {frozenset(f.test.columns): f.status for f in res.findings}
    assert status[frozenset({"metric0", "metric1"})] == "explained_by_time"
    assert status[frozenset({"order_date", "metric0"})] == "confirmed"
    assert frozenset({"metric0", "metric1"}) not in {
        frozenset(f.test.columns) for f in res.reported
    }


def test_a_real_correlation_survives_the_time_check():
    base = make_task("c", 3000, 0, seed=4).table
    df, p = plant(base, "correlation", 0, 0.4, np.random.default_rng(4))
    res = analyze(df, FAST)
    assert p.columns in {f.test.columns for f in res.reported}
