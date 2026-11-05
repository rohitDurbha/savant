import numpy as np
import polars as pl

from savant.engine import Settings, analyze, split
from savant.grade import grade
from savant.ledger import Ledger, Test, false_discovery_control
from savant.planted import make_task, plant, shuffle_columns
from savant.profile import build_card

FAST = Settings(stability_runs=20)


def test_bh_matches_textbook_example():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    # Step-up at q=0.05: the largest k with p_(k) <= k/10*0.05 is k=2.
    assert false_discovery_control(p, 0.05).tolist() == [True, True] + [False] * 8


def test_by_is_never_less_strict_than_bh():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.05, 40)
    assert (
        false_discovery_control(p, 0.05, dependent=True).sum()
        <= false_discovery_control(p, 0.05).sum()
    )


def test_ledger_counts_every_test_not_just_shown_ones():
    ledger = Ledger()
    for i, p in enumerate([0.001] + [0.5] * 99):
        ledger.add(
            Test(f"t{i}", "correlation", ("a", "b"), "spearman", 0.1, p, 0.1, "abs_rho", 100)
        )
    # Alone, p = 0.001 would pass. Among 100 tests its BH threshold is 0.05 / 100 = 0.0005.
    assert ledger.discoveries(0.05) == []


def test_null_table_reports_nothing():
    task = make_task("null", 3000, 0, seed=11)
    assert analyze(task.table, FAST).reported == []


def test_planted_patterns_are_found_and_nothing_else():
    task = make_task("plant", 3000, 3, seed=7)
    score = grade(task, analyze(task.table, FAST))
    assert score.found == score.planted
    assert score.false_findings == 0


def test_leak_is_flagged_not_reported():
    rng = np.random.default_rng(3)
    y = rng.normal(size=2000)
    df = pl.DataFrame({"revenue": y, "revenue_net": y * 0.9 + rng.normal(0, 0.01, 2000)})
    res = analyze(df, FAST)
    assert res.reported == []
    assert [f.status for f in res.findings] == ["leakage"]


def test_small_file_has_no_confirmed_findings():
    task = make_task("small", 300, 3, seed=5)
    res = analyze(task.table, FAST)
    assert not res.confirmed_split
    assert res.reported == []


def test_shuffle_columns_keeps_values_and_breaks_pairs():
    task = make_task("p", 2000, 3, seed=7)
    shuffled = shuffle_columns(task.table, seed=1)
    assert sorted(shuffled["price"].to_list()) == sorted(task.table["price"].to_list())
    assert analyze(shuffled, FAST).reported == []


def test_functional_dependency_pairs_are_not_tested():
    df = pl.DataFrame({"city": ["a", "b", "c", "d"] * 100, "state": ["x", "x", "y", "y"] * 100})
    card = build_card(df)
    assert ("city", "state") in card.dependencies
    assert analyze(df, FAST).ledger.tests == []


def test_time_split_halves_both_cover_the_full_period():
    task = make_task("t", 2000, 0, seed=1)
    card = build_card(task.table)
    explore, confirm = split(task.table, card, Settings())
    dates = sorted(task.table["order_date"].to_list())
    early, late = dates[len(dates) // 10], dates[9 * len(dates) // 10]
    for part in (explore, confirm):
        assert part["order_date"].min() <= early
        assert part["order_date"].max() >= late


def test_moderate_trend_is_confirmed():
    base = make_task("t", 2000, 0, seed=4).table
    df, p = plant(base, "trend", 0, 0.3, np.random.default_rng(4))
    reported = {f.test.columns for f in analyze(df, FAST).reported}
    assert p.columns in reported


def test_parallel_calibration_finishes_and_matches_serial():
    from savant.calibrate import evaluate
    from savant.planted import make_suite

    tasks = make_suite(2, 2, 800, seed=3)
    assert evaluate(tasks, FAST, workers=2) == evaluate(tasks, FAST, workers=1)


def test_two_trends_on_one_clock_are_derived_not_false():
    from savant.grade import derived_pairs
    from savant.planted import Plant, Task

    plants = [
        Plant("trend", ("order_date", "metric0"), 0.4),
        Plant("trend", ("order_date", "metric1"), 0.4),
    ]
    task = Task("x", pl.DataFrame({"a": [1]}), plants)
    assert derived_pairs(task) == {frozenset({"metric0", "metric1"})}
