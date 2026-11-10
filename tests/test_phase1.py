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


def _long_table(seed: int = 0) -> pl.DataFrame:
    """Three measures by state, year and sex. Obesity differs by state, smoking falls over
    the years, and sleep has no pattern. Each measure comes as a crude and an adjusted
    rate on different scales, with confidence limits and a footnote, like CDC tables."""
    rng = np.random.default_rng(seed)
    states = [f"S{i:02d}" for i in range(40)]
    state_effect = dict(zip(states, rng.normal(0, 1, 40), strict=True))
    rows = []
    for state in states:
        for year in range(2010, 2022):
            for sex in ("Male", "Female"):
                for measure in ("Obesity", "Smoking", "Sleep"):
                    base = {"Obesity": 30.0, "Smoking": 20.0, "Sleep": 35.0}[measure]
                    v = base + rng.normal(0, 1.5)
                    if measure == "Obesity":
                        v += 3 * state_effect[state]
                    if measure == "Smoking":
                        v -= 0.6 * (year - 2010)
                    for vtype, scale in (
                        ("Crude Prevalence", 1.0),
                        ("Age-adjusted Prevalence", 1.4),
                    ):
                        rows.append(
                            {
                                "YearStart": year,
                                "LocationAbbr": state,
                                "Stratification1": sex,
                                "Question": measure,
                                "DataValueType": vtype,
                                "DataValue": round(v * scale, 1),
                                "LowConfidenceLimit": round(v * scale - 2, 1),
                                "HighConfidenceLimit": round(v * scale + 2, 1),
                                "DataValueFootnoteSymbol": rng.choice(["", "~"]),
                            }
                        )
    return pl.DataFrame(rows)


def test_long_table_is_detected():
    from savant.longform import detect

    layout = detect(_long_table())
    assert layout is not None
    assert (layout.measure, layout.value) == ("Question", "DataValue")
    assert "DataValueType" in layout.variants
    assert {"LowConfidenceLimit", "HighConfidenceLimit", "DataValueFootnoteSymbol"} <= set(
        layout.ignored
    )


def test_wide_tables_are_not_mistaken_for_long_ones():
    from savant.longform import detect

    assert detect(make_task("w", 2000, 3, seed=1).table) is None


def test_each_measure_is_analyzed_on_its_own_with_one_value_type():
    from savant.longform import analyze_long

    results = analyze_long(_long_table(), FAST)
    found = {m: {frozenset(f.test.columns) for f in a.reported} for m, a in results.items()}
    assert frozenset({"LocationAbbr", "DataValue"}) in found["Obesity"]
    assert frozenset({"YearStart", "DataValue"}) in found["Smoking"]
    assert found["Sleep"] == set()
    assert not any("DataValueType" in pair for pairs in found.values() for pair in pairs)


def test_the_model_sees_findings_by_measure_for_a_long_table():
    from savant.agent import ask

    class Scripted:
        def __init__(self):
            self.seen = []

        def complete(self, messages):
            self.seen.append(messages[-1]["content"])
            return "<Answer>PAIR: LocationAbbr | DataValue</Answer>"

    client = Scripted()
    run = ask(_long_table(), "What matters?", client, settings=FAST)
    assert "long table" in client.seen[0] and "Measure Obesity" in client.seen[0]
    assert run.verified_pairs == {frozenset({"LocationAbbr", "DataValue"})}
