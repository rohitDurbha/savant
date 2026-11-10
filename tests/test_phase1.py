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


def _finding(kind, cols, effect, effect_name, status="confirmed", stat=None, note="", support=None):
    from savant.engine import Finding
    from savant.ledger import Test

    t = Test(
        "t", kind, cols, "m", effect if stat is None else stat, 0.0, effect, effect_name, 1000, note
    )
    t.support = support
    return Finding(t, status)


def test_insights_are_ordered_by_strength_and_coverage():
    from savant.insights import rank

    findings = [
        _finding(
            "segment",
            ("Make", "Range"),
            0.25,
            "max_group_d",
            note="Make=TESLA stands out, standardized difference +0.25 against the rest",
            support=0.02,
        ),
        _finding("correlation", ("Year", "Range"), 0.55, "abs_rho", stat=0.55),
        _finding("association", ("Type", "Eligible"), 0.30, "cramers_v"),
    ]
    insights = rank(findings)
    assert [i.lead.test.columns for i in insights][0] == ("Year", "Range")
    assert insights[-1].lead.test.columns == ("Make", "Range") or len(insights) < 3


def test_findings_sharing_a_column_are_grouped_under_the_strongest():
    from savant.insights import rank

    findings = [
        _finding("correlation", ("Year", "Range"), 0.55, "abs_rho", stat=0.55),
        _finding(
            "segment",
            ("Make", "Range"),
            0.5,
            "max_group_d",
            note="Make=CHEVROLET stands out, standardized difference +0.50 against the rest",
            support=0.1,
        ),
        _finding("association", ("Type", "Eligible"), 0.7, "cramers_v"),
    ]
    insights = rank(findings)
    leads = {i.lead.test.columns for i in insights}
    assert ("Type", "Eligible") in leads and ("Year", "Range") in leads
    year = next(i for i in insights if i.lead.test.columns == ("Year", "Range"))
    assert [f.test.columns for f in year.related] == [("Make", "Range")]


def test_headlines_are_plain_language_and_honest_about_direction_and_cause():
    from savant.insights import headline

    up = _finding("correlation", ("Model Year", "Electric Range"), 0.55, "abs_rho", stat=0.55)
    down = _finding("trend", ("order_date", "Smoking"), 0.4, "abs_rho", stat=-0.4)
    seg = _finding(
        "segment",
        ("Make", "Electric Range"),
        0.53,
        "max_group_d",
        note="Make=CHEVROLET stands out, standardized difference +0.53 against the rest",
        support=0.05,
    )
    near = _finding("segment", ("County", "Postal"), 0.99, "max_group_d", status="leakage")
    assert headline(up) == "Model Year and Electric Range rise together."
    assert headline(down) == "Smoking falls over order_date."
    assert headline(seg) == "Electric Range is higher when Make is CHEVROLET than in other rows."
    assert headline(near) == "County nearly determines Postal, likely by definition."
    assert not any(w in headline(up).lower() for w in ("cause", "drive", "because"))


def test_only_confirmed_and_near_deterministic_findings_become_insights():
    from savant.insights import rank

    findings = [
        _finding("correlation", ("a", "b"), 0.5, "abs_rho", stat=0.5, status="exploratory"),
        _finding("correlation", ("c", "d"), 0.5, "abs_rho", stat=0.5, status="explained_by_time"),
    ]
    assert rank(findings) == []
