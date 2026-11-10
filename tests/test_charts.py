import json

import numpy as np

from savant.charts import chart, lint
from savant.engine import Settings, analyze
from savant.planted import make_task, plant

FAST = Settings(stability_runs=20)


def _findings():
    base = make_task("c", 3000, 0, seed=8).table
    rng = np.random.default_rng(8)
    df = base
    for kind in ("correlation", "segment", "trend", "association", "shift"):
        df, _ = plant(df, kind, 0 if kind != "shift" else 9, 0.6, rng)
    return df, analyze(df, FAST).reported


def test_every_reported_kind_gets_a_chart_that_passes_the_linter():
    df, reported = _findings()
    kinds = {f.test.kind for f in reported}
    assert {"correlation", "segment", "trend", "association", "shift"} <= kinds
    for f in reported:
        spec = chart(f, df)
        assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite/v5")
        assert lint(spec) == [], (f.test.kind, lint(spec))
        assert len(json.dumps(spec)) < 400_000


def test_segment_chart_shows_intervals_and_highlights_the_standout_group():
    df, reported = _findings()
    seg = next(f for f in reported if f.test.kind == "segment")
    text = json.dumps(chart(seg, df))
    assert "errorbar" in text and "gamma" in text


def test_linter_catches_misleading_designs():
    bad_bar = {
        "mark": "bar",
        "encoding": {"y": {"field": "v", "type": "quantitative", "scale": {"zero": False}}},
    }
    assert any("zero" in p for p in lint(bad_bar))
    dual = {
        "layer": [{"mark": "line"}, {"mark": "line"}],
        "resolve": {"scale": {"y": "independent"}},
    }
    assert any("dual" in p for p in lint(dual))
    pie = {
        "mark": "arc",
        "data": {"values": [{"k": i} for i in range(8)]},
        "encoding": {"theta": {"field": "v"}, "color": {"field": "k"}},
    }
    assert any("slices" in p for p in lint(pie))
    unsorted = {
        "mark": "bar",
        "encoding": {
            "x": {"field": "g", "type": "nominal"},
            "y": {"field": "v", "type": "quantitative"},
        },
    }
    assert any("order" in p for p in lint(unsorted))


def test_html_report_holds_insights_charts_and_cleaning_notes(tmp_path):
    from savant.report_html import write_report

    df, _ = _findings()
    out = tmp_path / "report.html"
    write_report(df, out, title="Planted demo", cleaning="cleaning report, no changes were needed")
    html = out.read_text()
    assert "Planted demo" in html and "vegaEmbed" in html and "cleaning report" in html
    assert html.count('class="insight"') >= 4
    assert "cause" not in html.lower()
