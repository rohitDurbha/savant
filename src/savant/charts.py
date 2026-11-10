"""A Vega-Lite chart for each kind of finding, and a linter for misleading designs.

Each chart is built from the data by code, with the finding's own sentence as its title.
The linter applies the rules from the reliability research, such as bars that must start
at zero and no dual axes, and it is used on every chart before it reaches a report.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

from .engine import Finding
from .insights import headline

SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
SIZE = {"width": 560, "height": 300}
ACCENT, QUIET = "#2f6fdf", "#9aa3ad"
MAX_POINTS = 1500
MAX_LEVELS = 12
STANDOUT = re.compile(r"^(.*?)=(.*) stands out")
SHIFT = re.compile(r"level shift around ([0-9-]+)")


def _base(f: Finding) -> dict:
    return {"$schema": SCHEMA, "title": headline(f), **SIZE}


def _correlation(f: Finding, df: pl.DataFrame) -> dict:
    a, b = f.test.columns
    sub = df.select(a, b).drop_nulls()
    if sub.height > MAX_POINTS:
        sub = sub.sample(MAX_POINTS, seed=0)
    enc = {
        "x": {"field": a, "type": "quantitative", "scale": {"zero": False}},
        "y": {"field": b, "type": "quantitative", "scale": {"zero": False}},
    }
    return {
        **_base(f),
        "data": {"values": sub.to_dicts()},
        "layer": [
            {"mark": {"type": "point", "opacity": 0.3, "color": QUIET}, "encoding": enc},
            {
                "mark": {"type": "line", "color": ACCENT, "strokeWidth": 2},
                "transform": [{"regression": b, "on": a}],
                "encoding": enc,
            },
        ],
    }


def _monthly(df: pl.DataFrame, time: str, num: str) -> pl.DataFrame:
    return (
        df.select(time, num)
        .drop_nulls()
        .with_columns(pl.col(time).cast(pl.Date).dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.col(num).mean().alias("mean"), pl.col(num).std().alias("sd"), pl.len().alias("n"))
        .with_columns(
            (pl.col("mean") - 1.96 * pl.col("sd") / pl.col("n").sqrt()).alias("low"),
            (pl.col("mean") + 1.96 * pl.col("sd") / pl.col("n").sqrt()).alias("high"),
            pl.col("month").cast(pl.String),
        )
        .sort("month")
    )


def _over_time(f: Finding, df: pl.DataFrame) -> dict:
    time, num = f.test.columns
    data = _monthly(df, time, num)
    x = {"field": "month", "type": "temporal", "title": time}
    layers = [
        {
            "mark": {"type": "errorband", "color": ACCENT, "opacity": 0.2},
            "encoding": {
                "x": x,
                "y": {"field": "low", "type": "quantitative", "title": f"{num}, monthly mean"},
                "y2": {"field": "high"},
            },
        },
        {
            "mark": {"type": "line", "color": ACCENT, "point": True},
            "encoding": {
                "x": x,
                "y": {"field": "mean", "type": "quantitative", "scale": {"zero": False}},
            },
        },
    ]
    m = SHIFT.search(f.test.note)
    if f.test.kind == "shift" and m:
        layers.append(
            {
                "data": {"values": [{"when": m.group(1)}]},
                "mark": {"type": "rule", "strokeDash": [4, 4], "color": "#444"},
                "encoding": {"x": {"field": "when", "type": "temporal"}},
            }
        )
    return {**_base(f), "data": {"values": data.to_dicts()}, "layer": layers}


def _segment(f: Finding, df: pl.DataFrame) -> dict:
    cat, num = f.test.columns
    stats = (
        df.select(cat, num)
        .drop_nulls()
        .group_by(cat)
        .agg(pl.col(num).mean().alias("mean"), pl.col(num).std().alias("sd"), pl.len().alias("n"))
        .filter(pl.col("n") >= 5)
        .sort("n", descending=True)
        .head(MAX_LEVELS)
        .with_columns(
            (pl.col("mean") - 1.96 * pl.col("sd") / pl.col("n").sqrt()).alias("low"),
            (pl.col("mean") + 1.96 * pl.col("sd") / pl.col("n").sqrt()).alias("high"),
            pl.col(cat).cast(pl.String).alias("group"),
        )
    )
    m = STANDOUT.match(f.test.note)
    standout = m.group(2) if m else None
    stats = stats.with_columns((pl.col("group") == standout).alias("standout"))
    y = {
        "field": "group",
        "type": "nominal",
        "title": cat,
        "sort": {"field": "mean", "order": "descending"},
    }
    color = {"condition": {"test": "datum.standout", "value": ACCENT}, "value": QUIET}
    return {
        **_base(f),
        "data": {
            "values": stats.select("group", "mean", "low", "high", "n", "standout").to_dicts()
        },
        "layer": [
            {
                "mark": {"type": "errorbar", "ticks": True},
                "encoding": {
                    "y": y,
                    "x": {
                        "field": "low",
                        "type": "quantitative",
                        "title": f"{num}, mean with 95% interval",
                    },
                    "x2": {"field": "high"},
                    "color": color,
                },
            },
            {
                "mark": {"type": "point", "filled": True, "size": 70},
                "encoding": {
                    "y": y,
                    "x": {"field": "mean", "type": "quantitative", "scale": {"zero": False}},
                    "color": color,
                },
            },
        ],
    }


def _association(f: Finding, df: pl.DataFrame) -> dict:
    a, b = f.test.columns
    sub = (
        df.select(a, b)
        .drop_nulls()
        .with_columns(pl.col(a).cast(pl.String), pl.col(b).cast(pl.String))
    )
    top_a = sub[a].value_counts(sort=True)[a].head(MAX_LEVELS).to_list()
    top_b = sub[b].value_counts(sort=True)[b].head(MAX_LEVELS).to_list()
    sub = sub.filter(pl.col(a).is_in(top_a) & pl.col(b).is_in(top_b))
    shares = (
        sub.group_by(a, b)
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum().over(a)).alias("share"))
        .sort(a, b)
    )
    return {
        **_base(f),
        "data": {"values": shares.to_dicts()},
        "mark": "rect",
        "encoding": {
            "x": {"field": b, "type": "nominal", "sort": "ascending", "axis": {"labelAngle": 0}},
            "y": {"field": a, "type": "nominal", "sort": "ascending"},
            "color": {
                "field": "share",
                "type": "quantitative",
                "title": f"share of {a} rows",
                "scale": {"scheme": "blues"},
            },
        },
    }


BUILDERS = {
    "correlation": _correlation,
    "trend": _over_time,
    "shift": _over_time,
    "segment": _segment,
    "association": _association,
}


def chart(f: Finding, df: pl.DataFrame) -> dict | None:
    """A chart for a finding, or None for a kind that has no chart yet."""
    build = BUILDERS.get(f.test.kind)
    spec = build(f, df) if build else None
    if spec is not None:
        problems = lint(spec)
        if problems:
            raise ValueError(f"chart for {f.test.columns} breaks lint rules: {problems}")
    return spec


def _views(spec: dict) -> list[dict]:
    views = [spec]
    for layer in spec.get("layer", []):
        views += _views(layer)
    return views


def lint(spec: dict) -> list[str]:
    """Problems that would make a chart mislead, from the reliability research."""
    problems = []
    if spec.get("layer") and spec.get("resolve", {}).get("scale", {}).get("y") == "independent":
        problems.append("dual axes show two measures on different scales in one chart")
    for view in _views(spec):
        mark = view.get("mark")
        mark = mark.get("type") if isinstance(mark, dict) else mark
        enc = view.get("encoding", {})
        if mark in ("bar", "area"):
            for ch in ("x", "y"):
                e = enc.get(ch, {})
                if e.get("type") == "quantitative" and e.get("scale", {}).get("zero") is False:
                    problems.append(f"a {mark} axis must start at zero")
                if e.get("type") in ("nominal", "ordinal") and "sort" not in e:
                    problems.append("categories need a meaningful order, so sort them")
        if mark == "arc":
            values = (view.get("data") or spec.get("data") or {}).get("values", [])
            field = enc.get("color", {}).get("field")
            if field and len({v.get(field) for v in values}) > 5:
                problems.append("a pie with more than 5 slices is hard to read, so use bars")
    return problems


def _jsonable(spec: dict) -> dict:
    """Plain Python values only, so the spec can be written as JSON."""

    def fix(v):
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return v

    return fix(spec)
