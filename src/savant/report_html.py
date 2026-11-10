"""One self-contained HTML page with the cleaning report, ranked insights and charts."""

from __future__ import annotations

import html
import json
from pathlib import Path

import polars as pl

from .charts import _jsonable, chart
from .engine import Settings, analyze
from .insights import Insight, headline, rank
from .longform import detect, split

SCRIPTS = (
    "https://cdn.jsdelivr.net/npm/vega@5.30.0",
    "https://cdn.jsdelivr.net/npm/vega-lite@5.21.0",
    "https://cdn.jsdelivr.net/npm/vega-embed@6.26.0",
)
STYLE = """
body{font:15px/1.5 system-ui,sans-serif;max-width:760px;margin:32px auto;padding:0 16px;
color:#1d232a}
h1{font-size:24px}h2{font-size:18px;margin-top:32px}.insight{margin:24px 0}
.insight h3{font-size:16px;margin:0 0 4px}.related{color:#5b6570;font-size:14px;margin:4px 0 8px}
pre{background:#f4f6f8;padding:12px;overflow-x:auto;font-size:13px}.note{color:#5b6570}
"""
NOTE = (
    "Every finding below survived false discovery control, a check on rows the scan never "
    "saw and a stability check. They are associations in this data, and they do not show "
    "what makes what happen."
)


def _section(insights: list[Insight], df: pl.DataFrame, prefix: str) -> tuple[str, list[str]]:
    parts, scripts = [], []
    if not insights:
        return "<p>No finding survived the checks.</p>", []
    for n, ins in enumerate(insights):
        related = "".join(f"<div>{html.escape(headline(f))}</div>" for f in ins.related[:3])
        spec = chart(ins.lead, df)
        if spec:
            spec.pop("title", None)  # the insight heading above already says it
        div = f"{prefix}{n}"
        parts.append(
            f'<div class="insight"><h3>{html.escape(ins.headline)}</h3>'
            + (f'<div class="related">Related. {related}</div>' if related else "")
            + (f'<div id="{div}"></div>' if spec else "")
            + "</div>"
        )
        if spec:
            payload = json.dumps(_jsonable(spec)).replace("</", "<\\/")
            scripts.append(f'vegaEmbed("#{div}", {payload}, {{"actions": false}});')
    return "".join(parts), scripts


def write_report(
    df: pl.DataFrame,
    out: str | Path,
    title: str,
    cleaning: str = "",
    settings: Settings | None = None,
) -> Path:
    s = settings or Settings()
    body, scripts = [], []
    layout = detect(df)
    if layout is not None:
        body.append(f'<p class="note">This is a {html.escape(layout.describe())}.</p>')
        for i, (measure, (sub, notes)) in enumerate(split(df, layout).items()):
            res = analyze(sub, s)
            ranked = rank(res.findings, limit=4, time_column=res.card.time_column)
            section, js = _section(ranked, sub, f"m{i}c")
            extra = f' <span class="note">({html.escape("; ".join(notes))})</span>' if notes else ""
            body.append(f"<h2>{html.escape(measure)}{extra}</h2>{section}")
            scripts += js
    else:
        res = analyze(df, s)
        ranked = rank(res.findings, time_column=res.card.time_column)
        section, scripts = _section(ranked, df, "c")
        body.append(f"<h2>Insights</h2>{section}")
    if cleaning:
        body.append(f"<h2>Cleaning</h2><pre>{html.escape(cleaning)}</pre>")
    page = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style>"
        + "".join(f"<script src='{u}'></script>" for u in SCRIPTS)
        + f"</head><body><h1>{html.escape(title)}</h1><p class='note'>{NOTE}</p>"
        + "".join(body)
        + "<script>"
        + "\n".join(scripts)
        + "</script></body></html>"
    )
    path = Path(out)
    path.write_text(page)
    return path
