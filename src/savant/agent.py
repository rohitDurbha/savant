"""The plan, code, execute and answer loop.

The model sees the data card and a summary of guarded findings, never the raw rows. It may
only cite numbers through placeholders that point at results its own code emitted.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from . import sandbox
from .engine import Analysis, Settings, analyze
from .llm import Client
from .report import MissingResult, render, ungrounded_numbers

ACTION = re.compile(r"<(Analyze|Code|Answer)>(.*?)</\1>", re.S)

SYSTEM = """You are Savant, a careful data analyst. You work in steps.

Each reply holds an optional <Analyze>...</Analyze> block for reasoning, then exactly one action.
- <Code>python</Code> runs Python. A Polars DataFrame named df is already loaded, with numpy as np.
  Call emit("name", value) for every number you may want to cite. Keep each step short.
- <Answer>text</Answer> ends the analysis. Aim to answer within four steps.

Rules for the answer.
1. Never type a number. Cite every figure as {{rN.name}}, where rN is the step that emitted it.
2. Describe relationships as associations. Never say one column causes another.
3. Prefer the guarded findings listed below, which already passed false discovery control,
   confirmation on held-out rows and a stability check. Treat anything else as exploratory.
4. If the data cannot answer the question, say so plainly.

Example of a valid exchange.
<Code>emit("mean_price", df["price"].mean())
emit("rows", df.height)</Code>
<Execute>{"step": "r1", "ok": true, "emitted": {"mean_price": 41.2, "rows": 2000}}</Execute>
<Answer>The average price is {{r1.mean_price}} across {{r1.rows}} orders.</Answer>"""

FEEDBACK_CHARS = 1500


@dataclass
class Step:
    id: str
    code: str
    ok: bool
    results: dict[str, object]
    error: str = ""


@dataclass
class Run:
    question: str
    answer: str | None = None
    rendered: str | None = None
    steps: list[Step] = field(default_factory=list)
    rounds: int = 0
    grounded: bool = False
    transcript: list[dict[str, str]] = field(default_factory=list)


def window(messages: list[dict[str, str]], keep: int) -> list[dict[str, str]]:
    """The system prompt and opening request, then only the most recent messages."""
    head, tail = messages[:2], messages[2:]
    return head + tail[-keep:] if len(tail) > keep else messages


def findings_text(analysis: Analysis) -> str:
    rows = []
    for f in analysis.findings:
        t = f.test
        rows.append(
            f"- [{f.status}] {t.kind} between {t.columns[0]} and {t.columns[1]}, "
            f"{t.method}, {t.effect_name} {t.effect:.3f}, p {t.p_value:.2g}"
            + (f", {t.note}" if t.note else "")
        )
    if not rows:
        return "No guarded findings. The scan found nothing that survived its checks."
    return "\n".join(rows)


def _allowed_tokens(analysis: Analysis) -> set[str]:
    tokens = {c.name for c in analysis.card.columns}
    for c in analysis.card.columns:
        if c.kind in ("categorical", "boolean"):
            tokens.update(c.examples)
    return tokens


def ask(
    df: pl.DataFrame,
    question: str,
    client: Client,
    settings: Settings | None = None,
    max_rounds: int = 10,
    step_timeout: int = 30,
    keep_messages: int = 8,
) -> Run:
    """Answer one question about df.

    Only the system prompt, the opening request and the latest keep_messages messages are
    sent to the model each round, so memory and latency stay flat as a run goes on.
    """
    analysis = analyze(df, settings)
    run = Run(question)
    allowed = _allowed_tokens(analysis)
    results: dict[str, dict[str, object]] = {}
    messages = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Data card\n{analysis.card.to_text()}\n\nGuarded findings\n"
            f"{findings_text(analysis)}\n\nQuestion\n{question}",
        },
    ]
    with tempfile.TemporaryDirectory(prefix="savant-data-") as tmp:
        data_path = Path(tmp) / "data.parquet"
        df.write_parquet(data_path)
        for round_no in range(1, max_rounds + 1):
            run.rounds = round_no
            if round_no == max_rounds:
                messages.append(
                    {"role": "user", "content": "This is the last step. Reply with <Answer>."}
                )
            reply = client.complete(window(messages, keep_messages))
            messages.append({"role": "assistant", "content": reply})
            actions = [(m.group(1), m.group(2).strip()) for m in ACTION.finditer(reply)]
            action = next(((k, v) for k, v in reversed(actions) if k != "Analyze"), None)
            if action is None:
                messages.append({"role": "user", "content": "Reply with one <Code> or <Answer>."})
                continue
            kind, body = action
            if kind == "Code":
                step_id = f"r{len(run.steps) + 1}"
                ex = sandbox.run(body, data_path, timeout=step_timeout)
                tail = ex.stderr.strip().splitlines()
                error = "timed out" if ex.timed_out else (tail[-1] if tail else "")
                run.steps.append(Step(step_id, body, ex.ok, ex.results, "" if ex.ok else error))
                if ex.ok:
                    results[step_id] = ex.results
                feedback = {
                    "step": step_id,
                    "ok": ex.ok,
                    "emitted": ex.results,
                    "stdout": ex.stdout[-800:],
                    "error": "" if ex.ok else error,
                }
                payload = json.dumps(feedback, default=str)[:FEEDBACK_CHARS]
                messages.append({"role": "user", "content": f"<Execute>{payload}</Execute>"})
                continue
            stray = ungrounded_numbers(body, allowed)
            if stray:
                messages.append(
                    {
                        "role": "user",
                        "content": "Your answer typed these numbers without a placeholder: "
                        + ", ".join(stray[:10])
                        + ". Emit them from code and cite them as {{rN.name}}.",
                    }
                )
                continue
            try:
                run.rendered = render(body, results)
            except MissingResult as e:
                messages.append(
                    {"role": "user", "content": f"The placeholder {e} has no emitted result."}
                )
                continue
            run.answer, run.grounded = body, True
            break
    run.transcript = messages
    return run
