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
from .profile import DataCard, build_card
from .report import MissingResult, render, ungrounded_numbers

ACTION = re.compile(r"<(Analyze|Code|Answer)>(.*?)</\1>", re.S)

SYSTEM = """You are Savant, a careful data analyst. You work in steps.

Each reply holds an optional <Analyze>...</Analyze> block for reasoning, then exactly one action.
- <Code>python</Code> runs Python. A pandas DataFrame named df is already loaded, with pandas as pd
  and numpy as np.
  Call emit("name", value) for every number you may want to cite. Keep each step short.
- <Answer>text</Answer> ends the analysis. Aim to answer within four steps.

Rules for the answer.
1. Never type a number. Cite every figure as {{rN.name}}, where rN is the step that emitted it.
2. Describe relationships as associations. Never say one column causes another.
3. {findings_rule}
4. If the data cannot answer the question, say so plainly.

Example of a valid exchange.
<Code>emit("mean_price", df["price"].mean())
emit("rows", len(df))</Code>
<Execute>{"step": "r1", "ok": true, "emitted": {"mean_price": 41.2, "rows": 2000}}</Execute>
<Answer>The average price is {{r1.mean_price}} across {{r1.rows}} orders.</Answer>"""

FEEDBACK_CHARS = 1500
GUIDED_RULE = """Prefer the guarded findings listed below, which already passed false discovery
   control, confirmation on held-out rows and a stability check. Treat the rest as exploratory."""
BLIND_RULE = """Report a relationship only when your own code shows clear evidence for it, and say
   nothing when the evidence is weak. Reporting noise as a finding is the worst mistake."""
PAIR = re.compile(r"^\s*PAIR:\s*(.+?)\s*\|\s*(.+?)\s*$", re.I | re.M)


def parse_pairs(text: str, columns: list[str]) -> set[frozenset[str]]:
    """Column pairs named on PAIR lines, matched to real columns ignoring case."""
    by_lower = {c.lower(): c for c in columns}
    pairs = set()
    for a, b in PAIR.findall(text or ""):
        ca, cb = by_lower.get(a.strip().lower()), by_lower.get(b.strip().lower())
        if ca and cb and ca != cb:
            pairs.add(frozenset({ca, cb}))
    return pairs


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
    verified_pairs: set[frozenset[str]] = field(default_factory=set)
    unverified_pairs: set[frozenset[str]] = field(default_factory=set)
    transcript: list[dict[str, str]] = field(default_factory=list)


def window(messages: list[dict[str, str]], keep: int) -> list[dict[str, str]]:
    """The system prompt and opening request, then only the most recent messages."""
    head, tail = messages[:2], messages[2:]
    return head + tail[-keep:] if len(tail) > keep else messages


LABELS = {
    "confirmed": "confirmed",
    "leakage": "near-deterministic, likely a definition or a leak",
}


def findings_text(analysis: Analysis, limit: int = 15) -> str:
    """The strongest guarded findings for the prompt, confirmed ones first.

    Exploratory findings did not pass every check, so the model never sees them. A long
    list would crowd the prompt and run a small GPU out of memory, so the rest are only
    counted.
    """

    def strongest(status: str) -> list:
        return sorted(
            (f for f in analysis.findings if f.status == status), key=lambda f: -f.test.effect
        )

    near = strongest("leakage")[:3]  # always leave room to say which columns define others
    shown = strongest("confirmed")[: limit - len(near)] + near
    rows = []
    for f in shown:
        t = f.test
        rows.append(
            f"- [{LABELS[f.status]}] {t.kind} between {t.columns[0]} and {t.columns[1]}, "
            f"{t.method}, {t.effect_name} {t.effect:.3f}, p {t.p_value:.2g}"
            + (f", {t.note}" if t.note else "")
        )
    if not rows:
        return "No guarded findings. The scan found nothing that survived its checks."
    hidden = len(analysis.findings) - len(rows)
    if hidden:
        rows.append(f"({hidden} more findings are not shown, weaker or unconfirmed.)")
    return "\n".join(rows)


def _allowed_tokens(card: DataCard) -> set[str]:
    tokens = {c.name for c in card.columns}
    for c in card.columns:
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
    show_findings: bool = True,
) -> Run:
    """Answer one question about df.

    Only the system prompt, the opening request and the latest keep_messages messages are
    sent to the model each round, so memory and latency stay flat as a run goes on.
    With show_findings off, the model gets only the data card and must find patterns
    with its own code, which measures the model apart from the engine.
    """
    guarded: set[frozenset[str]] = set()
    if show_findings:
        analysis = analyze(df, settings)
        card = analysis.card
        context = f"Guarded findings\n{findings_text(analysis)}\n\n"
        guarded = {
            frozenset(f.test.columns)
            for f in analysis.findings
            if f.status in ("confirmed", "leakage")
        }
    else:
        card, context = build_card(df), ""
    run = Run(question)
    allowed = _allowed_tokens(card)
    results: dict[str, dict[str, object]] = {}
    last_error = ""
    messages = [
        {
            "role": "system",
            "content": SYSTEM.replace(
                "{findings_rule}", GUIDED_RULE if show_findings else BLIND_RULE
            ),
        },
        {
            "role": "user",
            "content": f"Data card\n{card.to_text()}\n\n{context}Question\n{question}",
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
            loose = ACTION.sub("", reply).strip()
            if action is None and loose and "<" not in loose:
                # Small models often drop the tags on their final reply. Plain text with no
                # code is treated as the answer, and it still has to pass grounding.
                action = ("Answer", loose)
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
                repeated = not ex.ok and error and error == last_error
                last_error = "" if ex.ok else error
                feedback = {
                    "step": step_id,
                    "ok": ex.ok,
                    "emitted": ex.results,
                    "stdout": ex.stdout[-800:],
                    "error": "" if ex.ok else error,
                }
                if repeated:
                    feedback["hint"] = (
                        "This is the same error as your last step. Change approach, "
                        "for example use plain pandas methods such as groupby, mean and corr."
                    )
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
            if show_findings:
                claimed = parse_pairs(body, df.columns)
                run.verified_pairs = claimed & guarded
                run.unverified_pairs = claimed - guarded
                if run.unverified_pairs:
                    listed = "; ".join(
                        " | ".join(sorted(p)) for p in sorted(run.unverified_pairs, key=sorted)
                    )
                    run.rendered += (
                        "\n\nNote. These pairs were not confirmed by the checks and may be "
                        f"noise: {listed}."
                    )
            break
    run.transcript = messages
    return run
