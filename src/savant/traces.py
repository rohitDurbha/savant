"""Training conversations written by a scripted analyst and kept only when verified.

The analyst sees the same opening the model sees, writes pandas code that measures each
guarded finding, runs it in the sandbox, and answers with placeholders and PAIR lines. A
conversation is kept only when its code ran, every number in the answer is a placeholder
that resolves, and every pair it names is true in the planted truth of the table. Tables come
from development seeds only, and the bench seed range is refused.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from . import sandbox
from .agent import _allowed_tokens, execute_message, parse_pairs, prepare
from .engine import Finding, Settings
from .grade import derived_pairs
from .planted import Task, hard_null_table, make_task
from .report import MissingResult, render, ungrounded_numbers

BENCH_SEED = 20261105
BENCH_SPAN = 100_000  # the bench uses seeds from BENCH_SEED up to about BENCH_SEED + 60,000
PAIR_RULE = (
    "End your answer with one line per relationship in the form PAIR: column_a | column_b, "
    "or the single line PAIR: none."
)
PAIR_QUESTION = (
    "What are the most important relationships between columns in this data? "
    "Report only what the data supports. " + PAIR_RULE
)
PAIR_QUESTIONS = (
    PAIR_QUESTION,
    "Which columns in this table are related to each other? Leave out anything the checks "
    "did not support. " + PAIR_RULE,
    "Summarize the relationships worth knowing about in this dataset. " + PAIR_RULE,
)
STANDOUT = re.compile(r"^(.*?)=(.*) stands out")
LIMIT = 15  # the same number of findings the agent shows


def _slug(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()[:24] or "col"


def _shown(findings: list[Finding]) -> list[Finding]:
    """The confirmed findings the model is shown, strongest first, as findings_text does."""

    def strongest(status: str) -> list[Finding]:
        return sorted((f for f in findings if f.status == status), key=lambda f: -f.test.effect)

    extra = len(strongest("leakage")[:3]) + len(
        strongest("explained_by_time")[:2] + strongest("zero_artifact")[:2]
    )
    return strongest("confirmed")[: LIMIT - extra]


def _measure(f: Finding) -> tuple[list[str], str]:
    """Code lines that measure one confirmed finding, and the sentence that cites them."""
    t = f.test
    a, b = t.columns
    rising = (t.statistic or 0) >= 0
    if t.kind == "correlation":
        name = f"{_slug(a)}_{_slug(b)}_rho"
        code = [f'emit("{name}", df[{a!r}].corr(df[{b!r}], method="spearman"))']
        way = "rise together" if rising else "move in opposite directions"
        return code, f"{a} and {b} {way}, with a rank correlation of {{{{r1.{name}}}}}."
    if t.kind == "trend":
        name = f"{_slug(b)}_time_rho"
        code = [
            f'emit("{name}", df[{b!r}].corr(pd.to_datetime(df[{a!r}]).rank(), method="spearman"))'
        ]
        way = "rises" if rising else "falls"
        return code, f"{b} {way} over {a}, with a rank correlation of {{{{r1.{name}}}}} with time."
    if t.kind == "shift":
        m = re.search(r"level shift around ([0-9-]+)", t.note)
        if not m:
            return [], ""
        cut, before, after = m.group(1), f"{_slug(b)}_before", f"{_slug(b)}_after"
        code = [
            f"when = pd.to_datetime(df[{a!r}])",
            f'emit("{before}", df.loc[when < {cut!r}, {b!r}].mean())',
            f'emit("{after}", df.loc[when >= {cut!r}, {b!r}].mean())',
        ]
        return code, (
            f"{b} moves to a new level partway through {a}, from an average of "
            f"{{{{r1.{before}}}}} before the change to {{{{r1.{after}}}}} after it."
        )
    if t.kind == "segment":
        m = STANDOUT.match(t.note)
        if not m:
            return [], ""
        group = m.group(2)
        inside, rest = f"{_slug(b)}_{_slug(group)}", f"{_slug(b)}_other_{_slug(a)}"
        code = [
            f"is_group = df[{a!r}].astype(str) == {group!r}",
            f'emit("{inside}", df.loc[is_group, {b!r}].mean())',
            f'emit("{rest}", df.loc[~is_group, {b!r}].mean())',
        ]
        return code, (
            f"{b} differs by {a}. Rows where {a} is {group} average {{{{r1.{inside}}}}}, "
            f"against {{{{r1.{rest}}}}} for the other groups."
        )
    if t.kind == "association":
        name = f"{_slug(a)}_{_slug(b)}_v"
        code = [
            f"ct = pd.crosstab(df[{a!r}], df[{b!r}])",
            f'emit("{name}", (chi2_contingency(ct)[0] / (ct.values.sum() * (min(ct.shape) - 1))) ** 0.5)',  # noqa: E501
        ]
        return code, f"{a} and {b} are associated, with a Cramér's V of {{{{r1.{name}}}}}."
    return [], ""


def _caveats(findings: list[Finding]) -> list[str]:
    out = []
    for f in findings:
        a, b = f.test.columns
        if f.status == "leakage":
            out.append(
                f"{a} and {b} are almost copies of each other, so one is likely defined from "
                "the other. I leave that pair out of the relationships."
            )
        elif f.status == "explained_by_time":
            out.append(
                f"{a} and {b} only move together because both change over time, so I leave "
                "that pair out as well."
            )
        elif f.status == "zero_artifact":
            out.append(
                f"The link between {a} and {b} rests on zeros that likely mean unknown, so I "
                "do not report it."
            )
    return out[:4]


def _execute(code: str, df: pl.DataFrame) -> tuple[sandbox.Execution, str]:
    with tempfile.TemporaryDirectory(prefix="savant-trace-") as tmp:
        path = Path(tmp) / "data.parquet"
        df.write_parquet(path)
        ex = sandbox.run(code, path, timeout=60)
    tail = ex.stderr.strip().splitlines()
    return ex, ("timed out" if ex.timed_out else (tail[-1] if tail else ""))


def _finish(
    opening_messages: list[dict[str, str]],
    analyze_text: str,
    code: str,
    answer: str,
    df: pl.DataFrame,
    allowed: set[str],
) -> dict | None:
    """Run the code, check the answer is grounded, and return the conversation."""
    ex, error = _execute(code, df)
    if not ex.ok:
        return None
    results = {"r1": ex.results}
    if ungrounded_numbers(answer, allowed):
        return None
    try:
        rendered = render(answer, results)
    except MissingResult:
        return None
    if "None" in rendered or "nan" in rendered.split():
        return None  # a measure came back empty, which would teach a broken answer
    messages = [
        *opening_messages,
        {"role": "assistant", "content": f"<Analyze>{analyze_text}</Analyze>\n<Code>{code}</Code>"},
        execute_message("r1", ex, error),
        {"role": "assistant", "content": f"<Answer>{answer}</Answer>"},
    ]
    return {"messages": messages, "results": results, "rendered": rendered}


def build_trace(task: Task, question: str, settings: Settings | None = None) -> dict | None:
    """A verified relationship conversation for one table, or None when it fails a check."""
    opening = prepare(task.table, question, settings)
    findings = opening.analysis.findings if opening.analysis else []
    shown = _shown(findings)
    lines, sentences, pairs = [], [], []
    for f in shown:
        code, sentence = _measure(f)
        if not code or f.test.columns in pairs:
            continue  # a trend and a shift on the same pair are told once, strongest first
        lines += code
        sentences.append(sentence)
        pairs.append(f.test.columns)
    caveats = _caveats(findings)
    if pairs:
        imports = (
            ["from scipy.stats import chi2_contingency"] if "chi2_contingency" in str(lines) else []
        )
        code = "\n".join(imports + lines)
        plan = (
            "The guarded findings list relationships that passed every check. I will measure "
            "each one in code so the answer can cite real numbers."
        )
        these = "These relationships" if len(pairs) > 1 else "This relationship"
        answer = (
            f"{these} passed false discovery control, a check on held-out rows and a stability "
            "check. They describe associations in this data and do not show cause.\n\n"
            + "\n".join(f"- {s}" for s in sentences)
            + ("\n\n" + " ".join(caveats) if caveats else "")
            + "\n\n"
            + "\n".join(f"PAIR: {a} | {b}" for a, b in pairs)
        )
    else:
        code = 'emit("rows", len(df))\nemit("columns", df.shape[1])'
        plan = (
            "No guarded finding passed every check. I will confirm the size of the table and "
            "report that nothing held up."
        )
        answer = (
            "Across {{r1.rows}} rows and {{r1.columns}} columns, no relationship survived false "
            "discovery control, the check on held-out rows and the stability check, so I report "
            "none." + ("\n\n" + " ".join(caveats) if caveats else "") + "\n\nPAIR: none"
        )
    trace = _finish(opening.messages, plan, code, answer, task.table, _allowed_tokens(opening.card))
    if trace is None:
        return None
    claimed = parse_pairs(answer, task.table.columns)
    real = {frozenset(p.columns) for p in task.plants if p.kind != "leak"}
    leaks = {frozenset(p.columns) for p in task.plants if p.kind == "leak"}
    if claimed - real - leaks - derived_pairs(task):
        return None  # a false pair would teach the model to report noise
    # A weak plant the engine missed is not claimed, which is honest given the evidence,
    # so the trace is kept and its recall is recorded.
    trace["recall"] = len(claimed & real) / len(real) if real else None
    trace.update(id=task.id, family="relationships", truth=task.truth())
    return trace


def lookup_trace(
    df: pl.DataFrame, cat: str, num: str, settings: Settings | None = None
) -> dict | None:
    """A verified answer to which level of cat has the highest average num."""
    question = f"Which {cat} has the highest average {num}?"
    opening = prepare(df, question, settings)
    findings = opening.analysis.findings if opening.analysis else []
    confirmed = any(f.status == "confirmed" and set(f.test.columns) == {cat, num} for f in findings)
    code = "\n".join(
        [
            f"means = df.groupby({cat!r})[{num!r}].mean().sort_values(ascending=False)",
            'emit("top_group", means.index[0])',
            'emit("top_mean", means.iloc[0])',
            'emit("next_group", means.index[1])',
            'emit("next_mean", means.iloc[1])',
            'emit("groups", len(means))',
        ]
    )
    plan = f"I will average {num} within each {cat} and compare the two highest groups."
    check = (
        f"The guarded scan also confirmed that {num} differs by {cat}, so the gap held up on "
        "held-out rows."
        if confirmed
        else f"The guarded scan did not confirm a difference in {num} across {cat}, so the gap "
        "may be noise and should not be read as a real pattern."
    )
    answer = (
        f"{{{{r1.top_group}}}} has the highest average {num} at {{{{r1.top_mean}}}}, ahead of "
        f"{{{{r1.next_group}}}} at {{{{r1.next_mean}}}}, across {{{{r1.groups}}}} groups of "
        f"{cat}. {check}"
    )
    trace = _finish(opening.messages, plan, code, answer, df, _allowed_tokens(opening.card))
    if trace is None:
        return None
    means = (
        df.select(cat, num)
        .drop_nulls()
        .group_by(cat)
        .agg(pl.col(num).mean())
        .sort(num, descending=True)
    )
    if means.height < 2 or str(means[cat][0]) != str(trace["results"]["r1"]["top_group"]):
        return None
    trace.update(family="lookup", confirmed=confirmed)
    return trace


def _check_seeds(seed: int, n: int) -> None:
    if seed < BENCH_SEED + BENCH_SPAN and seed + n > BENCH_SEED:
        raise ValueError(
            f"seeds {seed} to {seed + n} overlap the bench range starting at {BENCH_SEED}"
        )


def _one(args: tuple[int, int, int]) -> list[dict]:
    i, seed, rows = args
    rng = np.random.default_rng(seed)
    draw = rng.random()
    if draw < 0.2:
        task = make_task(f"null-{seed}", rows, 0, seed)
    elif draw < 0.35:
        task = Task(f"hard-null-{seed}", hard_null_table(rng, rows), [])
    else:
        task = make_task(f"plant-{seed}", rows, int(rng.integers(1, 4)), seed)
    order = list(task.table.columns)
    rng.shuffle(order)
    task = Task(task.id, task.table.select(order), task.plants)
    question = PAIR_QUESTIONS[int(rng.integers(len(PAIR_QUESTIONS)))]
    out = []
    trace = build_trace(task, question)
    out.append(
        {"seed": seed, "family": "relationships", "kept": trace is not None, **(trace or {})}
    )
    cats = [c for c in task.table.columns if task.table[c].dtype == pl.String]
    nums = [
        c for c in task.table.columns if task.table[c].dtype.is_float() and not c.endswith("_id")
    ]
    if cats and nums and rng.random() < 0.5:
        cat, num = cats[int(rng.integers(len(cats)))], nums[int(rng.integers(len(nums)))]
        lk = lookup_trace(task.table, cat, num)
        out.append({"seed": seed, "family": "lookup", "kept": lk is not None, **(lk or {})})
    return out


def make_traces(
    n: int, seed: int = 3_000_000, rows: tuple[int, int] = (800, 3000), workers: int = 4
) -> list[dict]:  # noqa: E501
    """Attempts on n development tables, with each record marked kept or not."""
    _check_seeds(seed, n)
    rng = np.random.default_rng(seed)
    jobs = [(i, seed + i, int(rng.integers(rows[0], rows[1]))) for i in range(n)]
    if workers <= 1:
        batches = [_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            batches = list(pool.map(_one, jobs))
    return [r for b in batches for r in b]


def write_traces(records: list[dict], out: Path) -> dict:
    """Write kept conversations as JSON Lines and return counts for the plan."""
    kept = [r for r in records if r["kept"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in kept:
            f.write(
                json.dumps(
                    {
                        "id": r.get("id", r["seed"]),
                        "family": r["family"],
                        "messages": r["messages"],
                    },
                    default=str,
                )
                + "\n"
            )  # noqa: E501
    summary: dict[str, object] = {"attempted": len(records), "kept": len(kept)}
    for fam in ("relationships", "lookup"):
        rs = [r for r in records if r["family"] == fam]
        summary[fam] = {"attempted": len(rs), "kept": sum(r["kept"] for r in rs)}
    rel = [r for r in kept if r["family"] == "relationships"]
    summary["relationship_answers_with_pair_none"] = sum(
        "PAIR: none" in r["messages"][-1]["content"] for r in rel
    )
    return summary
