"""Training conversations written by the model itself and kept only when the checks agree.

This follows rejection sampling as in ReST-EM. The base model answers each development
table several times, the planted truth and the guarded engine judge every run, and only
runs that name every confirmed pair and no false pair are kept. Because the kept turns are
in the model's own words, training stays close to how it already writes, which avoids the
template memorization of the scripted pilots. Failed code steps, empty turns and rejected
answers stay in the context but carry no labels, so the model never learns to repeat them.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

from .agent import Client, Opening, Run, ask, parse_pairs, prepare
from .grade import derived_pairs
from .planted import Task
from .traces import check_seeds, dev_table


def dev_tasks(
    n: int, seed: int = 4_000_000, rows: tuple[int, int] = (800, 3000)
) -> list[tuple[Task, str]]:
    """n development tables with their questions, never from the bench seed range."""
    check_seeds(seed, n)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        task, question, _ = dev_table(seed + i, int(rng.integers(rows[0], rows[1])))
        out.append((task, question))
    return out


def learnable(messages: list[dict[str, str]]) -> list[bool]:
    """One flag per message, true only for assistant turns that worked.

    A code turn is learned when its execution succeeded. The final answer is learned
    because the run was accepted. Anything the loop answered with a correction, such as an
    empty turn or an answer that typed numbers, is kept as context only.
    """
    flags = []
    for k, m in enumerate(messages):
        if m["role"] != "assistant":
            flags.append(False)
            continue
        after = messages[k + 1]["content"] if k + 1 < len(messages) else None
        if after is None:
            flags.append(True)
        elif after.startswith("<Execute>"):
            flags.append('"ok": true' in after)
        else:
            flags.append(False)
    return flags


def judge(run: Run, task: Task, keep_messages: int = 6) -> dict[str, object]:
    """Compare a run with the planted truth and the guarded findings.

    A run is kept when it is grounded, names no false pair, names every real pair the
    engine confirmed and never lost context to the message window, so every kept turn was
    written from exactly the context the training example shows.
    """
    columns = list(task.table.columns)
    claimed = parse_pairs(run.answer or "", columns) if run.grounded else set()
    real = {frozenset(p.columns) for p in task.plants if p.kind != "leak"}
    leaks = {frozenset(p.columns) for p in task.plants if p.kind == "leak"}
    confirmed = run.guarded & real
    false = len(claimed - real - leaks - derived_pairs(task))
    fidelity = len(claimed & confirmed) / len(confirmed) if confirmed else 1.0
    uncut = len(run.transcript) <= keep_messages + 3
    return {
        "planted": len(real),
        "confirmed": len(confirmed),
        "found": len(claimed & real),
        "false": false,
        "fidelity": round(fidelity, 3),
        "uncut": uncut,
        "keep": bool(run.grounded and false == 0 and fidelity == 1.0 and uncut),
    }


def _opening(args: tuple[Task, str]) -> Opening:
    task, question = args
    return prepare(task.table, question)


def collect(
    tasks: list[tuple[Task, str]],
    client_factory: Callable[[], Client],
    samples: int = 4,
    workers: int = 16,
    keep_messages: int = 6,
    max_rounds: int = 10,
    scan_workers: int = 1,
    on_record: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run every table samples times and judge each run.

    The engine scans each table once, in separate processes when scan_workers is above
    one, and the samples share that opening. Model calls run in threads, so a batching
    server such as vLLM sees many conversations at once.
    """
    if scan_workers > 1:
        with ProcessPoolExecutor(scan_workers, mp_context=mp.get_context("spawn")) as pool:
            openings = list(pool.map(_opening, tasks))
    else:
        openings = [_opening(t) for t in tasks]

    def one(job: tuple[int, int]) -> dict:
        i, s = job
        task, question = tasks[i]
        t0 = time.time()
        try:
            run = ask(
                task.table,
                question,
                client_factory(),
                keep_messages=keep_messages,
                max_rounds=max_rounds,
                opening=openings[i],
            )
            error = ""
        except Exception as e:  # a server timeout loses one sample, never the batch
            run, error = Run(question, guarded=set(openings[i].guarded)), repr(e)[:300]
        record = {
            "task": task.id,
            "kind": task.id.rsplit("-", 1)[0],
            "sample": s,
            "question": question,
            "grounded": run.grounded,
            "rounds": run.rounds,
            "turns": sum(m["role"] == "assistant" for m in run.transcript),
            "failed": sum(not st.ok for st in run.steps),
            "steps": len(run.steps),
            "answer": run.answer,
            "seconds": round(time.time() - t0, 1),
            "error": error,
            **judge(run, task, keep_messages),
            "messages": run.transcript,
            "learn": learnable(run.transcript),
        }
        if on_record:
            on_record(record)
        return record

    jobs = [(i, s) for i in range(len(tasks)) for s in range(samples)]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, jobs))


def select(records: list[dict], per_table: int = 2) -> list[dict]:
    """At most per_table kept runs per table, shortest and cleanest first, no duplicates.

    The cap stops easy tables from dominating the training set, which ReST-EM found
    protects against overfitting on the easiest problems.
    """
    by_task: dict[str, list[dict]] = {}
    for r in records:
        if r["keep"]:
            by_task.setdefault(r["task"], []).append(r)
    chosen = []
    for task in sorted(by_task):
        seen: set[str] = set()
        for r in sorted(by_task[task], key=lambda r: (r["turns"], r["failed"])):
            if r["answer"] in seen:
                continue
            seen.add(r["answer"])
            chosen.append(r)
            if len(seen) == per_table:
                break
    return chosen


def summarize(records: list[dict], chosen: list[dict]) -> dict[str, object]:
    """Counts that show how often the base model already does the job well."""
    by_kind: dict[str, list[dict]] = {}
    for r in records:
        by_kind.setdefault(r["kind"], []).append(r)
    tables = {r["task"] for r in records}
    return {
        "tables": len(tables),
        "runs": len(records),
        "kept_runs": sum(r["keep"] for r in records),
        "tables_with_a_kept_run": len({r["task"] for r in records if r["keep"]}),
        "chosen": len(chosen),
        "chosen_pair_none": sum("PAIR: none" in (r["answer"] or "") for r in chosen),
        "keep_rate_by_kind": {
            k: round(sum(r["keep"] for r in rs) / len(rs), 3) for k, rs in sorted(by_kind.items())
        },
        "mean_fidelity_planted": round(
            float(np.mean([r["fidelity"] for r in by_kind.get("plant", [])] or [0])), 3
        ),
        "failed_step_share": round(
            sum(r["failed"] for r in records) / max(1, sum(r["steps"] for r in records)), 3
        ),
        "errors": sum(bool(r["error"]) for r in records),
        "mean_seconds": round(float(np.mean([r["seconds"] for r in records])), 1),
    }
