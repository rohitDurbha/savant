"""Training conversations written by a scripted analyst and kept only when verified."""

import numpy as np
import polars as pl
import pytest

from savant.agent import ACTION, parse_pairs, prepare
from savant.planted import Plant, Task, make_task
from savant.report import render, ungrounded_numbers
from savant.traces import BENCH_SEED, PAIR_QUESTION, build_trace, lookup_trace, make_traces

DEV = 5_000_000


def _answer(trace):
    last = trace["messages"][-1]
    assert last["role"] == "assistant"
    kinds = [m.group(1) for m in ACTION.finditer(last["content"])]
    assert kinds[-1] == "Answer"
    return ACTION.findall(last["content"])[-1][1]


def _results(trace):
    return trace["results"]


def test_opening_matches_what_the_agent_sends():
    task = make_task("plant-x", 1500, 2, DEV)
    trace = build_trace(task, PAIR_QUESTION)
    opening = prepare(task.table, PAIR_QUESTION)
    assert trace["messages"][:2] == opening.messages


def test_planted_trace_cites_numbers_and_names_the_true_pairs():
    task = make_task("plant-x", 3000, 3, DEV + 1)
    trace = build_trace(task, PAIR_QUESTION)
    assert trace is not None, "a clear planted table should give a verified trace"
    answer = _answer(trace)
    assert ungrounded_numbers(answer, set(task.table.columns)) == []
    render(answer, _results(trace))  # every placeholder points at an emitted result
    real = {frozenset(p.columns) for p in task.plants if p.kind != "leak"}
    claimed = parse_pairs(answer, task.table.columns)
    assert claimed and claimed <= real
    assert trace["recall"] == len(claimed) / len(real)
    assert all(m["role"] in ("system", "user", "assistant") for m in trace["messages"])


def test_null_trace_says_none():
    task = make_task("null-x", 1500, 0, DEV + 2)
    trace = build_trace(task, PAIR_QUESTION)
    assert trace is not None
    answer = _answer(trace)
    assert "PAIR: none" in answer
    assert parse_pairs(answer, task.table.columns) == set()


def test_trace_is_dropped_when_it_names_a_false_pair():
    task = make_task("plant-x", 3000, 3, DEV + 1)
    # Hide the truth, so every pair the answer names counts as false.
    fake = Task(task.id, task.table, [Plant("correlation", ("price", "rating"), 0.4)])
    assert build_trace(fake, PAIR_QUESTION) is None


def test_bench_seed_is_refused():
    with pytest.raises(ValueError):
        make_traces(n=2, seed=BENCH_SEED - 1)


def test_lookup_trace_names_the_top_group_from_code():
    rng = np.random.default_rng(0)
    n = 900
    region = rng.choice(["north", "south", "east"], n)
    price = rng.normal(50, 5, n) + 20 * (region == "south")
    df = pl.DataFrame({"region": region, "price": price})
    trace = lookup_trace(df, "region", "price")
    assert trace is not None
    rendered = render(_answer(trace), _results(trace))
    assert "south" in rendered
