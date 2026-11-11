"""Training data written by the model itself and kept only when the checks agree."""

import pytest

from savant.agent import Run, ask, prepare
from savant.planted import make_task
from savant.selfplay import collect, dev_tasks, judge, learnable, select

DEV = 6_000_000


def _msgs(*contents):
    roles = ["system", "user"] + ["assistant", "user"] * 10
    return [{"role": r, "content": c} for r, c in zip(roles, contents, strict=False)]


def test_dev_tasks_refuse_the_bench_seeds():
    from savant.traces import BENCH_SEED

    with pytest.raises(ValueError):
        dev_tasks(3, seed=BENCH_SEED)


def test_dev_tasks_are_deterministic_and_mixed():
    a = dev_tasks(12, seed=DEV)
    b = dev_tasks(12, seed=DEV)
    assert [t.id for t, _ in a] == [t.id for t, _ in b]
    kinds = {t.id.rsplit("-", 1)[0] for t, _ in a}
    assert "plant" in kinds and len(kinds) >= 2


def test_learnable_marks_only_turns_that_worked():
    msgs = _msgs(
        "sys",
        "card",
        "<Code>x</Code>",
        '<Execute>{"step": "r1", "ok": false, "error": "NameError"}</Execute>',
        "<Code>y</Code>",
        '<Execute>{"step": "r2", "ok": true, "error": ""}</Execute>',
        "thinking without an action",
        "Reply with one <Code> or <Answer>.",
        "<Answer>12</Answer>",
        "Your answer typed these numbers without a placeholder: 12.",
        "<Answer>{{r2.n}}</Answer>",
    )
    assert learnable(msgs) == [
        False, False,  # system and opening are context
        False, False,  # failed code
        True, False,  # code that ran
        False, False,  # void turn
        False, False,  # rejected answer
        True,  # accepted answer
    ]  # fmt: skip


def _scripted_run(task, answer_pairs, extra=()):
    opening = prepare(task.table, "q")
    pairs = "\n".join(f"PAIR: {a} | {b}" for a, b in [*answer_pairs, *extra])
    run = Run("q", answer=pairs, grounded=True, guarded=opening.guarded)
    run.transcript = [
        *opening.messages,
        {"role": "assistant", "content": f"<Answer>{pairs}</Answer>"},
    ]
    return run, opening


def test_judge_keeps_a_run_that_names_every_confirmed_pair():
    task = make_task("plant-j", 3000, 3, DEV + 1)
    run, opening = _scripted_run(task, [])
    real = {frozenset(p.columns) for p in task.plants if p.kind != "leak"}
    confirmed = opening.guarded & real
    assert confirmed, "the engine should confirm something on a clear planted table"
    run, _ = _scripted_run(task, [tuple(sorted(p)) for p in confirmed])
    verdict = judge(run, task)
    assert verdict["keep"] and verdict["fidelity"] == 1.0 and verdict["false"] == 0


def test_judge_drops_a_run_that_misses_a_confirmed_pair_or_names_a_false_one():
    task = make_task("plant-j", 3000, 3, DEV + 1)
    run, opening = _scripted_run(task, [])
    real = {frozenset(p.columns) for p in task.plants if p.kind != "leak"}
    confirmed = sorted(tuple(sorted(p)) for p in opening.guarded & real)
    missing, _ = _scripted_run(task, confirmed[:-1])
    assert not judge(missing, task)["keep"]
    wrong, _ = _scripted_run(task, confirmed, extra=[("price", "rating")])
    assert not judge(wrong, task)["keep"] and judge(wrong, task)["false"] == 1


def test_judge_drops_a_run_whose_context_was_cut_by_the_window():
    task = make_task("null-j", 1500, 0, DEV + 2)
    run, _ = _scripted_run(task, [])
    filler = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "<Code>1</Code>"}]
    run.transcript = run.transcript[:2] + filler * 5 + run.transcript[2:]
    assert not judge(run, task, keep_messages=6)["keep"]


def test_select_caps_runs_per_table_and_prefers_short_ones():
    recs = [
        {"task": "a", "keep": True, "turns": 4, "failed": 1, "answer": "x"},
        {"task": "a", "keep": True, "turns": 2, "failed": 0, "answer": "y"},
        {"task": "a", "keep": True, "turns": 2, "failed": 0, "answer": "y"},  # duplicate
        {"task": "a", "keep": True, "turns": 3, "failed": 0, "answer": "z"},
        {"task": "b", "keep": False, "turns": 2, "failed": 0, "answer": "w"},
    ]
    chosen = select(recs, per_table=2)
    assert [(r["task"], r["answer"]) for r in chosen] == [("a", "y"), ("a", "z")]


class SaysNone:
    """Always answers PAIR: none, which is right on null tables and wrong on planted ones."""

    def complete(self, messages):
        return "<Answer>PAIR: none</Answer>"


def test_collect_runs_each_table_several_times_and_judges_every_run():
    tasks = dev_tasks(3, seed=DEV + 10)
    records = collect(tasks, SaysNone, samples=2, workers=2)
    assert len(records) == 6
    assert {r["task"] for r in records} == {t.id for t, _ in tasks}
    for r in records:
        assert set(r) >= {
            "task",
            "keep",
            "turns",
            "failed",
            "fidelity",
            "false",
            "messages",
            "learn",
        }
        assert len(r["learn"]) == len(r["messages"])
    # A null table answered with PAIR: none is a correct negative example.
    for r in records:
        assert r["keep"] == (r["kind"] != "plant" or r["confirmed"] == 0)


def test_ask_accepts_a_prepared_opening():
    task = make_task("null-o", 1200, 0, DEV + 3)
    opening = prepare(task.table, "q")

    class One:
        def complete(self, messages):
            assert messages[:2] == opening.messages
            return "<Answer>PAIR: none</Answer>"

    run = ask(task.table, "q", One(), opening=opening)
    assert run.grounded and run.guarded == opening.guarded
