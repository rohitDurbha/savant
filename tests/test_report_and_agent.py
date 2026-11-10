import pytest

from savant import sandbox
from savant.agent import ask
from savant.planted import make_task
from savant.report import MissingResult, render, ungrounded_numbers


def test_render_fills_placeholders():
    out = render(
        "Mean price was {{r1.mean}} over {{r1.n}} orders.", {"r1": {"mean": 33.456, "n": 2000}}
    )
    assert out == "Mean price was 33.456 over 2,000 orders."


def test_render_rejects_missing_result():
    with pytest.raises(MissingResult):
        render("{{r2.x}}", {"r1": {"x": 1}})


def test_typed_numbers_are_caught_but_column_names_are_not():
    text = "score1 rose by 12% in group1, and {{r1.delta}} overall."
    assert ungrounded_numbers(text, {"score1", "group1"}) == ["12%"]


def test_sandbox_returns_emitted_results(tmp_path):
    task = make_task("s", 200, 0, seed=1)
    path = tmp_path / "d.parquet"
    task.table.write_parquet(path)
    code = 'emit("rows", len(df))\nemit("r", df["price"].corr(df["quantity"]))'
    ex = sandbox.run(code, path)
    assert ex.ok, ex.stderr
    assert ex.results["rows"] == 200
    assert isinstance(ex.results["r"], float)


def test_sandbox_times_out(tmp_path):
    task = make_task("s", 50, 0, seed=1)
    path = tmp_path / "d.parquet"
    task.table.write_parquet(path)
    ex = sandbox.run("while True:\n    pass", path, timeout=2)
    assert not ex.ok and ex.timed_out


def test_sandbox_has_no_inherited_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("SAVANT_API_KEY", "secret")
    task = make_task("s", 50, 0, seed=1)
    path = tmp_path / "d.parquet"
    task.table.write_parquet(path)
    ex = sandbox.run('import os\nemit("key", os.environ.get("SAVANT_API_KEY"))', path)
    assert ex.results["key"] is None


class Scripted:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def complete(self, messages):
        self.seen.append(messages[-1]["content"])
        return self.replies.pop(0)


def test_agent_rejects_typed_numbers_then_accepts_grounded_answer():
    task = make_task("a", 600, 0, seed=2)
    client = Scripted(
        [
            '<Analyze>count rows</Analyze><Code>emit("n", len(df))</Code>',
            "<Answer>There are 600 orders.</Answer>",
            "<Answer>There are {{r1.n}} orders.</Answer>",
        ]
    )
    run = ask(task.table, "How many orders?", client)
    assert run.grounded and run.rendered == "There are 600 orders."
    assert "without a placeholder" in client.seen[2]
    assert run.rounds == 3


def test_model_context_stays_bounded_over_a_long_run():
    task = make_task("w", 600, 0, seed=2)
    sizes = []

    class Counting:
        def complete(self, messages):
            sizes.append(len(messages))
            return '<Code>emit("n", len(df))</Code>'

    run = ask(task.table, "How many orders?", Counting(), max_rounds=10, keep_messages=4)
    assert not run.grounded and run.rounds == 10
    assert max(sizes) <= 2 + 4
    assert "last step" in run.transcript[-3]["content"]


def test_repeated_error_gets_a_change_of_approach_hint():
    task = make_task("e", 600, 0, seed=2)
    client = Scripted(
        [
            "<Code>df.nope()</Code>",
            "<Code>df.nope()</Code>",
            '<Code>emit("n", len(df))</Code>',
            "<Answer>There are {{r3.n}} orders.</Answer>",
        ]
    )
    run = ask(task.table, "How many orders?", client)
    assert run.grounded
    assert "same error" not in client.seen[1]
    assert "same error" in client.seen[2]
