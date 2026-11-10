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


def test_blind_mode_hides_the_guarded_findings():
    task = make_task("b", 2000, 3, seed=7)
    client = Scripted(["<Answer>PAIR: none</Answer>"])
    ask(task.table, "What matters?", client, show_findings=False)
    assert "Guarded findings" not in client.seen[0]


def test_pairs_are_parsed_from_an_answer():
    from savant.agent import parse_pairs

    text = "Price rises with quantity.\nPAIR: price | quantity\nPAIR: Region | score0\n"
    assert parse_pairs(text, ["price", "quantity", "region", "score0"]) == {
        frozenset({"price", "quantity"}),
        frozenset({"region", "score0"}),
    }
    assert parse_pairs("PAIR: none", ["price"]) == set()


def test_sandbox_can_import_scipy_and_threaded_numpy(tmp_path):
    task = make_task("s", 200, 0, seed=1)
    path = tmp_path / "d.parquet"
    task.table.write_parquet(path)
    code = (
        "from scipy import stats\n"
        "r, p = stats.pearsonr(df['price'], df['quantity'])\n"
        "emit('p', p)\n"
        "emit('inv', float(np.linalg.inv(np.eye(300))[0, 0]))"
    )
    ex = sandbox.run(
        code + "\nimport os\nemit('threads', os.environ.get('OPENBLAS_NUM_THREADS'))", path
    )
    assert ex.ok, ex.stderr
    assert 0 <= ex.results["p"] <= 1
    # BLAS reserves memory per thread, so a many-core machine exhausts the address limit
    # unless threads are pinned. Kaggle hit this, so the pin is part of the contract.
    assert ex.results["threads"] == "1"


def test_findings_text_ranks_caps_and_names_near_deterministic_pairs():
    from types import SimpleNamespace

    from savant.agent import findings_text
    from savant.engine import Finding
    from savant.ledger import Test

    def f(i, status, effect):
        t = Test(
            f"t{i}",
            "correlation",
            (f"a{i}", f"b{i}"),
            "spearman",
            effect,
            0.0,
            effect,
            "abs_rho",
            100,
        )
        return Finding(t, status)

    findings = [f(i, "confirmed", 0.1 + i / 100) for i in range(30)]
    findings += [f(100 + i, "leakage", 0.99) for i in range(8)]
    findings += [f(200 + i, "exploratory", 0.5) for i in range(5)]
    text = findings_text(SimpleNamespace(findings=findings), limit=12)
    lines = text.splitlines()
    assert lines[0].startswith("- [confirmed]") and "a29" in lines[0]  # strongest first
    assert sum(line.startswith("- [") for line in lines) == 12
    assert "near-deterministic" in text and "leakage" not in text
    assert "exploratory" not in text
    assert "31 more" in text
