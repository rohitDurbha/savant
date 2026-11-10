"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",")]


def cmd_profile(a: argparse.Namespace) -> None:
    from .ingest import load
    from .profile import build_card

    print(build_card(load(a.file)).to_text())


def cmd_analyze(a: argparse.Namespace) -> None:
    from .agent import findings_text
    from .engine import Settings, analyze
    from .ingest import load

    res = analyze(load(a.file), Settings(stability_runs=a.stability_runs))
    if a.json:
        out = {
            "tests": len(res.ledger),
            "dropped": res.dropped,
            "confirmation_split": res.confirmed_split,
            "findings": [f.to_dict() for f in res.findings],
        }
        print(json.dumps(out, indent=1, default=str))
        return
    print(f"{len(res.ledger)} tests run, dropped {res.dropped}")
    if not res.confirmed_split:
        print("The file is too small for a confirmation split, so every finding is exploratory.")
    print(findings_text(res))


def cmd_plant(a: argparse.Namespace) -> None:
    from .planted import make_hard_suite, make_suite, write_suite

    tasks = make_suite(a.null, a.planted, a.rows, a.seed)
    tasks += make_hard_suite(a.hard, a.rows, a.seed + 50_000)
    write_suite(tasks, Path(a.out))
    print(f"wrote {len(tasks)} tasks to {a.out}")


def cmd_calibrate(a: argparse.Namespace) -> None:
    from .calibrate import sweep
    from .engine import Settings

    rows = sweep(
        a.null,
        a.planted,
        a.rows,
        a.seed,
        _floats(a.q),
        _floats(a.effect_scale),
        Settings(stability_runs=a.stability_runs),
        a.workers,
        a.hard,
    )
    text = json.dumps(rows, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(text)


def cmd_ask(a: argparse.Namespace) -> None:
    from .agent import ask
    from .ingest import load
    from .llm import OpenAICompatible

    run = ask(load(a.file), a.question, OpenAICompatible(a.base_url, a.model))
    print(run.rendered or f"No grounded answer after {run.rounds} rounds.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="savant", description="Guarded analytics engine")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("profile", help="print the data card for a file")
    s.add_argument("file")
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("analyze", help="run the guarded scan on a file")
    s.add_argument("file")
    s.add_argument("--json", action="store_true")
    s.add_argument("--stability-runs", type=int, default=100)
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("plant", help="write null and planted benchmark tables")
    s.add_argument("out")
    s.add_argument("--null", type=int, default=50)
    s.add_argument("--planted", type=int, default=50)
    s.add_argument("--hard", type=int, default=50, help="hard null tables")
    s.add_argument("--rows", type=int, default=2000)
    s.add_argument("--seed", type=int, default=20261105)
    s.set_defaults(fn=cmd_plant)

    s = sub.add_parser("calibrate", help="measure false findings and recall across settings")
    s.add_argument("--null", type=int, default=100)
    s.add_argument("--planted", type=int, default=100)
    s.add_argument("--hard", type=int, default=100, help="hard null tables")
    s.add_argument("--rows", type=int, default=2000)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--q", default="0.05")
    s.add_argument("--effect-scale", default="1.0")
    s.add_argument("--stability-runs", type=int, default=30)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--out")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("ask", help="ask a question through an OpenAI-compatible model server")
    s.add_argument("file")
    s.add_argument("question")
    s.add_argument("--base-url", default="http://localhost:11434/v1")
    s.add_argument("--model", default="qwen3.5:4b")
    s.set_defaults(fn=cmd_ask)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
