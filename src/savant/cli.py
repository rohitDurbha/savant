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


def _load_clean(path: str, quiet: bool = False):
    from .clean import clean_file

    df, report = clean_file(path)
    if not quiet:
        print(report.to_text(), end="\n\n")
    return df


def cmd_clean(a: argparse.Namespace) -> None:
    df = _load_clean(a.file)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(a.out) if a.out.endswith(".parquet") else df.write_csv(a.out)
        print(f"wrote {df.height} rows to {a.out}")


def cmd_analyze(a: argparse.Namespace) -> None:
    from .agent import findings_text
    from .engine import Settings, analyze
    from .ingest import load
    from .longform import analyze_long, detect

    df = load(a.file) if a.no_clean else _load_clean(a.file, quiet=a.json)
    settings = Settings(stability_runs=a.stability_runs)
    layout = detect(df)
    if layout is not None and not a.json:
        print(f"This is a {layout.describe()}.\n")
        from .insights import insights_text, rank

        for measure, res in analyze_long(df, settings).items():
            print(f"Measure {measure}\n{insights_text(rank(res.findings, limit=4))}\n")
        return
    res = analyze(df, settings)
    if a.json:
        out = {
            "tests": len(res.ledger),
            "dropped": res.dropped,
            "confirmation_split": res.confirmed_split,
            "findings": [f.to_dict() for f in res.findings],
        }
        print(json.dumps(out, indent=1, default=str))
        return
    from .insights import insights_text, rank

    print(f"{len(res.ledger)} tests run, dropped {res.dropped}")
    if not res.confirmed_split:
        print("The file is too small for a confirmation split, so every finding is exploratory.")
    print("\nInsights\n" + insights_text(rank(res.findings)))
    print("\nAll guarded findings\n" + findings_text(res, limit=50))


def cmd_drivers(a: argparse.Namespace) -> None:
    from .drivers import drivers
    from .engine import Settings

    df = _load_clean(a.file, quiet=True)
    print(drivers(df, a.target, Settings()).to_text())


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
    from .llm import OpenAICompatible

    run = ask(_load_clean(a.file), a.question, OpenAICompatible(a.base_url, a.model))
    print(run.rendered or f"No grounded answer after {run.rounds} rounds.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="savant", description="Guarded analytics engine")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(required=True)

    s = sub.add_parser("profile", help="print the data card for a file")
    s.add_argument("file")
    s.set_defaults(fn=cmd_profile)

    s = sub.add_parser("clean", help="clean a file and print the report of every change")
    s.add_argument("file")
    s.add_argument("--out", help="write the cleaned table as CSV or Parquet")
    s.set_defaults(fn=cmd_clean)

    s = sub.add_parser("analyze", help="clean a file, then run the guarded scan")
    s.add_argument("file")
    s.add_argument("--no-clean", action="store_true", help="skip the cleaning step")
    s.add_argument("--json", action="store_true")
    s.add_argument("--stability-runs", type=int, default=100)
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("drivers", help="columns most associated with a target, on held-out rows")
    s.add_argument("file")
    s.add_argument("target")
    s.set_defaults(fn=cmd_drivers)

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
