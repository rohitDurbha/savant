"""Measure false findings on null tables and recall on planted tables for given settings."""

from __future__ import annotations

import dataclasses
from concurrent.futures import ProcessPoolExecutor

from .engine import Settings, analyze
from .grade import Score, grade, summarize
from .planted import Task, make_suite


def _one(args: tuple[Task, Settings]) -> Score:
    task, settings = args
    return grade(task, analyze(task.table, settings))


def evaluate(tasks: list[Task], settings: Settings, workers: int = 4) -> dict:
    jobs = [(t, settings) for t in tasks]
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            scores = list(pool.map(_one, jobs))
    else:
        scores = [_one(j) for j in jobs]
    return summarize(scores)


def sweep(
    n_null: int,
    n_planted: int,
    rows: int,
    seed: int,
    qs: list[float],
    effect_scales: list[float],
    base: Settings,
    workers: int = 4,
) -> list[dict]:
    """Try each FDR level and effect-size scale on one fixed development suite."""
    tasks = make_suite(n_null, n_planted, rows, seed)
    results = []
    for q in qs:
        for scale in effect_scales:
            effects = {k: v * scale for k, v in Settings().min_effect.items()}
            s = dataclasses.replace(base, q=q, min_effect=effects)
            summary = evaluate(tasks, s, workers)
            results.append({"q": q, "effect_scale": scale, **summary})
    return results
