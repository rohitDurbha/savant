"""Tables with known truth, used to calibrate thresholds and to train and test the model.

A null table has no real association between any two columns, so every reported finding
on it is false. A planted table adds patterns on fresh columns, so each planted pattern
is the only true association involving those columns.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

REGIONS = ["north", "south", "east", "west", "central"]
CHANNELS = ["web", "store", "phone", "partner"]
SEGMENTS = ["consumer", "small_business", "enterprise"]
PAYMENTS = ["card", "cash", "invoice", "wallet"]
PLANT_KINDS = ("correlation", "segment", "trend", "association", "leak", "shift")


@dataclass
class Plant:
    kind: str
    columns: tuple[str, str]
    strength: float


@dataclass
class Task:
    id: str
    table: pl.DataFrame
    plants: list[Plant]

    @property
    def is_null(self) -> bool:
        return not self.plants

    def truth(self) -> dict:
        return {"id": self.id, "plants": [asdict(p) for p in self.plants]}


def base_table(rng: np.random.Generator, n: int) -> pl.DataFrame:
    """Independent columns that look like an orders table."""
    start = date(2024, 1, 1)
    days = rng.integers(0, 730, n)
    return pl.DataFrame(
        {
            "order_id": np.arange(1, n + 1),
            "order_date": [start + timedelta(days=int(d)) for d in days],
            "region": rng.choice(REGIONS, n),
            "channel": rng.choice(CHANNELS, n),
            "segment": rng.choice(SEGMENTS, n, p=[0.6, 0.3, 0.1]),
            "payment_method": rng.choice(PAYMENTS, n),
            "price": np.round(rng.lognormal(3.5, 0.6, n), 2),
            "quantity": rng.poisson(3, n) + 1,
            "discount": np.round(rng.uniform(0, 0.3, n), 3),
            "rating": rng.integers(1, 6, n),
            "delivery_days": rng.gamma(2.0, 2.0, n).round(1),
            "basket_items": rng.poisson(5, n) + 1,
        }
    )


def hard_null_table(rng: np.random.Generator, n: int) -> pl.DataFrame:
    """A table with no association in how it was generated, built to fool naive tests.

    Repeat customers carry their own level on two metrics and a fixed tier, so rows are
    not independent. Two metrics drift slowly over time without any trend, and their
    paths can look related. Values are skewed, have outliers and missing cells, and one
    category has rare levels. Any finding here is false.
    """
    start = date(2024, 1, 1)
    days = np.sort(rng.integers(0, 730, n))
    customers = rng.integers(0, max(10, n // 12), n)
    n_cust = customers.max() + 1
    level_spend = rng.normal(0, 1, n_cust)
    level_visits = rng.normal(0, 1, n_cust)
    tier = rng.choice(["bronze", "silver", "gold"], n_cust)

    def ar1(phi: float) -> np.ndarray:
        x = np.zeros(n)
        e = rng.normal(0, 1, n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x

    spend = np.exp(1.2 * level_spend[customers] + rng.normal(0, 0.8, n)) * 30
    spend[rng.random(n) < 0.01] *= 25
    visits = np.round(np.exp(0.8 * level_visits[customers] + rng.normal(0, 0.5, n)) * 3)
    rare = rng.choice(
        ["a", "b", "c", "d", "e", "f", "g", "h"], n, p=[0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01]
    )
    df = pl.DataFrame(
        {
            "customer_id": customers,
            "order_date": [start + timedelta(days=int(d)) for d in days],
            "customer_tier": tier[customers],
            "store": rng.choice(["s1", "s2", "s3", "s4", "s5", "s6"], n),
            "promo_code": rare,
            "spend": np.round(spend, 2),
            "visits": visits,
            "basket_value": np.round(rng.pareto(2.5, n) * 40 + 5, 2),
            "wait_minutes": np.round(rng.exponential(12, n), 1),
            "sensor_a": np.round(ar1(0.97) + 50, 3),
            "sensor_b": np.round(ar1(0.97) + 20, 3),
            "score": rng.integers(0, 11, n),
        }
    )
    holes = {c: rng.random(n) < 0.08 for c in ("spend", "wait_minutes", "store")}
    return df.with_columns(
        [pl.when(pl.Series(m)).then(None).otherwise(pl.col(c)).alias(c) for c, m in holes.items()]
    )


def make_hard_suite(n: int, rows: int, seed: int) -> list[Task]:
    return [
        Task(f"hard-null-{i:03d}", hard_null_table(np.random.default_rng(seed + i), rows), [])
        for i in range(n)
    ]


def shuffle_columns(df: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """Break every association in a real table while keeping each column's values."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({c: df[c].to_numpy()[rng.permutation(df.height)] for c in df.columns})


def _standard(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() or 1.0)


def plant(df: pl.DataFrame, kind: str, k: int, strength: float, rng: np.random.Generator):
    """Add one pattern on fresh columns and return the new table and its Plant."""
    n = df.height
    if kind == "correlation":
        x = rng.normal(0, 1, n)
        y = strength * x + np.sqrt(max(1 - strength**2, 1e-6)) * rng.normal(0, 1, n)
        a, b = f"signal_x{k}", f"signal_y{k}"
        df = df.with_columns(pl.Series(a, x * 10 + 50), pl.Series(b, np.exp(y / 2) * 20))
        return df, Plant(kind, (a, b), strength)
    if kind == "segment":
        cat = f"group{k}"
        labels = rng.choice(["alpha", "beta", "gamma", "delta"], n)
        y = rng.normal(0, 1, n) + strength * (labels == "gamma")
        num = f"score{k}"
        df = df.with_columns(pl.Series(cat, labels), pl.Series(num, y * 15 + 100))
        return df, Plant(kind, (cat, num), strength)
    if kind == "trend":
        t = _standard(df["order_date"].cast(pl.Date).cast(pl.Int64).to_numpy().astype(float))
        y = strength * t + np.sqrt(max(1 - strength**2, 1e-6)) * rng.normal(0, 1, n)
        num = f"metric{k}"
        df = df.with_columns(pl.Series(num, y * 8 + 40))
        return df, Plant(kind, ("order_date", num), strength)
    if kind == "association":
        a_vals = rng.choice(["p", "q", "r"], n)
        b_vals = rng.choice(["u", "v", "w"], n)
        follow = rng.random(n) < strength
        mapped = np.select([a_vals == "p", a_vals == "q"], ["u", "v"], "w")
        b_vals = np.where(follow, mapped, b_vals)
        a, b = f"tier{k}", f"status{k}"
        df = df.with_columns(pl.Series(a, a_vals), pl.Series(b, b_vals))
        return df, Plant(kind, (a, b), strength)
    if kind == "shift":
        days = df["order_date"].cast(pl.Date).cast(pl.Int64).to_numpy()
        cut = np.quantile(days, rng.uniform(0.3, 0.7))
        y = rng.normal(0, 1, n) + strength * (days > cut)
        num = f"level{k}"
        df = df.with_columns(pl.Series(num, y * 12 + 60))
        return df, Plant(kind, ("order_date", num), strength)
    if kind == "leak":
        y = rng.normal(0, 1, n)
        a, b = f"outcome{k}", f"outcome{k}_adjusted"
        noise = rng.normal(0, 0.01, n)
        df = df.with_columns(pl.Series(a, y * 30 + 200), pl.Series(b, y * 30 * 0.97 + 190 + noise))
        return df, Plant(kind, (a, b), 1.0)
    raise ValueError(kind)


STRENGTH = {
    "correlation": (0.1, 0.5),
    "segment": (0.15, 0.8),
    "trend": (0.1, 0.5),
    "association": (0.1, 0.5),
    "leak": (1.0, 1.0),
    "shift": (0.3, 1.0),
}


def make_task(task_id: str, rows: int, n_plants: int, seed: int) -> Task:
    rng = np.random.default_rng(seed)
    df = base_table(rng, rows)
    plants = []
    for k in range(n_plants):
        kind = PLANT_KINDS[rng.integers(len(PLANT_KINDS))]
        lo, hi = STRENGTH[kind]
        df, p = plant(df, kind, k, float(rng.uniform(lo, hi)), rng)
        plants.append(p)
    return Task(task_id, df, plants)


def make_suite(n_null: int, n_planted: int, rows: int, seed: int) -> list[Task]:
    tasks = [make_task(f"null-{i:03d}", rows, 0, seed + i) for i in range(n_null)]
    tasks += [make_task(f"plant-{i:03d}", rows, 3, seed + 10_000 + i) for i in range(n_planted)]
    return tasks


def write_suite(tasks: list[Task], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for t in tasks:
        t.table.write_csv(out / f"{t.id}.csv")
        (out / f"{t.id}.truth.json").write_text(json.dumps(t.truth(), indent=1))
