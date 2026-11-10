"""Which columns are most associated with a chosen target, measured on held-out rows.

A gradient boosted model is fitted on the exploration rows. Each feature is then shuffled
on the confirmation rows, and the drop in held-out score is its importance. A feature
counts as a driver only when its importance stays above zero across repeats and above a
small floor, and only when the model beats a trivial baseline at all. Drivers are
associations in this data, never causes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, r2_score

from .engine import Settings, split
from .profile import build_card

MIN_SKILL = 0.02  # held-out score above the baseline before any driver is trusted
MIN_IMPORTANCE = 0.005  # held-out score lost when the feature is shuffled
LEAK_SCORE = 0.95  # one column alone predicting the target this well is likely a leak
REPEATS = 10


@dataclass
class Driver:
    feature: str
    importance: float
    spread: float


@dataclass
class DriverResult:
    target: str
    task: str
    score: float
    baseline: float
    drivers: list[Driver] = field(default_factory=list)
    leaks: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def reliable(self) -> bool:
        return self.score - self.baseline >= MIN_SKILL

    def to_text(self) -> str:
        lines = [
            f"drivers of {self.target}, {self.task}, held-out score {self.score:.3f} "
            f"against a baseline of {self.baseline:.3f}"
        ]
        if self.note:
            lines.append(self.note)
        if self.leaks:
            lines.append(f"set aside as likely leaks: {', '.join(self.leaks)}")
        for d in self.drivers:
            lines.append(f"- {d.feature}: held-out score drops {d.importance:.3f} when shuffled")
        if not self.drivers:
            lines.append("No column is reliably associated with the target.")
        return "\n".join(lines)


def _frame(df: pl.DataFrame, cols: list[str], cats: set[str]) -> pd.DataFrame:
    out = df.select(cols).to_pandas()
    for c in cols:
        if c in cats:
            out[c] = out[c].astype("category")
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_datetime(out[c]).astype("int64") / 86_400e9
    return out


def _model(classify: bool, seed: int):
    # Early stopping and mild regularization keep a boosted model from fitting noise when
    # a table has only a few thousand rows and many unrelated columns.
    kwargs = {
        "categorical_features": "from_dtype",
        "random_state": seed,
        "max_iter": 300,
        "early_stopping": True,
        "learning_rate": 0.05,
        "min_samples_leaf": 50,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
    }
    return (
        HistGradientBoostingClassifier(**kwargs)
        if classify
        else HistGradientBoostingRegressor(**kwargs)
    )


def _score(classify: bool, y_true, y_pred) -> float:
    return balanced_accuracy_score(y_true, y_pred) if classify else r2_score(y_true, y_pred)


def drivers(df: pl.DataFrame, target: str, settings: Settings | None = None) -> DriverResult:
    s = settings or Settings()
    card = build_card(df)
    kind = card.column(target).kind
    classify = kind in ("categorical", "boolean")
    task = "classification" if classify else "regression"
    dupes = {d for d, _ in card.duplicates}
    features = [
        c.name
        for c in card.columns
        if c.name != target
        and c.name not in dupes
        and c.kind in ("numeric", "categorical", "boolean", "datetime")
    ]
    cats = {c.name for c in card.columns if c.kind in ("categorical", "boolean")}
    data = df.filter(pl.col(target).is_not_null())
    explore, confirm = split(data, card, s)
    if confirm is None or not features:
        return DriverResult(target, task, 0.0, 0.0, note="Too few rows for held-out drivers.")

    y_train = explore[target].to_pandas()
    y_test = confirm[target].to_pandas()
    if classify:
        y_train, y_test = y_train.astype(str), y_test.astype(str)
        majority = y_train.mode().iloc[0]
        baseline = _score(True, y_test, [majority] * len(y_test))
    else:
        baseline = _score(False, y_test, [float(y_train.mean())] * len(y_test))

    # A single column that predicts the target almost perfectly is a definition or a leak.
    leaks = []
    for f in features:
        m = _model(classify, s.seed).fit(_frame(explore, [f], cats), y_train)
        if _score(classify, y_test, m.predict(_frame(confirm, [f], cats))) >= LEAK_SCORE:
            leaks.append(f)
    features = [f for f in features if f not in leaks]
    if not features:
        return DriverResult(target, task, baseline, baseline, leaks=leaks)

    X_train, X_test = _frame(explore, features, cats), _frame(confirm, features, cats)
    model = _model(classify, s.seed).fit(X_train, y_train)
    score = _score(classify, y_test, model.predict(X_test))
    result = DriverResult(target, task, score, baseline, leaks=leaks)
    if not result.reliable:
        result.note = "The model does no better than a simple baseline, so no driver is trusted."
        return result
    scorer = "balanced_accuracy" if classify else "r2"
    imp = permutation_importance(
        model, X_test, y_test, scoring=scorer, n_repeats=REPEATS, random_state=s.seed
    )
    for f, mean, std in zip(features, imp.importances_mean, imp.importances_std, strict=True):
        if mean >= MIN_IMPORTANCE and mean - 2 * std > 0:
            result.drivers.append(Driver(f, float(mean), float(std)))
    result.drivers.sort(key=lambda d: -d.importance)
    return result
