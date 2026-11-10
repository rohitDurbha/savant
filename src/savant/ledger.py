"""Every statistical test the engine runs, shown or not, so false discovery control is honest."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class Test:
    __test__ = False  # not a pytest test class

    id: str
    kind: str
    columns: tuple[str, str]
    method: str
    statistic: float
    p_value: float
    effect: float
    effect_name: str
    n: int
    note: str = ""
    explained: float | None = None  # share of variance explained, where it applies

    def to_dict(self) -> dict:
        return asdict(self)


def false_discovery_control(p_values: list[float], q: float, dependent: bool = False) -> np.ndarray:
    """Benjamini-Hochberg, or Benjamini-Yekutieli when dependent is True.

    Returns a boolean mask of rejected hypotheses in the original order.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    c = np.sum(1.0 / np.arange(1, m + 1)) if dependent else 1.0
    thresholds = q * np.arange(1, m + 1) / (m * c)
    below = np.nonzero(ranked <= thresholds)[0]
    rejected = np.zeros(m, dtype=bool)
    if below.size:
        rejected[order[: below[-1] + 1]] = True
    return rejected


class Ledger:
    def __init__(self) -> None:
        self.tests: list[Test] = []

    def add(self, test: Test) -> Test:
        self.tests.append(test)
        return test

    def __len__(self) -> int:
        return len(self.tests)

    def discoveries(self, q: float = 0.05, dependent: bool = False) -> list[Test]:
        """Tests that survive FDR control over the whole ledger, in ledger order."""
        mask = false_discovery_control([t.p_value for t in self.tests], q, dependent)
        return [t for t, keep in zip(self.tests, mask, strict=True) if keep]
