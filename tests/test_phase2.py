import numpy as np
import polars as pl

from savant.engine import Settings, analyze
from savant.profile import build_card

FAST = Settings(stability_runs=20)


def _ev_like(seed: int = 0, n: int = 4000) -> pl.DataFrame:
    """Range truly rises with model year, but most recent cars have a range of zero
    because it was never researched, like the Washington vehicle file."""
    rng = np.random.default_rng(seed)
    year = rng.integers(2011, 2025, n)
    true_range = 80 + 12 * (year - 2011) + rng.normal(0, 30, n)
    unknown = (year >= 2020) & (rng.random(n) < 0.9)
    rng_col = np.where(unknown, 0, np.clip(true_range, 20, None)).round()
    status = np.where(
        unknown, "Eligibility unknown as battery range has not been researched", "Eligible"
    )
    return pl.DataFrame({"Model Year": year, "Electric Range": rng_col, "Eligibility": status})


def test_zeros_that_mean_unknown_are_flagged_in_the_card():
    card = build_card(_ev_like())
    c = card.column("Electric Range")
    assert c.stats["placeholder_zeros"] > 0.2
    assert "Electric Range" in card.placeholder_zeros
    assert "Eligibility" in card.placeholder_zeros["Electric Range"]
    assert "likely mean unknown" in card.to_text()


def test_real_zeros_in_counts_are_not_flagged():
    rng = np.random.default_rng(1)
    df = pl.DataFrame({"returns": rng.poisson(1.2, 3000), "visits": rng.poisson(4, 3000)})
    assert build_card(df).placeholder_zeros == {}


def test_a_finding_that_flips_without_the_zeros_is_not_reported():
    res = analyze(_ev_like(), FAST)
    status = {frozenset(f.test.columns): f for f in res.findings if f.test.kind == "correlation"}
    f = status[frozenset({"Model Year", "Electric Range"})]
    assert f.status == "zero_artifact"
    assert "zeros" in f.test.note
    assert frozenset({"Model Year", "Electric Range"}) not in {
        frozenset(x.test.columns) for x in res.reported
    }


def test_a_finding_that_holds_without_the_zeros_is_kept_with_a_note():
    rng = np.random.default_rng(2)
    n = 4000
    x = rng.normal(50, 10, n)
    y = 100 + 3 * x + rng.normal(0, 20, n)
    unknown = rng.random(n) < 0.3
    df = pl.DataFrame(
        {
            "x": x,
            "y": np.where(unknown, 0, y).round(),
            "flag": np.where(unknown, "not available", "ok"),
        }
    )
    res = analyze(df, FAST)
    kept = [f for f in res.reported if set(f.test.columns) == {"x", "y"}]
    assert kept and "holds with zeros set aside" in kept[0].test.note
