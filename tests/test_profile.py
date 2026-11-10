import numpy as np
import polars as pl

from savant.engine import Settings, analyze
from savant.profile import build_card


def test_numeric_codes_are_labels_not_quantities():
    rng = np.random.default_rng(0)
    n = 3000
    df = pl.DataFrame(
        {
            "Postal Code": rng.integers(98001, 99403, n),
            "Legislative District": rng.integers(1, 50, n),
            "2020 GEOID": rng.integers(53001000000, 53077999999, n),
            "LocationID": rng.integers(1, 60, n),
            "Electric Range": rng.integers(0, 330, n),
        }
    )
    kinds = {c.name: c.kind for c in build_card(df).columns}
    assert kinds["Postal Code"] == "code"
    assert kinds["Legislative District"] == "code"
    assert kinds["2020 GEOID"] == "code"
    assert kinds["LocationID"] == "code"
    assert kinds["Electric Range"] == "numeric"


def test_categories_with_hundreds_of_levels_are_not_tested_as_groups():
    rng = np.random.default_rng(1)
    n = 20000
    df = pl.DataFrame(
        {
            "City": [f"town{i}" for i in rng.integers(0, 700, n)],
            "Make": [f"make{i}" for i in rng.integers(0, 40, n)],
            "Vehicle Location": [f"POINT (-122.{i} 47.{i})" for i in rng.integers(0, 5000, n)],
        }
    )
    kinds = {c.name: c.kind for c in build_card(df).columns}
    assert kinds["Make"] == "categorical"
    assert kinds["City"] == "text"
    assert kinds["Vehicle Location"] == "text"


def test_duplicate_columns_are_tested_once():
    rng = np.random.default_rng(2)
    n = 3000
    states = rng.choice(["KY", "CA", "TX", "NY"], n)
    names = {"KY": "Kentucky", "CA": "California", "TX": "Texas", "NY": "New York"}
    value = rng.normal(size=n) + (states == "KY")
    df = pl.DataFrame(
        {
            "LocationAbbr": states,
            "LocationDesc": [names[s] for s in states],
            "DataValue": value,
            "DataValueAlt": value,
        }
    )
    card = build_card(df)
    assert set(card.duplicates) == {("LocationDesc", "LocationAbbr"), ("DataValueAlt", "DataValue")}
    tested = {c for t in analyze(df, Settings(stability_runs=20)).ledger.tests for c in t.columns}
    assert tested == {"LocationAbbr", "DataValue"}
