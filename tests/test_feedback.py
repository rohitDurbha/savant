"""Fewer failed code steps, through real datetimes, a clear prompt and hints on common errors."""

import datetime as dt
import json

import polars as pl

from savant import sandbox
from savant.agent import SYSTEM, execute_message, hint_for


def _run(code, df, tmp_path):
    path = tmp_path / "data.parquet"
    df.write_parquet(path)
    return sandbox.run(code, path)


def test_date_columns_arrive_as_pandas_datetimes(tmp_path):
    df = pl.DataFrame(
        {
            "order_date": [dt.date(2025, 1, 5), dt.date(2025, 3, 9), None],
            "price": [1.0, 2.0, 3.0],
        }
    )
    ex = _run(
        'emit("month", int(df["order_date"].dt.month.iloc[1]))\n'
        'emit("early", float(df.loc[df["order_date"] < "2025-02-01", "price"].mean()))',
        df,
        tmp_path,
    )
    assert ex.ok, ex.stderr
    assert ex.results == {"month": 3, "early": 1.0}


def test_text_columns_stay_text(tmp_path):
    df = pl.DataFrame({"region": ["north", "2025-01-01"], "price": [1.0, 2.0]})
    ex = _run('emit("kind", str(df["region"].dtype))', df, tmp_path)
    assert ex.ok and "datetime" not in ex.results["kind"]


def test_prompt_says_each_step_starts_fresh():
    assert "starts fresh" in SYSTEM and "Import" in SYSTEM


def test_common_errors_get_a_hint():
    cases = {
        "NameError: name 'df_numeric' is not defined": "starts fresh",
        "NameError: name 'chi2_contingency' is not defined": "starts fresh",
        "TypeError: Only valid with DatetimeIndex, TimedeltaIndex or PeriodIndex, "
        "but got an instance of 'Index'": "set_index",
        "AttributeError: 'Chi2ContingencyResult' object has no attribute 'chi2'": "statistic",
        "ValueError: could not convert string to float: 'north'": "numeric_only",
        "TypeError: agg function failed [how->mean,dtype->object]": "numeric_only",
        "AttributeError: 'Series' object has no attribute 'corrwith'": "Series",
        "FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'": "already loaded",
        "numpy.exceptions.DTypePromotionError: The DType <class "
        "'numpy.dtypes.DateTime64DType'> could not be promoted": "days",
    }
    for error, needle in cases.items():
        assert needle in hint_for(error), error
    assert hint_for("ZeroDivisionError: division by zero") == ""


def test_undefined_names_get_a_specific_fix():
    # A scipy function gets the exact import, and anything else is named so the model
    # rebuilds that variable in the same step.
    scipy = hint_for("NameError: name 'spearmanr' is not defined")
    assert "from scipy.stats import spearmanr" in scipy
    rebuilt = hint_for("NameError: name 'significant_correlations' is not defined")
    assert "significant_correlations" in rebuilt and "earlier step" in rebuilt


def test_unpacking_errors_get_a_hint():
    for error in (
        "ValueError: too many values to unpack (expected 2)",
        "ValueError: not enough values to unpack (expected 3, got 2)",
    ):
        assert "returns" in hint_for(error), error


def test_hint_is_sent_with_the_error():
    ex = sandbox.Execution(ok=False, stderr="NameError: name 'x' is not defined")
    msg = execute_message("r2", ex, "NameError: name 'x' is not defined")
    payload = json.loads(msg["content"][len("<Execute>") : -len("</Execute>")])
    assert "starts fresh" in payload["hint"]
