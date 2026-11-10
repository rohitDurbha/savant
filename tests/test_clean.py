from datetime import date

import polars as pl

from savant.clean import clean, parse_number


def test_number_words_and_formats_parse():
    assert parse_number("thirty") == 30
    assert parse_number("two hundred") == 200
    assert parse_number("SIXTY THOUSAND") == 60000
    assert parse_number("one hundred twenty five") == 125
    assert parse_number("$1,200.50") == 1200.5
    assert parse_number("45%") == 45
    assert parse_number("1.2k") == 1200
    assert parse_number("-3") == -3
    assert parse_number("abc") is None
    assert parse_number("12 apples") is None


def test_missing_placeholders_become_nulls():
    df = pl.DataFrame({"email": ["a@x.com", "nan", "", " ", "NaN", " NAN ", "n/a", "b@y.org"]})
    out, report = clean(df, drop_duplicates=False)
    assert out["email"].null_count() == 6
    assert any(c.action == "missing placeholders" and c.count == 6 for c in report.changes)


def test_numbers_stored_as_text_become_numbers_and_every_conversion_is_logged():
    df = pl.DataFrame({"age": ["25", "thirty", "40", "nan", "35", "50", "25", "40", "35", "50"]})
    out, report = clean(df, drop_duplicates=False)
    assert out["age"].dtype.is_numeric()
    assert out["age"].to_list() == [25, 30, 40, None, 35, 50, 25, 40, 35, 50]
    words = next(c for c in report.changes if c.action == "number words")
    assert words.count == 1 and "thirty" in words.examples[0]


def test_mixed_date_formats_become_dates_with_month_first_evidence():
    values = ["April 5, 2018", "2020/02/20", "01/15/2020", "03-25-2019", "2019.12.01", "nan"]
    out, _ = clean(pl.DataFrame({"joined": values * 3}), drop_duplicates=False)
    assert out["joined"].dtype == pl.Date
    assert out["joined"].to_list()[:6] == [
        date(2018, 4, 5),
        date(2020, 2, 20),
        date(2020, 1, 15),
        date(2019, 3, 25),
        date(2019, 12, 1),
        None,
    ]


def test_day_first_dates_are_read_day_first():
    out, _ = clean(pl.DataFrame({"restocked": ["20/12/2022", "05/03/2023", "15/01/2023"] * 4}))
    assert out["restocked"].to_list()[:3] == [
        date(2022, 12, 20),
        date(2023, 3, 5),
        date(2023, 1, 15),
    ]


def test_ambiguous_day_month_order_is_flagged_not_guessed():
    out, report = clean(pl.DataFrame({"d": ["01/02/2020", "03/04/2021", "05/06/2022"] * 4}))
    assert out["d"].dtype == pl.String
    assert any(c.action == "ambiguous dates" for c in report.changes)


def test_category_spellings_are_unified_to_the_most_common_one():
    vals = ["Sales"] * 6 + ["sales", "SALES ", " Sales"] + ["HR"] * 5 + ["Salse"]
    out, report = clean(pl.DataFrame({"dept": vals}), drop_duplicates=False)
    # " Sales" is fixed by trimming and "sales" and "SALES" by case. "Salse" may be a typo,
    # but real data is full of distinct values one letter apart, so it is only reported.
    assert set(out["dept"].to_list()) == {"Sales", "HR", "Salse"}
    changed = {c.action: c.count for c in report.changes}
    assert changed["unified case and spacing"] == 2
    assert "possible typos, not changed" in changed


def test_distinct_short_categories_are_never_merged():
    out, _ = clean(pl.DataFrame({"grade": ["A", "B", "C", "D", "F"] * 4}))
    assert sorted(set(out["grade"].to_list())) == ["A", "B", "C", "D", "F"]


def test_padding_is_trimmed():
    out, _ = clean(pl.DataFrame({"name": [" grace ", " david ", "eve"] * 4}))
    assert set(out["name"].to_list()) == {"grace", "david", "eve"}


def test_exact_duplicate_rows_are_removed_and_counted():
    df = pl.DataFrame({"a": ["1", "1", "2"], "b": ["x", "x", "y"]})
    out, report = clean(df)
    assert out.height == 2
    assert any(c.action == "removed duplicate rows" and c.count == 1 for c in report.changes)


def test_float_nan_in_numeric_columns_becomes_null():
    out, _ = clean(pl.DataFrame({"price": [9.99, float("nan"), 19.99]}))
    assert out["price"].null_count() == 1


def test_report_reads_as_plain_text():
    _, report = clean(pl.DataFrame({"age": ["25", "thirty", "nan"] * 4}))
    text = report.to_text()
    assert "age" in text and "number words" in text


def test_raw_csv_read_lets_the_cleaner_decide_types(tmp_path):
    from savant.ingest import load

    path = tmp_path / "hr.csv"
    path.write_text(
        "Age,Salary,Joined\n25,50000,2020/02/20\nthirty, NAN ,01/15/2020\n40,SIXTY THOUSAND,nan\n"
    )
    out, _ = clean(load(path, raw=True))
    assert out["Age"].to_list() == [25, 30, 40]
    assert out["Salary"].to_list() == [50000, None, 60000]
    assert out["Joined"].to_list() == [date(2020, 2, 20), date(2020, 1, 15), None]


def test_a_file_cut_off_mid_row_drops_the_partial_row_and_says_so(tmp_path):
    from savant.clean import clean_file

    cut = tmp_path / "cut.csv"
    cut.write_text("name,phone\nann,123-456-7890\nbob,555-555")
    df, report = clean_file(cut)
    assert df["name"].to_list() == ["ann"]
    assert "cut off" in report.to_text()

    quoted = tmp_path / "quoted.csv"
    quoted.write_text('a,b\n1,"x y"\n2,"unfinished')
    df, report = clean_file(quoted)
    assert df["a"].to_list() == [1]
    assert "cut off" in report.to_text()


def test_a_complete_file_keeps_every_row(tmp_path):
    from savant.clean import clean_file

    ok = tmp_path / "ok.csv"
    ok.write_text("name,age\nann,30\nbob,thirty\n")
    df, report = clean_file(ok)
    assert df["age"].to_list() == [30, 30]
    assert "cut off" not in report.to_text()


def test_real_values_one_letter_apart_are_never_changed():
    # From the Washington electric vehicle data, where a merge rule rewrote real values.
    vins = ["WAUSPBFF7H"] * 40 + ["WAUTPBFF7H"] * 2 + ["5YJ3E1EA3P"] * 60 + ["5YJ3E1EC3P"] * 3
    models = ["MODEL Y"] * 50 + ["MODEL X"] * 4 + ["MODEL S"] * 5
    towns = ["Preston"] * 30 + ["Creston"] * 2 + ["Arlington"] * 40 + ["Arrington"] * 1
    for col in (vins, models, towns):
        out, _ = clean(pl.DataFrame({"v": col}), drop_duplicates=False)
        assert out["v"].to_list() == col
