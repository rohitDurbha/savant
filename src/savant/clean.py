"""Deterministic cleaning with a report of every change.

Nothing is fixed silently. Each rule logs what it changed, how often and with examples,
and a rule that cannot decide safely, such as dates that read the same day first or
month first, leaves the column alone and says so.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime

import polars as pl

MISSING = {"", "nan", "none", "null", "na", "n/a", "-", "--", "?", "missing", "unknown", "nat"}
SHARE = 0.9  # share of present values that must parse before a column changes type

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
PLAIN = re.compile(r"[-+]?\d+(?:\.\d+)?")
YMD = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})$")
AMBIGUOUS = re.compile(r"^(\d{1,2})([-/.])(\d{1,2})\2(\d{4})$")
NAMED = ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y")


@dataclass
class Change:
    column: str
    action: str
    count: int
    examples: list[str] = field(default_factory=list)


@dataclass
class Report:
    rows_in: int
    rows_out: int = 0
    changes: list[Change] = field(default_factory=list)

    def add(self, column: str, action: str, count: int, examples: list[str]) -> None:
        if count:
            self.changes.append(Change(column, action, count, examples[:3]))

    def to_text(self) -> str:
        lines = [f"cleaning report, {self.rows_in} rows in and {self.rows_out} rows out"]
        for c in self.changes:
            ex = "; ".join(c.examples)
            lines.append(
                f"- {c.column}: {c.action}, {c.count} values" + (f" (e.g. {ex})" if ex else "")
            )
        if not self.changes:
            lines.append("- no changes were needed")
        return "\n".join(lines)


def _words(text: str) -> float | None:
    tokens = [t for t in re.split(r"[\s-]+", text.lower()) if t and t != "and"]
    if not tokens or not all(t in UNITS or t in SCALES for t in tokens):
        return None
    total, current = 0, 0
    for t in tokens:
        if t in UNITS:
            current += UNITS[t]
        elif t == "hundred":
            current = max(current, 1) * 100
        else:
            total += max(current, 1) * SCALES[t]
            current = 0
    return float(total + current)


def parse_number(text: str) -> float | None:
    """A number from text such as 30, thirty, $1,200.50, 45% or 1.2k, or None."""
    s = text.strip()
    word = _words(s)
    if word is not None:
        return word
    s = re.sub(r"[$€£,\s]", "", s.lower())
    scale = 1.0
    if s.endswith("%"):
        s = s[:-1]
    elif s[-1:] in ("k", "m") and PLAIN.fullmatch(s[:-1]):
        scale = 1_000.0 if s[-1] == "k" else 1_000_000.0
        s = s[:-1]
    return float(s) * scale if PLAIN.fullmatch(s) else None


def _date_order(values: list[str]) -> str | None:
    """Day first, month first, None when the column gives no evidence, or conflict."""
    day_first = month_first = False
    for v in values:
        m = AMBIGUOUS.match(v)
        if m:
            a, b = int(m.group(1)), int(m.group(3))
            day_first |= a > 12
            month_first |= b > 12
    if day_first and month_first:
        return "conflict"
    return "day" if day_first else "month" if month_first else None


def _parse_date(v: str, order: str | None) -> date | None:
    m = YMD.match(v)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return _safe(y, mo, d)
    m = AMBIGUOUS.match(v)
    if m:
        if order not in ("day", "month"):
            return None
        a, b, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
        return _safe(y, b, a) if order == "day" else _safe(y, a, b)
    for fmt in NAMED:
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def _safe(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _osa(a: str, b: str) -> int:
    """Edit distance that counts one swap of neighbouring letters as one edit."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _unify(values: list[str | None], name: str, report: Report) -> list[str | None]:
    present = [v for v in values if v is not None]
    counts = Counter(present)
    if not counts or len(counts) > max(50, 0.2 * len(present)):
        return values
    # Variants that differ only by case map to their most common spelling.
    by_fold: dict[str, Counter] = {}
    for v, n in counts.items():
        by_fold.setdefault(v.casefold(), Counter())[v] = n
    mapping = {v: group.most_common(1)[0][0] for group in by_fold.values() for v in group}
    case_changes = [v for v in present if mapping[v] != v]
    report.add(
        name,
        "unified case and spacing",
        len(case_changes),
        [f"{v} to {mapping[v]}" for v in dict.fromkeys(case_changes)],
    )
    merged_counts = Counter(mapping[v] for v in present)
    # A rare spelling one edit away from a far more common one, with the same digits,
    # is treated as a typo of it. Short codes and differently numbered labels never merge.
    near = {}
    for rare, n_rare in merged_counts.items():
        for common, n_common in merged_counts.most_common():
            if common == rare or n_rare > 0.2 * n_common or min(len(rare), len(common)) < 5:
                continue
            same_digits = re.findall(r"\d", rare) == re.findall(r"\d", common)
            if same_digits and _osa(rare.casefold(), common.casefold()) == 1:
                near[rare] = common
                break
    near_changes = [v for v in present if mapping[v] in near]
    report.add(
        name,
        "merged near duplicates",
        len(near_changes),
        [f"{mapping[v]} to {near[mapping[v]]}" for v in dict.fromkeys(near_changes)],
    )
    final = {v: near.get(mapping[v], mapping[v]) for v in counts}
    return [final[v] if v is not None else None for v in values]


def _clean_text(s: pl.Series, report: Report) -> pl.Series:
    name = s.name
    raw = s.to_list()
    stripped = [re.sub(r"\s+", " ", v).strip() if v is not None else None for v in raw]
    padded = [r for r, v in zip(raw, stripped, strict=True) if r is not None and r != v]
    report.add(name, "trimmed spacing", len(padded), [repr(v) for v in dict.fromkeys(padded)])
    holes = [v for v in stripped if v is not None and v.lower() in MISSING]
    report.add(name, "missing placeholders", len(holes), [repr(v) for v in dict.fromkeys(holes)])
    values = [None if v is None or v.lower() in MISSING else v for v in stripped]
    present = [v for v in values if v is not None]
    if not present:
        return pl.Series(name, values, dtype=pl.String)

    numbers = [parse_number(v) for v in present]
    if sum(n is not None for n in numbers) >= SHARE * len(present):
        words = [v for v in present if _words(v) is not None]
        report.add(
            name,
            "number words",
            len(words),
            [f"{v} to {_words(v):g}" for v in dict.fromkeys(words)],
        )
        styled = [
            v
            for v in present
            if _words(v) is None and parse_number(v) is not None and not PLAIN.fullmatch(v)
        ]
        report.add(
            name,
            "number formats",
            len(styled),
            [f"{v} to {parse_number(v):g}" for v in dict.fromkeys(styled)],
        )
        bad = [v for v, n in zip(present, numbers, strict=True) if n is None]
        report.add(name, "unreadable numbers set to missing", len(bad), list(dict.fromkeys(bad)))
        parsed = [parse_number(v) if v is not None else None for v in values]
        out = pl.Series(name, parsed, dtype=pl.Float64)
        whole = out.drop_nulls()
        return out.cast(pl.Int64) if whole.len() and (whole == whole.round(0)).all() else out

    order = _date_order(present)
    if order == "conflict":
        report.add(
            name, "dates left as text, day and month order conflict", len(present), present[:3]
        )
    else:
        dates = [_parse_date(v, order) for v in present]
        hits = sum(d is not None for d in dates)
        ambiguous = [v for v in present if AMBIGUOUS.match(v)]
        if order is None and ambiguous and hits + len(ambiguous) >= SHARE * len(present):
            report.add(name, "ambiguous dates", len(ambiguous), ambiguous[:3])
            return pl.Series(name, values, dtype=pl.String)
        if hits >= SHARE * len(present):
            formats = {re.sub(r"\d", "9", v) for v in present}
            report.add(name, f"parsed dates from {len(formats)} formats", hits, sorted(formats)[:3])
            bad = [v for v, d in zip(present, dates, strict=True) if d is None]
            report.add(name, "unreadable dates set to missing", len(bad), list(dict.fromkeys(bad)))
            parsed = [_parse_date(v, order) if v is not None else None for v in values]
            return pl.Series(name, parsed, dtype=pl.Date)

    return pl.Series(name, _unify(values, name, report), dtype=pl.String)


def clean(df: pl.DataFrame, drop_duplicates: bool = True) -> tuple[pl.DataFrame, Report]:
    """Clean every column, then drop exact duplicate rows unless told not to.

    Identical rows can be real, such as two identical orders in a table without ids, so
    the count of removed rows is always reported and the step can be turned off.
    """
    report = Report(rows_in=df.height)
    cols = []
    for name in df.columns:
        s = df[name]
        if s.dtype == pl.String:
            cols.append(_clean_text(s, report))
        elif s.dtype.is_float():
            report.add(name, "missing placeholders", int(s.is_nan().sum()), ["NaN"])
            cols.append(s.fill_nan(None))
        else:
            cols.append(s)
    out = pl.DataFrame(cols)
    deduped = out.unique(maintain_order=True) if drop_duplicates else out
    report.add("all columns", "removed duplicate rows", out.height - deduped.height, [])
    report.rows_out = deduped.height
    return deduped, report
