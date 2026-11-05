"""Read a data file into a Polars DataFrame without running anything from it."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

MAX_BYTES = 50 * 1024 * 1024
READERS = {
    ".csv": lambda p: pl.read_csv(p, try_parse_dates=True, infer_schema_length=10_000),
    ".tsv": lambda p: pl.read_csv(
        p, separator="\t", try_parse_dates=True, infer_schema_length=10_000
    ),
    ".parquet": pl.read_parquet,
    ".jsonl": pl.read_ndjson,
    ".xlsx": pl.read_excel,
    ".xls": pl.read_excel,
}


class IngestError(ValueError):
    """Raised when a file cannot be read safely."""


def load(path: str | Path, max_bytes: int = MAX_BYTES) -> pl.DataFrame:
    """Load CSV, TSV, Parquet, JSON, JSON Lines or Excel into a DataFrame."""
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"{path} is not a file")
    size = path.stat().st_size
    if size > max_bytes:
        raise IngestError(
            f"{path.name} is {size / 1e6:.1f} MB, above the {max_bytes / 1e6:.0f} MB cap"
        )
    suffix = path.suffix.lower()
    if suffix == ".json":
        return from_records(json.loads(path.read_text()))
    reader = READERS.get(suffix)
    if reader is None:
        raise IngestError(f"unsupported file type {suffix or '(none)'}")
    return reader(path)


def from_records(payload: object) -> pl.DataFrame:
    """Turn an API style JSON payload into a table.

    Accepts a list of objects, or an object holding one list of objects under a key
    such as "data", "results" or "items".
    """
    if isinstance(payload, dict):
        lists = [v for v in payload.values() if isinstance(v, list)]
        if len(lists) != 1:
            raise IngestError("JSON object must hold exactly one list of records")
        payload = lists[0]
    if not isinstance(payload, list) or not all(isinstance(r, dict) for r in payload):
        raise IngestError("JSON must be a list of objects")
    return pl.from_dicts(payload, infer_schema_length=None)
