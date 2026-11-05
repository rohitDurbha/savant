"""Numbers by reference. The model writes placeholders, and only executed code fills them."""

from __future__ import annotations

import re

PLACEHOLDER = re.compile(r"\{\{\s*(r\d+)\.([A-Za-z_]\w*)\s*\}\}")
NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?%?")


class MissingResult(KeyError):
    pass


def _format(v: object) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 0.001 or abs(v) >= 1e6):
            return f"{v:.3g}"
        return f"{v:,.3f}".rstrip("0").rstrip(".")
    return str(v)


def render(text: str, results: dict[str, dict[str, object]]) -> str:
    """Replace every {{rN.name}} with the value that step N emitted."""

    def fill(m: re.Match) -> str:
        step, name = m.group(1), m.group(2)
        if step not in results or name not in results[step]:
            raise MissingResult(f"{step}.{name}")
        return _format(results[step][name])

    return PLACEHOLDER.sub(fill, text)


def ungrounded_numbers(text: str, allowed: set[str] = frozenset()) -> list[str]:
    """Numbers typed outside placeholders, ignoring allowed tokens such as column names.

    Any result here means the model wrote a figure that no code produced.
    """
    stripped = PLACEHOLDER.sub(" ", text)
    for token in sorted(allowed, key=len, reverse=True):
        if any(ch.isdigit() for ch in token):
            stripped = stripped.replace(token, " ")
    return NUMBER.findall(stripped)
