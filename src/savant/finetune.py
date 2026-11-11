"""Turning verified conversations into token ids and labels for fine-tuning.

Only the analyst's own turns are learned. The system prompt, the data card and the
execution feedback are context, so their labels are set to IGNORE. The function takes
the chat template and the tokenizer as plain callables, which keeps it testable without
a GPU or a model download.
"""

from __future__ import annotations

from collections.abc import Callable

IGNORE = -100

Render = Callable[..., str]
Encode = Callable[[str], list[int]]


def encode_example(
    messages: list[dict[str, str]],
    render: Render,
    encode: Encode,
    max_len: int,
    learn: list[bool] | None = None,
) -> tuple[list[int], list[int]] | None:
    """Ids for the whole conversation and labels on assistant turns, or None if too long.

    learn marks which messages are learned. A failed code step or a rejected answer stays
    in the context, so the model sees the error, but it carries no labels, so the model is
    never trained to repeat the mistake.
    """
    if learn is not None and len(learn) != len(messages):
        raise ValueError("learn needs one flag per message")
    ids = encode(render(messages))
    if len(ids) > max_len:
        return None
    labels = [IGNORE] * len(ids)
    for k, m in enumerate(messages):
        if m["role"] != "assistant" or (learn is not None and not learn[k]):
            continue
        start = len(encode(render(messages[:k], add_generation_prompt=True)))
        end = len(encode(render(messages[: k + 1])))
        if ids[:end] != encode(render(messages[: k + 1])):
            raise ValueError("the chat template does not extend cleanly turn by turn")
        labels[start:end] = ids[start:end]
    return ids, labels
