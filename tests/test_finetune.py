"""Token labels for fine-tuning, so only the analyst's own turns are learned."""

from savant.finetune import IGNORE, encode_example


def chatml(messages, add_generation_prompt=False):
    text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
    return text + ("<|assistant|>" if add_generation_prompt else "")


def chars(text):
    return [ord(c) for c in text]


MESSAGES = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "card"},
    {"role": "assistant", "content": "<Code>x</Code>"},
    {"role": "user", "content": "<Execute>{}</Execute>"},
    {"role": "assistant", "content": "<Answer>ok</Answer>"},
]


def test_only_assistant_turns_carry_labels():
    ids, labels = encode_example(MESSAGES, chatml, chars, max_len=10_000)
    assert ids == chars(chatml(MESSAGES))
    learned = "".join(chr(i) for i, lab in zip(ids, labels, strict=True) if lab != IGNORE)
    assert learned == "<Code>x</Code><|end|><Answer>ok</Answer><|end|>"


def test_long_examples_are_skipped():
    assert encode_example(MESSAGES, chatml, chars, max_len=20) is None
