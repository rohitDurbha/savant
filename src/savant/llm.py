"""Model clients. Any OpenAI-compatible server works, including Ollama, vLLM and llama.cpp."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol

STOPS = ("</Code>", "</Answer>")


class Client(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


def cut_at_stop(text: str) -> str:
    """Keep text up to and including the first closing action tag."""
    ends = [text.find(s) + len(s) for s in STOPS if s in text]
    return text[: min(ends)] if ends else text


class OpenAICompatible:
    def __init__(self, base_url: str, model: str, temperature: float = 0.0, max_tokens: int = 1500):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.key = os.environ.get("SAVANT_API_KEY", "none")

    def complete(self, messages: list[dict[str, str]]) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": list(STOPS),
        }
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"] or ""
        # Servers drop the stop string itself, so close the tag again when it was cut.
        for tag in ("Code", "Answer"):
            if f"<{tag}>" in text and f"</{tag}>" not in text:
                text += f"</{tag}>"
                break
        return cut_at_stop(text)


class Transformers:
    """Local generation with Hugging Face transformers, used inside Kaggle notebooks."""

    def __init__(self, model_id: str, max_new_tokens: int = 1500, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        kwargs: dict = {"dtype": torch.float16, "device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.max_new_tokens = max_new_tokens

    def complete(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            stop_strings=list(STOPS),
            tokenizer=self.tok,
        )
        return cut_at_stop(
            self.tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True)
        )
