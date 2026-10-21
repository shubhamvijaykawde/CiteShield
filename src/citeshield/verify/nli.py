"""Natural-language inference: does a premise (cited legal text) entail a hypothesis (claim)?

Default model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (MIT, 435M params,
870 MB safetensors), a 3-way classifier (entailment / neutral / contradiction). Weights are
loaded from safetensors only and the revision is pinned in ``VerifierConfig``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from citeshield.contracts.verification import Entailment

_SENTENCE = re.compile(r"(?<=[.;:])\s+|\n+")


class NLIModel(Protocol):
    name: str
    max_tokens: int

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[Entailment]: ...

    def count_tokens(self, text: str) -> int: ...


class DebertaNLI:
    def __init__(
        self,
        model_name: str,
        *,
        revision: str | None = None,
        device: str = "cpu",
        batch_size: int = 8,
        max_tokens: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model_name
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            revision=revision,
            use_safetensors=True,
            dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        ).to(device)
        self.model.eval()
        labels = {i: str(lbl).lower() for i, lbl in self.model.config.id2label.items()}
        self._index = {name: idx for idx, name in labels.items()}
        missing = {"entailment", "neutral", "contradiction"} - set(self._index)
        if missing:
            raise ValueError(f"{model_name} is not a 3-way NLI model (missing {missing})")

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[Entailment]:
        import torch

        out: list[Entailment] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            enc = self.tokenizer(
                [p for p, _ in batch],
                [h for _, h in batch],
                truncation="only_first",
                max_length=self.max_tokens,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().tolist()
            for row in probs:
                out.append(
                    Entailment(
                        entailment=row[self._index["entailment"]],
                        neutral=row[self._index["neutral"]],
                        contradiction=row[self._index["contradiction"]],
                    )
                )
        return out


def premise_windows(premise: str, hypothesis: str, nli: NLIModel, *, margin: int = 8) -> list[str]:
    """Split a premise that would not fit next to the hypothesis into overlapping windows.

    Truncating instead could silently drop the very sentence that supports the claim.
    """
    budget = nli.max_tokens - nli.count_tokens(hypothesis) - margin
    if nli.count_tokens(premise) <= budget:
        return [premise]
    units = [u for u in _SENTENCE.split(premise) if u.strip()]
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        n = nli.count_tokens(unit)
        if current and size + n > budget:
            windows.append(" ".join(current))
            # one unit of overlap so a sentence pair split across windows is still seen together
            current, size = [current[-1]], nli.count_tokens(current[-1])
        current.append(unit)
        size += n
    if current:
        windows.append(" ".join(current))
    return windows
