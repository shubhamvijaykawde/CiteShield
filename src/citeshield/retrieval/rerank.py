"""Cross-encoder reranker (bge-reranker-v2-m3). Scores are sigmoid(logit) in [0, 1]."""

from __future__ import annotations

from typing import Protocol

from citeshield.retrieval.embed import resolve_device


class Reranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = resolve_device(device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype).to(
            self.device
        )
        self.model.eval()

    def score(self, query: str, passages: list[str]) -> list[float]:
        import torch

        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            chunk = passages[start : start + self.batch_size]
            batch = self.tokenizer(
                [query] * len(chunk),
                chunk,
                padding=True,
                truncation="only_second",
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.model(**batch).logits.view(-1).float()
            scores.extend(torch.sigmoid(logits).cpu().tolist())
        return scores
