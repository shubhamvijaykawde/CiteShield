"""bge-m3 encoder producing dense and sparse (lexical) vectors in one forward pass.

Implemented directly on ``transformers`` instead of FlagEmbedding:
  * dense  = L2-normalised [CLS] hidden state
  * sparse = relu(sparse_linear(hidden_states)) per token, max-pooled per vocabulary id,
             special tokens dropped (the "lexical weights" of the bge-m3 paper)
``sparse_linear.pt`` ships in the model repo next to the transformer weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class EncodedText:
    dense: list[float]
    sparse: dict[int, float]


class Encoder(Protocol):
    dim: int

    def encode(self, texts: list[str]) -> list[EncodedText]: ...


def resolve_device(device: str | None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class BGEM3Encoder:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        *,
        device: str | None = None,
        max_length: int = 512,
        batch_size: int = 16,
        dtype: str | None = None,
    ) -> None:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer

        self.device = resolve_device(device)
        if dtype is not None:
            weights: torch.dtype = getattr(torch, dtype)
        else:
            weights = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=weights).to(self.device)
        self.model.eval()

        sparse_state = torch.load(
            hf_hub_download(model_name, "sparse_linear.pt"), map_location="cpu", weights_only=True
        )
        self.sparse_linear = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.sparse_linear.load_state_dict(sparse_state)
        self.sparse_linear.to(self.device, dtype=weights).eval()

        self.dim: int = self.model.config.hidden_size
        self._special_ids = set(self.tokenizer.all_special_ids)

    def encode(self, texts: list[str]) -> list[EncodedText]:
        out: list[EncodedText] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._encode_batch(texts[start : start + self.batch_size]))
        return out

    def _encode_batch(self, texts: list[str]) -> list[EncodedText]:
        import torch

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            hidden: torch.Tensor = self.model(**batch).last_hidden_state
            dense = torch.nn.functional.normalize(hidden[:, 0].float(), dim=-1)
            weights = torch.relu(self.sparse_linear(hidden)).squeeze(-1).float()

        ids = batch["input_ids"].cpu().tolist()
        mask = batch["attention_mask"].cpu().tolist()
        token_weights = weights.cpu().tolist()
        results = []
        for row, (row_ids, row_mask, row_w) in enumerate(
            zip(ids, mask, token_weights, strict=True)
        ):
            sparse: dict[int, float] = {}
            for tok, m, w in zip(row_ids, row_mask, row_w, strict=True):
                if not m or tok in self._special_ids or w <= 0:
                    continue
                if w > sparse.get(tok, 0.0):
                    sparse[tok] = w
            results.append(EncodedText(dense=dense[row].cpu().tolist(), sparse=sparse))
        return results
