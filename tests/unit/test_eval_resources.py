"""Answer-eval resource settings: model devices and Ollama inference options."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from citeshield.agent import factory
from citeshield.agent.ollama import OllamaLLM
from citeshield.config import Settings, config_hash
from citeshield.evaluation import gate
from citeshield.evaluation.run_eval import (
    DEFAULT_EVAL_DEVICE,
    RUNTIME_IDENTITY,
    actual_devices,
    eval_settings,
)
from citeshield.retrieval import pipeline
from citeshield.retrieval.config import RetrieverConfig


def test_device_overrides_default_to_shared_device() -> None:
    cfg = RetrieverConfig()
    assert (cfg.effective_embed_device, cfg.effective_rerank_device) == (None, None)
    cfg = RetrieverConfig(device="cuda")
    assert (cfg.effective_embed_device, cfg.effective_rerank_device) == ("cuda", "cuda")
    cfg = RetrieverConfig(device="cuda", embed_device="cpu")
    assert (cfg.effective_embed_device, cfg.effective_rerank_device) == ("cpu", "cuda")


def test_unset_overrides_keep_the_config_hash() -> None:
    # Existing indexes, retrieval evals and traces are keyed by this hash.
    assert config_hash(RetrieverConfig()) == "7d3e26f9fa97"
    assert config_hash(RetrieverConfig(embed_device="cpu")) != config_hash(RetrieverConfig())


def test_eval_puts_retrieval_on_cpu_and_keeps_nli_on_cpu() -> None:
    assert DEFAULT_EVAL_DEVICE == "cpu"
    s = eval_settings(Settings(), DEFAULT_EVAL_DEVICE, DEFAULT_EVAL_DEVICE)
    assert s.retriever.effective_embed_device == "cpu"
    assert s.retriever.effective_rerank_device == "cpu"
    assert s.verifier.device == "cpu"
    # Everything else about retrieval is unchanged.
    assert s.retriever.model_copy(update={"embed_device": None, "rerank_device": None}) == (
        Settings().retriever
    )


def test_eval_auto_keeps_configured_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CITESHIELD_RETRIEVER__DEVICE", "cuda")
    s = eval_settings(Settings(), "auto", "cpu")
    assert s.retriever.effective_embed_device == "cuda"
    assert s.retriever.effective_rerank_device == "cpu"


def test_non_eval_defaults_unchanged() -> None:
    s = Settings()
    assert s.retriever.device is None  # cuda if available, as before
    assert s.retriever.embed_device is None and s.retriever.rerank_device is None
    assert s.verifier.device == "cpu"
    assert (s.llm.ollama_num_ctx, s.llm.ollama_num_predict) == (16384, 8192)


def test_retriever_builds_each_model_on_its_device(monkeypatch: pytest.MonkeyPatch) -> None:
    built: dict[str, Any] = {}

    class Enc:
        def __init__(self, name: str, *, device: str | None, **_: Any) -> None:
            built["embed"] = device

    class Rr:
        def __init__(self, name: str, *, device: str | None, **_: Any) -> None:
            built["rerank"] = device

    cfg = RetrieverConfig()
    monkeypatch.setattr(pipeline.Corpus, "load", classmethod(lambda cls, p: pipeline.Corpus([])))
    monkeypatch.setattr(
        pipeline,
        "read_manifest",
        lambda d: SimpleNamespace(
            corpus_hash=pipeline.corpus_hash([]), embed_model=cfg.embed_model
        ),
    )
    monkeypatch.setattr(pipeline, "BGEM3Encoder", Enc)
    monkeypatch.setattr(pipeline, "CrossEncoderReranker", Rr)
    monkeypatch.setattr(pipeline, "open_client", lambda *a: None)
    monkeypatch.setattr(pipeline, "HybridSearcher", lambda *a: None)
    pipeline.Retriever.from_settings(
        Settings(), RetrieverConfig(device="cuda", embed_device="cpu", rerank_device="cpu")
    )
    assert built == {"embed": "cpu", "rerank": "cpu"}


def test_eval_ollama_options_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CITESHIELD_LLM__PROVIDER", "ollama")
    monkeypatch.setenv("CITESHIELD_LLM__OLLAMA_NUM_CTX", "24576")
    llm, cfg = factory.build_llm(Settings())
    assert isinstance(llm, OllamaLLM)
    assert (llm.num_ctx, llm.num_predict, llm.think) == (24576, 8192, True)
    assert cfg.agent_model == cfg.guard_model == "qwen3:4b"


def test_ollama_request_carries_the_configured_options() -> None:
    sent: list[dict[str, Any]] = []

    class Client:
        def post(self, path: str, json: dict[str, Any]) -> Any:
            sent.append(json)
            data = {"message": {"content": "{}"}, "done_reason": "stop", "model": "qwen3:4b"}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)

    llm = OllamaLLM(num_ctx=24576, num_predict=8192, client=Client())  # type: ignore[arg-type]
    resp = llm.chat(model="qwen3:4b", system="s", messages=[{"role": "user", "content": "q"}])
    assert sent[0]["options"] == {"num_ctx": 24576, "num_predict": 8192}
    assert resp.meta["num_ctx"] == 24576  # recorded per call, aggregated into num_ctx_sent


def test_actual_devices_reads_the_loaded_models() -> None:
    agent = SimpleNamespace(
        tools=SimpleNamespace(
            retriever=SimpleNamespace(
                searcher=SimpleNamespace(encoder=SimpleNamespace(device="cpu")),
                reranker=SimpleNamespace(device="cpu"),
            )
        ),
        verifier=SimpleNamespace(nli=SimpleNamespace(device="cpu")),
    )
    assert actual_devices(agent) == {"embed": "cpu", "rerank": "cpu", "nli": "cpu"}


def test_devices_are_part_of_run_identity() -> None:
    assert "devices" in RUNTIME_IDENTITY
    meta = {
        "subset": "smoke",
        "runtime": {"models": {}, "llm_options": {}, "devices": {"embed": "cpu"}},
    }
    other = {**meta, "runtime": {**meta["runtime"], "devices": {"embed": "cuda"}}}
    assert gate.mismatches(meta, other) == [
        "runtime.devices: baseline {'embed': 'cpu'}, run {'embed': 'cuda'}"
    ]


def test_embed_dtype_is_opt_in_and_reaches_the_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    assert RetrieverConfig().embed_dtype is None
    assert config_hash(RetrieverConfig(embed_dtype=None)) == "7d3e26f9fa97"
    monkeypatch.setenv("CITESHIELD_RETRIEVER__EMBED_DTYPE", "bfloat16")
    assert Settings().retriever.embed_dtype == "bfloat16"
    seen: dict[str, Any] = {}

    class Enc:
        def __init__(self, name: str, *, dtype: str | None, **_: Any) -> None:
            seen["dtype"] = dtype

    cfg = RetrieverConfig(embed_dtype="bfloat16", use_reranker=False)
    monkeypatch.setattr(pipeline.Corpus, "load", classmethod(lambda cls, p: pipeline.Corpus([])))
    monkeypatch.setattr(
        pipeline,
        "read_manifest",
        lambda d: SimpleNamespace(
            corpus_hash=pipeline.corpus_hash([]), embed_model=cfg.embed_model
        ),
    )
    monkeypatch.setattr(pipeline, "BGEM3Encoder", Enc)
    monkeypatch.setattr(pipeline, "open_client", lambda *a: None)
    monkeypatch.setattr(pipeline, "HybridSearcher", lambda *a: None)
    pipeline.Retriever.from_settings(Settings(), cfg)
    assert seen == {"dtype": "bfloat16"}


def test_actual_dtypes_reads_parameters() -> None:
    from citeshield.evaluation.run_eval import actual_dtypes

    def module(*dtypes: str) -> Any:
        return SimpleNamespace(parameters=lambda: [SimpleNamespace(dtype=d) for d in dtypes])

    agent = SimpleNamespace(
        tools=SimpleNamespace(
            retriever=SimpleNamespace(
                searcher=SimpleNamespace(
                    encoder=SimpleNamespace(
                        model=module("torch.bfloat16"), sparse_linear=module("torch.bfloat16")
                    )
                ),
                reranker=SimpleNamespace(model=module("torch.float32")),
            )
        ),
        verifier=SimpleNamespace(nli=SimpleNamespace(model=module("torch.float32"))),
    )
    assert actual_dtypes(agent) == {
        "embed": ["torch.bfloat16"],
        "rerank": ["torch.float32"],
        "nli": ["torch.float32"],
    }


def test_dtypes_are_part_of_run_identity() -> None:
    assert "dtypes" in RUNTIME_IDENTITY
    meta = {"runtime": {"dtypes": {"embed": ["torch.float32"]}}}
    other = {"runtime": {"dtypes": {"embed": ["torch.bfloat16"]}}}
    assert any(m.startswith("runtime.dtypes") for m in gate.mismatches(meta, other))
