"""Optional observability: Langfuse traces for agent requests, MLflow runs for evaluations.

Both are off unless configured, and neither is needed to run, test or demo CiteShield.
Install with ``uv sync --extra observability``.

**Langfuse** (tracing) is on only when ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` are
set, the endpoint coming from ``LANGFUSE_BASE_URL`` (or ``LANGFUSE_HOST``; a self-hosted
Langfuse works the same way). ``CITESHIELD_TRACING=off`` disables it anyway. A trace is built
from the agent's own event stream, so it shows exactly what the API streams:

* one ``agent`` observation per request (input: question, regulations, mode; output:
  status, refusal reason, claim/checklist counts, cited labels);
* a ``guardrail`` observation for the guard, a ``retriever``/``tool`` observation per tool
  call (arguments, number of chunks, source labels, latency);
* events for rejected drafts and failed verifications (level WARNING), errors (ERROR);
* one ``generation`` per LLM call from the run's call log, with the model, stage, latency
  and only the token counts the provider reported (no invented costs).

Without keys, nothing is sent and no Langfuse client is created.

**MLflow** (experiment tracking) records an evaluation run when asked (``--mlflow`` on the
eval CLIs): parameters from the run identity (models, prompt versions, config hashes,
corpus, golden set, git commit), every numeric metric the scorer produced, and the run's
metadata as an artifact. The tracking URI defaults to ``./mlruns`` (gitignored); set
``MLFLOW_TRACKING_URI`` to use a server. Costs are not logged for providers that do not
bill (Ollama), since the harness only computes, not measures, them.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

log = logging.getLogger("citeshield.observability")

Emit = Callable[[str, dict[str, Any]], None]
TOOL_TYPES = {"retrieve": "retriever"}


# ------------------------------------------------------------------------ Langfuse


def langfuse_configured(env: Mapping[str, str] | None = None) -> bool:
    e = os.environ if env is None else env
    if e.get("CITESHIELD_TRACING", "").lower() in {"off", "0", "false"}:
        return False
    return bool(e.get("LANGFUSE_PUBLIC_KEY") and e.get("LANGFUSE_SECRET_KEY"))


def langfuse_client() -> Any | None:
    """A Langfuse client from the environment, or None (not configured / not installed)."""
    if not langfuse_configured():
        return None
    try:
        from langfuse import Langfuse
    except ImportError:
        log.warning("Langfuse keys are set but langfuse is not installed (extra 'observability')")
        return None
    return Langfuse()  # reads LANGFUSE_* (keys, base URL) from the environment


class RunTracer:
    """Turns one agent run's event stream into Langfuse observations. Never raises."""

    def __init__(self, client: Any, question: str, inputs: dict[str, Any], meta: dict[str, Any]):
        self.client = client
        self.root = client.start_observation(
            name="citeshield.ask",
            as_type="agent",
            input={"question": question, **inputs},
            metadata={k: meta.get(k) for k in ("llm_provider", "agent_model", "corpus_version",
                                                 "prompt_versions", "agent_config_hash")},
        )  # fmt: skip
        self.guard: Any | None = None
        self.tools: list[Any] = []

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._event(kind, payload)
        except Exception:  # tracing must never break a request
            log.exception("langfuse tracing failed on %s", kind)

    def _event(self, kind: str, p: dict[str, Any]) -> None:
        if kind == "status" and p.get("stage") == "guard":
            self.guard = self.root.start_observation(name="guard", as_type="guardrail")
        elif kind == "guard" and self.guard is not None:
            self.guard.update(output=p).end()
            self.guard = None
        elif kind == "tool_call":
            as_type = TOOL_TYPES.get(str(p.get("name")), "tool")
            obs = self.root.start_observation(name=str(p.get("name")), as_type=as_type,
                                              input=p.get("input"))  # fmt: skip
            self.tools.append(obs)
        elif kind == "tool_result" and self.tools:
            obs = self.tools.pop(0)
            obs.update(
                output={"n_chunks": p.get("n_chunks"),
                        "sources": [s.get("label") for s in p.get("sources", [])]},
                metadata={"ms": p.get("ms"), "is_error": p.get("is_error")},
                level="WARNING" if p.get("is_error") else None,
                status_message=p.get("error"),
            ).end()  # fmt: skip
        elif kind in {"draft_rejected", "verification_failed"}:
            self.root.create_event(name=kind, metadata=p, level="WARNING")
        elif kind == "final":
            self._final(p)
        elif kind == "error":
            self.fail(str(p.get("message")))

    def _final(self, p: dict[str, Any]) -> None:
        answer = p.get("answer") or {}
        for call in p.get("llm_log") or []:
            usage = {k: int(call[k]) for k in ("input_tokens", "output_tokens")
                     if isinstance(call.get(k), int)}  # fmt: skip
            gen = self.root.start_observation(
                name=f"llm.{call.get('stage')}", as_type="generation", model=call.get("model"),
                metadata={k: call.get(k) for k in ("stage", "provider", "stop_reason", "ms")},
                usage_details=usage or None,
            )  # fmt: skip
            gen.end()
        verification = p.get("verification") or {}
        self.root.update(
            output={
                "status": answer.get("status"),
                "refusal_reason": answer.get("refusal_reason"),
                "claims": len(answer.get("claims") or []),
                "checklist": len(answer.get("checklist") or []),
                "cited": [s.get("label") for s in (p.get("sources") or {}).values()],
            },
            metadata={
                "guard_label": p.get("guard_label"),
                "security_flags": p.get("security_flags"),
                "support_ratio": verification.get("support_ratio"),
                "draft_rejections": len(p.get("draft_rejections") or []),
                "timings_ms": p.get("timings_ms"),
            },
        )

    def fail(self, message: str) -> None:
        try:
            self.root.update(level="ERROR", status_message=message[:500])
        except Exception:
            log.exception("langfuse tracing failed on error")

    def close(self) -> None:
        try:
            for obs in self.tools:
                obs.end()
            self.root.end()
            self.client.flush()
        except Exception:
            log.exception("langfuse flush failed")


@contextmanager
def traced(
    emit: Emit,
    question: str,
    inputs: dict[str, Any],
    meta: dict[str, Any],
    client: Any | None = None,
) -> Iterator[Emit]:
    """``emit`` plus a Langfuse trace when configured; just ``emit`` otherwise."""
    client = client if client is not None else langfuse_client()
    if client is None:
        yield emit
        return
    tracer = RunTracer(client, question, inputs, meta)

    def both(kind: str, payload: dict[str, Any]) -> None:
        emit(kind, payload)
        tracer.event(kind, payload)

    try:
        yield both
    except BaseException as e:
        tracer.fail(f"{type(e).__name__}: {e}")
        raise
    finally:
        tracer.close()


# ------------------------------------------------------------------------ MLflow

DEFAULT_TRACKING_DIR = Path(__file__).resolve().parents[2] / "mlruns"
EXPERIMENT = "citeshield-eval"


def _flatten(prefix: str, value: Any, out: dict[str, str]) -> None:
    if isinstance(value, Mapping):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif value is not None and not isinstance(value, list):
        out[prefix] = str(value)[:500]


def run_params(meta: Mapping[str, Any]) -> dict[str, str]:
    """Identity parameters from an eval run's meta.json: what decides what it measured."""
    out: dict[str, str] = {}
    for key in ("run_id", "subset", "golden_sha", "corpus_hash", "cases_sha"):
        if meta.get(key) is not None:
            out[key] = str(meta[key])
    agent = meta.get("agent") or {}
    for key in ("llm_provider", "agent_model", "guard_model", "verifier_model",
                "agent_config_hash", "retriever_config_hash", "corpus_version"):  # fmt: skip
        if agent.get(key) is not None:
            out[f"agent.{key}"] = str(agent[key])
    _flatten("prompt", agent.get("prompt_versions") or {}, out)
    _flatten("verifier", agent.get("verifier_thresholds") or {}, out)
    runtime = meta.get("runtime") or {}
    for part in ("llm_options", "devices", "config_hashes", "code"):
        _flatten(part, runtime.get(part) or {}, out)
    _flatten("models", {k: v for k, v in (runtime.get("models") or {}).items()
                        if not isinstance(v, Mapping)}, out)  # fmt: skip
    profile = meta.get("profile") or {}
    for key in ("name", "sha"):
        if profile.get(key) is not None:
            out[f"profile.{key}"] = str(profile[key])
    return out


def numeric_metrics(metrics: Mapping[str, Any], provider: str | None) -> dict[str, float]:
    """Numeric, non-missing metrics; costs only for a provider that actually bills."""
    out = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, int | float) and not isinstance(v, bool)
    }
    if provider != "anthropic":
        out = {k: v for k, v in out.items() if "cost" not in k}
    return out


def log_eval_run(
    meta: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    kind: str,
    artifacts: Mapping[str, Any] | None = None,
    mlflow: Any | None = None,
) -> str:
    """Log one evaluation result to MLflow; returns the MLflow run id."""
    if mlflow is None:
        try:
            import mlflow as mlflow_module
        except ImportError as e:
            raise RuntimeError("--mlflow needs the 'observability' extra") from e
        mlflow = mlflow_module
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        # mlflow-skinny has no SQL backend; MLflow 3 keeps the file store working only on
        # explicit opt-in. A server or database URI in MLFLOW_TRACKING_URI needs no opt-in.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        mlflow.set_tracking_uri(DEFAULT_TRACKING_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    provider = (meta.get("agent") or {}).get("llm_provider")
    with mlflow.start_run(run_name=f"{kind}:{meta.get('run_id', 'run')}") as run:
        mlflow.set_tags({"kind": kind, "provider": str(provider)})
        mlflow.log_params(run_params(meta))
        mlflow.log_metrics(numeric_metrics(metrics, provider))
        mlflow.log_dict(json.loads(json.dumps(meta, default=str)), "meta.json")
        for name, obj in (artifacts or {}).items():
            mlflow.log_dict(json.loads(json.dumps(obj, default=str)), name)
        return str(run.info.run_id)
