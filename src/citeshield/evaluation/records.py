"""One answer-eval record per golden question: what the agent said and what it cost.

Records are self-contained: they carry the text of every cited source, so the metrics, the
LLM judge and the labelling CLI can all work from ``answers.jsonl`` without the index, the
GPU models or the LLM.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from citeshield.agent.graph import AgentRun
from citeshield.contracts.answer import AgentAnswer, AnswerStatus
from citeshield.evaluation.golden import Category, GoldenItem


class CitedSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    text: str


class AnswerRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    category: Category
    question: str
    should_refuse: bool
    expected: list[str]
    reference_answer: str | None = None

    answer: AgentAnswer | None = None
    error: str | None = None  # "ExcType: message" when the run crashed
    guard_label: str | None = None
    cited: dict[str, CitedSource] = Field(default_factory=dict)
    retrieved: list[str] = Field(default_factory=list, description="Tool order, deduplicated.")
    # Verifier support ratio of the draft behind the final answer (None: nothing verified).
    support_ratio: float | None = None
    first_draft_valid: bool = False
    schema_valid: bool = False

    latency_ms: float = 0.0
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    # Distinct context sizes the client actually sent (Ollama); empty for other providers.
    num_ctx_sent: list[int] = Field(default_factory=list)

    @classmethod
    def _base(cls, item: GoldenItem) -> dict[str, object]:
        return {
            "id": item.id,
            "category": item.category,
            "question": item.question,
            "should_refuse": item.should_refuse,
            "expected": item.expected,
            "reference_answer": item.reference_answer,
        }

    @classmethod
    def from_run(cls, item: GoldenItem, run: AgentRun) -> AnswerRecord:
        u = run.usage
        return cls.model_validate(
            {
                **cls._base(item),
                "answer": run.answer,
                "guard_label": run.guard_label,
                "cited": {
                    cid: CitedSource(label=it.label, text=it.text)
                    for cid, it in run.cited_sources().items()
                },
                "retrieved": list(run.evidence),
                "support_ratio": run.verification.support_ratio if run.verification else None,
                "first_draft_valid": not run.draft_rejections,
                "schema_valid": _round_trips(run.answer),
                "latency_ms": round(run.timings_ms.get("total", 0.0), 1),
                "cost_usd": round(u.cost_usd, 6),
                "input_tokens": u.input_tokens + u.cache_read_tokens + u.cache_write_tokens,
                "output_tokens": u.output_tokens,
                "llm_calls": u.calls,
                "num_ctx_sent": sorted(
                    {int(c["num_ctx"]) for c in run.llm_log if c.get("num_ctx") is not None}
                ),
            }
        )

    @classmethod
    def from_error(cls, item: GoldenItem, exc: BaseException, latency_ms: float) -> AnswerRecord:
        return cls.model_validate(
            {
                **cls._base(item),
                "error": f"{type(exc).__name__}: {exc}"[:2000],
                "latency_ms": round(latency_ms, 1),
            }
        )

    @property
    def crashed(self) -> bool:
        """Infrastructure or runtime failure: no answer was produced at all."""
        return self.answer is None

    @property
    def refused(self) -> bool:
        """A deliberate refusal by the agent. A crash is not a refusal."""
        return self.answer is not None and self.answer.status is AnswerStatus.REFUSED

    @property
    def cited_ids(self) -> list[str]:
        if self.answer is None:
            return []
        return list(dict.fromkeys(c.chunk_id for c in self.answer.all_citations()))

    @property
    def judgeable(self) -> bool:
        """The judge and the labeller grade substantive answers to answerable questions."""
        return (
            not self.should_refuse
            and not self.crashed
            and not self.refused
            and self.reference_answer is not None
        )

    def render_answer(self) -> str:
        """The answer as a reader sees it: summary, cited claims with quotes, caveats."""
        a = self.answer
        if a is None:
            return f"[no answer: {self.error}]"
        lines = [f"Status: {a.status.value}"]
        if a.refusal_reason:
            lines[0] += f" ({a.refusal_reason.value})"
        lines += ["", a.summary]
        if a.claims:
            lines += ["", "Claims:"]
        for i, claim in enumerate(a.claims, 1):
            lines.append(f"{i}. {claim.text}")
            for c in claim.citations:
                lines.append(f'   [{self._label(c.chunk_id)}] "{c.quote}"')
        if a.checklist:
            lines += ["", "Checklist:"]
        for i, item in enumerate(a.checklist or [], 1):
            lines.append(f"{i}. {item.obligation} (applies to: {item.applies_to})")
            for c in item.citations:
                lines.append(f'   [{self._label(c.chunk_id)}] "{c.quote}"')
        if a.caveats:
            lines += ["", "Caveats:", *(f"- {c}" for c in a.caveats)]
        return "\n".join(lines)

    def render_sources(self) -> str:
        return "\n\n".join(
            f'<source label="{s.label}">\n{s.text}\n</source>' for s in self.cited.values()
        )

    def view(self) -> str:
        """Everything the judge and the human labeller see, and nothing else."""
        return (
            f"<question>\n{self.question}\n</question>\n\n"
            f"<reference_answer>\n{self.reference_answer or '(none)'}\n</reference_answer>\n\n"
            f"<answer>\n{self.render_answer()}\n</answer>\n\n"
            f"<cited_sources>\n{self.render_sources() or '(none)'}\n</cited_sources>"
        )

    @property
    def view_sha(self) -> str:
        """Binds a human label or judge verdict to the exact text that was graded."""
        return hashlib.sha256(self.view().encode()).hexdigest()[:12]

    def _label(self, chunk_id: str) -> str:
        src = self.cited.get(chunk_id)
        return src.label if src else chunk_id


def _round_trips(answer: AgentAnswer) -> bool:
    try:
        AgentAnswer.model_validate_json(answer.model_dump_json())
    except ValidationError:
        return False
    return True


def load_records(path: Path) -> list[AnswerRecord]:
    with path.open(encoding="utf-8") as f:
        return [AnswerRecord.model_validate_json(line) for line in f if line.strip()]
