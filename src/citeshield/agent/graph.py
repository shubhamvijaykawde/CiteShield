"""The CiteShield agent as a LangGraph state machine.

    guard ──in scope──► agent ⇄ tools ──► draft ──► validate ──► finalize
      │                   (≤ max_tool_rounds)          │  ▲
      └──out of scope──► refuse                        └──┘ repair (≤ max_draft_attempts)

* guard     Haiku classifies the question; off-topic, advice and manipulation are refused
            before any retrieval. It fails open: the guard is a cost filter, not the
            security boundary.
* agent     Sonnet decides which tools to call (it cannot be forced to call one).
* tools     allowlisted, validated, read-only; every returned chunk goes into `evidence`.
* draft     a separate call with a JSON-schema output turns the research into an answer.
* validate  Pydantic contract + "cites only retrieved chunks"; errors go back for one repair,
            then unsupported claims are dropped rather than shown.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from citeshield.agent import prompts
from citeshield.agent.config import AgentConfig
from citeshield.agent.llm import LLM, LLMRefusal, LLMResponse, Usage
from citeshield.agent.schema import draft_schema, guard_schema
from citeshield.agent.tools import EvidenceItem, ToolExecutor
from citeshield.config import config_hash
from citeshield.contracts.answer import AgentAnswer, AnswerStatus, RefusalReason
from citeshield.contracts.chunk import corpus_hash

Emit = Callable[[str, dict[str, Any]], None]

_GUARD_REFUSALS: dict[str, tuple[RefusalReason, str]] = {
    "other_law": (
        RefusalReason.OUT_OF_SCOPE,
        "This question is about law or facts outside the EU texts I cover "
        "(GDPR, AI Act, DSA, NIS2, Data Act), so I can't answer it from my sources.",
    ),
    "not_legal": (
        RefusalReason.OUT_OF_SCOPE,
        "I only answer questions about what EU digital regulation says.",
    ),
    "legal_advice": (
        RefusalReason.LEGAL_ADVICE_REQUEST,
        "I can't assess your specific situation or confirm compliance. I can explain what the "
        "regulations require of a given type of organisation or system, with citations.",
    ),
    "manipulation": (
        RefusalReason.INJECTION_DETECTED,
        "This request tries to change how I work, so I won't act on it.",
    ),
}


class AgentState(TypedDict, total=False):
    question: str
    regulations: list[str] | None
    guard_label: str
    messages: list[Any]
    pending_calls: list[Any]
    tool_rounds: int
    tool_log: list[dict[str, Any]]
    evidence: dict[str, EvidenceItem]
    security_flags: list[str]
    draft: dict[str, Any] | None
    draft_errors: list[str]
    draft_attempts: int
    answer: AgentAnswer
    usage: Usage
    timings_ms: dict[str, float]


@dataclass
class AgentRun:
    answer: AgentAnswer
    evidence: dict[str, EvidenceItem]
    tool_log: list[dict[str, Any]]
    security_flags: list[str]
    usage: Usage
    timings_ms: dict[str, float]
    guard_label: str | None
    meta: dict[str, Any] = field(default_factory=dict)

    def cited_sources(self) -> dict[str, EvidenceItem]:
        ids = {c.chunk_id for c in self.answer.all_citations()}
        return {i: self.evidence[i] for i in ids if i in self.evidence}


class CiteShieldAgent:
    def __init__(self, llm: LLM, tools: ToolExecutor, cfg: AgentConfig | None = None) -> None:
        self.llm = llm
        self.tools = tools
        self.cfg = cfg or AgentConfig()
        corpus = tools.corpus
        self.corpus_version = corpus_hash(corpus)
        self.system = prompts.load("agent_system").text.format(
            corpus_listing=_corpus_listing(tools)
        )
        self.tool_specs = tools.specs()
        self.meta = {
            "agent_config_hash": config_hash(self.cfg),
            "retriever_config_hash": tools.retriever.config_hash,
            "corpus_version": self.corpus_version,
            "prompt_versions": prompts.versions(),
            "llm_provider": getattr(llm, "provider", type(llm).__name__),
            "agent_model": self.cfg.agent_model,
            "guard_model": self.cfg.guard_model,
        }
        self.graph = self._build()

    # ------------------------------------------------------------------ public

    def run(
        self, question: str, *, regulations: list[str] | None = None, emit: Emit | None = None
    ) -> AgentRun:
        emit = emit or (lambda _t, _p: None)
        state: AgentState = {
            "question": question.strip(),
            "regulations": regulations or None,
            "messages": [],
            "tool_rounds": 0,
            "tool_log": [],
            "evidence": {},
            "security_flags": [],
            "draft_errors": [],
            "draft_attempts": 0,
            "usage": Usage(),
            "timings_ms": {},
        }
        t0 = time.perf_counter()
        try:
            final: AgentState = self.graph.invoke(
                state, config={"configurable": {"emit": emit}, "recursion_limit": 60}
            )
        except LLMRefusal as e:
            final = {
                **state,
                "answer": _refusal(
                    RefusalReason.OUT_OF_SCOPE,
                    "The model declined to answer this request.",
                    caveats=[str(e)],
                ),
            }
        timings = dict(final.get("timings_ms", {}))
        timings["total"] = (time.perf_counter() - t0) * 1000
        answer = final["answer"].model_copy(update={"corpus_version": self.corpus_version})
        run = AgentRun(
            answer=answer,
            evidence=final.get("evidence", {}),
            tool_log=final.get("tool_log", []),
            security_flags=final.get("security_flags", []),
            usage=final.get("usage", Usage()),
            timings_ms=timings,
            guard_label=final.get("guard_label"),
            meta=self.meta,
        )
        emit("final", serialize_run(run))
        return run

    # ------------------------------------------------------------------ graph

    def _build(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("guard", self._guard)
        g.add_node("refuse", self._refuse)
        g.add_node("agent", self._agent)
        g.add_node("tools", self._tools)
        g.add_node("draft", self._draft)
        g.add_node("validate", self._validate)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "guard")
        g.add_conditional_edges("guard", _after_guard, {"agent": "agent", "refuse": "refuse"})
        g.add_conditional_edges("agent", _after_agent, {"tools": "tools", "draft": "draft"})
        g.add_conditional_edges("tools", self._after_tools, {"agent": "agent", "draft": "draft"})
        g.add_edge("draft", "validate")
        g.add_conditional_edges(
            "validate", self._after_validate, {"draft": "draft", "finalize": "finalize"}
        )
        g.add_edge("refuse", END)
        g.add_edge("finalize", END)
        return g.compile()

    def _call(self, state: AgentState, key: str, **kwargs: Any) -> LLMResponse:
        t0 = time.perf_counter()
        # Snapshot the history: callers keep appending to their list after the call.
        kwargs["messages"] = list(kwargs["messages"])
        resp = self.llm.chat(max_tokens=self.cfg.max_tokens, **kwargs)
        state["usage"].add(resp.usage)
        timings = state["timings_ms"]
        timings[key] = timings.get(key, 0.0) + (time.perf_counter() - t0) * 1000
        return resp

    def _guard(self, state: AgentState, config: RunnableConfig) -> AgentState:
        emit = _emitter(config)
        emit("status", {"stage": "guard"})
        question = state["question"]
        if not question or len(question) > self.cfg.max_question_chars:
            return {"guard_label": "not_legal"}
        try:
            resp = self._call(
                state,
                "guard",
                model=self.cfg.guard_model,
                system=prompts.load("guard").text,
                messages=[{"role": "user", "content": f"<question>\n{question}\n</question>"}],
                json_schema=guard_schema(),
                effort=self.cfg.guard_effort,
            )
            label = str(resp.json().get("label", "in_scope"))
        except (ValueError, KeyError, AttributeError):
            label = "in_scope"  # fail open; the research step can still refuse
        emit("guard", {"label": label})
        return {"guard_label": label, "usage": state["usage"], "timings_ms": state["timings_ms"]}

    def _refuse(self, state: AgentState, config: RunnableConfig) -> AgentState:
        reason, summary = _GUARD_REFUSALS.get(
            state.get("guard_label", ""), _GUARD_REFUSALS["not_legal"]
        )
        return {"answer": _refusal(reason, summary)}

    def _agent(self, state: AgentState, config: RunnableConfig) -> AgentState:
        emit = _emitter(config)
        messages = list(state["messages"])
        if not messages:
            messages.append({"role": "user", "content": _question_block(state)})
        emit("status", {"stage": "research", "round": state["tool_rounds"] + 1})
        resp = self._call(
            state,
            "agent",
            model=self.cfg.agent_model,
            system=self.system,
            messages=messages,
            tools=self.tool_specs,
            effort=self.cfg.agent_effort,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.text.strip():
            emit("note", {"text": resp.text.strip()})
        return {
            "messages": messages,
            "pending_calls": resp.tool_calls,
            "tool_rounds": state["tool_rounds"] + (1 if resp.tool_calls else 0),
            "usage": state["usage"],
            "timings_ms": state["timings_ms"],
        }

    def _tools(self, state: AgentState, config: RunnableConfig) -> AgentState:
        emit = _emitter(config)
        evidence = dict(state["evidence"])
        tool_log = list(state["tool_log"])
        flags = list(state["security_flags"])
        results = []
        t0 = time.perf_counter()
        for call in state.get("pending_calls", []):
            emit("tool_call", {"name": call.name, "input": call.input})
            outcome = self.tools.execute(call.name, call.input)
            for item in outcome.evidence:
                evidence.setdefault(item.chunk_id, item)
            if outcome.security_flag:
                flags.append(outcome.security_flag)
            entry = {
                "name": call.name,
                "input": call.input,
                "is_error": outcome.is_error,
                "n_chunks": len(outcome.evidence),
                "ms": round(outcome.elapsed_ms, 1),
            }
            tool_log.append(entry)
            emit(
                "tool_result",
                {
                    **entry,
                    "sources": [
                        {
                            "chunk_id": i.chunk_id,
                            "label": i.label,
                            "score": i.score,
                            "via": i.origin,
                        }
                        for i in outcome.evidence
                    ],
                    "error": outcome.content if outcome.is_error else None,
                },
            )
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": call.id,
                "content": outcome.content,
            }
            if outcome.is_error:
                block["is_error"] = True
            results.append(block)
        timings = state["timings_ms"]
        timings["tools"] = timings.get("tools", 0.0) + (time.perf_counter() - t0) * 1000
        # All results of one turn go back in a single user message.
        messages = [*state["messages"], {"role": "user", "content": results}]
        return {
            "messages": messages,
            "pending_calls": [],
            "evidence": evidence,
            "tool_log": tool_log,
            "security_flags": flags,
            "timings_ms": timings,
        }

    def _after_tools(self, state: AgentState) -> Literal["agent", "draft"]:
        return "draft" if state["tool_rounds"] >= self.cfg.max_tool_rounds else "agent"

    def _draft(self, state: AgentState, config: RunnableConfig) -> AgentState:
        emit = _emitter(config)
        attempt = state["draft_attempts"] + 1
        emit("status", {"stage": "draft", "attempt": attempt})
        messages = list(state["messages"])
        if attempt == 1:
            messages.append({"role": "user", "content": prompts.load("draft").text})
        resp = self._call(
            state,
            "draft",
            model=self.cfg.agent_model,
            system=self.system,
            messages=messages,
            tools=self.tool_specs,
            tool_choice_none=True,
            json_schema=draft_schema(),
            effort=self.cfg.draft_effort,
        )
        messages.append({"role": "assistant", "content": resp.content})
        try:
            draft = resp.json()
            if not isinstance(draft, dict):
                raise ValueError("draft is not a JSON object")
            errors: list[str] = []
        except ValueError as e:
            draft, errors = None, [f"output is not valid JSON: {e}"]
        return {
            "messages": messages,
            "draft": draft,
            "draft_errors": errors,
            "draft_attempts": attempt,
            "usage": state["usage"],
            "timings_ms": state["timings_ms"],
        }

    def _validate(self, state: AgentState, config: RunnableConfig) -> AgentState:
        draft = state.get("draft")
        errors = list(state["draft_errors"])
        answer: AgentAnswer | None = None
        if draft is not None:
            normalised = {**draft, "checklist": draft.get("checklist") or None}
            try:
                answer = AgentAnswer.model_validate(normalised)
            except ValidationError as e:
                errors += [f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()]
            if answer is not None:
                errors += citation_errors(answer, state["evidence"])

        if not errors and answer is not None:
            return {"answer": answer, "draft_errors": []}

        _emitter(config)("draft_rejected", {"errors": errors[:10]})
        if state["draft_attempts"] < self.cfg.max_draft_attempts:
            feedback = (
                "Your answer was rejected. Fix these problems and return the full corrected "
                "JSON:\n- " + "\n- ".join(errors[:20])
            )
            messages = [*state["messages"], {"role": "user", "content": feedback}]
            return {"messages": messages, "draft_errors": errors}
        return {"answer": salvage(draft, state["evidence"], errors), "draft_errors": errors}

    def _after_validate(self, state: AgentState) -> Literal["draft", "finalize"]:
        if state.get("answer") is not None:
            return "finalize"
        return "draft"

    def _finalize(self, state: AgentState, config: RunnableConfig) -> AgentState:
        return {}


# ---------------------------------------------------------------------------- helpers


def _emitter(config: RunnableConfig) -> Emit:
    emit: Emit | None = (config.get("configurable") or {}).get("emit")
    return emit or (lambda _t, _p: None)


def _after_guard(state: AgentState) -> Literal["agent", "refuse"]:
    return "agent" if state.get("guard_label") == "in_scope" else "refuse"


def _after_agent(state: AgentState) -> Literal["tools", "draft"]:
    return "tools" if state.get("pending_calls") else "draft"


def _question_block(state: AgentState) -> str:
    scope = ""
    if state.get("regulations"):
        scope = f"\n\nRestrict the research to: {', '.join(state['regulations'] or [])}."
    return f"<question>\n{state['question']}\n</question>{scope}"


def _corpus_listing(tools: ToolExecutor) -> str:
    seen: dict[str, str] = {}
    for c in tools.corpus:
        seen.setdefault(
            c.regulation, f"- {c.regulation} (CELEX {c.celex}, OJ text of {c.version_date})"
        )
    return "\n".join(seen[r] for r in sorted(seen))


def _refusal(reason: RefusalReason, summary: str, caveats: list[str] | None = None) -> AgentAnswer:
    return AgentAnswer(
        status=AnswerStatus.REFUSED, summary=summary, refusal_reason=reason, caveats=caveats or []
    )


def citation_errors(answer: AgentAnswer, evidence: dict[str, EvidenceItem]) -> list[str]:
    """Every cited chunk must have been returned by a tool in this run."""
    errors = []
    for c in answer.all_citations():
        if c.chunk_id not in evidence:
            errors.append(
                f"citation chunk_id {c.chunk_id!r} was not returned by any tool in this "
                "conversation; cite only ids from <source> elements"
            )
    return errors


def salvage(
    draft: dict[str, Any] | None, evidence: dict[str, EvidenceItem], errors: list[str]
) -> AgentAnswer:
    """Last resort after repairs: keep only claims whose citations all resolve."""
    note = f"Answer reduced after validation errors: {'; '.join(errors[:3])}"
    if draft:
        good = [
            cl
            for cl in draft.get("claims") or []
            if cl.get("citations") and all(ci.get("chunk_id") in evidence for ci in cl["citations"])
        ]
        if good:
            try:
                return AgentAnswer.model_validate(
                    {
                        "status": "partial",
                        "summary": str(draft.get("summary", ""))[:2000],
                        "claims": good,
                        "refusal_reason": RefusalReason.INSUFFICIENT_EVIDENCE.value,
                        "caveats": [*(draft.get("caveats") or []), note],
                    }
                )
            except ValidationError:
                pass
    return _refusal(
        RefusalReason.INSUFFICIENT_EVIDENCE,
        "I could not produce an answer that is fully supported by the sources I retrieved.",
        caveats=[note],
    )


def serialize_run(run: AgentRun) -> dict[str, Any]:
    return {
        "answer": run.answer.model_dump(mode="json"),
        "sources": {
            cid: {
                "label": it.label,
                "text": it.text,
                "url": it.source_url,
                "kind": it.kind,
            }
            for cid, it in run.cited_sources().items()
        },
        "tool_log": run.tool_log,
        "security_flags": run.security_flags,
        "guard_label": run.guard_label,
        "usage": {
            "llm_calls": run.usage.calls,
            "input_tokens": run.usage.input_tokens,
            "output_tokens": run.usage.output_tokens,
            "cache_read_tokens": run.usage.cache_read_tokens,
            "cache_write_tokens": run.usage.cache_write_tokens,
            "cost_usd": round(run.usage.cost_usd, 5),
        },
        "timings_ms": {k: round(v, 1) for k, v in run.timings_ms.items()},
        "meta": run.meta,
    }
