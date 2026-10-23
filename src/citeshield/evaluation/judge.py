"""LLM judge: correctness 1-3 against the reference answer, faithfulness, citation adequacy.

The judge sees ``AnswerRecord.view()``, the same text the human labeller sees in
``citeshield-label``, so judge and human grades are comparable item by item. Every verdict
records the prompt version and the ``view_sha`` of the text it graded; calibration only
compares a verdict and a label that graded the same text.

Refusals get their own rubric (``prompts/judge_refusal.md``, versioned separately) and a
single grade, ``refusal_appropriate``. On Ollama the output budget must cover qwen3's
thinking (3k-4.5k tokens before it writes anything), so ``build_judge`` passes the
provider's ``num_predict``; a verdict cut off by the budget is an error, never a grade.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from citeshield.agent.llm import LLM
from citeshield.agent.prompts import Prompt
from citeshield.evaluation.records import AnswerRecord

PROMPTS_DIR = Path(__file__).parent / "prompts"
PROMPT_PATH = PROMPTS_DIR / "judge.md"


def judge_prompt(name: str = "judge") -> Prompt:
    return Prompt(name, (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8"))


class Grades(BaseModel):
    """The three grades, shared by judge verdicts and human labels."""

    model_config = ConfigDict(extra="forbid")

    correctness: int = Field(ge=1, le=3)
    faithful: bool
    citation_adequate: bool


class JudgeVerdict(Grades):
    id: str
    view_sha: str
    rationale: str = ""
    model: str
    prompt_version: str


class RefusalVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    view_sha: str
    refusal_appropriate: bool
    rationale: str = ""
    model: str
    prompt_version: str


def refusal_schema() -> dict[str, Any]:
    props: dict[str, Any] = {
        "rationale": {"type": "string"},
        "refusal_appropriate": {"type": "boolean"},
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def judge_schema() -> dict[str, Any]:
    props: dict[str, Any] = {
        "rationale": {"type": "string"},
        "correctness": {"type": "integer", "enum": [1, 2, 3]},
        "faithful": {"type": "boolean"},
        "citation_adequate": {"type": "boolean"},
    }
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


class Judge:
    def __init__(
        self, llm: LLM, model: str, *, effort: str = "low", max_tokens: int = 4000
    ) -> None:
        self.llm = llm
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.prompt = judge_prompt()
        self.refusal_prompt = judge_prompt("judge_refusal")

    @property
    def meta(self) -> dict[str, str]:
        return {
            "model": self.model,
            "prompt_version": self.prompt.version,
            "refusal_prompt_version": self.refusal_prompt.version,
        }

    def _ask(self, system: str, rec: AnswerRecord, schema: dict[str, Any]) -> dict[str, Any]:
        resp = self.llm.chat(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": rec.view()}],
            json_schema=schema,
            effort=self.effort,
            max_tokens=self.max_tokens,
        )
        if resp.stop_reason == "max_tokens":
            raise ValueError(f"judge output hit the {self.max_tokens}-token budget")
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("judge output is not a JSON object")
        return data

    def grade_refusal(self, rec: AnswerRecord) -> RefusalVerdict:
        data = self._ask(self.refusal_prompt.text, rec, refusal_schema())
        return RefusalVerdict.model_validate(
            {
                **data,
                "id": rec.id,
                "view_sha": rec.view_sha,
                "model": self.model,
                "prompt_version": self.refusal_prompt.version,
            }
        )

    def grade(self, rec: AnswerRecord) -> JudgeVerdict:
        data = self._ask(self.prompt.text, rec, judge_schema())
        return JudgeVerdict.model_validate(
            {
                **data,
                "id": rec.id,
                "view_sha": rec.view_sha,
                "model": self.model,
                "prompt_version": self.prompt.version,
            }
        )


def load_verdicts(path: Path) -> list[JudgeVerdict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [JudgeVerdict.model_validate_json(line) for line in f if line.strip()]
