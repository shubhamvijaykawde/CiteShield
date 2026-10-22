"""LLM judge: correctness 1-3 against the reference answer, faithfulness, citation adequacy.

The judge sees ``AnswerRecord.view()``, the same text the human labeller sees in
``citeshield-label``, so judge and human grades are comparable item by item. Every verdict
records the prompt version and the ``view_sha`` of the text it graded; calibration only
compares a verdict and a label that graded the same text.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from citeshield.agent.llm import LLM
from citeshield.agent.prompts import Prompt
from citeshield.evaluation.records import AnswerRecord

PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.md"


def judge_prompt() -> Prompt:
    return Prompt("judge", PROMPT_PATH.read_text(encoding="utf-8"))


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

    @property
    def meta(self) -> dict[str, str]:
        return {"model": self.model, "prompt_version": self.prompt.version}

    def grade(self, rec: AnswerRecord) -> JudgeVerdict:
        resp = self.llm.chat(
            model=self.model,
            system=self.prompt.text,
            messages=[{"role": "user", "content": rec.view()}],
            json_schema=judge_schema(),
            effort=self.effort,
            max_tokens=self.max_tokens,
        )
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("judge output is not a JSON object")
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
