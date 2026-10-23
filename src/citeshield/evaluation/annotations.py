"""Human-annotation data: the frozen labelling sample and the labels a person gives it.

``eval/labels/sample.jsonl`` holds one ``LabelItem`` per line: a complete ``AnswerRecord``
(so labelling and judging need no gitignored run data), its split, and where it came from.
``origin``, ``outcome`` and ``perturbation`` describe how the item was made; they are never
shown to the labeller or the judge, both of which see only ``record.view()``.

``eval/labels/human_labels.jsonl`` holds ``HumanLabel`` lines written by ``citeshield-label``.
A label is tied to the exact text that was graded through ``view_sha``.

Two kinds of item, with their own grades:

* ``answer``: ``correctness`` (1-3), ``faithful``, ``citation_adequate``
* ``refusal``: ``refusal_appropriate``
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from citeshield.config import PROJECT_ROOT
from citeshield.evaluation.records import AnswerRecord

ItemKind = Literal["answer", "refusal"]
Origin = Literal["real", "synthetic"]
Outcome = Literal[
    "answered",
    "partial",
    "correct_refusal",
    "wrong_refusal",
    "perturbed_unsupported",
    "perturbed_wrong_citation",
    "perturbed_contradiction",
]
Split = Literal["dev", "holdout"]

ANSWER_GRADES = ("correctness", "faithful", "citation_adequate")
REFUSAL_GRADES = ("refusal_appropriate",)
GRADES: dict[str, tuple[str, ...]] = {"answer": ANSWER_GRADES, "refusal": REFUSAL_GRADES}

SAMPLE_PATH = PROJECT_ROOT / "eval" / "labels" / "sample.jsonl"
LABELS_PATH = SAMPLE_PATH.with_name("human_labels.jsonl")
VERDICTS_PATH = SAMPLE_PATH.with_name("judge_verdicts.jsonl")


class LabelItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    question_id: str
    kind: ItemKind
    origin: Origin
    outcome: Outcome
    split: Split
    source_run: str
    # What a synthetic perturbation changed. Never shown to the labeller or the judge.
    perturbation: str | None = None
    record: AnswerRecord
    view_sha: str

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.view_sha != self.record.view_sha:
            raise ValueError(f"{self.item_id}: view_sha does not match the record text")
        if self.question_id != self.record.id:
            raise ValueError(f"{self.item_id}: question_id differs from the record id")
        if (self.kind == "refusal") != self.record.refused:
            raise ValueError(
                f"{self.item_id}: kind {self.kind} but record refused={self.record.refused}"
            )
        synthetic = self.outcome.startswith("perturbed_")
        if synthetic != (self.origin == "synthetic") or synthetic != (
            self.perturbation is not None
        ):
            raise ValueError(f"{self.item_id}: origin, outcome and perturbation disagree")
        return self


class HumanLabel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    view_sha: str
    kind: ItemKind
    correctness: int | None = Field(None, ge=1, le=3)
    faithful: bool | None = None
    citation_adequate: bool | None = None
    refusal_appropriate: bool | None = None
    note: str = ""
    annotator: str = "author"
    labelled_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    @model_validator(mode="after")
    def _grades_match_kind(self) -> Self:
        own, other = GRADES[self.kind], GRADES["refusal" if self.kind == "answer" else "answer"]
        missing = [g for g in own if getattr(self, g) is None]
        extra = [g for g in other if getattr(self, g) is not None]
        if missing or extra:
            raise ValueError(f"{self.kind} label: missing {missing}, not allowed {extra}")
        return self

    def grades(self) -> dict[str, int | bool]:
        return {g: getattr(self, g) for g in GRADES[self.kind]}


def load_sample(path: Path = SAMPLE_PATH) -> list[LabelItem]:
    with path.open(encoding="utf-8") as f:
        items = [LabelItem.model_validate_json(line) for line in f if line.strip()]
    ids = [i.item_id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item_id in the labelling sample")
    return items


def load_labels(path: Path, sample: list[LabelItem]) -> tuple[dict[str, HumanLabel], list[str]]:
    """Valid labels by item_id (a later label for an item replaces an earlier one), and
    the problems found: malformed lines, unknown items, stale text, wrong kind."""
    by_id = {i.item_id: i for i in sample}
    labels: dict[str, HumanLabel] = {}
    problems: list[str] = []
    if not path.exists():
        return labels, problems
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                lb = HumanLabel.model_validate_json(line)
            except ValidationError as e:
                problems.append(f"line {n}: {e.errors()[0]['msg']}")
                continue
            item = by_id.get(lb.item_id)
            if item is None:
                problems.append(f"line {n}: unknown item {lb.item_id}")
            elif lb.view_sha != item.view_sha:
                problems.append(f"line {n}: {lb.item_id} labels different text (stale view_sha)")
            elif lb.kind != item.kind:
                problems.append(
                    f"line {n}: {lb.item_id} is a {item.kind} item, label says {lb.kind}"
                )
            else:
                labels[lb.item_id] = lb
    return labels, problems
