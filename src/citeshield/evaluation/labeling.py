"""Hand-label agent answers for judge calibration.

    uv run citeshield-label eval/runs/<run_id>            # label what is not labelled yet
    uv run citeshield-label eval/runs/<run_id> --limit 20

Shows exactly what the judge sees (``AnswerRecord.view()``) and asks for the same three
grades. Only substantive answers to answerable questions are labelled; refusals are scored
deterministically. Labels are appended to ``eval/labels/human_labels.jsonl`` one at a time,
so quitting loses nothing. The judge's own verdict is never shown while labelling.

Rubric (same as the judge prompt, evaluation/prompts/judge.md):
  correctness  3 = all key points of the reference, nothing contradicts it
               2 = main point right, a key point missing or a detail wrong
               1 = main point wrong, missing or off-question
  faithful     every statement is supported by the cited source text
  citations    each claim cites the provision that establishes it
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from citeshield.evaluation.judge_calibration import LABELS_PATH, HumanLabel, load_labels
from citeshield.evaluation.records import AnswerRecord, load_records

Ask = Callable[[str], str]


class Quit(Exception):
    pass


def _ask_choice(ask: Ask, prompt: str, choices: dict[str, object]) -> object:
    while True:
        raw = ask(prompt).strip().lower()
        if raw == "q":
            raise Quit
        if raw in choices:
            return choices[raw]
        print(f"  answer one of {', '.join(choices)} (or q to quit)")


YES_NO: dict[str, object] = {"y": True, "n": False}


def label_one(rec: AnswerRecord, run_id: str, ask: Ask) -> HumanLabel | None:
    """Ask for one label; None when skipped. Raises ``Quit``."""
    print("\n" + "=" * 100)
    print(f"{run_id}/{rec.id} ({rec.category.value})")
    print(rec.view())
    print("-" * 100)
    first = _ask_choice(ask, "correctness 1/2/3 (s = skip): ", {"1": 1, "2": 2, "3": 3, "s": None})
    if first is None:
        return None
    faithful = _ask_choice(ask, "faithful y/n: ", YES_NO)
    adequate = _ask_choice(ask, "citations adequate y/n: ", YES_NO)
    note = ask("note (optional): ").strip()
    return HumanLabel(
        id=rec.id,
        run_id=run_id,
        view_sha=rec.view_sha,
        correctness=int(str(first)),
        faithful=bool(faithful),
        citation_adequate=bool(adequate),
        note=note,
    )


def pending(
    records: list[AnswerRecord], run_id: str, labels: list[HumanLabel]
) -> list[AnswerRecord]:
    done = {(lb.run_id, lb.id, lb.view_sha) for lb in labels}
    return [r for r in records if r.judgeable and (run_id, r.id, r.view_sha) not in done]


def label_run(
    run_dir: Path, labels_path: Path = LABELS_PATH, *, limit: int | None = None, ask: Ask = input
) -> int:
    run_id = run_dir.name
    todo = pending(load_records(run_dir / "answers.jsonl"), run_id, load_labels(labels_path))
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(todo)} answers to label in {run_id}")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    for rec in todo:
        try:
            label = label_one(rec, run_id, ask)
        except (Quit, EOFError):
            break
        if label is None:
            continue
        with labels_path.open("a", encoding="utf-8") as f:
            f.write(label.model_dump_json() + "\n")
        n += 1
    print(f"saved {n} labels to {labels_path}")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--labels", type=Path, default=LABELS_PATH)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    label_run(args.run_dir, args.labels, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
