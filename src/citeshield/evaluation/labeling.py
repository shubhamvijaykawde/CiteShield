"""Hand-label the frozen sample (eval/labels/sample.jsonl) for judge calibration.

    uv run citeshield-label                        # every unlabelled item
    uv run citeshield-label --split dev --limit 10
    uv run citeshield-label --coverage             # what is labelled, what is left, problems
    uv run citeshield-label --item q031-cdf5c90b1feb   # relabel one item (replaces its label)

Read eval/labels/GUIDE.md first. Each item shows exactly what the judge sees
(``AnswerRecord.view()``) and nothing else: not the split's purpose, not whether an item is
synthetic, not the judge's verdict. Items come in a fixed shuffled order so a perturbed
answer is not next to its original. Labels are appended one at a time to
eval/labels/human_labels.jsonl, so quitting loses nothing; relabelling an item replaces the
earlier label.

Answer items ask for three grades, refusal items for one:
  correctness  3 = all key points of the reference, nothing contradicts it
               2 = main point right, a key point missing or a detail wrong
               1 = main point wrong, missing or off-question
  faithful     every statement is supported by the cited source text
  citations    each claim cites the provision that establishes it
  refusal      refusing was the right decision for this question
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

from citeshield.evaluation.annotations import (
    LABELS_PATH,
    SAMPLE_PATH,
    HumanLabel,
    LabelItem,
    load_labels,
    load_sample,
)

Ask = Callable[[str], str]
GUIDE = "eval/labels/GUIDE.md"
ORDER_SEED = "citeshield-label-order-v1"


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


def labelling_order(items: Sequence[LabelItem]) -> list[LabelItem]:
    return sorted(
        items, key=lambda i: hashlib.sha256(f"{ORDER_SEED}:{i.item_id}".encode()).digest()
    )


def label_one(item: LabelItem, ask: Ask, annotator: str) -> HumanLabel | None:
    """Ask for one label; None when skipped. Raises ``Quit``."""
    print("\n" + "=" * 100)
    print(f"{item.item_id}  [{item.kind}]")
    print(item.record.view())
    print("-" * 100)
    common = {"item_id": item.item_id, "view_sha": item.view_sha, "kind": item.kind}
    if item.kind == "refusal":
        verdict = _ask_choice(ask, "refusal appropriate y/n (s = skip): ", {**YES_NO, "s": None})
        if verdict is None:
            return None
        note = ask("note (optional): ").strip()
        return HumanLabel(
            **common, refusal_appropriate=bool(verdict), note=note, annotator=annotator
        )
    first = _ask_choice(ask, "correctness 1/2/3 (s = skip): ", {"1": 1, "2": 2, "3": 3, "s": None})
    if first is None:
        return None
    faithful = _ask_choice(ask, "faithful y/n: ", YES_NO)
    adequate = _ask_choice(ask, "citations adequate y/n: ", YES_NO)
    note = ask("note (optional): ").strip()
    return HumanLabel(
        **common,
        correctness=int(str(first)),
        faithful=bool(faithful),
        citation_adequate=bool(adequate),
        note=note,
        annotator=annotator,
    )


def coverage(items: Sequence[LabelItem], labels: dict[str, HumanLabel]) -> str:
    lines = ["| split | kind | labelled | total |", "|---|---|---|---|"]
    for split in ("dev", "holdout"):
        for kind in ("answer", "refusal"):
            group = [i for i in items if i.split == split and i.kind == kind]
            done = sum(1 for i in group if i.item_id in labels)
            lines.append(f"| {split} | {kind} | {done} | {len(group)} |")
    lines.append(f"| all | all | {len(labels)} | {len(items)} |")
    return "\n".join(lines)


def label_sample(
    sample_path: Path = SAMPLE_PATH,
    labels_path: Path = LABELS_PATH,
    *,
    split: str | None = None,
    limit: int | None = None,
    annotator: str = "author",
    item: str | None = None,
    ask: Ask = input,
) -> int:
    items = load_sample(sample_path)
    labels, problems = load_labels(labels_path, items)
    for p in problems:
        print(f"warning: {p}")
    if item is not None:
        # Relabel exactly one item, labelled or not. The new line is appended; load_labels
        # keeps the latest label per item, so the earlier one stays in the file as history.
        todo = [i for i in items if i.item_id == item]
        if not todo:
            raise SystemExit(f"unknown item {item!r}")
        if item in labels:
            print(f"current label: {labels[item].grades()} note: {labels[item].note!r}")
    else:
        todo = [i for i in labelling_order(items) if i.item_id not in labels]
    if split:
        todo = [i for i in todo if i.split == split]
    if limit is not None:
        todo = todo[:limit]
    print(f"{len(todo)} items to label (guide: {GUIDE})")
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    for entry in todo:
        try:
            label = label_one(entry, ask, annotator)
        except (Quit, EOFError):
            break
        if label is None:
            continue
        with labels_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(label.model_dump_json() + "\n")
        n += 1
    print(f"saved {n} labels to {labels_path}")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    ap.add_argument("--labels", type=Path, default=LABELS_PATH)
    ap.add_argument("--split", choices=("dev", "holdout"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--annotator", default="author")
    ap.add_argument("--item", help="Relabel this one item_id, replacing its current label.")
    ap.add_argument("--coverage", action="store_true", help="Report coverage and exit.")
    args = ap.parse_args(argv)
    if args.coverage:
        items = load_sample(args.sample)
        labels, problems = load_labels(args.labels, items)
        print(coverage(items, labels))
        for p in problems:
            print(f"problem: {p}")
        return 1 if problems else 0
    label_sample(args.sample, args.labels, split=args.split, limit=args.limit, item=args.item,
                 annotator=args.annotator)  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
