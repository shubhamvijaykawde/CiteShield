"""Calibrate the LLM judge against human labels: Cohen's kappa and confusion matrices.

    uv run citeshield-calibrate-judge             # judge any labelled answer not yet judged
    uv run citeshield-calibrate-judge --cached    # only verdicts already in judge.jsonl

Labels come from ``citeshield-label`` (``eval/labels/human_labels.jsonl``). Each label names
the run and the ``view_sha`` of the text it graded; a verdict is only compared with a label
that graded the same text under the current judge prompt.

Labels are split into dev and held-out halves by a stable hash of ``run_id/id``, so adding
labels never moves an existing one to the other half. Iterate on the judge prompt while
looking at dev disagreements only; the reported kappa is the held-out one (target >= 0.7).
Correctness is ordinal (1-3), so its headline kappa is quadratic-weighted; the unweighted
value is reported too. The two yes/no grades use plain Cohen's kappa.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from citeshield.config import PROJECT_ROOT, Settings
from citeshield.evaluation.judge import Grades, Judge, JudgeVerdict, judge_prompt, load_verdicts
from citeshield.evaluation.records import AnswerRecord, load_records

LABELS_PATH = PROJECT_ROOT / "eval" / "labels" / "human_labels.jsonl"
RUNS_DIR = PROJECT_ROOT / "eval" / "runs"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
TARGET_KAPPA = 0.7
GRADES = ("correctness", "faithful", "citation_adequate")

Split = Literal["dev", "holdout"]


class HumanLabel(Grades):
    id: str
    run_id: str
    view_sha: str
    note: str = ""
    labelled_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


def load_labels(path: Path = LABELS_PATH) -> list[HumanLabel]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        labels = [HumanLabel.model_validate_json(line) for line in f if line.strip()]
    # A relabel of the same answer replaces the earlier label.
    latest = {(lb.run_id, lb.id, lb.view_sha): lb for lb in labels}
    return list(latest.values())


def split_of(label: HumanLabel) -> Split:
    digest = hashlib.sha256(f"{label.run_id}/{label.id}".encode()).digest()
    return "dev" if digest[0] % 2 == 0 else "holdout"


# ------------------------------------------------------------------------ statistics


def cohen_kappa(
    a: Sequence[Any], b: Sequence[Any], categories: Sequence[Any], *, quadratic: bool = False
) -> float | None:
    """Cohen's kappa of two raters; quadratic weights for ordinal categories."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}

    def w(i: int, j: int) -> float:  # disagreement weight
        if quadratic:
            return ((i - j) / (k - 1)) ** 2 if k > 1 else 0.0
        return 0.0 if i == j else 1.0

    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        obs[idx[x]][idx[y]] += 1 / n
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    observed = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k))
    expected = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k))
    if expected == 0:
        # Both raters used a single identical category: perfect but uninformative agreement.
        return 1.0 if observed == 0 else 0.0
    return 1 - observed / expected


def confusion(a: Sequence[Any], b: Sequence[Any], categories: Sequence[Any]) -> list[list[int]]:
    """Rows: human; columns: judge."""
    idx = {c: i for i, c in enumerate(categories)}
    m = [[0] * len(categories) for _ in categories]
    for x, y in zip(a, b, strict=True):
        m[idx[x]][idx[y]] += 1
    return m


@dataclass(frozen=True)
class Pair:
    label: HumanLabel
    verdict: JudgeVerdict
    split: Split


def _categories(grade: str) -> list[Any]:
    return [1, 2, 3] if grade == "correctness" else [False, True]


def agreement(pairs: Sequence[Pair]) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(pairs)}
    for g in GRADES:
        h = [getattr(p.label, g) for p in pairs]
        j = [getattr(p.verdict, g) for p in pairs]
        cats = _categories(g)
        entry: dict[str, Any] = {
            "kappa": cohen_kappa(h, j, cats, quadratic=g == "correctness"),
            "accuracy": sum(x == y for x, y in zip(h, j, strict=True)) / len(h) if h else None,
            "confusion": confusion(h, j, cats),
        }
        if g == "correctness":
            entry["kappa_unweighted"] = cohen_kappa(h, j, cats)
        out[g] = entry
    return out


# ------------------------------------------------------------------------ pairing


def pair_labels(
    labels: Sequence[HumanLabel],
    judge: Judge | None,
    prompt_version: str,
    runs_dir: Path = RUNS_DIR,
) -> tuple[list[Pair], list[str]]:
    """Match each label with a current verdict, judging (and caching) when allowed."""
    pairs: list[Pair] = []
    skipped: list[str] = []
    records: dict[str, dict[str, AnswerRecord]] = {}
    for lb in labels:
        run_dir = runs_dir / lb.run_id
        if lb.run_id not in records:
            path = run_dir / "answers.jsonl"
            records[lb.run_id] = {r.id: r for r in load_records(path)} if path.exists() else {}
        rec = records[lb.run_id].get(lb.id)
        if rec is None or rec.view_sha != lb.view_sha:
            skipped.append(f"{lb.run_id}/{lb.id}: answer missing or changed since labelling")
            continue
        verdict = next(
            (v for v in load_verdicts(run_dir / "judge.jsonl")
             if v.id == lb.id and v.view_sha == lb.view_sha and v.prompt_version == prompt_version),
            None,
        )  # fmt: skip
        if verdict is None and judge is not None:
            try:
                verdict = judge.grade(rec)
            except Exception as e:
                skipped.append(f"{lb.run_id}/{lb.id}: judge failed ({type(e).__name__}: {e})")
                continue
            with (run_dir / "judge.jsonl").open("a", encoding="utf-8") as f:
                f.write(verdict.model_dump_json() + "\n")
        if verdict is None:
            skipped.append(f"{lb.run_id}/{lb.id}: no cached verdict for prompt {prompt_version}")
            continue
        pairs.append(Pair(lb, verdict, split_of(lb)))
    return pairs, skipped


def calibrate(pairs: Sequence[Pair]) -> dict[str, Any]:
    dev = [p for p in pairs if p.split == "dev"]
    hold = [p for p in pairs if p.split == "holdout"]
    held = agreement(hold)
    kappas = [held[g]["kappa"] for g in GRADES]
    return {
        "dev": agreement(dev),
        "holdout": held,
        "meets_target": bool(hold) and all(k is not None and k >= TARGET_KAPPA for k in kappas),
        "dev_disagreements": [
            {
                "run_id": p.label.run_id,
                "id": p.label.id,
                "human": p.label.model_dump(include=set(GRADES)),
                "judge": p.verdict.model_dump(include=set(GRADES)),
                "rationale": p.verdict.rationale,
                "note": p.label.note,
            }
            for p in dev
            if any(getattr(p.label, g) != getattr(p.verdict, g) for g in GRADES)
        ],
    }


def _matrix(m: list[list[int]], cats: list[Any]) -> list[str]:
    head = "| human \\ judge | " + " | ".join(str(c) for c in cats) + " |"
    lines = [head, "|---" * (len(cats) + 1) + "|"]
    lines += [
        f"| {c} | " + " | ".join(str(x) for x in row) + " |" for c, row in zip(cats, m, strict=True)
    ]
    return lines


def markdown(result: dict[str, Any], meta: dict[str, Any], skipped: list[str]) -> str:
    def k(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.3f}"

    lines = [
        "# LLM judge calibration",
        "",
        f"Judge `{meta['model']}`, prompt `{meta['prompt_version']}`. Human labels: "
        "`eval/labels/human_labels.jsonl`, labelled by the project author with "
        "`citeshield-label`. Split by a stable hash of run/id; the headline is held-out.",
        "",
        "| split | n | correctness kappa (quadratic) | correctness kappa | faithful kappa | "
        "citation kappa |",
        "|---|---|---|---|---|---|",
    ]
    for split in ("dev", "holdout"):
        a = result[split]
        lines.append(
            f"| {split} | {a['n']} | {k(a['correctness']['kappa'])} | "
            f"{k(a['correctness']['kappa_unweighted'])} | {k(a['faithful']['kappa'])} | "
            f"{k(a['citation_adequate']['kappa'])} |"
        )
    verdict = "meets" if result["meets_target"] else "does not meet"
    lines += [
        "",
        f"Held-out {verdict} the target (kappa >= {TARGET_KAPPA} on all three grades).",
        "",
    ]
    for g in GRADES:
        lines += [f"Held-out confusion, {g}:", ""]
        lines += _matrix(result["holdout"][g]["confusion"], _categories(g))
        lines.append("")
    lines += [f"Dev disagreements: {len(result['dev_disagreements'])}."]
    if skipped:
        lines += ["", f"Skipped labels ({len(skipped)}):", "", *(f"- {s}" for s in skipped)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--labels", type=Path, default=LABELS_PATH)
    ap.add_argument("--cached", action="store_true", help="Do not call the judge.")
    ap.add_argument("--judge-model")
    args = ap.parse_args(argv)

    labels = load_labels(args.labels)
    if not labels:
        print(f"no labels in {args.labels}; run citeshield-label first")
        return 1
    prompt = judge_prompt()
    judge = None
    model = args.judge_model or "cached"
    if not args.cached:
        from citeshield.evaluation.run_eval import build_judge

        judge = build_judge(Settings(), args.judge_model)
        model = judge.model
    pairs, skipped = pair_labels(labels, judge, prompt.version)
    result = calibrate(pairs)
    meta = {"model": model, "prompt_version": prompt.version}
    md = markdown(result, meta, skipped)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "judge_calibration.md").write_text(md, "utf-8")
    (RESULTS_DIR / "judge_calibration.json").write_text(
        json.dumps({"meta": meta, "result": result, "skipped": skipped}, indent=2), "utf-8"
    )
    print(md)
    return 0 if result["meets_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
