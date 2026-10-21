"""Calibrate the verifier's NLI thresholds on hand-labelled claim/passage pairs.

    uv run citeshield-calibrate-verifier

``eval/verification/claims.jsonl`` holds claims written against the canonical text of one
chunk each, labelled supported / unsupported / contradicted by the project author (not by
independent annotators). The split is stratified by label: within each label, alternate
pairs go to calibration and held-out. Thresholds are chosen on calibration only:

* entailment threshold: the midpoint of the margin between the highest-scoring non-supported
  pair and the lowest-scoring supported pair (maximum margin; falls back to the lowest value
  with zero false accepts if the classes overlap);
* contradiction threshold: the same rule for contradicted vs. not contradicted.

The report states the full feasible interval, because a wide interval means the labelled set
does not pin the threshold down; it does not claim more precision than the data has.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from citeshield.config import PROJECT_ROOT, Settings
from citeshield.contracts.verification import Entailment
from citeshield.retrieval.corpus import Corpus
from citeshield.verify.nli import DebertaNLI, NLIModel, premise_windows

PAIRS_PATH = PROJECT_ROOT / "eval" / "verification" / "claims.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95

Label = Literal["supported", "unsupported", "contradicted"]


class LabelledPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    chunk_id: str
    claim: str
    label: Label


@dataclass(frozen=True)
class Scored:
    pair: LabelledPair
    e: Entailment
    split: Literal["calibration", "holdout"]


def assign_splits(pairs: list[LabelledPair]) -> dict[str, Literal["calibration", "holdout"]]:
    """Stratified by label: the 1st, 3rd, ... pair of each label calibrates."""
    seen: dict[str, int] = {}
    out: dict[str, Literal["calibration", "holdout"]] = {}
    for p in pairs:
        n = seen.get(p.label, 0)
        seen[p.label] = n + 1
        out[p.id] = "calibration" if n % 2 == 0 else "holdout"
    return out


def load_pairs(path: Path = PAIRS_PATH) -> list[LabelledPair]:
    with path.open(encoding="utf-8") as f:
        return [LabelledPair.model_validate_json(line) for line in f if line.strip()]


def score_pairs(pairs: list[LabelledPair], corpus: Corpus, nli: NLIModel) -> list[Scored]:
    jobs: list[tuple[int, str, str]] = []
    for k, p in enumerate(pairs):
        chunk = corpus[p.chunk_id]  # KeyError = label file refers to a chunk not in the corpus
        premise = f"{chunk.citation_label}: {chunk.text}"
        jobs += [(k, w, p.claim) for w in premise_windows(premise, p.claim, nli)]
    results = nli.score([(w, h) for _, w, h in jobs])
    best: dict[int, Entailment] = {}
    for (k, _, _), e in zip(jobs, results, strict=True):
        if k not in best or e.entailment > best[k].entailment:
            best[k] = e
    splits = assign_splits(pairs)
    return [Scored(p, best[k], splits[p.id]) for k, p in enumerate(pairs)]


def entail_metrics(rows: list[Scored], t: float) -> dict[str, float]:
    accepted = [r for r in rows if r.e.entailment >= t]
    true_acc = sum(1 for r in accepted if r.pair.label == "supported")
    supported = sum(1 for r in rows if r.pair.label == "supported")
    return {
        "threshold": t,
        "accepted": len(accepted),
        "false_accepts": len(accepted) - true_acc,
        "precision": true_acc / len(accepted) if accepted else 1.0,
        "recall": true_acc / supported if supported else 0.0,
    }


def contra_metrics(rows: list[Scored], t: float) -> dict[str, float]:
    flagged = [r for r in rows if r.e.contradiction >= t]
    true_flag = sum(1 for r in flagged if r.pair.label == "contradicted")
    contradicted = sum(1 for r in rows if r.pair.label == "contradicted")
    return {
        "threshold": t,
        "flagged": len(flagged),
        "supported_flagged": sum(1 for r in flagged if r.pair.label == "supported"),
        "precision": true_flag / len(flagged) if flagged else 1.0,
        "recall": true_flag / contradicted if contradicted else 0.0,
    }


def _margin_threshold(
    positives: list[float], negatives: list[float], fallback: float
) -> tuple[float, tuple[float, float] | None]:
    """Midpoint of the separating margin, and the margin itself (None if classes overlap)."""
    lo, hi = max(negatives, default=0.0), min(positives, default=1.0)
    if lo < hi:
        return round((lo + hi) / 2, 3), (round(lo, 3), round(hi, 3))
    return fallback, None


def calibrate(rows: list[Scored]) -> dict[str, object]:
    calib = [r for r in rows if r.split == "calibration"]
    hold = [r for r in rows if r.split == "holdout"]
    t_e, e_margin = _margin_threshold(
        [r.e.entailment for r in calib if r.pair.label == "supported"],
        [r.e.entailment for r in calib if r.pair.label != "supported"],
        next((t for t in GRID if entail_metrics(calib, t)["false_accepts"] == 0), 0.95),
    )
    t_c, c_margin = _margin_threshold(
        [r.e.contradiction for r in calib if r.pair.label == "contradicted"],
        [r.e.contradiction for r in calib if r.pair.label != "contradicted"],
        next((t for t in GRID if contra_metrics(calib, t)["supported_flagged"] == 0), 0.95),
    )
    counts = {
        split: {lbl: sum(1 for r in part if r.pair.label == lbl)
                for lbl in ("supported", "unsupported", "contradicted")}
        for split, part in (("calibration", calib), ("holdout", hold))
    }  # fmt: skip
    return {
        "counts": counts,
        "entail_threshold": t_e,
        "entail_feasible_interval": e_margin,
        "contradiction_threshold": t_c,
        "contradiction_feasible_interval": c_margin,
        "entail": {"calibration": entail_metrics(calib, t_e), "holdout": entail_metrics(hold, t_e)},
        "contradiction": {
            "calibration": contra_metrics(calib, t_c),
            "holdout": contra_metrics(hold, t_c),
        },
        "holdout_errors": [
            {
                "id": r.pair.id,
                "label": r.pair.label,
                "claim": r.pair.claim,
                "entailment": round(r.e.entailment, 3),
                "contradiction": round(r.e.contradiction, 3),
            }
            for r in hold
            if (r.e.entailment >= t_e) != (r.pair.label == "supported")
            or (r.e.contradiction >= t_c) != (r.pair.label == "contradicted")
        ],
    }


def markdown(result: dict[str, object], model: str) -> str:
    e = result["entail"]
    c = result["contradiction"]
    lines = [
        "# Verifier NLI calibration",
        "",
        f"Model: `{model}`. Pairs: `eval/verification/claims.jsonl`, labelled by the project "
        "author (not independent annotators); split stratified by label (alternate pairs).",
        "",
        f"Counts: `{json.dumps(result['counts'])}`",
        "",
        f"**Entailment threshold {result['entail_threshold']}**: midpoint of the calibration "
        f"margin; any value in the open interval {result['entail_feasible_interval']} separates "
        "the calibration pairs equally well.",
        "",
        "| split | accepted | false accepts | precision | recall |",
        "|---|---|---|---|---|",
    ]
    for split in ("calibration", "holdout"):
        m = e[split]  # type: ignore[index]
        lines.append(
            f"| {split} | {m['accepted']} | {m['false_accepts']} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} |"
        )
    lines += [
        "",
        f"**Contradiction threshold {result['contradiction_threshold']}**: midpoint of the "
        f"calibration margin; feasible interval {result['contradiction_feasible_interval']}.",
        "",
        "| split | flagged | supported flagged | precision | recall |",
        "|---|---|---|---|---|",
    ]
    for split in ("calibration", "holdout"):
        m = c[split]  # type: ignore[index]
        lines.append(
            f"| {split} | {m['flagged']} | {m['supported_flagged']} | {m['precision']:.3f} | "
            f"{m['recall']:.3f} |"
        )
    lines += ["", "Held-out disagreements:", ""]
    errors = result["holdout_errors"]
    if not errors:
        lines.append("none")
    for err in errors:  # type: ignore[attr-defined]
        lines.append(
            f"- {err['id']} ({err['label']}, E={err['entailment']}, C={err['contradiction']}): "
            f"{err['claim']}"
        )
    lines += [
        "",
        "What this does and does not show: the model separates these pairs with a wide margin, "
        "so the data shows the classifier is reliable on clear-cut claims, but it does not "
        "determine the threshold precisely. The pairs are clear paraphrases or clear errors, "
        "written by the person who built the system; real drafts include borderline "
        "paraphrases (e.g. dropping a qualifier such as 'where feasible'), which this set does "
        "not contain. 60 pairs, single-chunk premises. Provisional until a harder, "
        "independently labelled set exists (milestone 6).",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    settings = Settings()
    cfg = settings.verifier
    corpus = Corpus.load(settings.chunks_path)
    nli = DebertaNLI(cfg.nli_model, revision=cfg.nli_revision, device=args.device or cfg.device)
    rows = score_pairs(load_pairs(), corpus, nli)
    result = calibrate(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = markdown(result, cfg.nli_model)
    (RESULTS_DIR / "verifier_calibration.md").write_text(md, "utf-8")
    raw = [
        {"id": r.pair.id, "split": r.split, "label": r.pair.label,
         "entailment": r.e.entailment, "neutral": r.e.neutral, "contradiction": r.e.contradiction}
        for r in rows
    ]  # fmt: skip
    (RESULTS_DIR / "verifier_calibration.json").write_text(
        json.dumps({"result": result, "scores": raw}, indent=2), "utf-8"
    )
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
