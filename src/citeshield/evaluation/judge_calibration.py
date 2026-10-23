"""Calibrate the LLM judge against human labels on the frozen sample.

    uv run citeshield-calibrate-judge             # judge labelled items not yet judged, report
    uv run citeshield-calibrate-judge --cached    # report from judge_verdicts.jsonl only

Inputs: eval/labels/sample.jsonl (items, split by question), human_labels.jsonl
(citeshield-label) and judge_verdicts.jsonl (cache; a verdict counts only for the exact
item text and the current prompt version). The judge runs on the configured provider; set
CITESHIELD_LLM__PROVIDER=ollama to keep it local and free.

Per grade and split: Cohen's kappa (quadratic-weighted for the ordinal correctness grade,
unweighted reported too), raw agreement, a confusion matrix (rows human, columns judge),
the human class counts, and a seeded percentile bootstrap 95% interval for kappa.

The verdict per grade is deliberately conservative. "meets target" needs, on held-out:
at least ``MIN_N`` pairs, at least ``MIN_CLASS`` human labels in each of two classes, and
both the kappa and the lower end of its interval at or above ``TARGET_KAPPA``. Otherwise it
is "below target" (enough data, not good enough) or "insufficient data". Iterate on the
judge prompt while looking at dev disagreements only; held-out errors are counted, not
listed, so they do not leak into prompt changes. Real and synthetic items are reported
separately.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from citeshield.config import PROJECT_ROOT, Settings
from citeshield.evaluation.annotations import (
    GRADES,
    LABELS_PATH,
    SAMPLE_PATH,
    VERDICTS_PATH,
    HumanLabel,
    LabelItem,
    load_labels,
    load_sample,
)
from citeshield.evaluation.judge import Judge, JudgeVerdict, RefusalVerdict, judge_prompt

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
# The judge under evaluation: prompt versions and model are frozen before held-out labels
# are used, so the held-out result measures a fixed judge (see frozen_judge.json).
FROZEN_PATH = PROJECT_ROOT / "eval" / "labels" / "frozen_judge.json"
TARGET_KAPPA = 0.7
MIN_N = 20
MIN_CLASS = 5
N_BOOT = 2000
BOOT_SEED = 20261022

CATEGORIES: dict[str, list[Any]] = {
    "correctness": [1, 2, 3],
    "faithful": [False, True],
    "citation_adequate": [False, True],
    "refusal_appropriate": [False, True],
}
Verdict = JudgeVerdict | RefusalVerdict


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
        # Both raters used one identical category: agreement, but kappa is undefined.
        return None
    return 1 - observed / expected


def confusion(a: Sequence[Any], b: Sequence[Any], categories: Sequence[Any]) -> list[list[int]]:
    """Rows: human; columns: judge."""
    idx = {c: i for i, c in enumerate(categories)}
    m = [[0] * len(categories) for _ in categories]
    for x, y in zip(a, b, strict=True):
        m[idx[x]][idx[y]] += 1
    return m


def bootstrap_ci(
    a: Sequence[Any],
    b: Sequence[Any],
    categories: Sequence[Any],
    *,
    quadratic: bool = False,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float] | None:
    """Percentile 95% interval of kappa over resampled pairs; None if mostly undefined."""
    if len(a) < 2:
        return None
    rng = random.Random(seed)  # noqa: S311 - reproducible resampling, not cryptography
    n = len(a)
    values = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        k = cohen_kappa([a[i] for i in idx], [b[i] for i in idx], categories, quadratic=quadratic)
        if k is not None:
            values.append(k)
    if len(values) < n_boot * 0.9:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


@dataclass(frozen=True)
class Pair:
    item: LabelItem
    label: HumanLabel
    verdict: Verdict


def agreement(pairs: Sequence[Pair], grade: str) -> dict[str, Any]:
    cats = CATEGORIES[grade]
    rows = [p for p in pairs if grade in GRADES[p.item.kind]]
    h = [getattr(p.label, grade) for p in rows]
    j = [getattr(p.verdict, grade) for p in rows]
    ordinal = grade == "correctness"
    counts = {str(c): sum(1 for x in h if x == c) for c in cats}
    big_classes = sum(1 for v in counts.values() if v >= MIN_CLASS)
    out: dict[str, Any] = {
        "n": len(rows),
        "kappa": cohen_kappa(h, j, cats, quadratic=ordinal),
        "ci95": bootstrap_ci(h, j, cats, quadratic=ordinal),
        "accuracy": sum(x == y for x, y in zip(h, j, strict=True)) / len(rows) if rows else None,
        "confusion": confusion(h, j, cats),
        "human_counts": counts,
        "sufficient": len(rows) >= MIN_N and big_classes >= 2,
    }
    if ordinal:
        out["kappa_unweighted"] = cohen_kappa(h, j, cats)
    return out


def grade_verdict(a: dict[str, Any]) -> str:
    if not a["sufficient"] or a["kappa"] is None or a["ci95"] is None:
        return "insufficient data"
    if a["kappa"] >= TARGET_KAPPA and a["ci95"][0] >= TARGET_KAPPA:
        return "meets target"
    return "below target"


def calibrate(pairs: Sequence[Pair]) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}, "by_origin": {}, "verdicts": {}}
    for split in ("dev", "holdout"):
        part = [p for p in pairs if p.item.split == split]
        out["splits"][split] = {g: agreement(part, g) for g in CATEGORIES}
    hold = [p for p in pairs if p.item.split == "holdout"]
    for origin in ("real", "synthetic"):
        part = [p for p in hold if p.item.origin == origin]
        out["by_origin"][origin] = {
            g: {"n": a["n"], "accuracy": a["accuracy"]}
            for g in CATEGORIES
            for a in [agreement(part, g)]
        }
    out["verdicts"] = {g: grade_verdict(out["splits"]["holdout"][g]) for g in CATEGORIES}
    out["dev_disagreements"] = [
        {
            "item_id": p.item.item_id,
            "origin": p.item.origin,
            "outcome": p.item.outcome,
            "human": p.label.grades(),
            "judge": {g: getattr(p.verdict, g) for g in GRADES[p.item.kind]},
            "rationale": p.verdict.rationale,
            "note": p.label.note,
        }
        for p in pairs
        if p.item.split == "dev"
        and any(getattr(p.label, g) != getattr(p.verdict, g) for g in GRADES[p.item.kind])
    ]
    out["holdout_disagreements"] = sum(
        1
        for p in hold
        if any(getattr(p.label, g) != getattr(p.verdict, g) for g in GRADES[p.item.kind])
    )
    return out


# ------------------------------------------------------------------------ pairing


def load_verdicts(path: Path) -> list[Verdict]:
    if not path.exists():
        return []
    out: list[Verdict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                cls = RefusalVerdict if "refusal_appropriate" in data else JudgeVerdict
                out.append(cls.model_validate(data))
    return out


def prompt_versions() -> dict[str, str]:
    return {"answer": judge_prompt().version, "refusal": judge_prompt("judge_refusal").version}


def frozen_mismatches(frozen: dict[str, Any], model: str | None) -> list[str]:
    """How the current judge differs from the frozen one (empty when it is the frozen one)."""
    current = prompt_versions()
    out = [
        f"{kind} prompt is {current[kind]}, frozen {spec['version']}"
        for kind, spec in frozen["prompts"].items()
        if current[kind] != spec["version"]
    ]
    if model is not None and model != frozen["model"]:
        out.append(f"judge model is {model}, frozen {frozen['model']}")
    return out


def pair_up(
    items: Sequence[LabelItem],
    labels: dict[str, HumanLabel],
    judge: Judge | None,
    verdicts_path: Path,
    model: str | None,
) -> tuple[list[Pair], list[str]]:
    """Pair each labelled item with a current verdict, judging (and caching) when allowed."""
    versions = prompt_versions()
    cache: dict[tuple[str, str], Verdict] = {}
    for v in load_verdicts(verdicts_path):
        kind = "refusal" if isinstance(v, RefusalVerdict) else "answer"
        if v.prompt_version == versions[kind] and (model is None or v.model == model):
            cache[(v.id, v.view_sha)] = v
    pairs: list[Pair] = []
    skipped: list[str] = []
    for item in items:
        label = labels.get(item.item_id)
        if label is None:
            continue
        verdict = cache.get((item.record.id, item.view_sha))
        if verdict is None and judge is not None:
            try:
                rec = item.record
                verdict = judge.grade_refusal(rec) if item.kind == "refusal" else judge.grade(rec)
            except Exception as e:  # report and continue with the rest
                skipped.append(f"{item.item_id}: judge failed ({type(e).__name__}: {e})")
                continue
            verdicts_path.parent.mkdir(parents=True, exist_ok=True)
            with verdicts_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(verdict.model_dump_json() + "\n")
        if verdict is None:
            skipped.append(f"{item.item_id}: no cached verdict for the current prompt")
            continue
        pairs.append(Pair(item, label, verdict))
    return pairs, skipped


# ------------------------------------------------------------------------ report


def _k(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def _matrix(m: list[list[int]], cats: list[Any]) -> list[str]:
    head = "| human \\ judge | " + " | ".join(str(c) for c in cats) + " |"
    lines = [head, "|---" * (len(cats) + 1) + "|"]
    lines += [
        f"| {c} | " + " | ".join(str(x) for x in row) + " |" for c, row in zip(cats, m, strict=True)
    ]
    return lines


def markdown(result: dict[str, Any], meta: dict[str, Any], skipped: Sequence[str]) -> str:
    lines = [
        "# LLM judge calibration",
        "",
        f"Judge `{meta['model']}`, answer prompt `{meta['prompt_versions']['answer']}`, refusal "
        f"prompt `{meta['prompt_versions']['refusal']}`. Labels: {meta['n_labels']} of "
        f"{meta['n_items']} sample items, by one annotator (the project author). Split by "
        "golden question; the held-out half decides the verdicts.",
        "",
        "This is the frozen judge (eval/labels/frozen_judge.json). Its prompt was revised on "
        "the dev split, so dev agreement is optimistic; held-out is the independent measure."
        if meta.get("frozen")
        else "NOT the frozen judge (dev iteration, --allow-unfrozen): no result here is an "
        "evaluation of the judge.",
        "",
        f"Target: kappa >= {TARGET_KAPPA} with the lower end of the bootstrap 95% interval "
        f"also >= {TARGET_KAPPA}, on at least {MIN_N} held-out pairs with at least "
        f"{MIN_CLASS} human labels in each of two classes.",
        "",
        "| grade | split | n | kappa | 95% CI | agreement | human classes | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for g in CATEGORIES:
        for split in ("dev", "holdout"):
            a = result["splits"][split][g]
            ci = "n/a" if a["ci95"] is None else f"{a['ci95'][0]:.2f} to {a['ci95'][1]:.2f}"
            verdict = result["verdicts"][g] if split == "holdout" else ""
            kappa = _k(a["kappa"])
            if g == "correctness":
                kappa += f" (unweighted {_k(a['kappa_unweighted'])})"
            lines.append(
                f"| {g} | {split} | {a['n']} | {kappa} | {ci} | {_k(a['accuracy'])} | "
                f"{a['human_counts']} | {verdict} |"
            )
    lines += ["", "Held-out agreement by origin (synthetic = deliberately perturbed answers):", ""]
    lines += ["| grade | real n | real agreement | synthetic n | synthetic agreement |",
              "|---|---|---|---|---|"]  # fmt: skip
    for g in CATEGORIES:
        r, s = result["by_origin"]["real"][g], result["by_origin"]["synthetic"][g]
        lines.append(f"| {g} | {r['n']} | {_k(r['accuracy'])} | {s['n']} | {_k(s['accuracy'])} |")
    for g in CATEGORIES:
        a = result["splits"]["holdout"][g]
        if a["n"]:
            lines += ["", f"Held-out confusion, {g}:", "", *_matrix(a["confusion"], CATEGORIES[g])]
    lines += [
        "",
        f"Held-out disagreements: {result['holdout_disagreements']} (not listed, to keep "
        "them out of prompt iteration).",
        "",
        f"Dev disagreements ({len(result['dev_disagreements'])}):",
        "",
    ]
    for d in result["dev_disagreements"]:
        lines.append(
            f"- {d['item_id']} ({d['origin']}, {d['outcome']}): human {d['human']}, judge "
            f"{d['judge']}. Judge: {d['rationale'][:200]}"
            + (f" Note: {d['note']}" if d["note"] else "")
        )
    if skipped:
        lines += ["", f"Skipped ({len(skipped)}):", "", *(f"- {s}" for s in skipped)]
    lines += [
        "",
        "Limits: one annotator, no inter-annotator agreement; a small sample, so the intervals "
        "are wide; synthetic perturbations are simpler than real failures.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    ap.add_argument("--labels", type=Path, default=LABELS_PATH)
    ap.add_argument("--verdicts", type=Path, default=VERDICTS_PATH)
    ap.add_argument("--cached", action="store_true", help="Do not call the judge.")
    ap.add_argument("--judge-model")
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--frozen", type=Path, default=FROZEN_PATH)
    ap.add_argument(
        "--allow-unfrozen",
        action="store_true",
        help="Run with a prompt or model other than the frozen judge (dev iteration only).",
    )
    args = ap.parse_args(argv)
    # Rationales and notes contain typographic characters ("≥", curly quotes); a
    # redirected Windows console is cp1252. The files are UTF-8; the echo must not crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    items = load_sample(args.sample)
    labels, problems = load_labels(args.labels, items)
    for p in problems:
        print(f"label problem: {p}")
    if not labels:
        print(f"no valid labels in {args.labels}; run citeshield-label first")
        return 1
    judge = None
    if not args.cached:
        from citeshield.evaluation.run_eval import build_judge

        judge = build_judge(Settings(), args.judge_model)
    model = judge.model if judge else args.judge_model
    frozen = json.loads(args.frozen.read_text("utf-8")) if args.frozen.exists() else None
    drift = frozen_mismatches(frozen, model) if frozen else []
    if drift and not args.allow_unfrozen:
        print("the judge is not the frozen one:", *drift, sep="\n- ")
        print("use --allow-unfrozen for dev iteration; the frozen judge is evaluated as is")
        return 2
    pairs, skipped = pair_up(items, labels, judge, args.verdicts, model)
    result = calibrate(pairs)
    meta = {
        "model": model or "cached (any model)",
        "prompt_versions": prompt_versions(),
        "n_labels": len(labels),
        "n_items": len(items),
        "label_problems": problems,
        "frozen": bool(frozen) and not drift,
        "frozen_drift": drift,
    }
    md = markdown(result, meta, skipped)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "judge_calibration.md").write_text(md, "utf-8", newline="\n")
    (args.out_dir / "judge_calibration.json").write_text(
        json.dumps({"meta": meta, "result": result, "skipped": list(skipped)}, indent=2), "utf-8"
    )
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
