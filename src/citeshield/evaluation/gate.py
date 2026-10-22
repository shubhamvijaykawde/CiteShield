"""CI gate: compare an eval run's metrics with the committed baseline.

    uv run citeshield-eval-gate eval/runs/<run_id>/metrics.json [--markdown comment.md]
    uv run citeshield-eval-gate eval/runs/<run_id>/metrics.json --write-baseline

The rules depend on the provider (``RULES_BY_PROVIDER``).

Anthropic (the CI gate, Claude): the run fails when, against the baseline,

* faithfulness drops by more than 0.03, or is below 0.85 in absolute terms,
* citation precision drops by more than 0.03,
* refusal F1 drops by more than 0.05,
* schema validity is below 100%.

Ollama (local, qwen3:4b): only the deterministic checks block: schema validity below 100%
and any crash. The quality rules are reported as advisory and never fail the run. Reasons:
the agent model samples, so one question's outcome changes between identical runs (q010 was
refused in some runs and answered in others); the local subset is small (14 questions, so
one flipped item moves a share by about 0.07); and the 0.85 faithfulness floor was set for
Claude. The advisory thresholds are uncalibrated until run-to-run variance has been
measured on repeated runs of the same configuration.

A run is only comparable with a baseline made on the same subset and golden items, the same
corpus, the same models (provider, agent, guard, embedder, reranker, verifier and the Ollama
model digests), the same LLM decoding options (num_ctx, num_predict, think) and the same
model devices and weight dtypes (fp16/bf16 and fp32 can rank near-ties differently). Prompts and
configs may differ: those are what the gate is there to test. A non-comparable run never
passes, and neither does a run whose questions were not all sent the same num_ctx.
There is one baseline per provider (``eval/baselines/<provider>.json``); a provisional one
(``--write-baseline --provisional``) goes to ``<provider>_provisional.json`` and carries its
limitations in ``baseline_info``. Baselines change only through ``--write-baseline`` in an
explicit PR, which refuses runs without recorded runtime metadata, runs with mixed options,
and (unless ``--allow-crashes``) runs with crashes.

Exit codes: 0 pass, 1 fail, 2 not comparable or no baseline.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from citeshield.config import PROJECT_ROOT
from citeshield.evaluation.metrics import METRIC_KEYS

BASELINES_DIR = PROJECT_ROOT / "eval" / "baselines"
EPS = 1e-9

# meta paths that must be equal for two runs to be comparable.
IDENTITY: tuple[tuple[str, ...], ...] = (
    ("subset",),
    ("golden_sha",),
    ("corpus_hash",),
    ("agent", "llm_provider"),
    ("agent", "agent_model"),
    ("agent", "guard_model"),
    ("agent", "verifier_model"),
    ("runtime", "models"),
    ("runtime", "llm_options"),
    ("runtime", "devices"),
    ("runtime", "dtypes"),
)

Value = float | None


@dataclass(frozen=True)
class Rule:
    metric: str
    description: str
    failed: Callable[[Value, Value], bool]
    blocking: bool = True  # advisory rules are reported but never fail the run


def _drop_more_than(limit: float) -> Callable[[Value, Value], bool]:
    def failed(base: Value, cur: Value) -> bool:
        if cur is None:
            return base is not None
        return base is not None and base - cur > limit + EPS

    return failed


def _below(floor: float) -> Callable[[Value, Value], bool]:
    return lambda _base, cur: cur is None or cur < floor - EPS


def _above(ceiling: float) -> Callable[[Value, Value], bool]:
    return lambda _base, cur: cur is not None and cur > ceiling + EPS


RULES = (
    Rule("faithfulness", "drop > 0.03", _drop_more_than(0.03)),
    Rule("faithfulness", "< 0.85", _below(0.85)),
    Rule("citation_precision", "drop > 0.03", _drop_more_than(0.03)),
    Rule("refusal_f1", "drop > 0.05", _drop_more_than(0.05)),
    Rule("schema_validity", "< 1.00", _below(1.0)),
)
OLLAMA_RULES = (
    Rule("schema_validity", "< 1.00", _below(1.0)),
    Rule("n_crashed", "> 0", _above(0.0)),
    Rule("faithfulness", "drop > 0.10", _drop_more_than(0.10), blocking=False),
    Rule("citation_precision", "drop > 0.10", _drop_more_than(0.10), blocking=False),
    Rule("refusal_f1", "drop > 0.15", _drop_more_than(0.15), blocking=False),
)
RULES_BY_PROVIDER = {"anthropic": RULES, "ollama": OLLAMA_RULES}


def rules_for(meta: dict[str, Any]) -> tuple[Rule, ...]:
    return RULES_BY_PROVIDER.get(_get(meta, ("agent", "llm_provider")), RULES)


def _get(meta: dict[str, Any], path: tuple[str, ...]) -> Any:
    v: Any = meta
    for key in path:
        v = v.get(key) if isinstance(v, dict) else None
    return v


def mismatches(base_meta: dict[str, Any], cur_meta: dict[str, Any]) -> list[str]:
    return [
        f"{'.'.join(p)}: baseline {_get(base_meta, p)!r}, run {_get(cur_meta, p)!r}"
        for p in IDENTITY
        if _get(base_meta, p) != _get(cur_meta, p)
    ]


@dataclass
class GateResult:
    status: str  # pass | fail | not_comparable
    failures: list[str]
    mismatches: list[str]
    markdown: str
    advisories: list[str] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return {"pass": 0, "fail": 1}.get(self.status, 2)


def _fmt(v: Value) -> str:
    return "n/a" if v is None else f"{v:.3f}"


def evaluate(baseline: dict[str, Any], current: dict[str, Any]) -> GateResult:
    diff = mismatches(baseline["meta"], current["meta"])
    base, cur = baseline["metrics"], current["metrics"]
    rules = rules_for(current["meta"])

    def broken(blocking: bool) -> list[str]:
        return [
            f"{r.metric} {r.description} (baseline {_fmt(base.get(r.metric))}, "
            f"run {_fmt(cur.get(r.metric))})"
            for r in rules
            if r.blocking is blocking and r.failed(base.get(r.metric), cur.get(r.metric))
        ]

    failures, advisories = broken(True), broken(False)
    if current["meta"].get("config_consistent") is False:
        diff.append("the run sent different num_ctx values to different questions")
    status = "not_comparable" if diff else "fail" if failures else "pass"

    gated = {r.metric for r in rules}
    icon = {"pass": "PASS", "fail": "FAIL", "not_comparable": "NOT COMPARABLE"}[status]
    info = baseline.get("baseline_info") or {}
    lines = [f"### Eval gate: {icon}", ""]
    if info.get("status") == "provisional":
        lines += ["**Provisional baseline.** " + " ".join(info.get("notes", [])), ""]
    lines += [
        f"Run `{current['meta'].get('run_id')}` vs baseline "
        f"`{baseline['meta'].get('run_id')}` ({current['meta'].get('subset')}, "
        f"{int(cur.get('n') or 0)} questions, provider "
        f"`{_get(current['meta'], ('agent', 'llm_provider'))}`).",
        "",
        "| metric | baseline | run | delta | gate |",
        "|---|---|---|---|---|",
    ]
    for k in METRIC_KEYS:
        if k == "n" or (k not in base and k not in cur):
            continue
        b, c = base.get(k), cur.get(k)
        delta = "" if b is None or c is None else f"{c - b:+.3f}"
        gate = ""
        if k in gated:
            own = [r for r in rules if r.metric == k]
            bad = [r.description for r in own if r.blocking and r.failed(b, c)]
            soft = [r.description for r in own if not r.blocking and r.failed(b, c)]
            parts = (["fail: " + ", ".join(bad)] if bad else []) + (
                ["advisory: " + ", ".join(soft)] if soft else []
            )
            default = "ok" if any(r.blocking for r in own) else "ok (advisory)"
            gate = "; ".join(parts) or default
        lines.append(f"| {k} | {_fmt(b)} | {_fmt(c)} | {delta} | {gate} |")
    if diff:
        lines += ["", "Not comparable with the baseline:", "", *(f"- {d}" for d in diff)]
    if failures and not diff:
        lines += ["", "Failed rules:", "", *(f"- {f}" for f in failures)]
    if advisories and not diff:
        lines += ["", "Advisory (not blocking):", "", *(f"- {a}" for a in advisories)]
    return GateResult(status, failures, diff, "\n".join(lines) + "\n", advisories)


def baseline_problems(current: dict[str, Any], *, allow_crashes: bool = False) -> list[str]:
    meta, m = current["meta"], current["metrics"]
    problems = []
    if not meta.get("runtime"):
        problems.append("the run has no runtime metadata (models, LLM options, config hashes)")
    if meta.get("config_consistent") is not True:
        problems.append(
            f"LLM options not consistent across questions: {meta.get('num_ctx_observed')}"
        )
    if (m.get("n_crashed") or 0) > 0 and not allow_crashes:
        problems.append(f"{int(m['n_crashed'])} questions crashed")
    return problems


def baseline_path(current: dict[str, Any], *, provisional: bool = False) -> Path:
    provider = _get(current["meta"], ("agent", "llm_provider")) or "unknown"
    return BASELINES_DIR / f"{provider}{'_provisional' if provisional else ''}.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("metrics", type=Path, help="metrics.json of the run to check")
    ap.add_argument("--baseline", type=Path, help="Default: eval/baselines/<provider>.json")
    ap.add_argument("--markdown", type=Path, help="Also write the report here (PR comment).")
    ap.add_argument("--write-baseline", action="store_true", help="Make this run the baseline.")
    ap.add_argument("--allow-crashes", action="store_true", help="Baseline despite crashes.")
    ap.add_argument(
        "--provisional",
        action="store_true",
        help="Label the baseline provisional (default path <provider>_provisional.json).",
    )
    ap.add_argument("--note", action="append", default=[], help="Limitation to record (repeat).")
    args = ap.parse_args(argv)

    current = json.loads(args.metrics.read_text("utf-8"))
    path = args.baseline or baseline_path(current, provisional=args.provisional)
    if args.write_baseline:
        problems = baseline_problems(current, allow_crashes=args.allow_crashes)
        if problems:
            print("refusing to write a baseline:", *problems, sep="\n- ")
            return 1
        path.parent.mkdir(parents=True, exist_ok=True)
        keep: dict[str, Any] = {
            "baseline_info": {
                "status": "provisional" if args.provisional else "reference",
                "notes": args.note,
                "source_run": current["meta"].get("run_id"),
                "n_questions": current["metrics"].get("n"),
            },
            "meta": current["meta"],
            "metrics": current["metrics"],
        }
        if "by_category" in current:
            keep["by_category"] = current["by_category"]
        path.write_text(json.dumps(keep, indent=2) + "\n", "utf-8")
        print(f"baseline written to {path}")
        return 0
    if not path.exists():
        md = f"### Eval gate: NO BASELINE\n\n`{path.name}` does not exist yet.\n"
        result = GateResult("not_comparable", [], ["no baseline"], md)
    else:
        result = evaluate(json.loads(path.read_text("utf-8")), current)
    if args.markdown:
        args.markdown.write_text(result.markdown, "utf-8")
    print(result.markdown)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
