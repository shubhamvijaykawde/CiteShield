"""Answer evaluation: run the agent on golden questions, then score, judge and report.

    uv run citeshield-eval --subset smoke                  # 40 stratified questions
    uv run citeshield-eval --subset local                  # 14 hand-picked (local Ollama)
    uv run citeshield-eval --subset full --judge           # every item, plus the LLM judge
    uv run citeshield-eval --ids q002,q054 --run-id sanity
    uv run citeshield-eval --rescore eval/runs/<run_id> [--judge]   # no agent calls
    uv run citeshield-eval --write-smoke                   # regenerate eval/golden/smoke.json

Each run writes ``eval/runs/<run_id>/``: ``meta.json`` (configs, models, prompt versions,
corpus and golden hashes), ``answers.jsonl`` (one ``AnswerRecord`` per question, appended as
it finishes), ``judge.jsonl``, ``metrics.json`` and ``report.md``. Re-running the same
``--run-id`` resumes: finished questions are skipped, but only if the agent configuration is
unchanged. The report is also copied to ``eval/results/answers_latest.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from citeshield.config import PROJECT_ROOT, Settings, config_hash
from citeshield.evaluation.golden import Category, GoldenItem, load_golden
from citeshield.evaluation.judge import Judge, JudgeVerdict, load_verdicts
from citeshield.evaluation.metrics import METRIC_KEYS, by_category, compute
from citeshield.evaluation.records import AnswerRecord, load_records
from citeshield.retrieval.corpus import Corpus

GOLDEN_PATH = PROJECT_ROOT / "eval" / "golden" / "questions.jsonl"
SMOKE_PATH = PROJECT_ROOT / "eval" / "golden" / "smoke.json"
# Small hand-picked subset for local Ollama runs (provisional baseline); see its rationale.
LOCAL_PATH = PROJECT_ROOT / "eval" / "golden" / "smoke_local.json"
RUNS_DIR = PROJECT_ROOT / "eval" / "runs"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
SMOKE_SIZE = 40
# Meta keys that must match for a resumed run or a baseline comparison to be meaningful.
AGENT_IDENTITY = (
    "agent_config_hash",
    "retriever_config_hash",
    "corpus_version",
    "prompt_versions",
    "llm_provider",
    "agent_model",
    "guard_model",
    "verifier_model",
    "verifier_thresholds",
)
# Parts of meta["runtime"] that must match for a resumed run (``code`` is informational).
RUNTIME_IDENTITY = ("models", "llm_options", "devices", "dtypes", "config_hashes")
# The eval keeps the embedder and reranker off the GPU by default: on the 8 GB dev GPU,
# qwen3:4b at num_ctx 24576 plus both retrieval models left ~125 MiB free. CPU is also what
# CI and HF Spaces run, so fp32 retrieval scores match across machines. "auto" = settings.
DEFAULT_EVAL_DEVICE = "cpu"


def _ollama_info(base_url: str, models: set[str]) -> tuple[dict[str, str | None], str | None]:
    """Digest of each model tag (a tag can be re-pulled to new weights) and server version."""
    import httpx

    try:
        tags = httpx.get(f"{base_url}/api/tags", timeout=10).json().get("models", [])
        version = httpx.get(f"{base_url}/api/version", timeout=10).json().get("version")
    except (httpx.HTTPError, ValueError):
        return dict.fromkeys(sorted(models)), None
    digests = {t.get("name"): t.get("digest") for t in tags}
    return {m: digests.get(m) for m in sorted(models)}, version


def _git() -> dict[str, Any]:
    def run(*cmd: str) -> str:
        out = subprocess.run(  # noqa: S603 - fixed git arguments
            ["git", *cmd],  # noqa: S607 - git from PATH
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip()

    return {
        "git_commit": run("rev-parse", "HEAD") or None,
        "git_dirty": bool(run("status", "--porcelain")),
    }


def eval_settings(settings: Settings, embed_device: str, rerank_device: str) -> Settings:
    """Settings with the eval's retrieval devices; "auto" keeps the configured value."""
    ret = settings.retriever
    update = {
        k: v
        for k, v in (("embed_device", embed_device), ("rerank_device", rerank_device))
        if v != "auto"
    }
    return settings.model_copy(update={"retriever": ret.model_copy(update=update)})


def actual_devices(agent: Any) -> dict[str, str | None]:
    """Devices the loaded models really ended up on (after cuda-if-available resolution)."""
    retriever = agent.tools.retriever
    return {
        "embed": getattr(retriever.searcher.encoder, "device", None),
        "rerank": getattr(retriever.reranker, "device", None) if retriever.reranker else None,
        "nli": getattr(agent.verifier.nli, "device", None),
    }


def actual_dtypes(agent: Any) -> dict[str, list[str] | None]:
    """Weight dtypes of the loaded models, read from their parameters (not from settings)."""

    def of(*modules: Any) -> list[str] | None:
        found = {str(p.dtype) for m in modules if m is not None for p in m.parameters()}
        return sorted(found) or None

    retriever = agent.tools.retriever
    enc = retriever.searcher.encoder
    return {
        "embed": of(getattr(enc, "model", None), getattr(enc, "sparse_linear", None)),
        "rerank": of(getattr(retriever.reranker, "model", None)),
        "nli": of(getattr(agent.verifier.nli, "model", None)),
    }


def runtime_identity(
    settings: Settings,
    agent_meta: dict[str, Any],
    devices: dict[str, str | None],
    dtypes: dict[str, list[str] | None] | None = None,
) -> dict[str, Any]:
    """Everything besides prompts and the golden items that decides what a run measures."""
    llm, ret, ver = settings.llm, settings.retriever, settings.verifier
    models: dict[str, Any] = {
        "llm_provider": llm.provider,
        "agent_model": agent_meta["agent_model"],
        "guard_model": agent_meta["guard_model"],
        "embed_model": ret.embed_model,
        "rerank_model": ret.rerank_model if ret.use_reranker else None,
        "nli_model": ver.nli_model,
        "nli_revision": ver.nli_revision,
    }
    llm_options: dict[str, Any] = {}
    code = _git()
    if llm.provider == "ollama":
        digests, version = _ollama_info(
            llm.ollama_base_url, {agent_meta["agent_model"], agent_meta["guard_model"]}
        )
        models["ollama_digests"] = digests
        llm_options = {
            "num_ctx": llm.ollama_num_ctx,
            "num_predict": llm.ollama_num_predict,
            "think": llm.ollama_think,
        }
        code["ollama_version"] = version
    return {
        "models": models,
        "llm_options": llm_options,
        # fp16 on cuda vs fp32 on cpu can shift rerank scores, so devices are identity too.
        "devices": devices,
        # bf16 and fp32 weights can rank near-ties differently, so dtypes are identity too.
        "dtypes": dtypes or {},
        "config_hashes": {
            "agent": agent_meta["agent_config_hash"],
            # Location and timeout do not change answers.
            "llm": config_hash(
                llm.model_copy(update={"ollama_base_url": "", "ollama_timeout_s": 1.0})
            ),
            "retriever": config_hash(ret),
            "verifier": config_hash(ver),
        },
        "code": code,
    }


# ------------------------------------------------------------------------ selection


def stratified_smoke(items: Sequence[GoldenItem], size: int = SMOKE_SIZE) -> list[str]:
    """Deterministic stratified sample: every category, in proportion, at least two each.

    Within a category the picks are evenly spaced over the id order, so both the original
    and the newer items are represented.
    """
    groups: dict[Category, list[GoldenItem]] = {}
    for it in sorted(items, key=lambda i: i.id):
        groups.setdefault(it.category, []).append(it)
    quota = {c: min(len(g), 2) for c, g in groups.items()}
    total = len(items)
    # Largest remainder on the proportional share, after the floor of two.
    while sum(quota.values()) < min(size, total):
        c = max(
            (c for c in groups if quota[c] < len(groups[c])),
            key=lambda c: (len(groups[c]) * size / total - quota[c], c.value),
        )
        quota[c] += 1
    picked = []
    for c, group in groups.items():
        n = quota[c]
        picked += [group[int((k + 0.5) * len(group) / n)].id for k in range(n)]
    return sorted(picked)


def select(
    items: list[GoldenItem],
    subset: str,
    ids: str | None,
    smoke_path: Path = SMOKE_PATH,
    local_path: Path = LOCAL_PATH,
) -> list[GoldenItem]:
    if ids:
        wanted = [s.strip() for s in ids.split(",") if s.strip()]
    elif subset in ("smoke", "local"):
        path = smoke_path if subset == "smoke" else local_path
        wanted = json.loads(path.read_text("utf-8"))["ids"]
    elif subset == "full":
        return items
    else:
        raise ValueError(f"unknown subset {subset!r}")
    by_id = {i.id: i for i in items}
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise ValueError(f"unknown golden ids: {', '.join(missing)}")
    return [by_id[w] for w in wanted]


def golden_sha(items: Sequence[GoldenItem]) -> str:
    """Hash of exactly the items run: editing a question outside the subset keeps it."""
    h = hashlib.sha256()
    for it in sorted(items, key=lambda i: i.id):
        h.update(it.model_dump_json().encode() + b"\n")
    return h.hexdigest()[:12]


# ------------------------------------------------------------------------ run


def _append(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_items(agent: Any, items: Sequence[GoldenItem], answers_path: Path) -> None:
    """Strictly one question at a time: no threads or concurrent requests to the LLM."""
    done = {r.id for r in load_records(answers_path)} if answers_path.exists() else set()
    todo = [i for i in items if i.id not in done]
    for n, item in enumerate(todo, 1):
        t0 = time.perf_counter()
        try:
            rec = AnswerRecord.from_run(item, agent.run(item.question))
        except Exception as e:
            rec = AnswerRecord.from_error(item, e, (time.perf_counter() - t0) * 1000)
        _append(answers_path, rec.model_dump_json())
        status = rec.error or (rec.answer.status.value if rec.answer else "?")
        print(f"[{n}/{len(todo)}] {item.id} {status[:120]} ({rec.latency_ms / 1000:.1f}s)")


def judge_records(judge: Judge, records: Sequence[AnswerRecord], path: Path) -> None:
    have = {
        (v.id, v.view_sha) for v in load_verdicts(path) if v.prompt_version == judge.prompt.version
    }
    for rec in records:
        if not rec.judgeable or (rec.id, rec.view_sha) in have:
            continue
        try:
            verdict = judge.grade(rec)
        except Exception as e:
            print(f"  judge failed on {rec.id}: {type(e).__name__}: {e}")
            continue
        _append(path, verdict.model_dump_json())
        v = verdict
        print(f"  judged {rec.id}: {v.correctness}/{v.faithful}/{v.citation_adequate}")


def current_verdicts(
    records: Sequence[AnswerRecord], path: Path, prompt_version: str | None
) -> dict[str, JudgeVerdict | None]:
    """Latest verdict per record that graded its current text with the given prompt."""
    latest: dict[tuple[str, str], JudgeVerdict] = {}
    for v in load_verdicts(path):
        if prompt_version is None or v.prompt_version == prompt_version:
            latest[(v.id, v.view_sha)] = v
    return {r.id: latest.get((r.id, r.view_sha)) for r in records if r.judgeable}


# ------------------------------------------------------------------------ report


def _fmt(v: float | None, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _digits(metric: str) -> int:
    if metric.startswith("n_"):
        return 0
    if metric.startswith("latency"):
        return 1
    return 4 if metric.startswith("cost") else 3


def markdown(
    meta: dict[str, Any],
    metrics: dict[str, Any],
    cats: dict[str, Any],
    crashes: Sequence[dict[str, Any]] = (),
) -> str:
    agent = meta["agent"]
    lines = [
        f"# Answer eval `{meta['run_id']}`",
        "",
        f"{meta['n_items']} items ({meta['subset']}), provider `{agent['llm_provider']}`, "
        f"agent `{agent['agent_model']}`, corpus `{meta['corpus_hash']}`, golden "
        f"`{meta['golden_sha']}`, {meta['run_at'][:19]}Z.",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for k in METRIC_KEYS:
        if k == "n":
            continue
        lines.append(f"| {k} | {_fmt(metrics.get(k), _digits(k))} |")
    lines += [
        "",
        "Shares below are over each category's completed (non-crashed) items.",
        "",
        "| category | n | crashed | refused | citation P | citation R | faithfulness |",
        "|---|---|---|---|---|---|---|",
    ]
    for cat, m in cats.items():
        lines.append(
            f"| {cat} | {int(m['n'])} | {int(m.get('crashed') or 0)} | {_fmt(m['refused'], 2)} | "
            f"{_fmt(m['citation_precision'])} | {_fmt(m['citation_recall'])} | "
            f"{_fmt(m['faithfulness'])} |"
        )
    rt = meta.get("runtime")
    if rt:
        lines += [
            "",
            f"Models: `{json.dumps(rt['models'])}`.",
            "",
            f"LLM options: `{json.dumps(rt['llm_options'])}`; num_ctx observed "
            f"{meta.get('num_ctx_observed')}, consistent: {meta.get('config_consistent')}.",
            "",
            f"Config hashes: `{json.dumps(rt['config_hashes'])}`.",
            "",
            f"Code: `{json.dumps(rt['code'])}`.",
        ]
    if crashes:
        lines += ["", f"Crashes ({len(crashes)}), excluded from quality metrics:", ""]
        lines += [f"- {c['id']}: {(c['error'] or '')[:160]}" for c in crashes]
    return "\n".join(lines) + "\n"


def score_run(run_dir: Path, corpus: Corpus, judge_meta: dict[str, str] | None) -> dict[str, Any]:
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():  # runs made before meta.json existed kept it in metrics.json
        meta_path = run_dir / "metrics.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    meta = meta.get("meta", meta)
    records = load_records(run_dir / "answers.jsonl")
    if judge_meta:
        meta["judge"] = judge_meta
    prompt_version = (meta.get("judge") or {}).get("prompt_version")
    verdicts = current_verdicts(records, run_dir / "judge.jsonl", prompt_version)
    metrics = compute(records, corpus.get, verdicts)
    cats = by_category(records, corpus.get)
    meta["n_items"] = len(records)
    # What the client actually sent, per question; must be the configured value throughout.
    observed = sorted({n for r in records for n in r.num_ctx_sent})
    configured = ((meta.get("runtime") or {}).get("llm_options") or {}).get("num_ctx")
    meta["num_ctx_observed"] = observed
    meta["config_consistent"] = configured is None or observed in ([], [configured])
    crashes = [{"id": r.id, "error": r.error} for r in records if r.crashed]
    out = {"meta": meta, "metrics": metrics, "by_category": cats, "crashes": crashes}
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2), "utf-8")
    md = markdown(meta, metrics, cats, crashes)
    (run_dir / "report.md").write_text(md, "utf-8")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "answers_latest.md").write_text(md, "utf-8")
    return out


# ------------------------------------------------------------------------ CLI


def build_judge(settings: Settings, model: str | None) -> Judge:
    from citeshield.agent.factory import build_llm

    llm, cfg = build_llm(settings)
    # Haiku on Anthropic, the guard model on Ollama, unless overridden. On Ollama the budget
    # includes qwen3's thinking, so the judge gets the provider's full num_predict.
    max_tokens = settings.llm.ollama_num_predict if settings.llm.provider == "ollama" else 4000
    return Judge(llm, model or cfg.guard_model, max_tokens=max_tokens)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--subset", choices=("smoke", "local", "full"), default="smoke")
    ap.add_argument("--ids", help="Comma-separated golden ids (overrides --subset).")
    ap.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    ap.add_argument("--run-id")
    ap.add_argument("--judge", action="store_true", help="Grade answers with the LLM judge.")
    ap.add_argument("--judge-model")
    ap.add_argument("--rescore", type=Path, metavar="RUN_DIR", help="Score an existing run.")
    ap.add_argument("--write-smoke", action="store_true")
    ap.add_argument(
        "--embed-device",
        default=DEFAULT_EVAL_DEVICE,
        help=f"Embedder device for the run (default {DEFAULT_EVAL_DEVICE}; auto = settings).",
    )
    ap.add_argument(
        "--rerank-device",
        default=DEFAULT_EVAL_DEVICE,
        help=f"Reranker device for the run (default {DEFAULT_EVAL_DEVICE}; auto = settings).",
    )
    args = ap.parse_args(argv)

    items = load_golden(args.golden)
    if args.write_smoke:
        payload = {
            "description": "Stratified CI smoke subset; regenerate with `citeshield-eval "
            "--write-smoke` and re-baseline in the same PR.",
            "ids": stratified_smoke(items),
        }
        SMOKE_PATH.write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
        print(f"wrote {len(payload['ids'])} ids to {SMOKE_PATH}")
        return 0

    settings = eval_settings(Settings(), args.embed_device, args.rerank_device)
    corpus = Corpus.load(settings.chunks_path)
    judge = build_judge(settings, args.judge_model) if args.judge else None

    if args.rescore:
        run_dir = args.rescore
    else:
        from citeshield.agent.factory import build_agent

        subset = "ids" if args.ids else args.subset
        selected = select(items, args.subset, args.ids)
        agent = build_agent(settings)
        runtime = runtime_identity(
            settings, agent.meta, actual_devices(agent), actual_dtypes(agent)
        )
        print(
            "effective settings: "
            + json.dumps({k: runtime[k] for k in ("llm_options", "devices", "dtypes")})
            + f", agent {agent.meta['agent_model']} via {agent.meta['llm_provider']}"
        )
        run_id = args.run_id or (
            f"{agent.meta['llm_provider']}-{subset}-{datetime.now(UTC):%Y%m%dT%H%M%S}"
        )
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        meta: dict[str, Any] = {
            "run_id": run_id,
            "run_at": datetime.now(UTC).isoformat(),
            "subset": subset,
            "ids": [i.id for i in selected],
            "golden_sha": golden_sha(selected),
            "corpus_hash": agent.corpus_version,
            "agent": agent.meta,
            "runtime": runtime,
        }
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            old = json.loads(meta_path.read_text("utf-8"))
            changed = [k for k in AGENT_IDENTITY if old["agent"].get(k) != agent.meta.get(k)]
            old_rt = old.get("runtime") or {}
            changed += [f"runtime.{k}" for k in RUNTIME_IDENTITY if old_rt.get(k) != runtime[k]]
            if changed or old["golden_sha"] != meta["golden_sha"]:
                print(f"cannot resume {run_id}: changed {changed or ['golden items']}")
                return 2
            meta["run_at"] = old["run_at"]
        meta_path.write_text(json.dumps(meta, indent=2), "utf-8")
        run_items(agent, selected, run_dir / "answers.jsonl")

    if judge:
        judge_records(judge, load_records(run_dir / "answers.jsonl"), run_dir / "judge.jsonl")
    result = score_run(run_dir, corpus, judge.meta if judge else None)
    print(markdown(result["meta"], result["metrics"], result["by_category"], result["crashes"]))
    print(f"run written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
