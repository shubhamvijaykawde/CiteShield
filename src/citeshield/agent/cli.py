"""``citeshield-ask "question"``: run the agent once and print its events and answer."""

from __future__ import annotations

import argparse
import json
from typing import Any

from citeshield.agent.factory import build_agent


def _print_event(kind: str, payload: dict[str, Any]) -> None:
    if kind == "final":
        return
    if kind == "tool_call":
        print(f"  -> {payload['name']}({json.dumps(payload['input'], ensure_ascii=False)})")
    elif kind == "tool_result":
        labels = ", ".join(s["label"] for s in payload["sources"][:6])
        print(f"     {payload['n_chunks']} chunks in {payload['ms']:.0f}ms: {labels}")
    elif kind == "note":
        print(f"  note: {payload['text'][:300]}")
    else:
        print(f"[{kind}] {json.dumps(payload, ensure_ascii=False)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question")
    ap.add_argument("--regulation", action="append")
    ap.add_argument("--json", action="store_true", help="Print the full final payload.")
    args = ap.parse_args(argv)

    agent = build_agent()
    run = agent.run(args.question, regulations=args.regulation, emit=_print_event)
    a = run.answer
    print(f"\n== {a.status.value}" + (f" ({a.refusal_reason.value})" if a.refusal_reason else ""))
    print(a.summary)
    sources = run.evidence
    for claim in a.claims:
        labels = ", ".join(
            sources[c.chunk_id].label for c in claim.citations if c.chunk_id in sources
        )
        print(f"  * {claim.text}  [{labels}]")
    for item in a.checklist or []:
        print(f"  [ ] {item.obligation} ({item.applies_to})")
    for caveat in a.caveats:
        print(f"  ! {caveat}")
    u = run.usage
    print(
        f"\n{u.calls} LLM calls, {u.input_tokens}+{u.cache_read_tokens} cached in / "
        f"{u.output_tokens} out, ~${u.cost_usd:.4f}, {run.timings_ms['total'] / 1000:.1f}s"
    )
    if args.json:
        from citeshield.agent.graph import serialize_run

        print(json.dumps(serialize_run(run), indent=2, ensure_ascii=False))
    return 0
