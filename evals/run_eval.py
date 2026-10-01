"""Offline evaluation of the RAG pipeline against a golden question set.

Metrics:
  retrieval_hit@k  - expected source appears in the top-k retrieved chunks
  answer_contains  - answer contains the required facts (string match)
  refusal_rate     - unanswerable questions correctly refused
  hallucination    - unanswerable questions that got a confident answer (should be 0)
  cost / latency   - tokens, USD, p50/p95 latency

Usage:
  python -m evals.run_eval                      # current settings (.env)
  python -m evals.run_eval --prompt-version 1   # compare prompt versions
  python -m evals.run_eval --fail-under 0.8     # CI gate
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from app.config import Settings
from app.main import build_app_state
from app.observability import estimate_cost
from app.rag.pipeline import NO_ANSWER


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="evals/golden_set.jsonl")
    ap.add_argument("--prompt-version", type=int, default=None)
    ap.add_argument("--fail-under", type=float, default=0.0, help="exit 1 if answer accuracy is below this")
    ap.add_argument("--out", default="evals/results")
    args = ap.parse_args()

    settings = Settings()
    state = build_app_state(settings)
    rag = state["rag"]
    if state["store"].count() == 0:
        rag.ingest_directory(settings.docs_dir)

    cases = [json.loads(line) for line in Path(args.golden).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows, latencies = [], []
    tokens_in = tokens_out = 0

    for case in cases:
        t0 = time.perf_counter()
        hits = rag.retrieve(case["question"])
        res = rag.answer(case["question"], prompt_version=args.prompt_version)
        latency = time.perf_counter() - t0
        latencies.append(latency)
        tokens_in += res.usage.prompt_tokens
        tokens_out += res.usage.completion_tokens

        retrieved = [h.chunk.source for h in hits]
        refused = NO_ANSWER in res.answer or not res.grounded
        if case["answerable"]:
            hit = case["expected_source"] in retrieved
            correct = (not refused) and all(s.lower() in res.answer.lower() for s in case["must_contain"])
            cited_ok = any(c.source == case["expected_source"] for c in res.citations)
        else:
            hit = None
            correct = refused
            cited_ok = None
        rows.append({**case, "retrieved": retrieved, "answer": res.answer, "retrieval_hit": hit,
                     "correct": correct, "cited_expected_source": cited_ok, "refused": refused,
                     "latency_s": round(latency, 3), "prompt_version": res.prompt_version})

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    summary = {
        "llm": f"{state['llm'].provider}/{state['llm'].model}",
        "embeddings": f"{state['embedder'].provider}/{state['embedder'].model}",
        "prompt_version": rows[0]["prompt_version"] if rows else None,
        "n": len(rows),
        "retrieval_hit_at_k": _rate(r["retrieval_hit"] for r in answerable),
        "answer_accuracy": _rate(r["correct"] for r in answerable),
        "citation_accuracy": _rate(r["cited_expected_source"] for r in answerable),
        "refusal_rate_on_unanswerable": _rate(r["correct"] for r in unanswerable),
        "hallucinations": sum(not r["correct"] for r in unanswerable),
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3),
        "tokens": {"prompt": tokens_in, "completion": tokens_out},
        "est_cost_usd": round(estimate_cost(state["llm"].model, tokens_in, tokens_out), 5),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False)
    (out / f"{stamp}.json").write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print("\nFailures:")
    for r in rows:
        if not r["correct"]:
            print(f"  - {r['question']}\n      answer: {r['answer'][:120]}\n      retrieved: {r['retrieved'][:2]}")
    print(f"\nFull report: {out / f'{stamp}.json'}")

    if summary["answer_accuracy"] < args.fail_under:
        print(f"\nFAIL: accuracy {summary['answer_accuracy']} < {args.fail_under}")
        return 1
    return 0


def _rate(values) -> float:
    values = list(values)
    return round(sum(bool(v) for v in values) / len(values), 3) if values else 0.0


if __name__ == "__main__":
    sys.exit(main())
