"""Reproducible retrieval evaluation; --answers optionally calls Groq."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from src.knowledge import ROOT, read_manifest
from src.rag import retrieve_chunks, generate_response, normalize, response_text


def evaluate(split="all", answers=False, output=None):
    cases = json.loads((ROOT / "eval/questions.json").read_text(encoding="utf-8"))
    cases = [case for case in cases if split == "all" or case["split"] == split]
    # Warm up model and database; report steady-state retrieval times separately.
    retrieve_chunks("warmup", top_k=3)
    results = []
    for case in cases:
        for mode in ("semantic", "hybrid"):
            if not case["expected_sources"] and not answers:
                continue
            start = time.perf_counter()
            chunks = retrieve_chunks(case["question"], top_k=3, mode=mode)
            elapsed = time.perf_counter() - start
            names = [c["metadata"]["filename"] for c in chunks]
            expected = set(case["expected_sources"])
            evidence = normalize(" ".join(c["text"] for c in chunks)).lower()
            row = {"id": case["id"], "split": case["split"], "mode": mode,
                   "retrieved": names, "seconds": elapsed,
                   "source_recall_at_3": len(expected & set(names)) / len(expected) if expected else None,
                   "all_sources_at_3": expected.issubset(names) if expected else None,
                   "reciprocal_rank": next((1 / (i+1) for i, name in enumerate(names) if name in expected), 0) if expected else None,
                   "evidence_coverage_at_3": sum(normalize(fact).lower() in evidence for fact in case["required_facts"]) / len(case["required_facts"]) if case["required_facts"] else None}
            if answers:
                start = time.perf_counter()
                try:
                    response = generate_response(case["question"], chunks)
                    row["answer"] = response_text(response)
                    row["status"] = response["status"]
                    row["status_correct"] = response["status"] == case["expected_status"]
                    row["required_fact_match"] = all(normalize(f).lower() in normalize(row["answer"]).lower() for f in case["required_facts"]) if case["required_facts"] else None
                except Exception as exc:
                    row["error"] = type(exc).__name__
                    row["status_correct"] = False
                row["generation_seconds"] = time.perf_counter() - start
            results.append(row)
    summaries = {}
    for mode in ("semantic", "hybrid"):
        selected = [row for row in results if row["mode"] == mode]
        metrics = ("source_recall_at_3", "all_sources_at_3", "reciprocal_rank", "evidence_coverage_at_3", "seconds")
        summaries[mode] = {metric: mean(values) for metric in metrics
                           if (values := [r[metric] for r in selected if r.get(metric) is not None])}
        if answers:
            summaries[mode]["status_accuracy"] = mean(r["status_correct"] for r in selected)
            summaries[mode]["generation_errors"] = sum("error" in r for r in selected)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "split": split,
              "collection": read_manifest()["synthetic"]["collection"],
              "corpus_hashes": {d["document_id"]: d["content_hash"] for d in read_manifest()["synthetic"]["documents"]},
              "answer_evaluation": answers, "case_count": len(cases), "summary": summaries, "results": results}
    destination = Path(output) if output else ROOT / "eval/results.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(f"Saved {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "development", "test"), default="all")
    parser.add_argument("--answers", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    evaluate(args.split, args.answers, args.output)
