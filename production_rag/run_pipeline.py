from __future__ import annotations

import argparse
import csv

from rag.config import DEFAULT_VECTOR_BACKEND, EVAL_PATH, load_env
from rag.chunking import split_metadata_values
from rag.pipeline import run_query


def run_eval(
    *,
    allowed_scopes: set[str] | None = None,
    rebuild_index: bool = False,
    vector_backend: str = DEFAULT_VECTOR_BACKEND,
    monitoring_enabled: bool = True,
) -> None:
    rows = list(csv.DictReader(EVAL_PATH.read_text(encoding="utf-8").splitlines()))
    passed = 0
    for index, row in enumerate(rows):
        trace = run_query(
            row["query"],
            quiet=True,
            allowed_scopes=allowed_scopes,
            rebuild_index=rebuild_index and index == 0,
            vector_backend=vector_backend,
            monitoring_enabled=monitoring_enabled,
        )
        answer_text = trace["answer"]["answer"]
        selected_doc_ids = {item["doc_id"] for item in trace["context_packet"]["evidence"]}
        must_answer = row["must_answer"].lower() == "true"
        expected_doc_ok = (not row["expected_doc_id"]) or row["expected_doc_id"] in selected_doc_ids
        expected_terms_ok = row["expected_terms"] in answer_text
        refusal_ok = (not must_answer) and trace["answer"]["mode"] == "refusal"
        ok = (must_answer and expected_doc_ok and expected_terms_ok) or refusal_ok
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {row['case_id']} {row['query']}")
        if not ok:
            print(f"  selected_doc_ids={sorted(selected_doc_ids)}")
            print(f"  answer={answer_text}")
    print(f"\nEval: {passed}/{len(rows)} passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runnable production_rag practice pipeline.")
    parser.add_argument("--query", help="Question to answer.")
    parser.add_argument("--trace-only", action="store_true", help="Print the full JSON trace instead of the answer view.")
    parser.add_argument("--save-trace", action="store_true", help="Also save the full JSON trace.")
    parser.add_argument("--no-monitoring", action="store_true", help="Do not append the per-query monitoring event JSONL.")
    parser.add_argument("--blocked-hint", action="store_true", help="Surface query-relevant blocked-title hints in the trace.")
    parser.add_argument("--eval", action="store_true", help="Run eval_cases.csv.")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuilding the configured vector store.")
    parser.add_argument(
        "--scopes",
        default="internal,public",
        help="Comma-separated permission scopes for the current user. Default: internal,public.",
    )
    parser.add_argument(
        "--vector-backend",
        choices=("qdrant", "local"),
        default=DEFAULT_VECTOR_BACKEND,
        help="Vector store backend. Default: qdrant. Use local only for tests or offline debugging.",
    )
    return parser.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    scopes = split_metadata_values(args.scopes)
    if args.eval:
        run_eval(
            allowed_scopes=scopes,
            rebuild_index=args.rebuild_index,
            vector_backend=args.vector_backend,
            monitoring_enabled=not args.no_monitoring,
        )
        return
    if not args.query:
        raise SystemExit("Provide --query or --eval")
    run_query(
        args.query,
        trace_only=args.trace_only,
        save_trace=args.save_trace,
        allowed_scopes=scopes,
        rebuild_index=args.rebuild_index,
        vector_backend=args.vector_backend,
        monitoring_enabled=not args.no_monitoring,
        blocked_hint=args.blocked_hint,
    )


if __name__ == "__main__":
    main()
