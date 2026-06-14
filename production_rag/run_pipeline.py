from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from rag.config import *  # noqa: F401,F403 - re-export shim during modularization
from rag.models import *  # noqa: F401,F403
from rag.http import *  # noqa: F401,F403
from rag.chunking import *  # noqa: F401,F403
from rag.vectorstore.filters import *  # noqa: F401,F403
from rag.vectorstore.sqlite import *  # noqa: F401,F403
from rag.vectorstore.qdrant import *  # noqa: F401,F403
from rag.vectorstore.mirrored import *  # noqa: F401,F403
from rag.embedding import *  # noqa: F401,F403
from rag.retrieval import *  # noqa: F401,F403
from rag.rerank import *  # noqa: F401,F403
from rag.selection import *  # noqa: F401,F403
from rag.context import *  # noqa: F401,F403
from rag.generation import *  # noqa: F401,F403
from rag.access import *  # noqa: F401,F403
from rag.indexing import *  # noqa: F401,F403
from rag.monitoring import *  # noqa: F401,F403
from rag.pipeline import run_query  # noqa: F401


def build_corpus() -> tuple[list[ParentSection], list[Chunk]]:
    parents: list[ParentSection] = []
    chunks: list[Chunk] = []
    for metadata, body in read_documents():
        doc_parents = split_sections(metadata, body)
        parents.extend(doc_parents)
        for parent in doc_parents:
            chunks.extend(chunk_parent(parent))
    return parents, chunks


def print_answer(trace: dict, trace_path: Path | None) -> None:
    print(f"Query: {trace['query']}")
    print(f"Trace id: {trace['trace_id']}")
    if trace_path:
        print(f"Trace file: {trace_path}")
    if trace.get("trace_save_error"):
        print(f"Trace save skipped: {trace['trace_save_error']}")
    if trace.get("monitoring_write_error"):
        print(f"Monitoring event write skipped: {trace['monitoring_write_error']}")
    elif trace.get("monitoring_metrics_path"):
        print(f"Monitoring event: {trace['monitoring_metrics_path']}")
    print("\nSelected evidence:")
    for evidence in trace["context_packet"]["evidence"]:
        print(
            f"- [{evidence['citation_id']}] {evidence['doc_id']} "
            f"{' > '.join(evidence['title_path'])} score={evidence['rerank_score']}"
        )
    print("\nAnswer:")
    print(trace["answer"]["answer"])
    print("\nValidation:")
    print(json.dumps(trace["validation"], ensure_ascii=False, indent=2))


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
    )


if __name__ == "__main__":
    main()
