from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from rag.config import METRICS_PATH

__all__ = ["build_monitoring_event", "append_jsonl", "persist_monitoring_event"]


def build_monitoring_event(trace: dict, latency_ms: int, status: str) -> dict:
    sufficiency = trace.get("context_packet", {}).get("sufficiency", {})
    selected_doc_ids = sorted({item.get("doc_id", "") for item in trace.get("context_packet", {}).get("evidence", []) if item.get("doc_id")})
    validation = trace.get("validation", {})
    model_config = trace.get("model_config", {})
    selection_strategy = trace.get("selection_strategy", {})
    reranker = trace.get("reranker", {})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trace_id": trace["trace_id"],
        "status": status,
        "latency_ms": latency_ms,
        "pipeline_version": trace.get("pipeline_version"),
        "index_version": trace.get("index_version"),
        "vector_backend": model_config.get("vector_backend"),
        "embedding_model": model_config.get("embedding_model"),
        "embedding_identity": model_config.get("embedding_identity"),
        "qdrant_collection": model_config.get("qdrant_collection"),
        "query_hash": hashlib.sha256(trace.get("query", "").encode("utf-8")).hexdigest()[:16],
        "query_chars": len(trace.get("query", "")),
        "answer_mode": trace.get("answer", {}).get("mode"),
        "sufficiency_enough": sufficiency.get("enough"),
        "sufficiency_reason": sufficiency.get("reason"),
        "permission_denied": sufficiency.get("reason") == "permission_denied",
        "citation_valid": validation.get("citation_valid"),
        "missing_citation_count": len(validation.get("missing_citations", [])),
        "dense_hits": len(trace.get("dense_top", [])),
        "bm25_hits": len(trace.get("bm25_top", [])),
        "rerank_candidates": len(trace.get("rerank_top", [])),
        "dedup_dropped_count": len(trace.get("dedup_dropped", [])),
        "selected_count": trace.get("truncation", {}).get("selected_count", 0),
        "selected_doc_ids": selected_doc_ids,
        "context_token_total": trace.get("context_packet", {}).get(
            "estimated_token_total",
            trace.get("truncation", {}).get("token_total", 0),
        ),
        "blocked_match_count": len(trace.get("permission_filter", {}).get("blocked_matches", [])),
        "stage_latencies_ms": trace.get("stage_latencies_ms", {}),
        "selection_strategy": selection_strategy.get("name", selection_strategy),
        "reranker_mode": reranker.get("mode"),
        "reranker_model": reranker.get("model"),
        "reranker_score_policy": reranker.get("score_policy"),
        "reranker_fallback_used": reranker.get("fallback_used", False),
    }


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def persist_monitoring_event(event: dict, metrics_path: Path = METRICS_PATH) -> None:
    append_jsonl(metrics_path, event)
