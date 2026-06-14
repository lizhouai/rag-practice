from __future__ import annotations

import re
import sys
from datetime import datetime

from rag.chunking import split_metadata_values, tokenize
from rag.models import Chunk

__all__ = [
    "is_effective",
    "filter_chunks_for_access",
    "find_permission_blocked_matches",
    "fallback_summary",
]


def _shim_value(name: str, default):
    shim = sys.modules.get("run_pipeline")
    return getattr(shim, name, default) if shim is not None else default


def is_effective(metadata: dict[str, str], today: str) -> tuple[bool, str | None]:
    effective_from = metadata.get("effective_from", "")
    effective_to = metadata.get("effective_to", "")
    if effective_from and effective_from > today:
        return False, "not_yet_effective"
    if effective_to and effective_to < today:
        return False, "expired"
    return True, None


def filter_chunks_for_access(
    chunks: list[Chunk],
    allowed_scopes: set[str],
    *,
    today: str | None = None,
) -> tuple[list[Chunk], list[dict]]:
    check_effective = _shim_value("is_effective", is_effective)
    today_value = today or datetime.now().date().isoformat()
    visible: list[Chunk] = []
    rejected: list[dict] = []
    for chunk in chunks:
        scopes = split_metadata_values(chunk.metadata.get("permission_scope", ""))
        if not scopes or scopes.isdisjoint(allowed_scopes):
            rejected.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "reason": "permission_scope"})
            continue
        effective, reason = check_effective(chunk.metadata, today_value)
        if not effective:
            rejected.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "reason": reason})
            continue
        visible.append(chunk)
    return visible, rejected


def find_permission_blocked_matches(
    query: str,
    all_chunks: list[Chunk],
    rejected_chunks: list[dict],
    *,
    limit: int = 5,
) -> list[dict]:
    tokenize_query = _shim_value("tokenize", tokenize)
    query_terms = set(tokenize_query(query))
    if not query_terms:
        return []
    rejected_by_id = {item["chunk_id"]: item for item in rejected_chunks if item.get("reason") == "permission_scope"}
    matches = []
    for chunk in all_chunks:
        if chunk.chunk_id not in rejected_by_id:
            continue
        matched_terms = sorted(query_terms & set(chunk.terms))
        if not matched_terms:
            continue
        overlap_ratio = len(matched_terms) / max(1, len(query_terms))
        has_strong_term = any(re.search(r"\d|[-_/]", term) or len(term) >= 4 for term in matched_terms)
        if overlap_ratio < 0.25 and not has_strong_term:
            continue
        matches.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title_path": " > ".join(chunk.title_path),
                "matched_terms": matched_terms,
                "overlap_ratio": round(overlap_ratio, 4),
            }
        )
    return sorted(matches, key=lambda item: (item["overlap_ratio"], len(item["matched_terms"])), reverse=True)[:limit]


def fallback_summary(component_status: dict[str, dict]) -> list[dict]:
    fallbacks: list[dict] = []
    for component, status in component_status.items():
        if not status.get("fallback_used"):
            continue
        fallbacks.append(
            {
                "component": component,
                "mode": status.get("mode", ""),
                "reason": status.get("reason", ""),
                "error": status.get("error", ""),
                "attempts": status.get("attempts", 0),
            }
        )
    return fallbacks
