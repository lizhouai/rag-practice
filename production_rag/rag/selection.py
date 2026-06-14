from __future__ import annotations

import sys
from collections import defaultdict

from rag.config import CONTEXT_TOKEN_BUDGET, DEDUP_THRESHOLD, FINAL_MAX_K, GAP_THRESHOLD
from rag.config import MIN_RERANK_SCORE, MMR_LAMBDA, PARENT_EXPANSION_MAX_CHARS, RERANK_TOP_N
from rag.config import SCORE_POLICY_EXTERNAL_RERANK, SCORE_POLICY_RRF_ONLY
from rag.models import Candidate, Chunk, SelectedEvidence
from rag.retrieval import cosine

__all__ = [
    "semantic_dedup",
    "mmr_select",
    "dynamic_truncate",
    "build_chunks_by_parent",
    "expand_parent_context",
]


def _shim_value(name: str, default):
    shim = sys.modules.get("run_pipeline")
    return getattr(shim, name, default) if shim is not None else default


def semantic_dedup(ranked: list[Candidate], chunks_by_id: dict[str, Chunk]) -> tuple[list[Candidate], list[dict]]:
    dedup_threshold = _shim_value("DEDUP_THRESHOLD", DEDUP_THRESHOLD)
    kept: list[Candidate] = []
    dropped: list[dict] = []
    for candidate in ranked:
        chunk = chunks_by_id[candidate.chunk_id]
        duplicate_of: tuple[Candidate, float] | None = None
        for selected in kept:
            selected_chunk = chunks_by_id[selected.chunk_id]
            similarity = cosine(chunk.dense_vector, selected_chunk.dense_vector)
            if similarity >= dedup_threshold:
                duplicate_of = (selected, similarity)
                break
        if duplicate_of:
            dropped.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "duplicate_of": duplicate_of[0].chunk_id,
                    "similarity": round(duplicate_of[1], 4),
                }
            )
        else:
            kept.append(candidate)
    return kept, dropped


def mmr_select(
    ranked: list[Candidate],
    chunks_by_id: dict[str, Chunk],
    *,
    limit: int = RERANK_TOP_N,
    lambda_mult: float = MMR_LAMBDA,
    duplicate_threshold: float = DEDUP_THRESHOLD,
) -> tuple[list[Candidate], list[dict]]:
    if not ranked:
        return [], []
    remaining = sorted(ranked, key=lambda item: item.rerank_score, reverse=True)
    max_score = max(abs(item.rerank_score) for item in remaining) or 1.0
    selected: list[Candidate] = []
    dropped: list[dict] = []

    while remaining and len(selected) < limit:
        best: Candidate | None = None
        best_score = float("-inf")
        next_remaining: list[Candidate] = []
        for candidate in remaining:
            chunk = chunks_by_id[candidate.chunk_id]
            if selected:
                similarities = [
                    cosine(chunk.dense_vector, chunks_by_id[item.chunk_id].dense_vector)
                    for item in selected
                ]
                max_similarity = max(similarities) if similarities else 0.0
                if max_similarity >= duplicate_threshold:
                    duplicate_of = selected[similarities.index(max_similarity)]
                    dropped.append(
                        {
                            "chunk_id": candidate.chunk_id,
                            "duplicate_of": duplicate_of.chunk_id,
                            "similarity": round(max_similarity, 4),
                            "reason": "near_duplicate",
                        }
                    )
                    continue
            else:
                max_similarity = 0.0
            relevance = candidate.rerank_score / max_score
            candidate.mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
            if candidate.mmr_score > best_score:
                if best is not None:
                    next_remaining.append(best)
                best = candidate
                best_score = candidate.mmr_score
            else:
                next_remaining.append(candidate)
        if best is None:
            break
        selected.append(best)
        remaining = next_remaining
    return selected, dropped


def dynamic_truncate(
    candidates: list[Candidate],
    chunks_by_id: dict[str, Chunk],
    *,
    chunks_by_parent: dict[str, list[Chunk]] | None = None,
    score_policy: str = SCORE_POLICY_EXTERNAL_RERANK,
    min_score: float | None = MIN_RERANK_SCORE,
    gap_threshold: float | None = GAP_THRESHOLD,
) -> tuple[list[SelectedEvidence], dict]:
    if score_policy == SCORE_POLICY_RRF_ONLY:
        min_score = None
        gap_threshold = None
    final_max_k = _shim_value("FINAL_MAX_K", FINAL_MAX_K)
    context_token_budget = _shim_value("CONTEXT_TOKEN_BUDGET", CONTEXT_TOKEN_BUDGET)
    filtered = list(candidates) if min_score is None else [item for item in candidates if item.rerank_score >= min_score]
    reason = {
        "score_policy": score_policy,
        "score_confidence": score_policy == SCORE_POLICY_EXTERNAL_RERANK,
        "min_score": min_score,
        "gap_threshold": gap_threshold,
        "max_k": final_max_k,
        "context_token_budget": context_token_budget,
        "budget_basis": "expanded_parent_context_tokens",
        "stop_reason": "max_k_or_budget",
    }
    if not filtered:
        reason["stop_reason"] = "no_candidate_above_min_score"
        return [], reason

    cutoff = len(filtered)
    if gap_threshold is not None:
        for index in range(len(filtered) - 1):
            gap = filtered[index].rerank_score - filtered[index + 1].rerank_score
            if gap >= gap_threshold:
                cutoff = index + 1
                reason["stop_reason"] = "gap_cutoff"
                reason["gap"] = round(gap, 4)
                break

    selected: list[SelectedEvidence] = []
    token_total = 0
    for candidate in filtered[:cutoff]:
        if len(selected) >= final_max_k:
            reason["stop_reason"] = "max_k_or_budget"
            break
        chunk = chunks_by_id[candidate.chunk_id]
        expanded_text, expanded_ids, expanded_tokens = expand_parent_context(chunk, chunks_by_parent)
        if token_total + expanded_tokens > context_token_budget:
            reason["stop_reason"] = "max_k_or_budget"
            break
        selected.append(
            SelectedEvidence(
                candidate=candidate,
                expanded_text=expanded_text,
                expanded_from_chunk_ids=expanded_ids,
                expanded_token_count=expanded_tokens,
            )
        )
        token_total += expanded_tokens
    reason["selected_count"] = len(selected)
    reason["token_total"] = token_total
    return selected, reason


def build_chunks_by_parent(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    chunks_by_parent: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_parent[chunk.parent_id].append(chunk)
    for parent_chunks in chunks_by_parent.values():
        parent_chunks.sort(key=lambda item: item.chunk_id)
    return chunks_by_parent


def expand_parent_context(
    chunk: Chunk,
    chunks_by_parent: dict[str, list[Chunk]] | None,
    *,
    max_chars: int = PARENT_EXPANSION_MAX_CHARS,
) -> tuple[str, list[str], int]:
    if not chunks_by_parent:
        return chunk.text, [chunk.chunk_id], chunk.token_count
    siblings = chunks_by_parent.get(chunk.parent_id) or [chunk]
    text_parts: list[str] = []
    expanded_ids: list[str] = []
    token_total = 0
    for sibling in siblings:
        candidate_text = "\n\n".join([*text_parts, sibling.text]).strip()
        if len(candidate_text) > max_chars and sibling.chunk_id != chunk.chunk_id:
            continue
        text_parts.append(sibling.text)
        expanded_ids.append(sibling.chunk_id)
        token_total += sibling.token_count
    if chunk.chunk_id not in expanded_ids:
        text_parts.append(chunk.text)
        expanded_ids.append(chunk.chunk_id)
        token_total += chunk.token_count
    return "\n\n".join(text_parts).strip(), expanded_ids, token_total
