from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from rag.chunking import tokenize
from rag.config import BM25_TOP_N, DENSE_TOP_N, RRF_K
from rag.models import Candidate, Chunk

__all__ = [
    "cosine",
    "dense_recall",
    "bm25_recall",
    "rrf_fuse",
    "summarize_results",
]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def dense_recall(query_vector: list[float], chunks: list[Chunk], top_n: int = DENSE_TOP_N) -> list[tuple[float, Chunk]]:
    scored = [(cosine(query_vector, chunk.dense_vector), chunk) for chunk in chunks]
    return sorted(scored, key=lambda item: item[0], reverse=True)[:top_n]


def bm25_recall(query: str, chunks: list[Chunk], top_n: int = BM25_TOP_N) -> list[tuple[float, Chunk]]:
    query_terms = tokenize(query)
    if not query_terms:
        return []

    doc_freq: Counter[str] = Counter()
    chunk_terms = {chunk.chunk_id: Counter(chunk.terms) for chunk in chunks}
    for counts in chunk_terms.values():
        for term in counts:
            doc_freq[term] += 1

    avg_len = sum(sum(counts.values()) for counts in chunk_terms.values()) / max(1, len(chunks))
    total_docs = len(chunks)
    k1 = 1.5
    b = 0.75
    scores: list[tuple[float, Chunk]] = []

    for chunk in chunks:
        counts = chunk_terms[chunk.chunk_id]
        length = sum(counts.values()) or 1
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avg_len))
        if score > 0:
            scores.append((score, chunk))

    return sorted(scores, key=lambda item: item[0], reverse=True)[:top_n]


def rrf_fuse(
    dense_results: list[tuple[float, Chunk]],
    bm25_results: list[tuple[float, Chunk]],
) -> dict[str, Candidate]:
    candidates: dict[str, Candidate] = {}
    for route, results in (("dense", dense_results), ("bm25", bm25_results)):
        for rank, (score, chunk) in enumerate(results, start=1):
            item = candidates.setdefault(chunk.chunk_id, Candidate(chunk_id=chunk.chunk_id))
            item.rrf_score += 1 / (RRF_K + rank)
            if route == "dense":
                item.dense_rank = rank
                item.dense_score = score
            else:
                item.bm25_rank = rank
                item.bm25_score = score
    return dict(sorted(candidates.items(), key=lambda pair: pair[1].rrf_score, reverse=True))


def summarize_results(results: Iterable[tuple[float, Chunk]]) -> list[dict]:
    return [
        {
            "score": round(score, 4),
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title_path": " > ".join(chunk.title_path),
        }
        for score, chunk in results
    ]
