from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

from rag.config import DEFAULT_RERANKER_MODEL, DEFAULT_RERANKER_URL, EXTERNAL_RERANKER_PROVIDERS
from rag.config import RERANK_TOP_N, SCORE_POLICY_EXTERNAL_RERANK, SCORE_POLICY_RRF_ONLY, parse_int_env
from rag.http import RetryExhausted, call_with_retries
from rag.models import Candidate, Chunk

__all__ = [
    "ExternalReranker",
    "parse_reranker_scores",
    "make_external_reranker",
    "make_configured_external_reranker",
    "skip_rerank",
    "rerank",
    "describe_reranker",
]


@dataclass
class ExternalReranker:
    url: str
    model: str = DEFAULT_RERANKER_MODEL
    timeout_seconds: int = 30
    last_error: str = ""
    last_attempts: int = 0
    fallback_used: bool = False

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        body = {
            "model": self.model,
            "query": query,
            "documents": [
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "title_path": " > ".join(chunk.title_path),
                    },
                }
                for chunk in chunks
            ],
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return parse_reranker_scores(payload, expected_count=len(chunks))


def parse_reranker_scores(payload: dict, expected_count: int) -> list[float]:
    if isinstance(payload.get("scores"), list):
        scores = [float(score) for score in payload["scores"]]
    elif isinstance(payload.get("results"), list):
        scores = [0.0] * expected_count
        for item in payload["results"]:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", -1))
            if 0 <= index < expected_count:
                scores[index] = float(item.get("score", item.get("relevance_score", 0.0)))
    else:
        raise RuntimeError("Reranker response must include `scores` or `results`.")
    if len(scores) != expected_count:
        raise RuntimeError(f"Reranker returned {len(scores)} scores for {expected_count} documents.")
    return scores


def make_external_reranker(
    url: str,
    *,
    model: str = DEFAULT_RERANKER_MODEL,
    timeout_seconds: int = 30,
) -> ExternalReranker:
    return ExternalReranker(url=url, model=model, timeout_seconds=timeout_seconds)


def make_configured_external_reranker() -> ExternalReranker | None:
    provider = os.getenv("RERANKER_PROVIDER", "").strip().lower()
    url = os.getenv("RERANKER_URL", "").strip()
    if not provider and not url:
        return None
    if not url and provider in EXTERNAL_RERANKER_PROVIDERS:
        url = DEFAULT_RERANKER_URL
    if not url:
        return None
    return make_external_reranker(
        url,
        model=os.getenv("RERANKER_MODEL", "").strip() or DEFAULT_RERANKER_MODEL,
        timeout_seconds=parse_int_env("RERANKER_TIMEOUT_SECONDS", 30),
    )


def skip_rerank(candidates: dict[str, Candidate], reason: str) -> list[Candidate]:
    ranked = sorted(candidates.values(), key=lambda item: item.rrf_score, reverse=True)[:RERANK_TOP_N]
    for candidate in ranked:
        candidate.rerank_score = candidate.rrf_score
        candidate.reason = f"rerank_skipped:{reason}"
    return ranked


def rerank(
    query: str,
    candidates: dict[str, Candidate],
    chunks_by_id: dict[str, Chunk],
    *,
    external_reranker: ExternalReranker | None = None,
) -> list[Candidate]:
    if not external_reranker:
        return skip_rerank(candidates, "no_configured_model")
    ordered_candidates = list(candidates.values())
    chunks = [chunks_by_id[candidate.chunk_id] for candidate in ordered_candidates]
    try:
        scores, attempts = call_with_retries(lambda: external_reranker.score(query, chunks))
        external_reranker.last_attempts = attempts
    except RetryExhausted as exc:
        external_reranker.last_error = str(exc)
        external_reranker.last_attempts = exc.attempts
        return skip_rerank(candidates, "reranker_error")
    for candidate, score in zip(ordered_candidates, scores):
        candidate.rerank_score = score
        candidate.reason = f"external_reranker:{external_reranker.model}"
    return sorted(ordered_candidates, key=lambda item: item.rerank_score, reverse=True)[:RERANK_TOP_N]


def describe_reranker(external_reranker: ExternalReranker | None) -> dict:
    if external_reranker is None:
        return {
            "mode": "skipped",
            "reason": "not_configured",
            "model": "",
            "url": "",
            "fallback_used": False,
            "error": "",
            "attempts": 0,
            "score_policy": SCORE_POLICY_RRF_ONLY,
        }
    if external_reranker.last_error:
        return {
            "mode": "skipped",
            "reason": "reranker_error",
            "model": external_reranker.model,
            "url": external_reranker.url,
            "fallback_used": False,
            "error": external_reranker.last_error,
            "attempts": external_reranker.last_attempts,
            "score_policy": SCORE_POLICY_RRF_ONLY,
        }
    return {
        "mode": "external",
        "reason": "configured_model",
        "model": external_reranker.model,
        "url": external_reranker.url,
        "fallback_used": False,
        "error": "",
        "attempts": external_reranker.last_attempts,
        "score_policy": SCORE_POLICY_EXTERNAL_RERANK,
    }
