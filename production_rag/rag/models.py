from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ParentSection:
    parent_id: str
    doc_id: str
    title: str
    title_path: list[str]
    text: str
    metadata: dict[str, str]


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    doc_id: str
    title_path: list[str]
    text: str
    metadata: dict[str, str]
    token_count: int
    dense_vector: list[float] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    chunk_id: str
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    mmr_score: float = 0.0
    reason: str = ""


@dataclass
class SelectedEvidence:
    candidate: Candidate
    expanded_text: str
    expanded_from_chunk_ids: list[str]
    expanded_token_count: int
