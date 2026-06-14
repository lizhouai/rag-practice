from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_pipeline as rag  # noqa: E402


def make_chunk(chunk_id: str, scope: str, *, effective_to: str = "") -> rag.Chunk:
    return rag.Chunk(
        chunk_id=chunk_id,
        parent_id=f"{chunk_id}:parent",
        doc_id=f"{chunk_id}:doc",
        title_path=["doc", "section"],
        text=f"{chunk_id} test text",
        metadata={
            "permission_scope": scope,
            "effective_from": "2026-01-01",
            "effective_to": effective_to,
        },
        token_count=8,
        dense_vector=rag.vectorize(["test", chunk_id]),
        terms=["test", chunk_id],
    )
