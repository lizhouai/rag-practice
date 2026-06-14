from __future__ import annotations

import csv
import hashlib
import sqlite3 as sqlite
import sys
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_pipeline as _cli  # noqa: E402
from rag import access as _access  # noqa: E402
from rag import chunking as _chunking  # noqa: E402
from rag import config as _config  # noqa: E402
from rag import context as _context  # noqa: E402
from rag import embedding as _embedding  # noqa: E402
from rag import generation as _generation  # noqa: E402
from rag import http as _http  # noqa: E402
from rag import indexing as _indexing  # noqa: E402
from rag import models as _models  # noqa: E402
from rag import monitoring as _monitoring  # noqa: E402
from rag import rerank as _rerank  # noqa: E402
from rag import retrieval as _retrieval  # noqa: E402
from rag import selection as _selection  # noqa: E402
from rag.pipeline import run_query  # noqa: E402
from rag.vectorstore import filters as _filters  # noqa: E402
from rag.vectorstore import mirrored as _mirrored  # noqa: E402
from rag.vectorstore import qdrant as _qdrant  # noqa: E402
from rag.vectorstore import sqlite as _sqlite_store  # noqa: E402


def _export_public(target: types.SimpleNamespace, module: object) -> None:
    public_names = getattr(module, "__all__", None)
    if public_names is None:
        public_names = [name for name in dir(module) if not name.startswith("_")]
    for name in public_names:
        setattr(target, name, getattr(module, name))


rag = types.SimpleNamespace(
    csv=csv,
    hashlib=hashlib,
    sqlite=sqlite,
    time=time,
    parse_args=_cli.parse_args,
    run_eval=_cli.run_eval,
    run_query=run_query,
)

for _module in (
    _config,
    _models,
    _http,
    _chunking,
    _filters,
    _sqlite_store,
    _qdrant,
    _mirrored,
    _embedding,
    _retrieval,
    _rerank,
    _selection,
    _context,
    _generation,
    _access,
    _indexing,
    _monitoring,
):
    _export_public(rag, _module)

# Phase-1 compatibility tests still patch this surface; production modules look
# it up dynamically through sys.modules during the shim-removal transition.
sys.modules["run_pipeline"] = rag  # type: ignore[assignment]


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
