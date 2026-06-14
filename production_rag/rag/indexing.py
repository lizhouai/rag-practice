from __future__ import annotations

import sys
from pathlib import Path

from rag.chunking import build_document_chunks, content_hash, normalize_text, parse_frontmatter, safe_relative
from rag.config import DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_EMBEDDING_MODEL, DEFAULT_RETRY_ATTEMPTS
from rag.config import DEFAULT_VECTOR_DB_PATH, EMBEDDING_PROVIDER_EXTERNAL, EMBEDDING_PROVIDER_LOCAL
from rag.config import LOCAL_EMBEDDING_MODEL, RAW_DIR, ROOT, embedding_identity, is_qdrant_configured
from rag.config import resolve_qdrant_api_key, resolve_qdrant_collection, resolve_qdrant_url, resolve_vector_dimensions
from rag.http import RetryExhausted, call_with_retries
from rag.vectorstore.mirrored import MirroredVectorStore
from rag.vectorstore.qdrant import QdrantVectorStore
from rag.vectorstore.sqlite import LocalVectorStore

__all__ = [
    "make_vector_store",
    "set_vector_store_fallback",
    "initialize_vector_store_with_fallback",
    "index_status_without_rebuild",
    "sync_index_with_store_fallback",
    "sync_index",
]


def _shim_value(name: str, default):
    shim = sys.modules.get("run_pipeline")
    return getattr(shim, name, default) if shim is not None else default


def make_vector_store(
    vector_backend: str,
    *,
    store_path: Path = DEFAULT_VECTOR_DB_PATH,
    vector_size: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> object:
    if vector_backend == "qdrant":
        return QdrantVectorStore(
            base_url=resolve_qdrant_url(),
            collection_name=resolve_qdrant_collection(),
            vector_size=vector_size,
            api_key=resolve_qdrant_api_key(),
        )
    if vector_backend == "local":
        return LocalVectorStore(store_path)
    raise RuntimeError(f"Unsupported vector backend: {vector_backend}")


def set_vector_store_fallback(status: dict, error: str, attempts: int) -> None:
    status.update(
        {
            "mode": "sqlite_fallback",
            "backend": "local",
            "fallback_used": True,
            "reason": "qdrant_error",
            "error": error,
            "attempts": attempts,
        }
    )


def initialize_vector_store_with_fallback(
    requested_backend: str,
    *,
    store_path: Path = DEFAULT_VECTOR_DB_PATH,
    vector_size: int = DEFAULT_EMBEDDING_DIMENSIONS,
    status: dict,
) -> tuple[object, str]:
    status.update(
        {
            "component": "vector_store",
            "requested_backend": requested_backend,
            "backend": requested_backend,
            "mode": requested_backend,
            "fallback_used": False,
            "reason": "configured_backend" if requested_backend == "qdrant" else "requested_local",
            "error": "",
            "attempts": 0,
            "qdrant_url": resolve_qdrant_url() if requested_backend == "qdrant" else "",
            "qdrant_collection": resolve_qdrant_collection() if requested_backend == "qdrant" else "",
        }
    )
    if requested_backend == "local":
        return LocalVectorStore(store_path), "local"

    if requested_backend == "qdrant" and not is_qdrant_configured():
        status.update(
            {
                "mode": "sqlite_fallback",
                "backend": "local",
                "fallback_used": True,
                "reason": "not_configured",
                "error": "Qdrant is not configured. Set QDRANT_URL or VECTOR_DB_URL to enable Qdrant.",
                "attempts": 0,
                "qdrant_url": "",
                "qdrant_collection": "",
            }
        )
        return LocalVectorStore(store_path), "local"

    vector_store = make_vector_store(requested_backend, store_path=store_path, vector_size=vector_size)
    if hasattr(vector_store, "ensure_collection"):
        try:
            _, attempts = call_with_retries(lambda: vector_store.ensure_collection())
            status["attempts"] = attempts
        except RetryExhausted as exc:
            set_vector_store_fallback(status, str(exc), exc.attempts)
            return LocalVectorStore(store_path), "local"
    return vector_store, requested_backend


def index_status_without_rebuild(
    *,
    vector_store: object,
    store_path: Path,
    embedding_model: str,
    embedding_identity_value: str,
    reason: str = "rebuild_not_requested",
) -> dict:
    chunks_count = 0
    if hasattr(vector_store, "chunks_count"):
        chunks_count = vector_store.chunks_count()
    return {
        "store": vector_store.describe() if hasattr(vector_store, "describe") else str(store_path),
        "embedding_model": embedding_model,
        "embedding_identity": embedding_identity_value,
        "changed_docs": [],
        "removed_docs": [],
        "chunks_count": chunks_count,
        "rebuild_requested": False,
        "reason": reason,
    }


def sync_index_with_store_fallback(
    *,
    vector_store: object,
    resolved_backend: str,
    vector_status: dict,
    raw_dir: Path = RAW_DIR,
    store_path: Path = DEFAULT_VECTOR_DB_PATH,
    embedder: object | None = None,
    rebuild: bool = False,
    embedding_model: str | None = None,
    embedding_identity_value: str | None = None,
) -> tuple[dict, object, str]:
    sync_index_func = _shim_value("sync_index", sync_index)
    attempts = DEFAULT_RETRY_ATTEMPTS if resolved_backend == "qdrant" else 1
    sync_store = (
        MirroredVectorStore(vector_store, LocalVectorStore(store_path))
        if resolved_backend == "qdrant"
        else vector_store
    )
    try:
        index_sync, used_attempts = call_with_retries(
            lambda: sync_index_func(
                raw_dir,
                store_path,
                vector_store=sync_store,
                embedder=embedder,
                rebuild=rebuild,
                embedding_model=embedding_model,
                embedding_identity_value=embedding_identity_value,
            ),
            attempts=attempts,
        )
        if resolved_backend == "qdrant":
            vector_status["attempts"] = max(vector_status.get("attempts", 0), used_attempts)
        return index_sync, vector_store, resolved_backend
    except RetryExhausted as exc:
        if resolved_backend != "qdrant":
            raise
        set_vector_store_fallback(vector_status, str(exc), exc.attempts)
        local_store = LocalVectorStore(store_path)
        index_sync = sync_index_func(
            raw_dir,
            store_path,
            vector_store=local_store,
            embedder=embedder,
            rebuild=rebuild,
            embedding_model=embedding_model,
            embedding_identity_value=embedding_identity_value,
        )
        return index_sync, local_store, "local"


def sync_index(
    raw_dir: Path = RAW_DIR,
    store_path: Path = DEFAULT_VECTOR_DB_PATH,
    *,
    vector_store: object | None = None,
    embedder: object | None = None,
    rebuild: bool = False,
    embedding_model: str | None = None,
    embedding_identity_value: str | None = None,
) -> dict:
    """Sync changed markdown documents into the configured vector store."""
    model_name = embedding_model or (DEFAULT_EMBEDDING_MODEL if embedder else LOCAL_EMBEDDING_MODEL)
    index_identity = embedding_identity_value or embedding_identity(
        EMBEDDING_PROVIDER_EXTERNAL if embedder else EMBEDDING_PROVIDER_LOCAL,
        model_name,
    )
    dimensions = resolve_vector_dimensions()
    store = vector_store or LocalVectorStore(store_path)
    if rebuild:
        store.reset()

    manifest = store.load_manifest()
    changed_docs: list[str] = []
    current_doc_ids: set[str] = set()

    for path in sorted(raw_dir.glob("*.md")):
        raw_text = path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw_text)
        metadata.setdefault("doc_id", path.stem)
        metadata.setdefault("title", path.stem)
        metadata.setdefault("source_path", safe_relative(path, ROOT))
        metadata.setdefault("permission_scope", "internal")
        metadata.setdefault("effective_from", "")
        metadata.setdefault("effective_to", "")
        doc_id = metadata["doc_id"]
        current_doc_ids.add(doc_id)
        hash_value = content_hash(raw_text, index_identity, dimensions)

        existing = manifest.get(doc_id)
        if (
            existing
            and existing["content_hash"] == hash_value
            and existing["embedding_model"] == index_identity
        ):
            continue

        chunks = build_document_chunks(metadata, normalize_text(body))
        if embedder and chunks:
            embeddings = embedder([chunk.text for chunk in chunks])  # type: ignore[operator]
            for chunk, vector in zip(chunks, embeddings):
                chunk.dense_vector = vector
        store.upsert_document(
            doc_id=doc_id,
            source_path=metadata.get("source_path", ""),
            hash_value=hash_value,
            embedding_model=index_identity,
            chunks=chunks,
        )
        changed_docs.append(doc_id)

    removed_docs = set(manifest) - current_doc_ids
    store.delete_documents(removed_docs)
    return {
        "store": store.describe() if hasattr(store, "describe") else str(store_path),
        "embedding_model": model_name,
        "embedding_identity": index_identity,
        "changed_docs": changed_docs,
        "removed_docs": sorted(removed_docs),
        "chunks_count": len(store.load_chunks()),
        "rebuild_requested": rebuild,
        "reason": "rebuilt" if rebuild else "incremental_sync",
    }
