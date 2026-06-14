from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rag.access import fallback_summary, find_permission_blocked_matches
from rag.chunking import tokenize, vectorize
from rag.config import BM25_TOP_N, DEFAULT_ALLOWED_SCOPES, DEFAULT_LLM_API_STYLE, DEFAULT_VECTOR_BACKEND
from rag.config import DEFAULT_VECTOR_DB_PATH, DENSE_TOP_N, EMBEDDING_PROVIDER_LOCAL, LOCAL_EMBEDDING_MODEL
from rag.config import METRICS_PATH, MMR_LAMBDA, PARENT_EXPANSION_MAX_CHARS, TRACE_DIR
from rag.config import embedding_identity, local_store_path_for_embedding_identity
from rag.config import resolve_qdrant_collection, resolve_qdrant_url, resolve_vector_dimensions
from rag.context import assemble_context, sufficiency_check
from rag.docstore import SqliteDocstore
from rag.embedding import build_embedding_status, make_retrying_embedding_function
from rag.generation import build_llm_status, generate_answer_resilient, validate_citations
from rag.http import ComponentFallback, RetryExhausted, call_with_retries
from rag.indexing import index_status_without_rebuild as _index_status_without_rebuild
from rag.indexing import initialize_vector_store_with_fallback, set_vector_store_fallback as _set_vector_store_fallback
from rag.indexing import sync_index_with_store_fallback as _sync_index_with_store_fallback
from rag.models import Chunk
from rag.monitoring import build_monitoring_event as _build_monitoring_event
from rag.monitoring import persist_monitoring_event as _persist_monitoring_event
from rag.rerank import describe_reranker, make_configured_external_reranker, rerank
from rag.retrieval import dense_recall, rrf_fuse, summarize_results
from rag.selection import build_chunks_by_parent, dynamic_truncate, mmr_select
from rag.vectorstore.filters import qdrant_access_filter
from rag.vectorstore.sqlite import LocalVectorStore

__all__ = ["run_query"]


def _shim_value(name: str, default):
    shim = sys.modules.get("run_pipeline")
    return getattr(shim, name, default) if shim is not None else default


def _print_answer(trace: dict, trace_path: Path | None) -> None:
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


def _hydrate_recall_results(
    results: list[tuple[float, Chunk]],
    hydrated_chunks: dict[str, Chunk],
    chunks_by_id: dict[str, Chunk],
) -> list[tuple[float, Chunk]]:
    hydrated_results: list[tuple[float, Chunk]] = []
    for score, recalled in results:
        source = hydrated_chunks.get(recalled.chunk_id)
        if source is None:
            chunk = recalled
        else:
            chunk = Chunk(
                chunk_id=source.chunk_id,
                parent_id=source.parent_id,
                doc_id=source.doc_id,
                title_path=list(source.title_path),
                text=source.text,
                metadata=dict(source.metadata),
                token_count=source.token_count,
                dense_vector=recalled.dense_vector or source.dense_vector,
                terms=list(source.terms),
            )
        chunks_by_id[chunk.chunk_id] = chunk
        hydrated_results.append((score, chunk))
    return hydrated_results


def run_query(
    query: str,
    trace_only: bool = False,
    save_trace: bool = False,
    quiet: bool = False,
    *,
    allowed_scopes: set[str] | None = None,
    rebuild_index: bool = False,
    vector_backend: str = DEFAULT_VECTOR_BACKEND,
    store_path: Path = DEFAULT_VECTOR_DB_PATH,
    monitoring_enabled: bool = True,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    started = time.perf_counter()
    stage_latencies_ms: dict[str, int] = {}
    llm_status = build_llm_status()
    embedding_status = build_embedding_status()
    vector_store_status: dict = {}
    scopes = allowed_scopes or DEFAULT_ALLOWED_SCOPES
    vector_size = resolve_vector_dimensions()
    embedder = make_retrying_embedding_function(embedding_status, vector_size) if embedding_status["configured"] else None
    embedding_model = embedding_status["model"]
    index_embedding_identity = embedding_status["identity"]
    active_store_path = local_store_path_for_embedding_identity(store_path, index_embedding_identity)
    lexical_store = LocalVectorStore(active_store_path)
    query_vector: list[float] | None = None
    stage_started = time.perf_counter()
    try:
        query_vector = embedder([query])[0] if embedder else vectorize(tokenize(query))
    except ComponentFallback as exc:
        embedding_status.update(
            {
                "mode": "hash_fallback",
                "provider": EMBEDDING_PROVIDER_LOCAL,
                "model": LOCAL_EMBEDDING_MODEL,
                "identity": embedding_identity(EMBEDDING_PROVIDER_LOCAL, LOCAL_EMBEDDING_MODEL),
                "fallback_used": True,
                "reason": exc.reason,
                "error": exc.error,
                "attempts": exc.attempts,
            }
        )
        embedder = None
        embedding_model = LOCAL_EMBEDDING_MODEL
        index_embedding_identity = embedding_status["identity"]
        active_store_path = local_store_path_for_embedding_identity(store_path, index_embedding_identity)
        lexical_store = LocalVectorStore(active_store_path)
        query_vector = vectorize(tokenize(query))
    stage_latencies_ms["embedding_probe"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    vector_store, resolved_vector_backend = initialize_vector_store_with_fallback(
        vector_backend,
        store_path=active_store_path,
        vector_size=vector_size,
        status=vector_store_status,
    )
    stage_latencies_ms["vector_store_probe"] = int((time.perf_counter() - stage_started) * 1000)

    sync_index_with_store_fallback = _shim_value("sync_index_with_store_fallback", _sync_index_with_store_fallback)
    index_status_without_rebuild = _shim_value("index_status_without_rebuild", _index_status_without_rebuild)
    set_vector_store_fallback = _shim_value("set_vector_store_fallback", _set_vector_store_fallback)
    build_monitoring_event = _shim_value("build_monitoring_event", _build_monitoring_event)
    persist_monitoring_event = _shim_value("persist_monitoring_event", _persist_monitoring_event)

    stage_started = time.perf_counter()
    if rebuild_index:
        try:
            index_sync, vector_store, resolved_vector_backend = sync_index_with_store_fallback(
                vector_store=vector_store,
                resolved_backend=resolved_vector_backend,
                vector_status=vector_store_status,
                store_path=active_store_path,
                embedder=embedder,
                rebuild=True,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
            )
        except ComponentFallback as exc:
            embedding_status.update(
                {
                    "mode": "hash_fallback",
                    "provider": EMBEDDING_PROVIDER_LOCAL,
                    "model": LOCAL_EMBEDDING_MODEL,
                    "identity": embedding_identity(EMBEDDING_PROVIDER_LOCAL, LOCAL_EMBEDDING_MODEL),
                    "fallback_used": True,
                    "reason": exc.reason,
                    "error": exc.error,
                    "attempts": exc.attempts,
                }
            )
            embedder = None
            embedding_model = LOCAL_EMBEDDING_MODEL
            index_embedding_identity = embedding_status["identity"]
            active_store_path = local_store_path_for_embedding_identity(store_path, index_embedding_identity)
            lexical_store = LocalVectorStore(active_store_path)
            if resolved_vector_backend == "local":
                vector_store = LocalVectorStore(active_store_path)
            query_vector = vectorize(tokenize(query))
            index_sync, vector_store, resolved_vector_backend = sync_index_with_store_fallback(
                vector_store=vector_store,
                resolved_backend=resolved_vector_backend,
                vector_status=vector_store_status,
                store_path=active_store_path,
                embedder=embedder,
                rebuild=True,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
            )
    else:
        index_sync = index_status_without_rebuild(
            vector_store=vector_store,
            store_path=active_store_path,
            embedding_model=embedding_model,
            embedding_identity_value=index_embedding_identity,
        )
    stage_latencies_ms["index_sync"] = int((time.perf_counter() - stage_started) * 1000)

    def load_access_state():
        visible_chunks, rejected, permission_blocked_chunks = lexical_store.load_access_chunks(scopes)
        access_checked_chunks = visible_chunks + permission_blocked_chunks
        blocked_matches = find_permission_blocked_matches(query, access_checked_chunks, rejected)
        visible_chunks_by_id = {chunk.chunk_id: chunk for chunk in visible_chunks}
        if not index_sync.get("chunks_count"):
            index_sync["chunks_count"] = len(visible_chunks)
        return (
            access_checked_chunks,
            visible_chunks,
            rejected,
            blocked_matches,
            len({chunk.parent_id for chunk in visible_chunks}),
            visible_chunks_by_id,
            build_chunks_by_parent(visible_chunks),
        )

    stage_started = time.perf_counter()
    if resolved_vector_backend == "qdrant":
        all_chunks = []
        chunks = []
        rejected_chunks = []
        permission_blocked_matches = []
        parents_count = 0
        chunks_by_id = {}
        chunks_by_parent = {}
    else:
        (
            all_chunks,
            chunks,
            rejected_chunks,
            permission_blocked_matches,
            parents_count,
            chunks_by_id,
            chunks_by_parent,
        ) = load_access_state()
    stage_latencies_ms["access_filter"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    # The query vector must come from the same vectorizer (and dimensions) as
    # the stored chunk vectors, regardless of which store serves the search.
    try:
        query_vector = query_vector or (embedder([query])[0] if embedder else vectorize(tokenize(query)))
    except ComponentFallback as exc:
        embedding_status.update(
            {
                "mode": "hash_fallback",
                "provider": EMBEDDING_PROVIDER_LOCAL,
                "model": LOCAL_EMBEDDING_MODEL,
                "identity": embedding_identity(EMBEDDING_PROVIDER_LOCAL, LOCAL_EMBEDDING_MODEL),
                "fallback_used": True,
                "reason": exc.reason,
                "error": exc.error,
                "attempts": exc.attempts,
            }
        )
        embedder = None
        embedding_model = LOCAL_EMBEDDING_MODEL
        index_embedding_identity = embedding_status["identity"]
        active_store_path = local_store_path_for_embedding_identity(store_path, index_embedding_identity)
        lexical_store = LocalVectorStore(active_store_path)
        if resolved_vector_backend == "local":
            vector_store = lexical_store
        if rebuild_index:
            index_sync, vector_store, resolved_vector_backend = sync_index_with_store_fallback(
                vector_store=vector_store,
                resolved_backend=resolved_vector_backend,
                vector_status=vector_store_status,
                store_path=active_store_path,
                embedder=embedder,
                rebuild=True,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
            )
        else:
            index_sync = index_status_without_rebuild(
                vector_store=vector_store,
                store_path=active_store_path,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
                reason="embedding_fallback_without_rebuild",
            )
        (
            all_chunks,
            chunks,
            rejected_chunks,
            permission_blocked_matches,
            parents_count,
            chunks_by_id,
            chunks_by_parent,
        ) = load_access_state()
        query_vector = vectorize(tokenize(query))

    if resolved_vector_backend == "qdrant":
        try:
            dense_results, attempts = call_with_retries(
                lambda: vector_store.search(
                    query_vector,
                    DENSE_TOP_N,
                    access_filter=qdrant_access_filter(scopes),
                )
            )
            vector_store_status["attempts"] = max(vector_store_status.get("attempts", 0), attempts)
        except RetryExhausted as exc:
            set_vector_store_fallback(vector_store_status, str(exc), exc.attempts)
            lexical_store = LocalVectorStore(active_store_path)
            vector_store = lexical_store
            resolved_vector_backend = "local"
            index_sync = index_status_without_rebuild(
                vector_store=lexical_store,
                store_path=active_store_path,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
                reason="qdrant_search_fallback",
            )
            (
                all_chunks,
                chunks,
                rejected_chunks,
                permission_blocked_matches,
                parents_count,
                chunks_by_id,
                chunks_by_parent,
            ) = load_access_state()
            dense_results = dense_recall(query_vector, chunks)
    else:
        dense_results = dense_recall(query_vector, chunks)
    stage_latencies_ms["dense_recall"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    if resolved_vector_backend == "qdrant":
        try:
            bm25_results, attempts = call_with_retries(
                lambda: vector_store.bm25_search(query, scopes, top_n=BM25_TOP_N)
            )
            vector_store_status["attempts"] = max(vector_store_status.get("attempts", 0), attempts)
        except RetryExhausted as exc:
            set_vector_store_fallback(vector_store_status, str(exc), exc.attempts)
            lexical_store = LocalVectorStore(active_store_path)
            vector_store = lexical_store
            resolved_vector_backend = "local"
            index_sync = index_status_without_rebuild(
                vector_store=lexical_store,
                store_path=active_store_path,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
                reason="qdrant_bm25_fallback",
            )
            (
                all_chunks,
                chunks,
                rejected_chunks,
                permission_blocked_matches,
                parents_count,
                chunks_by_id,
                chunks_by_parent,
            ) = load_access_state()
            dense_results = dense_recall(query_vector, chunks)
            bm25_results = lexical_store.bm25_search(query, scopes)
    else:
        bm25_results = lexical_store.bm25_search(query, scopes)
    stage_latencies_ms["bm25_recall"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    if resolved_vector_backend == "qdrant":
        candidate_ids = [chunk.chunk_id for _, chunk in [*dense_results, *bm25_results]]
        docstore = SqliteDocstore(active_store_path)
        hydrated_chunks = docstore.hydrate(candidate_ids)
        chunks_by_id = {}
        dense_results = _hydrate_recall_results(dense_results, hydrated_chunks, chunks_by_id)
        bm25_results = _hydrate_recall_results(bm25_results, hydrated_chunks, chunks_by_id)
        chunks = list(chunks_by_id.values())
        all_chunks = chunks
        rejected_chunks = []
        permission_blocked_matches = []
        parents_count = len({chunk.parent_id for chunk in chunks})
        if not index_sync.get("chunks_count"):
            index_sync["chunks_count"] = len(chunks)
    else:
        for _, chunk in bm25_results:
            chunks_by_id.setdefault(chunk.chunk_id, chunk)
        for _, chunk in [*dense_results, *bm25_results]:
            stored = chunks_by_id.get(chunk.chunk_id)
            if stored is not None and not stored.dense_vector and chunk.dense_vector:
                stored.dense_vector = chunk.dense_vector
    chunks_by_parent = build_chunks_by_parent(list(chunks_by_id.values()))
    stage_latencies_ms["docstore_hydrate"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    fused = rrf_fuse(dense_results, bm25_results)
    stage_latencies_ms["rrf_fusion"] = int((time.perf_counter() - stage_started) * 1000)

    external_reranker = make_configured_external_reranker()
    stage_started = time.perf_counter()
    reranked = rerank(query, fused, chunks_by_id, external_reranker=external_reranker)
    stage_latencies_ms["rerank"] = int((time.perf_counter() - stage_started) * 1000)
    reranker_info = describe_reranker(external_reranker)
    score_policy = reranker_info["score_policy"]

    stage_started = time.perf_counter()
    diversified, dedup_dropped = mmr_select(reranked, chunks_by_id)
    stage_latencies_ms["mmr"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    selected, truncation = dynamic_truncate(
        diversified,
        chunks_by_id,
        chunks_by_parent=chunks_by_parent,
        score_policy=score_policy,
    )
    stage_latencies_ms["truncate"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    selected_candidates = [item.candidate for item in selected]
    sufficiency = sufficiency_check(
        query,
        selected_candidates,
        chunks_by_id,
        permission_blocked_matches=permission_blocked_matches,
        score_policy=score_policy,
    )
    stage_latencies_ms["sufficiency"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    context_packet = assemble_context(
        query,
        selected,
        chunks_by_id,
        sufficiency,
        chunks_by_parent=chunks_by_parent,
    )
    stage_latencies_ms["context"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    answer = generate_answer_resilient(context_packet, llm_status)
    stage_latencies_ms["answer"] = int((time.perf_counter() - stage_started) * 1000)
    validation = validate_citations(answer, context_packet)
    component_status = {
        "llm": llm_status,
        "embedding": embedding_status,
        "vector_store": vector_store_status,
        "reranker": reranker_info,
    }

    trace = {
        "trace_id": f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{uuid.uuid4().hex[:8]}",
        "query": query,
        "pipeline_version": "production_rag_pipeline_v2",
        "index_version": index_sync["store"],
        "model_config": {
            "llm_model": llm_status["model"],
            "llm_api_style": DEFAULT_LLM_API_STYLE,
            "llm_base_url": llm_status["base_url"] if not llm_status["fallback_used"] else "offline_extractive",
            "embedding_model": embedding_model,
            "embedding_identity": index_embedding_identity,
            "embedding_base_url": (
                embedding_status["base_url"] if not embedding_status["fallback_used"] else "offline_hash"
            ),
            "vector_dimensions": vector_size,
            "rerank_mode": reranker_info["mode"],
            "rerank_score_policy": score_policy,
            "requested_vector_backend": vector_backend,
            "vector_backend": resolved_vector_backend,
            "qdrant_url": resolve_qdrant_url() if resolved_vector_backend == "qdrant" else "",
            "qdrant_collection": resolve_qdrant_collection() if resolved_vector_backend == "qdrant" else "",
        },
        "component_status": component_status,
        "fallbacks": fallback_summary(component_status),
        "index_sync": index_sync,
        "permission_filter": {
            "allowed_scopes": sorted(scopes),
            "visible_chunks": len(chunks),
            "rejected_chunks": rejected_chunks,
            "blocked_matches": permission_blocked_matches,
        },
        "parents_count": parents_count,
        "chunks_count": len(chunks),
        "dense_top": summarize_results(dense_results),
        "bm25_top": summarize_results(bm25_results),
        "rrf_top": [asdict(item) for item in list(fused.values())[:10]],
        "rerank_top": [asdict(item) for item in reranked],
        "selection_strategy": {
            "name": "mmr",
            "lambda": MMR_LAMBDA,
            "parent_expansion": True,
            "parent_expansion_max_chars": PARENT_EXPANSION_MAX_CHARS,
        },
        "reranker": reranker_info,
        "dedup_dropped": dedup_dropped,
        "truncation": truncation,
        "context_packet": context_packet,
        "answer": answer,
        "validation": validation,
        "stage_latencies_ms": stage_latencies_ms,
    }
    latency_ms = int((time.perf_counter() - started) * 1000)
    monitoring_event = build_monitoring_event(trace, latency_ms=latency_ms, status="ok")
    if monitoring_enabled:
        try:
            persist_monitoring_event(monitoring_event, metrics_path=metrics_path)
            trace["monitoring_metrics_path"] = str(metrics_path)
        except OSError as exc:
            trace["monitoring_write_error"] = str(exc)

    trace_path: Path | None = None
    if save_trace:
        try:
            TRACE_DIR.mkdir(exist_ok=True)
            trace_path = TRACE_DIR / f"{trace['trace_id']}.json"
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            trace["trace_save_error"] = str(exc)
            trace_path = None

    if not quiet and trace_only:
        print(json.dumps(trace, ensure_ascii=False, indent=2))
    elif not quiet:
        print_answer = _shim_value("print_answer", _print_answer)
        print_answer(trace, trace_path)
    return trace
