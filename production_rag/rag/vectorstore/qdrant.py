from __future__ import annotations

import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

from rag.config import BM25_TOP_N, DEFAULT_EMBEDDING_DIMENSIONS, DEFAULT_QDRANT_COLLECTION, DEFAULT_QDRANT_URL
from rag.config import DENSE_TOP_N, QDRANT_BM25_VECTOR_NAME, QDRANT_DENSE_VECTOR_NAME, QDRANT_PAYLOAD_INDEXES
from rag.chunking import metadata_access_fields, qdrant_sparse_vector_from_terms, split_metadata_values, tokenize
from rag.http import request_json as _http_request_json
from rag.models import Chunk
from rag.vectorstore.filters import qdrant_access_filter, qdrant_expired_filter, qdrant_filter_by_doc_id
from rag.vectorstore.filters import qdrant_not_yet_effective_filter, qdrant_permission_blocked_filter

__all__ = [
    "qdrant_point_dense_vector",
    "stable_point_id",
    "chunk_to_qdrant_payload",
    "chunk_from_qdrant_payload",
    "QdrantVectorStore",
]


def request_json(
    method: str,
    url: str,
    body: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    ok_statuses: tuple[int, ...] = (200,),
) -> dict:
    # During Phase 1, tests patch run_pipeline.request_json through the shim.
    # Keep that patch point alive while Qdrant code lives in rag.vectorstore.qdrant.
    shim = sys.modules.get("run_pipeline")
    request_json_func = getattr(shim, "request_json", _http_request_json) if shim is not None else _http_request_json
    return request_json_func(method, url, body=body, headers=headers, ok_statuses=ok_statuses)


def qdrant_point_dense_vector(raw_vector: object) -> list[float]:
    if isinstance(raw_vector, dict):
        vector = raw_vector.get(QDRANT_DENSE_VECTOR_NAME) or raw_vector.get("")
    else:
        vector = raw_vector
    return list(vector) if isinstance(vector, list) else []


def stable_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def chunk_to_qdrant_payload(
    chunk: Chunk,
    *,
    source_path: str,
    content_hash: str,
    embedding_model: str,
    doc_id: str | None = None,
) -> dict:
    metadata = dict(chunk.metadata)
    metadata.setdefault("source_path", source_path)
    resolved_doc_id = doc_id or chunk.doc_id
    access_fields = metadata_access_fields(metadata)
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "doc_id": resolved_doc_id,
        "title_path": chunk.title_path,
        "metadata": metadata,
        "permission_scope": access_fields["permission_scope"],
        "permission_scopes": sorted(split_metadata_values(str(access_fields["permission_scope"]))),
        "effective_from": access_fields["effective_from"],
        "effective_to": access_fields["effective_to"],
        "effective_from_day": access_fields["effective_from_day"],
        "effective_to_day": access_fields["effective_to_day"],
        "token_count": chunk.token_count,
        "source_path": source_path,
        "content_hash": content_hash,
        "embedding_model": embedding_model,
    }


def chunk_from_qdrant_payload(payload: dict, vector: list[float] | None = None) -> Chunk:
    return Chunk(
        chunk_id=payload["chunk_id"],
        parent_id=payload["parent_id"],
        doc_id=payload["doc_id"],
        title_path=list(payload.get("title_path", [])),
        text=payload.get("text", ""),
        metadata=dict(payload.get("metadata", {})),
        token_count=int(payload.get("token_count", 1)),
        dense_vector=vector or [],
        terms=list(payload.get("terms", [])),
    )


class QdrantVectorStore:
    def __init__(
        self,
        base_url: str = DEFAULT_QDRANT_URL,
        collection_name: str = DEFAULT_QDRANT_COLLECTION,
        vector_size: int = DEFAULT_EMBEDDING_DIMENSIONS,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.api_key = api_key

    def describe(self) -> str:
        return f"{self.base_url}/{self.collection_name}"

    def collection_url(self, suffix: str = "") -> str:
        return f"{self.base_url}/collections/{self.collection_name}{suffix}"

    def headers(self) -> dict[str, str] | None:
        if not self.api_key:
            return None
        return {"api-key": self.api_key}

    def ensure_collection(self) -> None:
        try:
            collection_info = request_json("GET", self.collection_url(), headers=self.headers(), ok_statuses=(200,))
            self.ensure_payload_indexes(collection_info)
            self.ensure_sparse_vectors(collection_info)
            return
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise RuntimeError(
                    "Qdrant is not reachable. Set QDRANT_URL to a reachable Qdrant endpoint, "
                    "set QDRANT_API_KEY if the endpoint requires authentication, start the local "
                    "service with `docker compose up -d qdrant`, or use `--vector-backend local` "
                    "only for offline debugging."
                ) from exc
        body = {
            "vectors": {
                QDRANT_DENSE_VECTOR_NAME: {
                    "size": self.vector_size,
                    "distance": "Cosine",
                },
            },
            "sparse_vectors": {
                QDRANT_BM25_VECTOR_NAME: {
                    "modifier": "idf",
                },
            },
            "optimizers_config": {
                "default_segment_number": 2,
            },
        }
        request_json("PUT", self.collection_url(), body=body, headers=self.headers(), ok_statuses=(200,))
        self.ensure_payload_indexes()

    def ensure_sparse_vectors(self, collection_info: dict | None = None) -> None:
        existing = self.sparse_vectors_schema(collection_info)
        if QDRANT_BM25_VECTOR_NAME in existing:
            return
        request_json(
            "POST",
            self.collection_url("/sparse_vectors"),
            body={"sparse_vectors": {QDRANT_BM25_VECTOR_NAME: {"modifier": "idf"}}},
            headers=self.headers(),
            ok_statuses=(200,),
        )

    def ensure_payload_indexes(self, collection_info: dict | None = None) -> None:
        existing = self.payload_schema(collection_info)
        for field_name, field_schema in QDRANT_PAYLOAD_INDEXES.items():
            if field_name in existing:
                continue
            try:
                request_json(
                    "PUT",
                    self.collection_url("/index?wait=true"),
                    body={"field_name": field_name, "field_schema": field_schema},
                    headers=self.headers(),
                    ok_statuses=(200,),
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if "already exists" in message or "already has" in message:
                    continue
                raise

    @staticmethod
    def payload_schema(collection_info: dict | None) -> dict:
        if not isinstance(collection_info, dict):
            return {}
        result = collection_info.get("result", {})
        if not isinstance(result, dict):
            return {}
        payload_schema = result.get("payload_schema", {})
        return payload_schema if isinstance(payload_schema, dict) else {}

    @staticmethod
    def sparse_vectors_schema(collection_info: dict | None) -> dict:
        if not isinstance(collection_info, dict):
            return {}
        result = collection_info.get("result", {})
        if not isinstance(result, dict):
            return {}
        config = result.get("config", {})
        if not isinstance(config, dict):
            return {}
        params = config.get("params", {})
        if not isinstance(params, dict):
            return {}
        sparse_vectors = params.get("sparse_vectors", {})
        return sparse_vectors if isinstance(sparse_vectors, dict) else {}


    def reset(self) -> None:
        try:
            request_json("DELETE", self.collection_url(), headers=self.headers(), ok_statuses=(200,))
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
        self.ensure_collection()

    def load_manifest(self) -> dict[str, dict[str, str]]:
        manifest: dict[str, dict[str, str]] = {}
        for point in self.scroll_points(with_vector=False):
            payload = point.get("payload", {})
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue
            manifest.setdefault(
                doc_id,
                {
                    "source_path": payload.get("source_path", ""),
                    "content_hash": payload.get("content_hash", ""),
                    "embedding_model": payload.get("embedding_model", ""),
                    "updated_at": "",
                },
            )
        return manifest

    def upsert_document(
        self,
        doc_id: str,
        source_path: str,
        hash_value: str,
        embedding_model: str,
        chunks: list[Chunk],
    ) -> None:
        self.delete_documents({doc_id})
        points = [
            {
                "id": stable_point_id(chunk.chunk_id),
                "vector": {
                    QDRANT_DENSE_VECTOR_NAME: chunk.dense_vector,
                    QDRANT_BM25_VECTOR_NAME: qdrant_sparse_vector_from_terms(chunk.terms),
                },
                "payload": chunk_to_qdrant_payload(
                    chunk,
                    source_path=source_path,
                    content_hash=hash_value,
                    embedding_model=embedding_model,
                    doc_id=doc_id,
                ),
            }
            for chunk in chunks
        ]
        if not points:
            return
        request_json(
            "PUT",
            self.collection_url("/points?wait=true"),
            body={"points": points},
            headers=self.headers(),
            ok_statuses=(200,),
        )

    def delete_documents(self, doc_ids: set[str]) -> None:
        for doc_id in sorted(doc_ids):
            request_json(
                "POST",
                self.collection_url("/points/delete?wait=true"),
                body={"filter": qdrant_filter_by_doc_id(doc_id)},
                headers=self.headers(),
                ok_statuses=(200,),
            )

    def load_chunks(self, *, access_filter: dict | None = None, with_vector: bool = True) -> list[Chunk]:
        chunks: list[Chunk] = []
        for point in self.scroll_points(with_vector=with_vector, access_filter=access_filter):
            payload = point.get("payload", {})
            vector = qdrant_point_dense_vector(point.get("vector") or [])
            chunks.append(chunk_from_qdrant_payload(payload, vector=vector))
        return sorted(chunks, key=lambda chunk: chunk.chunk_id)

    def load_access_chunks(
        self,
        allowed_scopes: set[str],
        *,
        today: str | None = None,
    ) -> tuple[list[Chunk], list[dict], list[Chunk]]:
        # Dense vectors for retrieval candidates come back with the search/bm25
        # responses, so even the visible scroll can skip vector payloads here.
        # The four category scrolls are independent reads; running them
        # concurrently pays one round trip of latency instead of four, which
        # matters against a remote Qdrant.
        filters = {
            "visible": qdrant_access_filter(allowed_scopes, today),
            "permission_blocked": qdrant_permission_blocked_filter(allowed_scopes, today),
            "not_yet_effective": qdrant_not_yet_effective_filter(allowed_scopes, today),
            "expired": qdrant_expired_filter(allowed_scopes, today),
        }
        with ThreadPoolExecutor(max_workers=len(filters)) as executor:
            futures = {
                name: executor.submit(self.load_chunks, access_filter=access_filter, with_vector=False)
                for name, access_filter in filters.items()
            }
            loaded = {name: future.result() for name, future in futures.items()}
        visible = loaded["visible"]
        permission_blocked = loaded["permission_blocked"]
        not_yet_effective = loaded["not_yet_effective"]
        expired = loaded["expired"]
        rejected = [
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "reason": "permission_scope"}
            for chunk in permission_blocked
        ]
        rejected.extend(
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "reason": "not_yet_effective"}
            for chunk in not_yet_effective
        )
        rejected.extend(
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "reason": "expired"}
            for chunk in expired
        )
        return visible, rejected, permission_blocked

    def scroll_points(self, with_vector: bool, access_filter: dict | None = None) -> list[dict]:
        points: list[dict] = []
        offset: object | None = None
        while True:
            body: dict[str, object] = {
                "limit": 256,
                "with_payload": True,
                "with_vector": with_vector,
            }
            if access_filter is not None:
                body["filter"] = access_filter
            if offset is not None:
                body["offset"] = offset
            payload = request_json(
                "POST",
                self.collection_url("/points/scroll"),
                body=body,
                headers=self.headers(),
                ok_statuses=(200,),
            )
            result = payload.get("result", {})
            batch = result.get("points", []) if isinstance(result, dict) else []
            points.extend(batch)
            offset = result.get("next_page_offset") if isinstance(result, dict) else None
            if not offset:
                return points

    def search(
        self,
        query_vector: list[float],
        top_n: int = DENSE_TOP_N,
        *,
        access_filter: dict | None = None,
    ) -> list[tuple[float, Chunk]]:
        body: dict[str, object] = {
            "query": query_vector,
            "using": QDRANT_DENSE_VECTOR_NAME,
            "limit": top_n,
            "with_payload": True,
            "with_vector": [QDRANT_DENSE_VECTOR_NAME],
        }
        if access_filter is not None:
            body["filter"] = access_filter
        payload = request_json(
            "POST",
            self.collection_url("/points/query"),
            body=body,
            headers=self.headers(),
            ok_statuses=(200,),
        )
        result = payload.get("result", {})
        points = result.get("points", result) if isinstance(result, dict) else result
        scored: list[tuple[float, Chunk]] = []
        for point in points or []:
            scored.append(
                (
                    float(point.get("score", 0.0)),
                    chunk_from_qdrant_payload(
                        point.get("payload", {}),
                        vector=qdrant_point_dense_vector(point.get("vector") or []),
                    ),
                )
            )
        return scored

    def bm25_search(
        self,
        query: str,
        allowed_scopes: set[str],
        *,
        top_n: int = BM25_TOP_N,
        today: str | None = None,
    ) -> list[tuple[float, Chunk]]:
        query_sparse = qdrant_sparse_vector_from_terms(tokenize(query))
        if not query_sparse["indices"]:
            return []
        body: dict[str, object] = {
            "query": query_sparse,
            "using": QDRANT_BM25_VECTOR_NAME,
            "limit": top_n,
            "with_payload": True,
            "with_vector": [QDRANT_DENSE_VECTOR_NAME],
            "filter": qdrant_access_filter(allowed_scopes, today),
        }
        payload = request_json(
            "POST",
            self.collection_url("/points/query"),
            body=body,
            headers=self.headers(),
            ok_statuses=(200,),
        )
        result = payload.get("result", {})
        points = result.get("points", result) if isinstance(result, dict) else result
        scored: list[tuple[float, Chunk]] = []
        for point in points or []:
            scored.append(
                (
                    float(point.get("score", 0.0)),
                    chunk_from_qdrant_payload(
                        point.get("payload", {}),
                        vector=qdrant_point_dense_vector(point.get("vector") or []),
                    ),
                )
            )
        return scored
