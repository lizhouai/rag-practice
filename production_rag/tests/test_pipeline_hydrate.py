from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import run_pipeline as rag


def _active_local_store_path(store_path: Path) -> Path:
    identity = rag.embedding_identity(rag.EMBEDDING_PROVIDER_LOCAL, rag.LOCAL_EMBEDDING_MODEL)
    return rag.local_store_path_for_embedding_identity(store_path, identity)


class PipelineHydrateTest(unittest.TestCase):
    def test_qdrant_path_issues_no_full_scroll_and_hydrates_from_docstore(self) -> None:
        seen_urls: list[str] = []

        def fake_request_json(method, url, body=None, headers=None, ok_statuses=(200,)):
            seen_urls.append(url)
            if method == "GET" and url.endswith("/collections/rag_test"):
                return {
                    "result": {
                        "payload_schema": {name: {} for name in rag.QDRANT_PAYLOAD_INDEXES},
                        "config": {"params": {"sparse_vectors": {rag.QDRANT_BM25_VECTOR_NAME: {}}}},
                    }
                }
            if url.endswith("/points/count"):
                return {"result": {"count": 0}}
            if url.endswith("/points/query") and body.get("using") == rag.QDRANT_DENSE_VECTOR_NAME:
                return {
                    "result": {
                        "points": [
                            {
                                "id": 1,
                                "score": 0.9,
                                "vector": {"dense": [1.0, 0.0]},
                                "payload": {
                                    "chunk_id": "c1",
                                    "parent_id": "p1",
                                    "doc_id": "d1",
                                    "title_path": ["t"],
                                    "token_count": 5,
                                    "permission_scopes": ["internal"],
                                },
                            },
                        ]
                    }
                }
            if url.endswith("/points/query") and body.get("using") == rag.QDRANT_BM25_VECTOR_NAME:
                return {"result": {"points": []}}
            raise AssertionError(f"unexpected request {method} {url}")

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "rag.sqlite"
            active_store_path = _active_local_store_path(store_path)
            rag.LocalVectorStore(active_store_path).upsert_document(
                "d1",
                "d.md",
                "h",
                "local:test",
                [
                    rag.Chunk(
                        chunk_id="c1",
                        parent_id="p1",
                        doc_id="d1",
                        title_path=["t"],
                        text="退款 7 到 15 个工作日",
                        metadata={"permission_scope": "internal"},
                        token_count=5,
                        dense_vector=[1.0, 0.0],
                        terms=rag.tokenize("退款 工作日"),
                    )
                ],
            )
            with (
                patch.dict(os.environ, {"QDRANT_URL": "http://qdrant.test", "QDRANT_COLLECTION": "rag_test"}, clear=True),
                patch.object(rag.time, "sleep", lambda _: None),
                patch.object(rag, "request_json", fake_request_json),
            ):
                trace = rag.run_query(
                    "退款多久",
                    quiet=True,
                    vector_backend="qdrant",
                    store_path=store_path,
                    metrics_path=Path(tmp) / "m.jsonl",
                )

        self.assertTrue(all(not url.endswith("/points/scroll") for url in seen_urls), f"unexpected scroll in {seen_urls}")
        self.assertLessEqual(len(seen_urls), 6)
        self.assertIn("退款", json.dumps(trace["context_packet"], ensure_ascii=False))
        self.assertEqual(trace["component_status"]["vector_store"]["backend"], "qdrant")

    def test_parent_expansion_pulls_unrecalled_sibling_from_docstore(self) -> None:
        def fake_request_json(method, url, body=None, headers=None, ok_statuses=(200,)):
            if method == "GET" and url.endswith("/collections/rag_test"):
                return {
                    "result": {
                        "payload_schema": {name: {} for name in rag.QDRANT_PAYLOAD_INDEXES},
                        "config": {"params": {"sparse_vectors": {rag.QDRANT_BM25_VECTOR_NAME: {}}}},
                    }
                }
            if url.endswith("/points/count"):
                return {"result": {"count": 0}}
            if url.endswith("/points/query") and body.get("using") == rag.QDRANT_DENSE_VECTOR_NAME:
                return {
                    "result": {
                        "points": [
                            {
                                "id": 1,
                                "score": 0.9,
                                "vector": {"dense": [1.0, 0.0]},
                                "payload": {
                                    "chunk_id": "c1",
                                    "parent_id": "p1",
                                    "doc_id": "d1",
                                    "title_path": ["t"],
                                    "token_count": 5,
                                    "permission_scopes": ["internal"],
                                },
                            }
                        ]
                    }
                }
            if url.endswith("/points/query"):
                return {"result": {"points": []}}
            raise AssertionError(f"unexpected request {method} {url}")

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "rag.sqlite"
            active_store_path = _active_local_store_path(store_path)
            rag.LocalVectorStore(active_store_path).upsert_document(
                "d1",
                "d.md",
                "h",
                "local:test",
                [
                    rag.Chunk(
                        chunk_id="c1",
                        parent_id="p1",
                        doc_id="d1",
                        title_path=["t"],
                        text="第一段 已召回",
                        metadata={"permission_scope": "internal"},
                        token_count=5,
                        dense_vector=[1.0, 0.0],
                        terms=rag.tokenize("第一段"),
                    ),
                    rag.Chunk(
                        chunk_id="c2",
                        parent_id="p1",
                        doc_id="d1",
                        title_path=["t"],
                        text="第二段 未召回兄弟",
                        metadata={"permission_scope": "internal"},
                        token_count=5,
                        dense_vector=[0.0, 1.0],
                        terms=rag.tokenize("第二段"),
                    ),
                ],
            )
            with (
                patch.dict(os.environ, {"QDRANT_URL": "http://qdrant.test", "QDRANT_COLLECTION": "rag_test"}, clear=True),
                patch.object(rag, "request_json", fake_request_json),
            ):
                trace = rag.run_query(
                    "第一段",
                    quiet=True,
                    vector_backend="qdrant",
                    store_path=store_path,
                    metrics_path=Path(tmp) / "m.jsonl",
                )
        self.assertIn("第二段 未召回兄弟", json.dumps(trace["context_packet"], ensure_ascii=False))

    def test_trace_uses_count_for_time_audit(self) -> None:
        def fake_request_json(method, url, body=None, headers=None, ok_statuses=(200,)):
            if method == "GET" and url.endswith("/collections/rag_test"):
                return {
                    "result": {
                        "payload_schema": {name: {} for name in rag.QDRANT_PAYLOAD_INDEXES},
                        "config": {"params": {"sparse_vectors": {rag.QDRANT_BM25_VECTOR_NAME: {}}}},
                    }
                }
            if url.endswith("/points/count"):
                rendered = json.dumps(body["filter"])
                return {"result": {"count": 3 if '"lt"' in rendered else 2}}
            if url.endswith("/points/query") and body.get("using") == rag.QDRANT_DENSE_VECTOR_NAME:
                return {
                    "result": {
                        "points": [
                            {
                                "id": 1,
                                "score": 0.9,
                                "vector": {"dense": [1.0, 0.0]},
                                "payload": {
                                    "chunk_id": "c1",
                                    "parent_id": "p1",
                                    "doc_id": "d1",
                                    "title_path": ["t"],
                                    "token_count": 5,
                                    "permission_scopes": ["internal"],
                                },
                            }
                        ]
                    }
                }
            if url.endswith("/points/query"):
                return {"result": {"points": []}}
            raise AssertionError(f"unexpected request {method} {url}")

        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "rag.sqlite"
            active_store_path = _active_local_store_path(store_path)
            rag.LocalVectorStore(active_store_path).upsert_document(
                "d1",
                "d.md",
                "h",
                "local:test",
                [
                    rag.Chunk(
                        chunk_id="c1",
                        parent_id="p1",
                        doc_id="d1",
                        title_path=["t"],
                        text="x",
                        metadata={"permission_scope": "internal"},
                        token_count=5,
                        dense_vector=[1.0, 0.0],
                        terms=rag.tokenize("x"),
                    )
                ],
            )
            with (
                patch.dict(os.environ, {"QDRANT_URL": "http://qdrant.test", "QDRANT_COLLECTION": "rag_test"}, clear=True),
                patch.object(rag, "request_json", fake_request_json),
            ):
                trace = rag.run_query(
                    "x",
                    quiet=True,
                    vector_backend="qdrant",
                    store_path=store_path,
                    metrics_path=Path(tmp) / "m.jsonl",
                )
        self.assertEqual(trace["permission_filter"]["rejected_counts"], {"expired": 3, "not_yet_effective": 2})


if __name__ == "__main__":
    unittest.main()
