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


class BlockedHintTest(unittest.TestCase):
    def _fake(self, blocked_recall_points: list[dict]):
        def fake_request_json(method, url, body=None, headers=None, ok_statuses=(200,)):
            if method == "GET":
                return {
                    "result": {
                        "payload_schema": {name: {} for name in rag.QDRANT_PAYLOAD_INDEXES},
                        "config": {"params": {"sparse_vectors": {rag.QDRANT_BM25_VECTOR_NAME: {}}}},
                    }
                }
            if url.endswith("/points/count"):
                return {"result": {"count": 0}}
            if url.endswith("/points/query"):
                rendered_filter = json.dumps(body.get("filter", {}))
                if "must_not" in rendered_filter:
                    return {"result": {"points": blocked_recall_points}}
                return {"result": {"points": []}}
            raise AssertionError(url)

        return fake_request_json

    def test_blocked_hint_off_by_default_no_extra_recall(self) -> None:
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
                patch.object(rag, "request_json", self._fake([])),
            ):
                trace = rag.run_query(
                    "x",
                    quiet=True,
                    vector_backend="qdrant",
                    store_path=store_path,
                    metrics_path=Path(tmp) / "m.jsonl",
                )
        self.assertEqual(trace["permission_filter"]["blocked_matches"], [])

    def test_blocked_hint_on_surfaces_relevant_blocked_titles(self) -> None:
        blocked_points = [
            {
                "id": 9,
                "score": 3.0,
                "vector": {"dense": [0.0, 1.0]},
                "payload": {
                    "chunk_id": "b1",
                    "parent_id": "pb",
                    "doc_id": "secret",
                    "title_path": ["机密", "薪酬"],
                    "token_count": 5,
                    "permission_scopes": ["finance_restricted"],
                },
            }
        ]
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
                patch.object(rag, "request_json", self._fake(blocked_points)),
            ):
                trace = rag.run_query(
                    "薪酬",
                    quiet=True,
                    vector_backend="qdrant",
                    blocked_hint=True,
                    store_path=store_path,
                    metrics_path=Path(tmp) / "m.jsonl",
                )
        titles = [match["title_path"] for match in trace["permission_filter"]["blocked_matches"]]
        self.assertIn("机密 > 薪酬", titles)


if __name__ == "__main__":
    unittest.main()
