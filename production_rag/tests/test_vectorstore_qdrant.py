from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.helpers import make_chunk, rag


class QdrantVectorStoreTest(unittest.TestCase):
    def test_qdrant_cloud_url_without_rest_port_gets_actionable_error(self) -> None:
        message = rag.format_transport_error(
            "https://cluster.us-east-1-1.aws.cloud.qdrant.io/collections/rag",
            "UNEXPECTED_EOF_WHILE_READING",
        )

        self.assertIn("port 6333", message)
        self.assertIn("https://cluster.us-east-1-1.aws.cloud.qdrant.io:6333", message)

    def test_request_json_error_includes_target_url(self) -> None:
        class FailingUrlopen:
            def __call__(self, request: urllib.request.Request, timeout: int) -> object:
                raise urllib.error.URLError("UNEXPECTED_EOF_WHILE_READING")

        with patch("urllib.request.urlopen", FailingUrlopen()):
            with self.assertRaises(RuntimeError) as raised:
                rag.request_json(
                    "GET",
                    "https://cluster.us-east-1-1.aws.cloud.qdrant.io/collections/rag",
                )

        message = str(raised.exception)
        self.assertIn("GET https://cluster.us-east-1-1.aws.cloud.qdrant.io/collections/rag", message)
        self.assertIn("https://cluster.us-east-1-1.aws.cloud.qdrant.io:6333", message)

    def test_ensure_collection_creates_doc_id_payload_index_when_missing(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeResponse:
            def __init__(self, body: dict[str, object]) -> None:
                self.body = body

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self.body).encode("utf-8")

        def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
            calls.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "headers": request.headers,
                    "body": request.data,
                }
            )
            if request.get_method() == "GET":
                return FakeResponse({"result": {"payload_schema": {}}})
            return FakeResponse({"result": {"status": "acknowledged"}})

        store = rag.QdrantVectorStore(
            base_url="https://qdrant.test",
            collection_name="rag_test",
            vector_size=64,
            api_key="qdrant-key",
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            store.ensure_collection()

        self.assertEqual(calls[0]["url"], "https://qdrant.test/collections/rag_test")
        self.assertEqual(calls[0]["method"], "GET")
        self.assertEqual(calls[1]["url"], "https://qdrant.test/collections/rag_test/index?wait=true")
        self.assertEqual(calls[1]["method"], "PUT")
        self.assertEqual(calls[1]["headers"]["Api-key"], "qdrant-key")
        self.assertEqual(
            json.loads(calls[1]["body"].decode("utf-8")),
            {"field_name": "doc_id", "field_schema": "keyword"},
        )

    def test_upsert_document_sends_chunks_to_qdrant_points_api(self) -> None:
        captured: dict[str, object] = {}
        chunk = make_chunk("chunk-1", "internal")

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"result":{"status":"ok"}}'

        def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["headers"] = request.headers
            captured["body"] = request.data
            captured["timeout"] = timeout
            return FakeResponse()

        store = rag.QdrantVectorStore(
            base_url="http://qdrant.test",
            collection_name="rag_test",
            vector_size=64,
            api_key="qdrant-key",
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            store.upsert_document("doc-1", "data/raw/doc.md", "hash-1", "embedding-3", [chunk])

        body = json.loads(captured["body"].decode("utf-8"))
        self.assertEqual(captured["url"], "http://qdrant.test/collections/rag_test/points?wait=true")
        self.assertEqual(captured["method"], "PUT")
        self.assertEqual(captured["headers"]["Api-key"], "qdrant-key")
        self.assertEqual(captured["timeout"], 90)
        self.assertEqual(body["points"][0]["id"], rag.stable_point_id("chunk-1"))
        self.assertEqual(body["points"][0]["vector"]["dense"], chunk.dense_vector)
        self.assertEqual(
            body["points"][0]["vector"]["bm25"],
            rag.qdrant_sparse_vector_from_terms(chunk.terms),
        )
        self.assertEqual(body["points"][0]["payload"]["doc_id"], "doc-1")
        self.assertEqual(body["points"][0]["payload"]["content_hash"], "hash-1")
        self.assertNotIn("text", body["points"][0]["payload"])
        self.assertNotIn("terms", body["points"][0]["payload"])
        self.assertEqual(body["points"][0]["payload"]["parent_id"], chunk.parent_id)

    def test_search_uses_qdrant_query_points_and_returns_payload_chunks(self) -> None:
        chunk = make_chunk("chunk-1", "internal")
        payload = rag.chunk_to_qdrant_payload(
            chunk,
            source_path="data/raw/doc.md",
            content_hash="hash-1",
            embedding_model="embedding-3",
        )

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "result": {
                            "points": [
                                {
                                    "id": rag.stable_point_id(chunk.chunk_id),
                                    "score": 0.87,
                                    "payload": payload,
                                    "vector": {"dense": [0.5, 0.6]},
                                }
                            ]
                        }
                    }
                ).encode("utf-8")

        captured: dict[str, object] = {}

        def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["body"] = request.data
            return FakeResponse()

        store = rag.QdrantVectorStore(
            base_url="http://qdrant.test",
            collection_name="rag_test",
            vector_size=64,
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            results = store.search([0.1, 0.2], top_n=3)

        body = json.loads(captured["body"].decode("utf-8"))
        self.assertEqual(captured["url"], "http://qdrant.test/collections/rag_test/points/query")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(body["query"], [0.1, 0.2])
        self.assertEqual(body["using"], "dense")
        self.assertEqual(body["limit"], 3)
        self.assertEqual(body["with_vector"], [rag.QDRANT_DENSE_VECTOR_NAME])
        self.assertEqual(results[0][0], 0.87)
        self.assertEqual(results[0][1].chunk_id, "chunk-1")
        self.assertEqual(results[0][1].dense_vector, [0.5, 0.6])

    def test_bm25_search_uses_qdrant_sparse_vector_query(self) -> None:
        chunk = make_chunk("chunk-1", "internal")
        chunk.terms = rag.tokenize("refund timeline")
        payload = rag.chunk_to_qdrant_payload(
            chunk,
            source_path="data/raw/doc.md",
            content_hash="hash-1",
            embedding_model="embedding-3",
        )

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "result": {
                            "points": [
                                {
                                    "id": rag.stable_point_id(chunk.chunk_id),
                                    "score": 4.2,
                                    "payload": payload,
                                    "vector": {"dense": [0.3, 0.4]},
                                }
                            ]
                        }
                    }
                ).encode("utf-8")

        captured: dict[str, object] = {}

        def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["body"] = request.data
            return FakeResponse()

        store = rag.QdrantVectorStore(
            base_url="http://qdrant.test",
            collection_name="rag_test",
            vector_size=64,
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            results = store.bm25_search("refund timeline", {"internal"}, top_n=3, today="2026-06-07")

        body = json.loads(captured["body"].decode("utf-8"))
        self.assertEqual(captured["url"], "http://qdrant.test/collections/rag_test/points/query")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(body["query"], rag.qdrant_sparse_vector_from_terms(rag.tokenize("refund timeline")))
        self.assertEqual(body["using"], "bm25")
        self.assertEqual(body["limit"], 3)
        self.assertEqual(body["with_vector"], [rag.QDRANT_DENSE_VECTOR_NAME])
        self.assertEqual(body["filter"], rag.qdrant_access_filter({"internal"}, "2026-06-07"))
        self.assertEqual(results[0][0], 4.2)
        self.assertEqual(results[0][1].chunk_id, "chunk-1")
        self.assertEqual(results[0][1].dense_vector, [0.3, 0.4])

    def test_count_posts_filtered_count_request(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"result":{"count":7}}'

        def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
            captured["url"] = request.full_url
            captured["body"] = request.data
            return FakeResponse()

        store = rag.QdrantVectorStore(
            base_url="http://qdrant.test",
            collection_name="rag_test",
            vector_size=64,
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            count = store.count(rag.qdrant_expired_filter({"internal"}, "2026-06-07"))

        body = json.loads(captured["body"].decode("utf-8"))
        self.assertEqual(captured["url"], "http://qdrant.test/collections/rag_test/points/count")
        self.assertEqual(body["filter"], rag.qdrant_expired_filter({"internal"}, "2026-06-07"))
        self.assertEqual(body["exact"], True)
        self.assertEqual(count, 7)

    @staticmethod
    def classify_scroll_filter(access_filter: dict) -> str:
        rendered = json.dumps(access_filter)
        if "must_not" in access_filter:
            return "permission_blocked"
        if '"gt"' in rendered:
            return "not_yet_effective"
        if '"lt"' in rendered:
            return "expired"
        return "visible"

    def test_load_access_chunks_scrolls_all_categories_without_vectors(self) -> None:
        visible_payload = rag.chunk_to_qdrant_payload(
            make_chunk("chunk-visible", "internal"),
            source_path="data/raw/doc.md",
            content_hash="hash-1",
            embedding_model="embedding-3",
        )
        blocked_payload = rag.chunk_to_qdrant_payload(
            make_chunk("chunk-blocked", "finance_restricted"),
            source_path="data/raw/doc.md",
            content_hash="hash-2",
            embedding_model="embedding-3",
        )
        expired_payload = rag.chunk_to_qdrant_payload(
            make_chunk("chunk-expired", "internal", effective_to="2025-01-01"),
            source_path="data/raw/doc.md",
            content_hash="hash-3",
            embedding_model="embedding-3",
        )
        points_by_category = {
            "visible": [{"id": 1, "payload": visible_payload}],
            "permission_blocked": [{"id": 2, "payload": blocked_payload}],
            "not_yet_effective": [],
            "expired": [{"id": 3, "payload": expired_payload}],
        }
        scroll_bodies: list[dict] = []

        def fake_request_json(
            method: str,
            url: str,
            body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            ok_statuses: tuple[int, ...] = (200,),
        ) -> dict:
            assert url.endswith("/points/scroll"), f"unexpected request {method} {url}"
            scroll_bodies.append(body)
            category = self.classify_scroll_filter(body["filter"])
            return {"result": {"points": points_by_category[category], "next_page_offset": None}}

        store = rag.QdrantVectorStore(
            base_url="http://qdrant.test",
            collection_name="rag_test",
            vector_size=64,
        )
        with patch.object(rag, "request_json", fake_request_json):
            visible, rejected, blocked = store.load_access_chunks({"internal"}, today="2026-06-07")

        self.assertEqual(len(scroll_bodies), 4)
        for body in scroll_bodies:
            self.assertEqual(body["with_vector"], False)
        self.assertEqual([chunk.chunk_id for chunk in visible], ["chunk-visible"])
        self.assertEqual([chunk.chunk_id for chunk in blocked], ["chunk-blocked"])
        self.assertEqual(
            {(item["chunk_id"], item["reason"]) for item in rejected},
            {("chunk-blocked", "permission_scope"), ("chunk-expired", "expired")},
        )

    def test_run_query_qdrant_recall_vectors_feed_mmr_dedup(self) -> None:
        chunk_a = make_chunk("chunk-a", "internal")
        chunk_b = make_chunk("chunk-b", "internal")
        payloads = {
            chunk.chunk_id: rag.chunk_to_qdrant_payload(
                chunk,
                source_path="data/raw/doc.md",
                content_hash=f"hash-{chunk.chunk_id}",
                embedding_model="embedding-3",
            )
            for chunk in (chunk_a, chunk_b)
        }
        shared_vector = [1.0, 0.0]

        def fake_request_json(
            method: str,
            url: str,
            body: dict[str, object] | None = None,
            headers: dict[str, str] | None = None,
            ok_statuses: tuple[int, ...] = (200,),
        ) -> dict:
            if method == "GET" and url.endswith("/collections/rag_test"):
                return {
                    "result": {
                        "payload_schema": {name: {} for name in rag.QDRANT_PAYLOAD_INDEXES},
                        "config": {"params": {"sparse_vectors": {rag.QDRANT_BM25_VECTOR_NAME: {}}}},
                    }
                }
            if url.endswith("/points/scroll"):
                category = self.classify_scroll_filter(body["filter"])
                if category == "visible":
                    # The access snapshot deliberately carries no vectors; MMR must
                    # get them from the recall responses below.
                    points = [{"id": 1, "payload": payloads["chunk-a"]}, {"id": 2, "payload": payloads["chunk-b"]}]
                else:
                    points = []
                return {"result": {"points": points, "next_page_offset": None}}
            if url.endswith("/points/count"):
                return {"result": {"count": 0}}
            if url.endswith("/points/query") and body["using"] == rag.QDRANT_DENSE_VECTOR_NAME:
                return {
                    "result": {
                        "points": [
                            {"id": 1, "score": 0.9, "payload": payloads["chunk-a"], "vector": {"dense": shared_vector}},
                            {"id": 2, "score": 0.8, "payload": payloads["chunk-b"], "vector": {"dense": shared_vector}},
                        ]
                    }
                }
            if url.endswith("/points/query") and body["using"] == rag.QDRANT_BM25_VECTOR_NAME:
                return {"result": {"points": []}}
            raise AssertionError(f"unexpected request {method} {url}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with (
                patch.dict(
                    os.environ,
                    {"QDRANT_URL": "http://qdrant.test", "QDRANT_COLLECTION": "rag_test"},
                    clear=True,
                ),
                patch.object(rag, "request_json", fake_request_json),
            ):
                trace = rag.run_query(
                    "chunk test text",
                    quiet=True,
                    vector_backend="qdrant",
                    store_path=tmp_path / "rag.sqlite",
                    metrics_path=tmp_path / "metrics.jsonl",
                )

        self.assertEqual(trace["component_status"]["vector_store"]["backend"], "qdrant")
        self.assertFalse(trace["component_status"]["vector_store"]["fallback_used"])
        self.assertEqual(
            [(item["chunk_id"], item["reason"]) for item in trace["dedup_dropped"]],
            [("chunk-b", "near_duplicate")],
        )


if __name__ == "__main__":
    unittest.main()
