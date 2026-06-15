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


class DefaultFallbackBehaviorTest(unittest.TestCase):
    def test_run_query_records_component_fallbacks_without_legacy_real_models_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store_path = tmp_path / "indexes" / "rag.sqlite"
            metrics_path = tmp_path / "metrics" / "online_metrics.jsonl"
            query = "\u8de8\u5883\u8ba2\u5355\u9000\u6b3e\u591a\u4e45\u5230\u8d26\uff1f"

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(rag.time, "sleep", lambda _: None),
                patch.object(rag, "request_json", side_effect=RuntimeError("qdrant down")),
            ):
                trace = rag.run_query(
                    query,
                    quiet=True,
                    rebuild_index=True,
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

            status = trace["component_status"]
            self.assertEqual(status["llm"]["mode"], "extractive_fallback")
            self.assertEqual(status["llm"]["reason"], "not_configured")
            self.assertEqual(status["embedding"]["mode"], "hash_fallback")
            self.assertEqual(status["embedding"]["reason"], "not_configured")
            self.assertEqual(status["vector_store"]["mode"], "sqlite_fallback")
            self.assertEqual(status["vector_store"]["reason"], "not_configured")
            self.assertEqual(status["vector_store"]["attempts"], 0)
            self.assertEqual(status["reranker"]["mode"], "skipped")
            self.assertEqual(status["reranker"]["score_policy"], rag.SCORE_POLICY_RRF_ONLY)
            self.assertEqual(trace["model_config"]["requested_vector_backend"], "qdrant")
            self.assertEqual(trace["model_config"]["vector_backend"], "local")
            self.assertEqual(trace["answer"]["mode"], "extractive")
            self.assertTrue(any(item["component"] == "vector_store" for item in trace["fallbacks"]))

    def test_configured_qdrant_failure_retries_then_uses_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store_path = tmp_path / "indexes" / "rag.sqlite"

            with (
                patch.dict(os.environ, {"QDRANT_URL": "http://qdrant.test:6333"}, clear=True),
                patch.object(rag.time, "sleep", lambda _: None),
                patch.object(rag, "request_json", side_effect=RuntimeError("qdrant down")),
            ):
                status: dict = {}
                vector_store, backend = rag.initialize_vector_store_with_fallback(
                    "qdrant",
                    store_path=store_path,
                    status=status,
                )

        self.assertIsInstance(vector_store, rag.LocalVectorStore)
        self.assertEqual(backend, "local")
        self.assertEqual(status["mode"], "sqlite_fallback")
        self.assertEqual(status["reason"], "qdrant_error")
        self.assertEqual(status["attempts"], 3)

    def test_missing_qdrant_configuration_uses_sqlite_without_network_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store_path = tmp_path / "indexes" / "rag.sqlite"

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(rag, "request_json", side_effect=AssertionError("qdrant should not be probed")),
            ):
                status: dict = {}
                vector_store, backend = rag.initialize_vector_store_with_fallback(
                    "qdrant",
                    store_path=store_path,
                    status=status,
                )

        self.assertIsInstance(vector_store, rag.LocalVectorStore)
        self.assertEqual(backend, "local")
        self.assertEqual(status["mode"], "sqlite_fallback")
        self.assertEqual(status["reason"], "not_configured")
        self.assertEqual(status["attempts"], 0)
        self.assertEqual(status["qdrant_url"], "")
        self.assertEqual(status["qdrant_collection"], "")

    def test_qdrant_cloud_without_api_key_uses_actionable_configuration_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store_path = tmp_path / "indexes" / "rag.sqlite"

            with (
                patch.dict(
                    os.environ,
                    {
                        "QDRANT_URL": "https://cluster.us-east-1-1.aws.cloud.qdrant.io:6333",
                        "QDRANT_COLLECTION": "rag_test",
                    },
                    clear=True,
                ),
                patch.object(rag, "request_json", side_effect=AssertionError("qdrant should not be probed")),
            ):
                status: dict = {}
                vector_store, backend = rag.initialize_vector_store_with_fallback(
                    "qdrant",
                    store_path=store_path,
                    status=status,
                )

        self.assertIsInstance(vector_store, rag.LocalVectorStore)
        self.assertEqual(backend, "local")
        self.assertEqual(status["mode"], "sqlite_fallback")
        self.assertEqual(status["reason"], "not_configured")
        self.assertEqual(status["attempts"], 0)
        self.assertIn("QDRANT_API_KEY", status["error"])
        self.assertEqual(status["qdrant_url"], "https://cluster.us-east-1-1.aws.cloud.qdrant.io:6333")
        self.assertEqual(status["qdrant_collection"], "rag_test")

    def test_parse_args_rejects_removed_real_models_flag(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["run_pipeline.py", "--query", "refund timeline", "--real-models"],
        ):
            with self.assertRaises(SystemExit):
                rag.parse_args()

    def test_configured_embedding_failure_retries_then_uses_hash_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store_path = tmp_path / "indexes" / "rag.sqlite"
            metrics_path = tmp_path / "metrics" / "online_metrics.jsonl"

            with (
                patch.dict(os.environ, {"EMBEDDING_API_KEY": "bad-key"}, clear=True),
                patch.object(rag.time, "sleep", lambda _: None),
                patch.object(rag, "request_json", side_effect=RuntimeError("embedding down")),
            ):
                trace = rag.run_query(
                    "\u8de8\u5883\u8ba2\u5355\u9000\u6b3e\u591a\u4e45\u5230\u8d26\uff1f",
                    quiet=True,
                    rebuild_index=True,
                    vector_backend="local",
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

        status = trace["component_status"]["embedding"]
        self.assertEqual(status["mode"], "hash_fallback")
        self.assertEqual(status["reason"], "embedding_error")
        self.assertEqual(status["attempts"], 3)
        self.assertEqual(trace["model_config"]["embedding_model"], rag.LOCAL_EMBEDDING_MODEL)

    def test_embedding_failure_uses_current_hash_index_without_rechunking_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store_path = tmp_path / "indexes" / "rag.sqlite"
            metrics_path = tmp_path / "metrics" / "online_metrics.jsonl"
            query = "\u8de8\u5883\u8ba2\u5355\u9000\u6b3e\u591a\u4e45\u5230\u8d26\uff1f"

            with patch.dict(os.environ, {}, clear=True):
                first = rag.run_query(
                    query,
                    quiet=True,
                    rebuild_index=True,
                    vector_backend="local",
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

            self.assertGreater(len(first["index_sync"]["changed_docs"]), 0)

            chunk_builds = 0
            original_build_document_chunks = rag.build_document_chunks

            def counting_build_document_chunks(metadata: dict[str, str], body: str) -> list[rag.Chunk]:
                nonlocal chunk_builds
                chunk_builds += 1
                return original_build_document_chunks(metadata, body)

            with (
                patch.dict(os.environ, {"EMBEDDING_API_KEY": "bad-key"}, clear=True),
                patch.object(rag.time, "sleep", lambda _: None),
                patch.object(rag, "request_json", side_effect=RuntimeError("embedding down")),
                patch.object(rag, "build_document_chunks", counting_build_document_chunks),
            ):
                second = rag.run_query(
                    query,
                    quiet=True,
                    rebuild_index=False,
                    vector_backend="local",
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

            self.assertEqual(second["index_sync"]["changed_docs"], [])
            self.assertEqual(second["component_status"]["embedding"]["mode"], "hash_fallback")
            self.assertEqual(chunk_builds, 0)

    def test_run_query_without_rebuild_reuses_existing_index_without_syncing_raw_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store_path = tmp_path / "indexes" / "rag.sqlite"
            metrics_path = tmp_path / "metrics" / "online_metrics.jsonl"
            query = "\u8de8\u5883\u8ba2\u5355\u9000\u6b3e\u591a\u4e45\u5230\u8d26\uff1f"

            with patch.dict(os.environ, {}, clear=True):
                rag.run_query(
                    query,
                    quiet=True,
                    rebuild_index=True,
                    vector_backend="local",
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(rag, "sync_index", side_effect=AssertionError("index sync should be manual")),
            ):
                trace = rag.run_query(
                    query,
                    quiet=True,
                    rebuild_index=False,
                    vector_backend="local",
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

            self.assertEqual(trace["index_sync"]["changed_docs"], [])
            self.assertEqual(trace["index_sync"]["reason"], "rebuild_not_requested")

    def test_sqlite_bm25_search_filters_permission_in_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "rag.sqlite"
            store = rag.LocalVectorStore(store_path)
            visible = make_chunk("visible", "internal")
            visible.terms = rag.tokenize("refund timeline")
            hidden = make_chunk("hidden", "finance_restricted")
            hidden.terms = rag.tokenize("refund timeline")
            store.upsert_document("visible_doc", "visible.md", "hash-1", "local:test", [visible])
            store.upsert_document("hidden_doc", "hidden.md", "hash-2", "local:test", [hidden])

            results = store.bm25_search("refund timeline", {"internal"}, top_n=5, today="2026-06-07")

            self.assertEqual([chunk.chunk_id for _, chunk in results], ["visible"])

    def test_external_embedding_success_records_external_identity_before_index_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store_path = tmp_path / "indexes" / "rag.sqlite"
            metrics_path = tmp_path / "metrics" / "online_metrics.jsonl"

            def fake_request_json(
                method: str,
                url: str,
                body: dict[str, object] | None = None,
                headers: dict[str, str] | None = None,
                ok_statuses: tuple[int, ...] = (200,),
            ) -> dict:
                inputs = body["input"] if body else []
                assert isinstance(inputs, list)
                return {
                    "data": [
                        {"embedding": [1.0, 0.0], "index": index, "object": "embedding"}
                        for index, _ in enumerate(inputs)
                    ]
                }

            with (
                patch.dict(os.environ, {"EMBEDDING_API_KEY": "ok-key", "EMBEDDING_MODEL": "embedding-3"}, clear=True),
                patch.object(rag, "request_json", fake_request_json),
            ):
                trace = rag.run_query(
                    "\u8de8\u5883\u8ba2\u5355\u9000\u6b3e\u591a\u4e45\u5230\u8d26\uff1f",
                    quiet=True,
                    rebuild_index=True,
                    vector_backend="local",
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

        self.assertEqual(trace["component_status"]["embedding"]["provider"], rag.EMBEDDING_PROVIDER_EXTERNAL)
        self.assertEqual(trace["component_status"]["embedding"]["model"], "embedding-3")
        self.assertEqual(trace["component_status"]["embedding"]["identity"], "external:embedding-3")
        self.assertEqual(trace["model_config"]["embedding_model"], "embedding-3")
        self.assertEqual(trace["model_config"]["embedding_identity"], "external:embedding-3")
        self.assertEqual(trace["index_sync"]["embedding_identity"], "external:embedding-3")

    def test_local_embedding_success_records_local_identity_before_index_sync(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store_path = tmp_path / "indexes" / "rag.sqlite"
            metrics_path = tmp_path / "metrics" / "online_metrics.jsonl"

            def fake_request_json(
                method: str,
                url: str,
                body: dict[str, object] | None = None,
                headers: dict[str, str] | None = None,
                ok_statuses: tuple[int, ...] = (200,),
            ) -> dict:
                inputs = body["input"] if body else []
                assert isinstance(inputs, list)
                return {
                    "data": [
                        {"embedding": [1.0, 0.0], "index": index, "object": "embedding"}
                        for index, _ in enumerate(inputs)
                    ]
                }

            with (
                patch.dict(
                    os.environ,
                    {
                        "EMBEDDING_BASE_URL": "http://127.0.0.1:11434/v1",
                        "EMBEDDING_MODEL": "nomic-embed-text",
                    },
                    clear=True,
                ),
                patch.object(rag, "request_json", fake_request_json),
            ):
                trace = rag.run_query(
                    "\u8de8\u5883\u8ba2\u5355\u9000\u6b3e\u591a\u4e45\u5230\u8d26\uff1f",
                    quiet=True,
                    rebuild_index=True,
                    vector_backend="local",
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

        self.assertEqual(trace["component_status"]["embedding"]["provider"], rag.EMBEDDING_PROVIDER_LOCAL)
        self.assertEqual(trace["component_status"]["embedding"]["model"], "nomic-embed-text")
        self.assertEqual(trace["component_status"]["embedding"]["identity"], "local:nomic-embed-text")
        self.assertEqual(trace["model_config"]["embedding_model"], "nomic-embed-text")
        self.assertEqual(trace["model_config"]["embedding_identity"], "local:nomic-embed-text")
        self.assertEqual(trace["index_sync"]["embedding_identity"], "local:nomic-embed-text")

    def test_local_vector_store_keeps_indexes_for_different_embedding_identities(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store_path = tmp_path / "indexes" / "rag.sqlite"
            metrics_path = tmp_path / "metrics" / "online_metrics.jsonl"

            def fake_request_json(
                method: str,
                url: str,
                body: dict[str, object] | None = None,
                headers: dict[str, str] | None = None,
                ok_statuses: tuple[int, ...] = (200,),
            ) -> dict:
                inputs = body["input"] if body else []
                assert isinstance(inputs, list)
                return {
                    "data": [
                        {"embedding": rag.vectorize(rag.tokenize(text)), "index": index, "object": "embedding"}
                        for index, text in enumerate(inputs)
                    ]
                }

            with (
                patch.dict(os.environ, {"EMBEDDING_API_KEY": "ok-key", "EMBEDDING_MODEL": "embedding-3"}, clear=True),
                patch.object(rag, "request_json", fake_request_json),
            ):
                first_external = rag.run_query(
                    "\u8de8\u5883\u8ba2\u5355\u9000\u6b3e\u591a\u4e45\u5230\u8d26\uff1f",
                    quiet=True,
                    rebuild_index=True,
                    vector_backend="local",
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

            with patch.dict(os.environ, {}, clear=True):
                hash_fallback = rag.run_query(
                    "\u8de8\u5883\u8ba2\u5355\u9000\u6b3e\u591a\u4e45\u5230\u8d26\uff1f",
                    quiet=True,
                    rebuild_index=True,
                    vector_backend="local",
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

            with (
                patch.dict(os.environ, {"EMBEDDING_API_KEY": "ok-key", "EMBEDDING_MODEL": "embedding-3"}, clear=True),
                patch.object(rag, "request_json", fake_request_json),
            ):
                second_external = rag.run_query(
                    "\u8de8\u5883\u8ba2\u5355\u9000\u6b3e\u591a\u4e45\u5230\u8d26\uff1f",
                    quiet=True,
                    vector_backend="local",
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

        self.assertEqual(first_external["index_sync"]["embedding_identity"], "external:embedding-3")
        self.assertEqual(hash_fallback["index_sync"]["embedding_identity"], "local:local-hash-embedding")
        self.assertGreater(len(first_external["index_sync"]["changed_docs"]), 0)
        self.assertGreater(len(hash_fallback["index_sync"]["changed_docs"]), 0)
        self.assertEqual(second_external["index_sync"]["embedding_identity"], "external:embedding-3")
        self.assertEqual(second_external["index_sync"]["changed_docs"], [])

    def test_configured_llm_failure_retries_then_uses_extractive_answer(self) -> None:
        context_packet = {
            "query": "refund timeline",
            "sufficiency": {"enough": True, "reason": "pass"},
            "evidence": [
                {
                    "citation_id": "E1",
                    "text": "Refunds usually arrive within three to seven business days.",
                }
            ],
        }

        with (
            patch.dict(os.environ, {"LLM_API_KEY": "bad-key"}, clear=True),
            patch.object(rag.time, "sleep", lambda _: None),
            patch.object(rag, "request_json", side_effect=RuntimeError("llm down")),
        ):
            status = rag.build_llm_status()
            answer = rag.generate_answer_resilient(context_packet, status)

        self.assertEqual(status["mode"], "extractive_fallback")
        self.assertEqual(status["reason"], "llm_error")
        self.assertEqual(status["attempts"], 3)
        self.assertEqual(answer["mode"], "extractive")


if __name__ == "__main__":
    unittest.main()
