from __future__ import annotations

import json
import importlib.util
import os
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch


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


class ProductionDefaultsTest(unittest.TestCase):
    def test_defaults_target_deepseek_v4_pro_and_zhipu_embedding_3(self) -> None:
        self.assertEqual(rag.DEFAULT_CHAT_MODEL, "deepseek-v4-pro")
        self.assertEqual(rag.DEFAULT_EMBEDDING_MODEL, "embedding-3")
        self.assertEqual(rag.DEFAULT_LLM_API_STYLE, "anthropic")
        self.assertEqual(rag.DEFAULT_DEEPSEEK_BASE_URL, "https://api.deepseek.com/anthropic")
        self.assertEqual(rag.DEFAULT_ZHIPU_EMBEDDING_BASE_URL, "https://open.bigmodel.cn/api/paas/v4")
        self.assertEqual(rag.DEFAULT_VECTOR_BACKEND, "qdrant")
        self.assertEqual(rag.DEFAULT_QDRANT_URL, "http://localhost:6333")
        self.assertEqual(rag.DEFAULT_QDRANT_COLLECTION, "production_rag_chunks")


class VectorDimensionConsistencyTest(unittest.TestCase):
    def test_hash_vectorize_follows_global_embedding_dimensions(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EMBEDDING_DIMENSIONS", None)
            self.assertEqual(rag.DEFAULT_EMBEDDING_DIMENSIONS, 1024)
            self.assertEqual(len(rag.vectorize(["alpha", "beta"])), 1024)
        with patch.dict(os.environ, {"EMBEDDING_DIMENSIONS": "256"}, clear=False):
            self.assertEqual(rag.resolve_vector_dimensions(), 256)
            self.assertEqual(len(rag.vectorize(["alpha", "beta"])), 256)

    def test_content_hash_changes_when_dimensions_change(self) -> None:
        self.assertNotEqual(
            rag.content_hash("same text", "embedding-3", 1024),
            rag.content_hash("same text", "embedding-3", 256),
        )

    def test_embedding_identity_distinguishes_provider_and_model_name(self) -> None:
        self.assertEqual(
            rag.embedding_identity("local", "embedding-3"),
            "local:embedding-3",
        )
        self.assertEqual(
            rag.embedding_identity("external", "embedding-3"),
            "external:embedding-3",
        )
        self.assertNotEqual(
            rag.content_hash("same text", rag.embedding_identity("local", "embedding-3"), 1024),
            rag.content_hash("same text", rag.embedding_identity("external", "embedding-3"), 1024),
        )
        self.assertNotEqual(
            rag.content_hash("same text", rag.embedding_identity("external", "embedding-3"), 1024),
            rag.content_hash("same text", rag.embedding_identity("external", "embedding-4"), 1024),
        )

    def test_embedding_provider_detects_local_and_external_base_urls(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                rag.resolve_embedding_provider("http://127.0.0.1:11434/v1"),
                rag.EMBEDDING_PROVIDER_LOCAL,
            )
            self.assertEqual(
                rag.resolve_embedding_provider("http://localhost:8000/v1"),
                rag.EMBEDDING_PROVIDER_LOCAL,
            )
            self.assertEqual(
                rag.resolve_embedding_provider("https://open.bigmodel.cn/api/paas/v4"),
                rag.EMBEDDING_PROVIDER_EXTERNAL,
            )

        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "local"}, clear=True):
            self.assertEqual(
                rag.resolve_embedding_provider("https://open.bigmodel.cn/api/paas/v4"),
                rag.EMBEDDING_PROVIDER_LOCAL,
            )

    def test_dense_recall_scores_with_provided_query_vector(self) -> None:
        matching = make_chunk("matching", "internal")
        matching.dense_vector = [1.0, 0.0]
        other = make_chunk("other", "internal")
        other.dense_vector = [0.0, 1.0]

        results = rag.dense_recall([1.0, 0.0], [matching, other], top_n=2)

        self.assertEqual(results[0][1].chunk_id, "matching")
        self.assertAlmostEqual(results[0][0], 1.0)
        self.assertAlmostEqual(results[1][0], 0.0)


class PermissionFilterTest(unittest.TestCase):
    def test_filter_chunks_for_access_removes_out_of_scope_and_expired_docs(self) -> None:
        chunks = [
            make_chunk("internal", "internal"),
            make_chunk("support", "support_only"),
            make_chunk("expired", "internal", effective_to="2026-01-31"),
        ]

        visible, rejected = rag.filter_chunks_for_access(
            chunks,
            allowed_scopes={"internal"},
            today="2026-06-07",
        )

        self.assertEqual([chunk.chunk_id for chunk in visible], ["internal"])
        self.assertEqual(
            {item["chunk_id"]: item["reason"] for item in rejected},
            {"support": "permission_scope", "expired": "expired"},
        )

    def test_default_scopes_include_public_but_exclude_restricted_docs(self) -> None:
        chunks = [
            make_chunk("internal", "internal"),
            make_chunk("public", "public"),
            make_chunk("finance", "finance_restricted"),
            make_chunk("security", "security_restricted"),
        ]

        visible, rejected = rag.filter_chunks_for_access(
            chunks,
            allowed_scopes=rag.DEFAULT_ALLOWED_SCOPES,
            today="2026-06-07",
        )

        self.assertEqual([chunk.chunk_id for chunk in visible], ["internal", "public"])
        self.assertEqual(
            {item["chunk_id"]: item["reason"] for item in rejected},
            {"finance": "permission_scope", "security": "permission_scope"},
        )
        self.assertEqual(rag.DEFAULT_ALLOWED_SCOPES, {"internal", "public"})

    def test_sufficiency_refuses_when_best_match_is_permission_blocked(self) -> None:
        visible = make_chunk("visible", "internal")
        visible.terms = rag.tokenize("普通处理说明")
        blocked = make_chunk("blocked", "finance_restricted")
        blocked.doc_id = "finance_refund_reconciliation_v1"
        blocked.terms = rag.tokenize("FR-21 差异工单 处理")
        rejected = [{"chunk_id": blocked.chunk_id, "doc_id": blocked.doc_id, "reason": "permission_scope"}]

        blocked_matches = rag.find_permission_blocked_matches(
            "FR-21 差异工单怎么处理？",
            [visible, blocked],
            rejected,
        )
        result = rag.sufficiency_check(
            "FR-21 差异工单怎么处理？",
            [rag.Candidate(chunk_id=visible.chunk_id, rerank_score=0.7, reason="visible")],
            {visible.chunk_id: visible},
            permission_blocked_matches=blocked_matches,
        )
        answer = rag.generate_answer(
            {
                "query": "FR-21 差异工单怎么处理？",
                "sufficiency": result,
                "evidence": [],
            }
        )

        self.assertEqual(result["reason"], "permission_denied")
        self.assertEqual(answer["answer"], "当前权限不足，不能可靠回答这个问题。")


class IncrementalSyncTest(unittest.TestCase):
    def test_sync_index_reuses_unchanged_embeddings_and_updates_changed_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            store_path = root / "indexes" / "rag.sqlite"
            (raw_dir / "a.md").write_text(
                "---\ndoc_id: doc_a\ntitle: A\npermission_scope: internal\n---\n\n# A\n\n## Rule\n\nalpha refund text",
                encoding="utf-8",
            )
            (raw_dir / "b.md").write_text(
                "---\ndoc_id: doc_b\ntitle: B\npermission_scope: internal\n---\n\n# B\n\n## Rule\n\nbeta shipping text",
                encoding="utf-8",
            )
            calls: list[list[str]] = []

            def fake_embedder(texts: list[str]) -> list[list[float]]:
                calls.append(texts)
                return [rag.vectorize(rag.tokenize(text)) for text in texts]

            first = rag.sync_index(raw_dir, store_path, embedder=fake_embedder, rebuild=True)
            calls_after_first = len(calls)
            second = rag.sync_index(raw_dir, store_path, embedder=fake_embedder, rebuild=False)
            calls_after_second = len(calls)
            (raw_dir / "b.md").write_text(
                "---\ndoc_id: doc_b\ntitle: B\npermission_scope: internal\n---\n\n# B\n\n## Rule\n\nbeta shipping text changed",
                encoding="utf-8",
            )
            third = rag.sync_index(raw_dir, store_path, embedder=fake_embedder, rebuild=False)

            self.assertEqual(first["changed_docs"], ["doc_a", "doc_b"])
            self.assertEqual(second["changed_docs"], [])
            self.assertEqual(third["changed_docs"], ["doc_b"])
            self.assertEqual(calls_after_first, 2)
            self.assertEqual(calls_after_second, calls_after_first)
            self.assertEqual(len(calls), calls_after_first + 1)
            stored_chunks = rag.LocalVectorStore(store_path).load_chunks()
            self.assertEqual({chunk.doc_id for chunk in stored_chunks}, {"doc_a", "doc_b"})
            self.assertTrue(all(chunk.dense_vector for chunk in stored_chunks))


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


class RetrievalPipelineEnhancementTest(unittest.TestCase):
    def test_dynamic_truncate_budgets_expanded_parent_context(self) -> None:
        chunks: list[rag.Chunk] = []
        candidates: list[rag.Candidate] = []
        for index, score in enumerate((0.9, 0.85, 0.8), start=1):
            anchor = make_chunk(f"parent{index}-anchor", "internal")
            sibling = make_chunk(f"parent{index}-sibling", "internal")
            parent_id = f"parent{index}"
            anchor.parent_id = parent_id
            sibling.parent_id = parent_id
            anchor.text = f"parent {index} anchor"
            sibling.text = f"parent {index} sibling expansion"
            anchor.token_count = 100
            sibling.token_count = 350
            chunks.extend([anchor, sibling])
            candidates.append(rag.Candidate(chunk_id=anchor.chunk_id, rerank_score=score))

        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        chunks_by_parent = rag.build_chunks_by_parent(chunks)

        with patch.object(rag, "CONTEXT_TOKEN_BUDGET", 900):
            selected, truncation = rag.dynamic_truncate(
                candidates,
                chunks_by_id,
                chunks_by_parent=chunks_by_parent,
            )
            context_packet = rag.assemble_context(
                "budgeted parent expansion",
                selected,
                chunks_by_id,
                {"enough": True, "reason": "pass"},
                chunks_by_parent=chunks_by_parent,
            )

        selected_chunk_ids = [item.candidate.chunk_id for item in selected]
        self.assertEqual(selected_chunk_ids, ["parent1-anchor", "parent2-anchor"])
        self.assertEqual(truncation["budget_basis"], "expanded_parent_context_tokens")
        self.assertEqual(truncation["token_total"], 900)
        self.assertEqual(context_packet["estimated_token_total"], truncation["token_total"])
        self.assertLessEqual(context_packet["estimated_token_total"], 900)
        self.assertEqual(selected[0].expanded_from_chunk_ids, ["parent1-anchor", "parent1-sibling"])

    def test_external_reranker_can_replace_rule_score_ordering(self) -> None:
        first = make_chunk("first", "internal")
        first.text = "generic refund text"
        second = make_chunk("second", "internal")
        second.text = "exact SKU-A17 no reason return policy"
        candidates = {
            "first": rag.Candidate(chunk_id="first", rrf_score=0.2, dense_rank=1),
            "second": rag.Candidate(chunk_id="second", rrf_score=0.1, bm25_rank=1),
        }
        chunks_by_id = {"first": first, "second": second}
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"results":[{"index":0,"score":0.2},{"index":1,"score":0.95}]}'

        def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
            captured["url"] = request.full_url
            captured["body"] = request.data
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("urllib.request.urlopen", fake_urlopen):
            external = rag.make_external_reranker("http://reranker.test/rerank", model="bge-reranker-v2-m3")
            ranked = rag.rerank("SKU-A17 是否支持无理由退货？", candidates, chunks_by_id, external_reranker=external)

        request_body = json.loads(captured["body"].decode("utf-8"))
        self.assertEqual(captured["url"], "http://reranker.test/rerank")
        self.assertEqual(captured["timeout"], 30)
        self.assertEqual(request_body["model"], "bge-reranker-v2-m3")
        self.assertEqual(request_body["documents"][1]["id"], "second")
        self.assertEqual([item.chunk_id for item in ranked], ["second", "first"])
        self.assertEqual(ranked[0].reason, "external_reranker:bge-reranker-v2-m3")

    def test_rerank_skips_without_configured_model_and_keeps_rrf_order(self) -> None:
        first = make_chunk("first", "internal")
        first.text = "generic refund text"
        second = make_chunk("second", "internal")
        second.text = "exact SKU-A17 no reason return policy"
        candidates = {
            "first": rag.Candidate(chunk_id="first", rrf_score=0.2, dense_rank=1),
            "second": rag.Candidate(chunk_id="second", rrf_score=0.1, bm25_rank=1),
        }
        chunks_by_id = {"first": first, "second": second}

        ranked = rag.rerank("SKU-A17 是否支持无理由退货？", candidates, chunks_by_id)

        self.assertEqual([item.chunk_id for item in ranked], ["first", "second"])
        self.assertEqual(ranked[0].rerank_score, ranked[0].rrf_score)
        self.assertEqual(ranked[0].reason, "rerank_skipped:no_configured_model")

    def test_rrf_only_truncate_uses_rank_budget_without_absolute_score_cutoffs(self) -> None:
        first = make_chunk("first", "internal")
        second = make_chunk("second", "internal")
        candidates = [
            rag.Candidate(chunk_id="first", rerank_score=0.032, rrf_score=0.032, bm25_rank=1),
            rag.Candidate(chunk_id="second", rerank_score=0.016, rrf_score=0.016, bm25_rank=2),
        ]
        chunks_by_id = {"first": first, "second": second}

        selected, truncation = rag.dynamic_truncate(candidates, chunks_by_id, score_policy="rrf_only")

        self.assertEqual([item.candidate.chunk_id for item in selected], ["first", "second"])
        self.assertEqual(truncation["score_policy"], "rrf_only")
        self.assertIsNone(truncation["min_score"])
        self.assertIsNone(truncation["gap_threshold"])
        self.assertEqual(truncation["stop_reason"], "max_k_or_budget")

    def test_rrf_only_sufficiency_requires_lexical_retrieval_signal(self) -> None:
        chunk = make_chunk("dense-only", "internal")
        chunk.terms = rag.tokenize("积分 提现")
        candidate = rag.Candidate(
            chunk_id=chunk.chunk_id,
            dense_rank=1,
            bm25_rank=None,
            rerank_score=0.032,
        )

        result = rag.sufficiency_check(
            "积分可以提现吗？",
            [candidate],
            {chunk.chunk_id: chunk},
            score_policy="rrf_only",
        )

        self.assertFalse(result["enough"])
        self.assertEqual(result["reason"], "rrf_only_missing_lexical_signal")

    def test_rrf_only_sufficiency_passes_with_lexical_coverage(self) -> None:
        chunk = make_chunk("bm25-hit", "internal")
        chunk.terms = rag.tokenize("积分 提现")
        candidate = rag.Candidate(
            chunk_id=chunk.chunk_id,
            dense_rank=2,
            bm25_rank=1,
            rerank_score=0.016,
        )

        result = rag.sufficiency_check(
            "积分可以提现吗？",
            [candidate],
            {chunk.chunk_id: chunk},
            score_policy="rrf_only",
        )

        self.assertTrue(result["enough"])
        self.assertEqual(result["reason"], "pass")
        self.assertEqual(result["lexical_coverage_ratio"], result["coverage_ratio"])

    def test_external_reranker_error_skips_without_rule_fallback(self) -> None:
        first = make_chunk("first", "internal")
        first.text = "generic refund text"
        second = make_chunk("second", "internal")
        second.text = "exact SKU-A17 no reason return policy"
        candidates = {
            "first": rag.Candidate(chunk_id="first", rrf_score=0.2, dense_rank=1),
            "second": rag.Candidate(chunk_id="second", rrf_score=0.1, bm25_rank=1),
        }
        chunks_by_id = {"first": first, "second": second}
        external = rag.make_external_reranker("http://reranker.test/rerank", model="bge-reranker-v2-m3")

        with patch.object(external, "score", side_effect=RuntimeError("service unavailable")):
            ranked = rag.rerank("SKU-A17 是否支持无理由退货？", candidates, chunks_by_id, external_reranker=external)

        self.assertEqual([item.chunk_id for item in ranked], ["first", "second"])
        self.assertEqual(external.last_error, "service unavailable")
        self.assertEqual(ranked[0].reason, "rerank_skipped:reranker_error")

    def test_configured_reranker_uses_flagembedding_service_defaults(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RERANKER_PROVIDER": "flagembedding",
                "RERANKER_URL": "",
                "RERANKER_MODEL": "",
            },
            clear=False,
        ):
            external = rag.make_configured_external_reranker()

        self.assertIsNotNone(external)
        assert external is not None
        self.assertEqual(external.url, "http://127.0.0.1:8008/rerank")
        self.assertEqual(external.model, "bge-reranker-v2-m3")

    def test_bge_reranker_service_keeps_scores_in_request_order(self) -> None:
        service_path = PROJECT_ROOT / "scripts" / "serve_bge_reranker.py"
        spec = importlib.util.spec_from_file_location("serve_bge_reranker", service_path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(service)

        class FakeBackend:
            def __init__(self) -> None:
                self.query = ""
                self.documents: list[str] = []

            def score(self, query: str, documents: list[str]) -> list[float]:
                self.query = query
                self.documents = documents
                return [0.2, 0.95]

        backend = FakeBackend()
        response = service.score_payload(
            {
                "model": "bge-reranker-v2-m3",
                "query": "SKU-A17 是否支持无理由退货？",
                "documents": [
                    {"id": "first", "text": "generic refund text"},
                    {"id": "second", "text": "exact SKU-A17 no reason return policy"},
                ],
            },
            backend=backend,
        )

        self.assertEqual(backend.query, "SKU-A17 是否支持无理由退货？")
        self.assertEqual(backend.documents[1], "exact SKU-A17 no reason return policy")
        self.assertEqual(response["model"], "BAAI/bge-reranker-v2-m3")
        self.assertEqual(response["scores"], [0.2, 0.95])
        self.assertEqual(response["results"][0]["id"], "second")
        self.assertEqual(response["results"][0]["index"], 1)

    def test_bge_reranker_service_can_use_local_model_directory(self) -> None:
        service_path = PROJECT_ROOT / "scripts" / "serve_bge_reranker.py"
        spec = importlib.util.spec_from_file_location("serve_bge_reranker", service_path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(service)

        self.assertEqual(
            service.resolve_model_reference("bge-reranker-v2-m3", "D:/models/bge-reranker-v2-m3"),
            "D:/models/bge-reranker-v2-m3",
        )
        self.assertEqual(
            service.resolve_model_reference("bge-reranker-v2-m3", ""),
            "BAAI/bge-reranker-v2-m3",
        )

    def test_bge_reranker_model_load_error_explains_hf_endpoint_and_local_dir(self) -> None:
        service_path = PROJECT_ROOT / "scripts" / "serve_bge_reranker.py"
        spec = importlib.util.spec_from_file_location("serve_bge_reranker", service_path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(service)

        message = service.format_model_load_error(
            "BAAI/bge-reranker-v2-m3",
            OSError("We couldn't connect to 'https://hf-mirror.com' to load the files."),
        )

        self.assertIn("BAAI/bge-reranker-v2-m3", message)
        self.assertIn("hf-mirror.com", message)
        self.assertIn("HF_ENDPOINT", message)
        self.assertIn("RERANKER_MODEL_DIR", message)
        self.assertIn("huggingface-cli download BAAI/bge-reranker-v2-m3", message)

    def test_bge_reranker_service_loads_dotenv_without_overriding_shell_env(self) -> None:
        service_path = PROJECT_ROOT / "scripts" / "serve_bge_reranker.py"
        spec = importlib.util.spec_from_file_location("serve_bge_reranker", service_path)
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        service = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(service)

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "RERANKER_MODEL_DIR=D:/models/from-dotenv\n"
                "RERANKER_BACKEND=flagembedding\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"RERANKER_MODEL_DIR": "D:/models/from-shell"}, clear=False):
                loaded = service.load_dotenv(env_path)
                self.assertEqual(os.environ["RERANKER_MODEL_DIR"], "D:/models/from-shell")
                self.assertEqual(os.environ["RERANKER_BACKEND"], "flagembedding")

        self.assertEqual(loaded, {"RERANKER_BACKEND": "flagembedding"})

    def test_mmr_select_keeps_diverse_candidate_over_near_duplicate(self) -> None:
        primary = make_chunk("primary", "internal")
        primary.dense_vector = [1.0, 0.0]
        duplicate = make_chunk("duplicate", "internal")
        duplicate.dense_vector = [0.99, 0.01]
        diverse = make_chunk("diverse", "internal")
        diverse.dense_vector = [0.0, 1.0]
        ranked = [
            rag.Candidate(chunk_id="primary", rerank_score=0.9),
            rag.Candidate(chunk_id="duplicate", rerank_score=0.85),
            rag.Candidate(chunk_id="diverse", rerank_score=0.7),
        ]

        selected, dropped = rag.mmr_select(
            ranked,
            {"primary": primary, "duplicate": duplicate, "diverse": diverse},
            limit=2,
            lambda_mult=0.65,
        )

        self.assertEqual([item.chunk_id for item in selected], ["primary", "diverse"])
        self.assertEqual(dropped[0]["chunk_id"], "duplicate")
        self.assertEqual(dropped[0]["reason"], "near_duplicate")

    def test_assemble_context_expands_selected_chunk_to_parent_siblings(self) -> None:
        first = make_chunk("doc:sec_00:chunk_00", "internal")
        first.parent_id = "doc:sec_00"
        first.doc_id = "doc"
        first.text = "第一段说明用户需要先核对订单。"
        second = make_chunk("doc:sec_00:chunk_01", "internal")
        second.parent_id = "doc:sec_00"
        second.doc_id = "doc"
        second.text = "第二段说明超过 15 个工作日要提交渠道核查。"
        candidate = rag.Candidate(chunk_id=second.chunk_id, rerank_score=0.8)

        packet = rag.assemble_context(
            "跨境订单退款超时怎么办？",
            [candidate],
            {first.chunk_id: first, second.chunk_id: second},
            {"enough": True, "reason": "pass"},
            chunks_by_parent={"doc:sec_00": [first, second]},
        )

        evidence = packet["evidence"][0]
        self.assertIn("第一段说明", evidence["text"])
        self.assertIn("第二段说明", evidence["text"])
        self.assertEqual(evidence["expanded_from_chunk_ids"], [first.chunk_id, second.chunk_id])
        self.assertGreater(packet["estimated_token_total"], second.token_count)


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
        self.assertEqual(body["with_vector"], False)
        self.assertEqual(results[0][0], 0.87)
        self.assertEqual(results[0][1].chunk_id, "chunk-1")
        self.assertEqual(results[0][1].dense_vector, [])

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
        self.assertEqual(body["with_vector"], False)
        self.assertEqual(body["filter"], rag.qdrant_access_filter({"internal"}, "2026-06-07"))
        self.assertEqual(results[0][0], 4.2)
        self.assertEqual(results[0][1].chunk_id, "chunk-1")


class ProviderClientTest(unittest.TestCase):
    def test_anthropic_messages_client_posts_messages_request(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"content":[{"type":"text","text":"answer [E1]"}]}'

        def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
            captured["url"] = request.full_url
            captured["headers"] = request.headers
            captured["body"] = request.data
            captured["timeout"] = timeout
            return FakeResponse()

        client = rag.AnthropicMessagesClient(
            api_key="test-key",
            base_url="https://api.deepseek.com/anthropic",
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            text = client.create_message(
                model="deepseek-v4-pro",
                system_prompt="system",
                prompt="prompt",
                max_tokens=800,
            )

        self.assertEqual(text, "answer [E1]")
        self.assertEqual(captured["url"], "https://api.deepseek.com/anthropic/v1/messages")
        self.assertEqual(captured["timeout"], 90)
        self.assertEqual(captured["headers"]["X-api-key"], "test-key")
        self.assertEqual(captured["headers"]["Anthropic-version"], "2023-06-01")
        self.assertEqual(
            json.loads(captured["body"].decode("utf-8"))["model"],
            "deepseek-v4-pro",
        )

    def test_openai_compatible_embedding_client_posts_embedding_request(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b'{"data":[{"embedding":[0.1,0.2],"index":0,"object":"embedding"}]}'

        def fake_urlopen(request: urllib.request.Request, timeout: int) -> FakeResponse:
            captured["url"] = request.full_url
            captured["headers"] = request.headers
            captured["body"] = request.data
            captured["timeout"] = timeout
            return FakeResponse()

        client = rag.OpenAICompatibleEmbeddingClient(
            api_key="zhipu-key",
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        with patch("urllib.request.urlopen", fake_urlopen):
            vectors = client.embed_texts(["hello"], model="embedding-3", dimensions=1024)

        self.assertEqual(vectors, [[0.1, 0.2]])
        self.assertEqual(captured["url"], "https://open.bigmodel.cn/api/paas/v4/embeddings")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer zhipu-key")
        self.assertEqual(captured["timeout"], 90)
        self.assertEqual(
            json.loads(captured["body"].decode("utf-8")),
            {"model": "embedding-3", "input": ["hello"], "dimensions": 1024},
        )


class CitationAndMonitoringTest(unittest.TestCase):
    def test_citation_validation_extracts_llm_bracket_citations(self) -> None:
        context_packet = {"evidence": [{"citation_id": "E1"}, {"citation_id": "E2"}]}
        answer = {"answer": "关键事实来自资料 [E1]，另一个引用不存在 [E404]。", "mode": "deepseek"}

        result = rag.validate_citations(answer, context_packet)

        self.assertFalse(result["citation_valid"])
        self.assertEqual(result["used_citations"], ["E1", "E404"])
        self.assertEqual(result["missing_citations"], ["E404"])

    def test_monitoring_event_captures_online_rag_health_fields(self) -> None:
        trace = {
            "trace_id": "trace-1",
            "query": "跨境订单退款多久到账？",
            "model_config": {
                "vector_backend": "local",
                "embedding_model": "local-hash-embedding",
                "embedding_identity": "local:local-hash-embedding",
            },
            "answer": {"mode": "extractive"},
            "validation": {"citation_valid": True, "missing_citations": []},
            "context_packet": {
                "evidence": [{"citation_id": "E1", "doc_id": "refund_policy_v3"}],
                "sufficiency": {"enough": True, "reason": "pass"},
            },
            "permission_filter": {"blocked_matches": []},
            "dense_top": [1, 2],
            "bm25_top": [1],
            "rerank_top": [1, 2, 3],
            "dedup_dropped": [{"chunk_id": "dup"}],
            "truncation": {"selected_count": 1, "token_total": 120},
            "stage_latencies_ms": {
                "index_sync": 7,
                "dense_recall": 11,
                "bm25_recall": 3,
                "rerank": 13,
                "mmr": 2,
                "answer": 17,
            },
            "selection_strategy": {"name": "mmr", "lambda": 0.72, "parent_expansion": True},
            "reranker": {
                "mode": "skipped",
                "reason": "not_configured",
                "fallback_used": False,
                "score_policy": "rrf_only",
            },
        }

        event = rag.build_monitoring_event(trace, latency_ms=123, status="ok")

        self.assertEqual(event["trace_id"], "trace-1")
        self.assertEqual(event["latency_ms"], 123)
        self.assertEqual(event["vector_backend"], "local")
        self.assertEqual(event["embedding_model"], "local-hash-embedding")
        self.assertEqual(event["embedding_identity"], "local:local-hash-embedding")
        self.assertEqual(event["query_hash"], rag.hashlib.sha256("跨境订单退款多久到账？".encode("utf-8")).hexdigest()[:16])
        self.assertEqual(event["answer_mode"], "extractive")
        self.assertEqual(event["citation_valid"], True)
        self.assertEqual(event["sufficiency_reason"], "pass")
        self.assertEqual(event["selected_doc_ids"], ["refund_policy_v3"])
        self.assertEqual(event["dense_hits"], 2)
        self.assertEqual(event["bm25_hits"], 1)
        self.assertEqual(event["dedup_dropped_count"], 1)
        self.assertEqual(event["stage_latencies_ms"]["rerank"], 13)
        self.assertEqual(event["selection_strategy"], "mmr")
        self.assertEqual(event["reranker_mode"], "skipped")
        self.assertEqual(event["reranker_score_policy"], "rrf_only")
        self.assertEqual(event["reranker_fallback_used"], False)

    def test_eval_cases_cover_practice_retrieval_and_permission_scenarios(self) -> None:
        rows = list(rag.csv.DictReader(rag.EVAL_PATH.read_text(encoding="utf-8").splitlines()))
        case_ids = {row["case_id"] for row in rows}

        self.assertGreaterEqual(len(rows), 10)
        self.assertIn("case_007", case_ids)
        self.assertIn("case_008", case_ids)
        self.assertIn("case_009", case_ids)
        self.assertIn("case_010", case_ids)

    def test_run_query_appends_monitoring_event_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            metrics_path = tmp_path / "metrics" / "online_metrics.jsonl"
            store_path = tmp_path / "indexes" / "rag.sqlite"

            trace = rag.run_query(
                "会员积分可以提现吗？",
                quiet=True,
                rebuild_index=True,
                vector_backend="local",
                store_path=store_path,
                metrics_path=metrics_path,
            )

            rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["trace_id"], trace["trace_id"])
            self.assertEqual(rows[0]["answer_mode"], "extractive")
            self.assertIn("monitoring_metrics_path", trace)
            self.assertNotIn("monitoring_event", trace)
            serialized_trace = json.dumps(trace, ensure_ascii=False)
            self.assertEqual(serialized_trace.count('"stage_latencies_ms"'), 1)
            self.assertEqual(serialized_trace.count('"selection_strategy"'), 1)

    def test_run_query_records_skipped_rerank_when_model_is_not_configured(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            store_path = tmp_path / "indexes" / "rag.sqlite"
            metrics_path = tmp_path / "metrics" / "online_metrics.jsonl"
            with patch.dict(os.environ, {"RERANKER_PROVIDER": "", "RERANKER_URL": ""}, clear=False):
                trace = rag.run_query(
                    "会员积分可以提现吗？",
                    quiet=True,
                    rebuild_index=True,
                    vector_backend="local",
                    store_path=store_path,
                    metrics_path=metrics_path,
                )

            self.assertEqual(trace["reranker"]["mode"], "skipped")
            self.assertEqual(trace["reranker"]["reason"], "not_configured")
            self.assertEqual(trace["reranker"]["score_policy"], "rrf_only")
            self.assertEqual(trace["model_config"]["rerank_mode"], "skipped")
            self.assertEqual(trace["model_config"]["rerank_score_policy"], "rrf_only")
            self.assertEqual(trace["truncation"]["score_policy"], "rrf_only")
            self.assertIsNone(trace["truncation"]["min_score"])
            self.assertIsNone(trace["truncation"]["gap_threshold"])
            self.assertEqual(trace["context_packet"]["sufficiency"]["score_policy"], "rrf_only")
            self.assertFalse(trace["context_packet"]["sufficiency"]["score_confidence"])


class DatasetImportTest(unittest.TestCase):
    def load_importer_module(self):
        script_path = PROJECT_ROOT / "scripts" / "import_customer_support_dataset.py"
        spec = importlib.util.spec_from_file_location("customer_support_importer", script_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module

    def test_customer_support_dataset_rows_render_to_long_markdown_documents(self) -> None:
        module = self.load_importer_module()

        rows = [
            {
                "issue_area": "Cancellations and returns",
                "issue_category": "Cash on Delivery (CoD) Refunds",
                "issue_sub_category": "Refund timelines for Cash on Delivery returns",
                "product_category": "Appliances",
                "product_sub_category": "Water Purifier",
                "issue_complexity": "medium",
                "conversation": "Agent: Hello. Customer: When will my refund arrive? Agent: Refunds are processed after pickup and quality check.",
                "qa": json.dumps(
                    {
                        "knowledge": [
                            {
                                "customer_summary_question": "When will my refund arrive?",
                                "agent_summary_solution": "Refunds are processed after pickup and quality check.",
                            }
                        ]
                    }
                ),
            }
        ]

        docs = module.rows_to_markdown_documents(rows, rows_per_doc=1)

        self.assertEqual(len(docs), 1)
        self.assertIn("dataset_source: rjac/e-commerce-customer-support-qa", docs[0].content)
        self.assertIn("## Case 1: Refund timelines for Cash on Delivery returns", docs[0].content)
        self.assertIn("Refunds are processed after pickup and quality check.", docs[0].content)
        self.assertGreater(len(docs[0].content), 700)

    def test_huggingface_download_error_mentions_optional_import_and_proxy(self) -> None:
        module = self.load_importer_module()

        message = module.format_huggingface_download_error(
            RuntimeError("LocalEntryNotFoundError"),
            RuntimeError("datasets-server unavailable"),
        )

        self.assertIn("optional Hugging Face dataset", message)
        self.assertIn("does not block the main production RAG experiment", message)
        self.assertIn("Dataset Viewer fallback", message)
        self.assertIn("HTTPS_PROXY", message)
        self.assertIn("HF_ENDPOINT", message)

    def test_dataset_viewer_rows_api_can_load_fallback_rows(self) -> None:
        module = self.load_importer_module()
        captured_urls: list[str] = []

        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "rows": [
                            {
                                "row": {
                                    "issue_area": "Order",
                                    "issue_category": "Delivery",
                                    "issue_sub_category": "Late delivery",
                                    "conversation": "Agent: hello",
                                    "qa": "{\"knowledge\": []}",
                                }
                            }
                        ]
                    }
                ).encode("utf-8")

        def fake_urlopen(url: str, timeout: int) -> FakeResponse:
            captured_urls.append(url)
            return FakeResponse()

        with patch("urllib.request.urlopen", fake_urlopen):
            rows = module.load_dataset_viewer_rows(limit=1)

        self.assertEqual(rows[0]["issue_area"], "Order")
        self.assertIn("datasets-server.huggingface.co/rows", captured_urls[0])
        self.assertIn("dataset=rjac%2Fe-commerce-customer-support-qa", captured_urls[0])


if __name__ == "__main__":
    unittest.main()
