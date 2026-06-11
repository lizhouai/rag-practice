from __future__ import annotations

import json
import importlib.util
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
        self.assertEqual(body["points"][0]["vector"], chunk.dense_vector)
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
        self.assertEqual(body["limit"], 3)
        self.assertEqual(results[0][0], 0.87)
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
            "model_config": {"vector_backend": "local", "embedding_model": "local-hash-embedding"},
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
        }

        event = rag.build_monitoring_event(trace, latency_ms=123, status="ok")

        self.assertEqual(event["trace_id"], "trace-1")
        self.assertEqual(event["latency_ms"], 123)
        self.assertEqual(event["vector_backend"], "local")
        self.assertEqual(event["embedding_model"], "local-hash-embedding")
        self.assertEqual(event["query_hash"], rag.hashlib.sha256("跨境订单退款多久到账？".encode("utf-8")).hexdigest()[:16])
        self.assertEqual(event["answer_mode"], "extractive")
        self.assertEqual(event["citation_valid"], True)
        self.assertEqual(event["sufficiency_reason"], "pass")
        self.assertEqual(event["selected_doc_ids"], ["refund_policy_v3"])
        self.assertEqual(event["dense_hits"], 2)
        self.assertEqual(event["bm25_hits"], 1)
        self.assertEqual(event["dedup_dropped_count"], 1)

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
