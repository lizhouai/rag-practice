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


if __name__ == "__main__":
    unittest.main()
