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

        with patch.object(rag, "CONTEXT_TOKEN_BUDGET", 899):
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
        self.assertEqual(selected_chunk_ids, ["parent1-anchor"])
        self.assertEqual(truncation["budget_basis"], "expanded_parent_context_tokens")
        self.assertEqual(truncation["context_token_budget"], 899)
        self.assertEqual(truncation["token_total"], 450)
        self.assertEqual(context_packet["estimated_token_total"], truncation["token_total"])
        self.assertLessEqual(context_packet["estimated_token_total"], 899)
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


if __name__ == "__main__":
    unittest.main()
