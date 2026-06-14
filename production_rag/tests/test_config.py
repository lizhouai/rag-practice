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


if __name__ == "__main__":
    unittest.main()
