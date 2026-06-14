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


if __name__ == "__main__":
    unittest.main()
