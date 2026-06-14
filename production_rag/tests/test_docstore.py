from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from tests.helpers import rag
from rag.docstore import SqliteDocstore


def _chunk(cid: str, parent: str, text: str) -> "rag.Chunk":
    return rag.Chunk(
        chunk_id=cid,
        parent_id=parent,
        doc_id="doc-1",
        title_path=["doc", "sec"],
        text=text,
        metadata={"permission_scope": "internal"},
        token_count=5,
        dense_vector=[0.1, 0.2],
        terms=rag.tokenize(text),
    )


class SqliteDocstoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        store_path = Path(self.tmp.name) / "rag.sqlite"
        self.store = rag.LocalVectorStore(store_path)
        self.store.upsert_document(
            "doc-1",
            "doc.md",
            "h1",
            "local:test",
            [
                _chunk("c1", "p1", "alpha text"),
                _chunk("c2", "p1", "beta text"),
                _chunk("c3", "p2", "gamma text"),
            ],
        )
        self.docstore = SqliteDocstore(store_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_hydrate_returns_only_requested_ids_with_text(self) -> None:
        out = self.docstore.hydrate(["c1", "c3", "missing"])
        self.assertEqual(set(out), {"c1", "c3"})
        self.assertEqual(out["c1"].text, "alpha text")
        self.assertEqual(out["c3"].parent_id, "p2")

    def test_siblings_groups_by_parent_id(self) -> None:
        out = self.docstore.siblings(["p1"])
        self.assertEqual(sorted(c.chunk_id for c in out["p1"]), ["c1", "c2"])


if __name__ == "__main__":
    unittest.main()
