from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from rag.models import Chunk
from rag.vectorstore.sqlite import LocalVectorStore

__all__ = ["SqliteDocstore"]


class SqliteDocstore:
    """Read-only hydration over the SQLite store: id->text and parent_id->siblings.

    Used by the Qdrant happy path, where recall returns ids + vectors and the
    full chunk text is fetched locally instead of crossing the network.
    """

    def __init__(self, path: Path) -> None:
        self._store = LocalVectorStore(path)

    def _connect(self) -> sqlite3.Connection:
        return self._store.connect()

    def hydrate(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        ids = [cid for cid in dict.fromkeys(chunk_ids) if cid]
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT {self._store.chunk_select_columns('c')} "
                f"FROM chunks c WHERE c.chunk_id IN ({placeholders})",
                ids,
            ).fetchall()
        return {row[0]: self._store.chunk_from_sql_row(row) for row in rows}

    def siblings(self, parent_ids: list[str]) -> dict[str, list[Chunk]]:
        pids = [pid for pid in dict.fromkeys(parent_ids) if pid]
        if not pids:
            return {}
        placeholders = ",".join("?" for _ in pids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT {self._store.chunk_select_columns('c')} "
                f"FROM chunks c WHERE c.parent_id IN ({placeholders}) "
                f"ORDER BY c.chunk_id",
                pids,
            ).fetchall()
        grouped: dict[str, list[Chunk]] = {pid: [] for pid in pids}
        for row in rows:
            chunk = self._store.chunk_from_sql_row(row)
            grouped[chunk.parent_id].append(chunk)
        return grouped
