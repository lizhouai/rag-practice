from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from rag.config import BM25_TOP_N, DEFAULT_VECTOR_DB_PATH
from rag.chunking import allowed_scope_key, date_to_sortable_day, metadata_access_fields, split_metadata_values
from rag.chunking import sqlite_fts_query, tokenize
from rag.models import Chunk

__all__ = ["LocalVectorStore"]


class LocalVectorStore:
    def __init__(self, path: Path = DEFAULT_VECTOR_DB_PATH) -> None:
        self.path = path

    def describe(self) -> str:
        return str(self.path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        self.ensure_schema(connection)
        self.register_functions(connection)
        return connection

    @staticmethod
    def ensure_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                parent_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                title_path_json TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                permission_scope TEXT NOT NULL DEFAULT '',
                effective_from TEXT NOT NULL DEFAULT '',
                effective_to TEXT NOT NULL DEFAULT '',
                effective_from_day INTEGER NOT NULL DEFAULT 0,
                effective_to_day INTEGER NOT NULL DEFAULT 99991231,
                token_count INTEGER NOT NULL,
                dense_vector_json TEXT NOT NULL,
                terms_json TEXT NOT NULL
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id)")
        for column_name, column_type in {
            "permission_scope": "TEXT NOT NULL DEFAULT ''",
            "effective_from": "TEXT NOT NULL DEFAULT ''",
            "effective_to": "TEXT NOT NULL DEFAULT ''",
            "effective_from_day": "INTEGER NOT NULL DEFAULT 0",
            "effective_to_day": "INTEGER NOT NULL DEFAULT 99991231",
        }.items():
            if not LocalVectorStore.column_exists(connection, "chunks", column_name):
                connection.execute(f"ALTER TABLE chunks ADD COLUMN {column_name} {column_type}")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_permission_scope ON chunks(permission_scope)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_chunks_effective_days ON chunks(effective_from_day, effective_to_day)")
        connection.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(chunk_id UNINDEXED, terms_text)
            """
        )
        LocalVectorStore.backfill_access_columns(connection)
        LocalVectorStore.refresh_fts_index(connection)
        connection.commit()

    @staticmethod
    def column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        return any(row[1] == column_name for row in rows)

    @staticmethod
    def register_functions(connection: sqlite3.Connection) -> None:
        def scope_allowed(scope_text: str | None, allowed_text: str | None) -> int:
            scopes = split_metadata_values(scope_text or "")
            allowed = {item for item in (allowed_text or "").split("\x1f") if item}
            return int(bool(scopes and not scopes.isdisjoint(allowed)))

        connection.create_function("scope_allowed", 2, scope_allowed)

    @staticmethod
    def backfill_access_columns(connection: sqlite3.Connection) -> None:
        rows = connection.execute("SELECT chunk_id, metadata_json FROM chunks").fetchall()
        for chunk_id, metadata_json in rows:
            metadata = json.loads(metadata_json)
            fields = metadata_access_fields(metadata)
            connection.execute(
                """
                UPDATE chunks
                SET permission_scope = ?,
                    effective_from = ?,
                    effective_to = ?,
                    effective_from_day = ?,
                    effective_to_day = ?
                WHERE chunk_id = ?
                """,
                (
                    fields["permission_scope"],
                    fields["effective_from"],
                    fields["effective_to"],
                    fields["effective_from_day"],
                    fields["effective_to_day"],
                    chunk_id,
                ),
            )

    @staticmethod
    def refresh_fts_index(connection: sqlite3.Connection) -> None:
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        if chunk_count == fts_count:
            return
        rows = connection.execute("SELECT chunk_id, terms_json FROM chunks").fetchall()
        connection.execute("DELETE FROM chunks_fts")
        connection.executemany(
            "INSERT INTO chunks_fts(chunk_id, terms_text) VALUES (?, ?)",
            [(row[0], " ".join(json.loads(row[1]))) for row in rows],
        )

    def reset(self) -> None:
        with closing(self.connect()) as connection:
            connection.execute("DELETE FROM chunks_fts")
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            connection.commit()

    def load_manifest(self) -> dict[str, dict[str, str]]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT doc_id, source_path, content_hash, embedding_model, updated_at FROM documents"
            ).fetchall()
        return {
            row[0]: {
                "source_path": row[1],
                "content_hash": row[2],
                "embedding_model": row[3],
                "updated_at": row[4],
            }
            for row in rows
        }

    def upsert_document(
        self,
        doc_id: str,
        source_path: str,
        hash_value: str,
        embedding_model: str,
        chunks: list[Chunk],
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with closing(self.connect()) as connection:
            connection.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            connection.execute(
                """
                INSERT INTO documents(doc_id, source_path, content_hash, embedding_model, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source_path = excluded.source_path,
                    content_hash = excluded.content_hash,
                    embedding_model = excluded.embedding_model,
                    updated_at = excluded.updated_at
                """,
                (doc_id, source_path, hash_value, embedding_model, now),
            )
            connection.executemany(
                """
                INSERT INTO chunks(
                    chunk_id, parent_id, doc_id, title_path_json, text, metadata_json,
                    permission_scope, effective_from, effective_to, effective_from_day, effective_to_day,
                    token_count, dense_vector_json, terms_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.parent_id,
                        chunk.doc_id,
                        json.dumps(chunk.title_path, ensure_ascii=False),
                        chunk.text,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                        metadata_access_fields(chunk.metadata)["permission_scope"],
                        metadata_access_fields(chunk.metadata)["effective_from"],
                        metadata_access_fields(chunk.metadata)["effective_to"],
                        metadata_access_fields(chunk.metadata)["effective_from_day"],
                        metadata_access_fields(chunk.metadata)["effective_to_day"],
                        chunk.token_count,
                        json.dumps(chunk.dense_vector, ensure_ascii=False),
                        json.dumps(chunk.terms, ensure_ascii=False),
                    )
                    for chunk in chunks
                ],
            )
            connection.executemany(
                "INSERT INTO chunks_fts(chunk_id, terms_text) VALUES (?, ?)",
                [(chunk.chunk_id, " ".join(chunk.terms)) for chunk in chunks],
            )
            connection.commit()

    def delete_documents(self, doc_ids: set[str]) -> None:
        if not doc_ids:
            return
        with closing(self.connect()) as connection:
            for doc_id in sorted(doc_ids):
                connection.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE doc_id = ?)",
                    (doc_id,),
                )
                connection.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
                connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            connection.commit()

    def load_chunks(self) -> list[Chunk]:
        with closing(self.connect()) as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, parent_id, doc_id, title_path_json, text, metadata_json,
                       token_count, dense_vector_json, terms_json
                FROM chunks
                ORDER BY doc_id, chunk_id
                """
            ).fetchall()
        return [self.chunk_from_sql_row(row) for row in rows]

    def load_access_chunks(
        self,
        allowed_scopes: set[str],
        *,
        today: str | None = None,
    ) -> tuple[list[Chunk], list[dict], list[Chunk]]:
        today_day = date_to_sortable_day(today or datetime.now().date().isoformat(), default=0)
        scope_key = allowed_scope_key(allowed_scopes)
        with closing(self.connect()) as connection:
            visible = self.fetch_chunks(
                connection,
                """
                scope_allowed(c.permission_scope, ?) = 1
                AND c.effective_from_day <= ?
                AND c.effective_to_day >= ?
                """,
                (scope_key, today_day, today_day),
            )
            permission_blocked = self.fetch_chunks(
                connection,
                "scope_allowed(c.permission_scope, ?) = 0",
                (scope_key,),
            )
            time_rejected_rows = connection.execute(
                """
                SELECT c.chunk_id, c.doc_id, c.effective_from_day, c.effective_to_day
                FROM chunks c
                WHERE scope_allowed(c.permission_scope, ?) = 1
                  AND (c.effective_from_day > ? OR c.effective_to_day < ?)
                ORDER BY c.doc_id, c.chunk_id
                """,
                (scope_key, today_day, today_day),
            ).fetchall()
        rejected = [
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "reason": "permission_scope"}
            for chunk in permission_blocked
        ]
        for chunk_id, doc_id, effective_from_day, effective_to_day in time_rejected_rows:
            reason = "not_yet_effective" if effective_from_day > today_day else "expired"
            rejected.append({"chunk_id": chunk_id, "doc_id": doc_id, "reason": reason})
        return visible, rejected, permission_blocked

    def bm25_search(
        self,
        query: str,
        allowed_scopes: set[str],
        *,
        top_n: int = BM25_TOP_N,
        today: str | None = None,
    ) -> list[tuple[float, Chunk]]:
        fts_query = sqlite_fts_query(tokenize(query))
        if not fts_query:
            return []
        today_day = date_to_sortable_day(today or datetime.now().date().isoformat(), default=0)
        scope_key = allowed_scope_key(allowed_scopes)
        try:
            with closing(self.connect()) as connection:
                rows = connection.execute(
                    f"""
                    SELECT {self.chunk_select_columns("c")}, bm25(chunks_fts) AS rank
                    FROM chunks_fts
                    JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                    WHERE chunks_fts MATCH ?
                      AND scope_allowed(c.permission_scope, ?) = 1
                      AND c.effective_from_day <= ?
                      AND c.effective_to_day >= ?
                    ORDER BY rank ASC
                    LIMIT ?
                    """,
                    (fts_query, scope_key, today_day, today_day, top_n),
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(-float(row[9]), self.chunk_from_sql_row(row)) for row in rows]

    def chunks_count(self) -> int:
        with closing(self.connect()) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    @staticmethod
    def chunk_select_columns(alias: str) -> str:
        return (
            f"{alias}.chunk_id, {alias}.parent_id, {alias}.doc_id, {alias}.title_path_json, "
            f"{alias}.text, {alias}.metadata_json, {alias}.token_count, "
            f"{alias}.dense_vector_json, {alias}.terms_json"
        )

    @staticmethod
    def chunk_from_sql_row(row: tuple) -> Chunk:
        return Chunk(
            chunk_id=row[0],
            parent_id=row[1],
            doc_id=row[2],
            title_path=json.loads(row[3]),
            text=row[4],
            metadata=json.loads(row[5]),
            token_count=row[6],
            dense_vector=json.loads(row[7]),
            terms=json.loads(row[8]),
        )

    def fetch_chunks(
        self,
        connection: sqlite3.Connection,
        where_clause: str,
        params: tuple,
    ) -> list[Chunk]:
        rows = connection.execute(
            f"""
            SELECT {self.chunk_select_columns("c")}
            FROM chunks c
            WHERE {where_clause}
            ORDER BY c.doc_id, c.chunk_id
            """,
            params,
        ).fetchall()
        return [self.chunk_from_sql_row(row) for row in rows]
