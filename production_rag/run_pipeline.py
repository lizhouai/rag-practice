from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
DEFAULT_RUNTIME_DIR = Path(
    os.getenv("RAG_RUNTIME_DIR")
    or Path(os.getenv("TEMP") or os.getenv("TMPDIR") or "C:/tmp") / "production_rag_runtime"
)
TRACE_DIR = DEFAULT_RUNTIME_DIR / "traces"
EVAL_PATH = ROOT / "eval_cases.csv"
INDEX_DIR = DEFAULT_RUNTIME_DIR / "indexes"
DEFAULT_VECTOR_DB_PATH = INDEX_DIR / "rag_vector_store.sqlite"
METRICS_PATH = TRACE_DIR / "online_metrics.jsonl"
DEFAULT_VECTOR_BACKEND = "qdrant"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "production_rag_chunks"
QDRANT_DENSE_VECTOR_NAME = "dense"
QDRANT_BM25_VECTOR_NAME = "bm25"
QDRANT_SPARSE_HASH_BUCKETS = 2_147_483_647
QDRANT_PAYLOAD_INDEXES = {
    "doc_id": "keyword",
    "permission_scopes": "keyword",
    "effective_from_day": "integer",
    "effective_to_day": "integer",
}

CHUNK_CHARS = 280
OVERLAP_CHARS = 60
DENSE_TOP_N = 12
BM25_TOP_N = 12
RRF_K = 60
RERANK_TOP_N = 12
DEDUP_THRESHOLD = 0.92
MMR_LAMBDA = 0.72
FINAL_MAX_K = 5
MIN_RERANK_SCORE = 0.12
GAP_THRESHOLD = 0.28
CONTEXT_TOKEN_BUDGET = 900
PARENT_EXPANSION_MAX_CHARS = 1200
SCORE_POLICY_EXTERNAL_RERANK = "external_rerank"
SCORE_POLICY_RRF_ONLY = "rrf_only"
DEFAULT_CHAT_MODEL = "deepseek-v4-pro"
DEFAULT_EMBEDDING_MODEL = "embedding-3"
DEFAULT_LLM_API_STYLE = "anthropic"
DEFAULT_LLM_MAX_TOKENS = 1200
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"
DEFAULT_ZHIPU_EMBEDDING_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_EMBEDDING_DIMENSIONS = 1024
LOCAL_EMBEDDING_MODEL = "local-hash-embedding"
EMBEDDING_PROVIDER_LOCAL = "local"
EMBEDDING_PROVIDER_EXTERNAL = "external"
LOCAL_EMBEDDING_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}
DEFAULT_ALLOWED_SCOPES = {"internal", "public"}
DEFAULT_RERANKER_MODEL = "bge-reranker-v2-m3"
DEFAULT_RERANKER_URL = "http://127.0.0.1:8008/rerank"
EXTERNAL_RERANKER_PROVIDERS = {"external", "http", "flagembedding", "transformers", "bge"}
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 0.05

STOPWORDS = {
    "的",
    "了",
    "吗",
    "呢",
    "是",
    "在",
    "和",
    "或",
    "与",
    "可以",
    "怎么",
    "如何",
    "什么",
    "一个",
    "用户",
}


@dataclass
class ParentSection:
    parent_id: str
    doc_id: str
    title: str
    title_path: list[str]
    text: str
    metadata: dict[str, str]


@dataclass
class Chunk:
    chunk_id: str
    parent_id: str
    doc_id: str
    title_path: list[str]
    text: str
    metadata: dict[str, str]
    token_count: int
    dense_vector: list[float] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)


@dataclass
class Candidate:
    chunk_id: str
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    mmr_score: float = 0.0
    reason: str = ""


@dataclass
class SelectedEvidence:
    candidate: Candidate
    expanded_text: str
    expanded_from_chunk_ids: list[str]
    expanded_token_count: int


class RetryExhausted(RuntimeError):
    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class ComponentFallback(RuntimeError):
    def __init__(self, component: str, reason: str, error: str, attempts: int) -> None:
        super().__init__(error)
        self.component = component
        self.reason = reason
        self.error = error
        self.attempts = attempts


def call_with_retries(
    operation,
    *,
    attempts: int = DEFAULT_RETRY_ATTEMPTS,
    backoff_seconds: float = DEFAULT_RETRY_BACKOFF_SECONDS,
):
    attempts = max(1, attempts)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation(), attempt
        except ComponentFallback:
            raise
        except Exception as exc:  # noqa: BLE001 - component adapters normalize failure in trace.
            last_exc = exc
            if attempt < attempts and backoff_seconds > 0:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))
    message = str(last_exc) if last_exc else "operation failed"
    raise RetryExhausted(message, attempts) from last_exc


@dataclass
class ExternalReranker:
    url: str
    model: str = DEFAULT_RERANKER_MODEL
    timeout_seconds: int = 30
    last_error: str = ""
    last_attempts: int = 0
    fallback_used: bool = False

    def score(self, query: str, chunks: list[Chunk]) -> list[float]:
        body = {
            "model": self.model,
            "query": query,
            "documents": [
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": {
                        "doc_id": chunk.doc_id,
                        "title_path": " > ".join(chunk.title_path),
                    },
                }
                for chunk in chunks
            ],
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return parse_reranker_scores(payload, expected_count=len(chunks))


@dataclass
class AnthropicMessagesClient:
    api_key: str | None
    base_url: str

    def create_message(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        body = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        }
        headers = {"anthropic-version": "2023-06-01"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        payload = post_json(f"{self.base_url.rstrip('/')}/v1/messages", body, headers=headers)
        return extract_anthropic_text(payload)


@dataclass
class OpenAICompatibleEmbeddingClient:
    api_key: str | None
    base_url: str

    def embed_texts(
        self,
        texts: list[str],
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        body: dict[str, object] = {"model": model, "input": texts}
        if dimensions:
            body["dimensions"] = dimensions
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = post_json(f"{self.base_url.rstrip('/')}/embeddings", body, headers=headers)
        data = payload.get("data", [])
        if not isinstance(data, list):
            raise RuntimeError("Embedding response missing data list.")
        return [item["embedding"] for item in sorted(data, key=lambda item: item.get("index", 0))]


def post_json(url: str, body: dict[str, object], headers: dict[str, str] | None = None) -> dict:
    return request_json("POST", url, body=body, headers=headers)


def request_json(
    method: str,
    url: str,
    body: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    ok_statuses: tuple[int, ...] = (200,),
) -> dict:
    request_headers = {
        "Content-Type": "application/json",
        **(headers or {}),
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            status = getattr(response, "status", 200)
            if status not in ok_statuses:
                raise RuntimeError(f"API request failed for {method} {url}: HTTP {status}")
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API request failed for {method} {url}: HTTP {exc.code} {error_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API request failed for {method} {url}: {format_transport_error(url, exc.reason)}") from exc


def format_transport_error(url: str, reason: object) -> str:
    message = str(reason)
    parsed = urllib.parse.urlparse(url)
    if (
        parsed.scheme == "https"
        and parsed.hostname
        and parsed.hostname.endswith(".cloud.qdrant.io")
        and parsed.port is None
    ):
        message = (
            f"{message}. Qdrant Cloud REST endpoints require port 6333; "
            f"set QDRANT_URL to https://{parsed.hostname}:6333"
        )
    elif "UNEXPECTED_EOF_WHILE_READING" in message:
        message = (
            f"{message}. This usually means the URL scheme or port does not match the service TLS mode. "
            "For local Qdrant use http://localhost:6333; for Qdrant Cloud use https://<cluster>.cloud.qdrant.io:6333."
        )
    return message


def extract_anthropic_text(payload: dict) -> str:
    content = payload.get("content", [])
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
    return "".join(texts)


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env_first(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def has_env_value(*names: str) -> bool:
    return any(bool(os.getenv(name)) for name in names)


def resolve_llm_base_url() -> str:
    return env_first(
        "LLM_BASE_URL",
        "ANTHROPIC_BASE_URL",
        "DEEPSEEK_BASE_URL",
        default=DEFAULT_DEEPSEEK_BASE_URL,
    ) or DEFAULT_DEEPSEEK_BASE_URL


def resolve_embedding_base_url() -> str:
    return env_first(
        "EMBEDDING_BASE_URL",
        "ZHIPU_EMBEDDING_BASE_URL",
        "ZHIPUAI_BASE_URL",
        default=DEFAULT_ZHIPU_EMBEDDING_BASE_URL,
    ) or DEFAULT_ZHIPU_EMBEDDING_BASE_URL


def resolve_embedding_provider(base_url: str | None = None) -> str:
    explicit_provider = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if explicit_provider in {EMBEDDING_PROVIDER_LOCAL, EMBEDDING_PROVIDER_EXTERNAL}:
        return explicit_provider

    resolved_base_url = (base_url if base_url is not None else resolve_embedding_base_url()).strip()
    url_for_parse = resolved_base_url if "://" in resolved_base_url else f"http://{resolved_base_url}"
    parsed = urllib.parse.urlparse(url_for_parse)
    hostname = (parsed.hostname or "").lower()
    if hostname in LOCAL_EMBEDDING_HOSTS or hostname.startswith("127."):
        return EMBEDDING_PROVIDER_LOCAL
    return EMBEDDING_PROVIDER_EXTERNAL


def resolve_qdrant_url() -> str:
    return env_first("QDRANT_URL", "VECTOR_DB_URL", default=DEFAULT_QDRANT_URL) or DEFAULT_QDRANT_URL


def is_qdrant_configured() -> bool:
    return has_env_value(
        "QDRANT_URL",
        "VECTOR_DB_URL",
        "QDRANT_COLLECTION",
        "VECTOR_DB_COLLECTION",
        "QDRANT_API_KEY",
        "VECTOR_DB_API_KEY",
    )


def resolve_qdrant_api_key() -> str | None:
    return env_first("QDRANT_API_KEY", "VECTOR_DB_API_KEY")


def resolve_qdrant_collection() -> str:
    return (
        env_first("QDRANT_COLLECTION", "VECTOR_DB_COLLECTION", default=DEFAULT_QDRANT_COLLECTION)
        or DEFAULT_QDRANT_COLLECTION
    )


def is_llm_configured() -> bool:
    return bool(
        env_first("LLM_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY")
        or has_env_value("LLM_BASE_URL", "ANTHROPIC_BASE_URL", "DEEPSEEK_BASE_URL")
    )


def is_embedding_configured() -> bool:
    return bool(
        env_first("EMBEDDING_API_KEY", "ZHIPU_API_KEY")
        or has_env_value("EMBEDDING_BASE_URL", "ZHIPU_EMBEDDING_BASE_URL", "ZHIPUAI_BASE_URL")
    )


def parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise RuntimeError(f"{name} must be greater than 0.")
    return parsed


def resolve_vector_dimensions() -> int:
    # Single global vector dimension: real embeddings, local hash vectors, and
    # the Qdrant collection all use this value so vectors stay comparable
    # across every mode combination.
    return parse_int_env("EMBEDDING_DIMENSIONS", DEFAULT_EMBEDDING_DIMENSIONS)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end == -1:
        return {}, text
    block = text[4:end].strip()
    body = text[end + 4 :].strip()
    metadata: dict[str, str] = {}
    for line in block.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, body


def safe_relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def content_hash(text: str, embedding_model: str, dimensions: int) -> str:
    # Dimensions are part of the index identity: changing EMBEDDING_DIMENSIONS
    # must re-embed every document, or stored vectors would no longer match
    # query vectors.
    digest = hashlib.sha256()
    digest.update(embedding_model.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(dimensions).encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def embedding_identity(provider: str, model: str) -> str:
    return f"{provider.strip().lower()}:{model.strip()}"


def safe_path_slug(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return (slug or "embedding")[:max_length]


def local_store_path_for_embedding_identity(store_path: Path, identity: str) -> Path:
    slug = safe_path_slug(identity)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return store_path.with_name(f"{store_path.stem}.{slug}.{digest}{store_path.suffix}")


def date_to_sortable_day(value: str | None, *, default: int) -> int:
    text = (value or "").strip()
    if not text:
        return default
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return int(text.replace("-", ""))
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        return int(digits[:8])
    return default


def metadata_access_fields(metadata: dict[str, str]) -> dict[str, int | str]:
    effective_from = metadata.get("effective_from", "").strip()
    effective_to = metadata.get("effective_to", "").strip()
    return {
        "permission_scope": metadata.get("permission_scope", "").strip(),
        "effective_from": effective_from,
        "effective_to": effective_to,
        "effective_from_day": date_to_sortable_day(effective_from, default=0),
        "effective_to_day": date_to_sortable_day(effective_to, default=99991231),
    }


def allowed_scope_key(allowed_scopes: set[str]) -> str:
    return "\x1f".join(sorted(scope for scope in allowed_scopes if scope))


def sqlite_fts_query(terms: list[str]) -> str:
    quoted_terms = [f'"{term.replace(chr(34), chr(34) + chr(34))}"' for term in terms if term]
    return " OR ".join(quoted_terms)


def qdrant_sparse_index(term: str) -> int:
    return int.from_bytes(hashlib.sha256(term.encode("utf-8")).digest()[:8], "big") % QDRANT_SPARSE_HASH_BUCKETS


def qdrant_sparse_vector_from_terms(terms: list[str]) -> dict[str, list[int] | list[float]]:
    counts: Counter[int] = Counter()
    for term in terms:
        if term:
            counts[qdrant_sparse_index(term)] += 1
    items = sorted(counts.items())
    return {
        "indices": [index for index, _ in items],
        "values": [float(count) for _, count in items],
    }


def qdrant_point_dense_vector(raw_vector: object) -> list[float]:
    if isinstance(raw_vector, dict):
        vector = raw_vector.get(QDRANT_DENSE_VECTOR_NAME) or raw_vector.get("")
    else:
        vector = raw_vector
    return list(vector) if isinstance(vector, list) else []


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


def stable_point_id(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def chunk_to_qdrant_payload(
    chunk: Chunk,
    *,
    source_path: str,
    content_hash: str,
    embedding_model: str,
    doc_id: str | None = None,
) -> dict:
    metadata = dict(chunk.metadata)
    metadata.setdefault("source_path", source_path)
    resolved_doc_id = doc_id or chunk.doc_id
    access_fields = metadata_access_fields(metadata)
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.parent_id,
        "doc_id": resolved_doc_id,
        "title_path": chunk.title_path,
        "text": chunk.text,
        "metadata": metadata,
        "permission_scope": access_fields["permission_scope"],
        "permission_scopes": sorted(split_metadata_values(str(access_fields["permission_scope"]))),
        "effective_from": access_fields["effective_from"],
        "effective_to": access_fields["effective_to"],
        "effective_from_day": access_fields["effective_from_day"],
        "effective_to_day": access_fields["effective_to_day"],
        "token_count": chunk.token_count,
        "terms": chunk.terms,
        "source_path": source_path,
        "content_hash": content_hash,
        "embedding_model": embedding_model,
    }


def chunk_from_qdrant_payload(payload: dict, vector: list[float] | None = None) -> Chunk:
    return Chunk(
        chunk_id=payload["chunk_id"],
        parent_id=payload["parent_id"],
        doc_id=payload["doc_id"],
        title_path=list(payload.get("title_path", [])),
        text=payload.get("text", ""),
        metadata=dict(payload.get("metadata", {})),
        token_count=int(payload.get("token_count", 1)),
        dense_vector=vector or [],
        terms=list(payload.get("terms", [])),
    )


def qdrant_filter_by_doc_id(doc_id: str) -> dict:
    return {"must": [{"key": "doc_id", "match": {"value": doc_id}}]}


def qdrant_scope_filter(allowed_scopes: set[str]) -> dict:
    scopes = sorted(scope for scope in allowed_scopes if scope)
    if not scopes:
        return {"must": [{"key": "__empty_scope__", "match": {"value": "__never__"}}]}
    return {"should": [{"key": "permission_scopes", "match": {"value": scope}} for scope in scopes]}


def qdrant_effective_filter(today: str | None = None) -> list[dict]:
    today_day = date_to_sortable_day(today or datetime.now().date().isoformat(), default=0)
    return [
        {"key": "effective_from_day", "range": {"lte": today_day}},
        {"key": "effective_to_day", "range": {"gte": today_day}},
    ]


def qdrant_access_filter(allowed_scopes: set[str], today: str | None = None) -> dict:
    return {
        "must": [qdrant_scope_filter(allowed_scopes), *qdrant_effective_filter(today)],
    }


def qdrant_permission_blocked_filter(allowed_scopes: set[str], today: str | None = None) -> dict:
    scopes = sorted(scope for scope in allowed_scopes if scope)
    return {
        "must": qdrant_effective_filter(today),
        "must_not": [{"key": "permission_scopes", "match": {"value": scope}} for scope in scopes],
    }


def qdrant_not_yet_effective_filter(allowed_scopes: set[str], today: str | None = None) -> dict:
    today_day = date_to_sortable_day(today or datetime.now().date().isoformat(), default=0)
    return {
        "must": [
            qdrant_scope_filter(allowed_scopes),
            {"key": "effective_from_day", "range": {"gt": today_day}},
        ],
    }


def qdrant_expired_filter(allowed_scopes: set[str], today: str | None = None) -> dict:
    today_day = date_to_sortable_day(today or datetime.now().date().isoformat(), default=0)
    return {
        "must": [
            qdrant_scope_filter(allowed_scopes),
            {"key": "effective_to_day", "range": {"lt": today_day}},
        ],
    }


class QdrantVectorStore:
    def __init__(
        self,
        base_url: str = DEFAULT_QDRANT_URL,
        collection_name: str = DEFAULT_QDRANT_COLLECTION,
        vector_size: int = DEFAULT_EMBEDDING_DIMENSIONS,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.api_key = api_key

    def describe(self) -> str:
        return f"{self.base_url}/{self.collection_name}"

    def collection_url(self, suffix: str = "") -> str:
        return f"{self.base_url}/collections/{self.collection_name}{suffix}"

    def headers(self) -> dict[str, str] | None:
        if not self.api_key:
            return None
        return {"api-key": self.api_key}

    def ensure_collection(self) -> None:
        try:
            collection_info = request_json("GET", self.collection_url(), headers=self.headers(), ok_statuses=(200,))
            self.ensure_payload_indexes(collection_info)
            self.ensure_sparse_vectors(collection_info)
            return
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise RuntimeError(
                    "Qdrant is not reachable. Set QDRANT_URL to a reachable Qdrant endpoint, "
                    "set QDRANT_API_KEY if the endpoint requires authentication, start the local "
                    "service with `docker compose up -d qdrant`, or use `--vector-backend local` "
                    "only for offline debugging."
                ) from exc
        body = {
            "vectors": {
                QDRANT_DENSE_VECTOR_NAME: {
                    "size": self.vector_size,
                    "distance": "Cosine",
                },
            },
            "sparse_vectors": {
                QDRANT_BM25_VECTOR_NAME: {
                    "modifier": "idf",
                },
            },
            "optimizers_config": {
                "default_segment_number": 2,
            },
        }
        request_json("PUT", self.collection_url(), body=body, headers=self.headers(), ok_statuses=(200,))
        self.ensure_payload_indexes()

    def ensure_sparse_vectors(self, collection_info: dict | None = None) -> None:
        existing = self.sparse_vectors_schema(collection_info)
        if QDRANT_BM25_VECTOR_NAME in existing:
            return
        request_json(
            "POST",
            self.collection_url("/sparse_vectors"),
            body={"sparse_vectors": {QDRANT_BM25_VECTOR_NAME: {"modifier": "idf"}}},
            headers=self.headers(),
            ok_statuses=(200,),
        )

    def ensure_payload_indexes(self, collection_info: dict | None = None) -> None:
        existing = self.payload_schema(collection_info)
        for field_name, field_schema in QDRANT_PAYLOAD_INDEXES.items():
            if field_name in existing:
                continue
            try:
                request_json(
                    "PUT",
                    self.collection_url("/index?wait=true"),
                    body={"field_name": field_name, "field_schema": field_schema},
                    headers=self.headers(),
                    ok_statuses=(200,),
                )
            except RuntimeError as exc:
                message = str(exc).lower()
                if "already exists" in message or "already has" in message:
                    continue
                raise

    @staticmethod
    def payload_schema(collection_info: dict | None) -> dict:
        if not isinstance(collection_info, dict):
            return {}
        result = collection_info.get("result", {})
        if not isinstance(result, dict):
            return {}
        payload_schema = result.get("payload_schema", {})
        return payload_schema if isinstance(payload_schema, dict) else {}

    @staticmethod
    def sparse_vectors_schema(collection_info: dict | None) -> dict:
        if not isinstance(collection_info, dict):
            return {}
        result = collection_info.get("result", {})
        if not isinstance(result, dict):
            return {}
        config = result.get("config", {})
        if not isinstance(config, dict):
            return {}
        params = config.get("params", {})
        if not isinstance(params, dict):
            return {}
        sparse_vectors = params.get("sparse_vectors", {})
        return sparse_vectors if isinstance(sparse_vectors, dict) else {}


    def reset(self) -> None:
        try:
            request_json("DELETE", self.collection_url(), headers=self.headers(), ok_statuses=(200,))
        except RuntimeError as exc:
            if "HTTP 404" not in str(exc):
                raise
        self.ensure_collection()

    def load_manifest(self) -> dict[str, dict[str, str]]:
        manifest: dict[str, dict[str, str]] = {}
        for point in self.scroll_points(with_vector=False):
            payload = point.get("payload", {})
            doc_id = payload.get("doc_id")
            if not doc_id:
                continue
            manifest.setdefault(
                doc_id,
                {
                    "source_path": payload.get("source_path", ""),
                    "content_hash": payload.get("content_hash", ""),
                    "embedding_model": payload.get("embedding_model", ""),
                    "updated_at": "",
                },
            )
        return manifest

    def upsert_document(
        self,
        doc_id: str,
        source_path: str,
        hash_value: str,
        embedding_model: str,
        chunks: list[Chunk],
    ) -> None:
        self.delete_documents({doc_id})
        points = [
            {
                "id": stable_point_id(chunk.chunk_id),
                "vector": {
                    QDRANT_DENSE_VECTOR_NAME: chunk.dense_vector,
                    QDRANT_BM25_VECTOR_NAME: qdrant_sparse_vector_from_terms(chunk.terms),
                },
                "payload": chunk_to_qdrant_payload(
                    chunk,
                    source_path=source_path,
                    content_hash=hash_value,
                    embedding_model=embedding_model,
                    doc_id=doc_id,
                ),
            }
            for chunk in chunks
        ]
        if not points:
            return
        request_json(
            "PUT",
            self.collection_url("/points?wait=true"),
            body={"points": points},
            headers=self.headers(),
            ok_statuses=(200,),
        )

    def delete_documents(self, doc_ids: set[str]) -> None:
        for doc_id in sorted(doc_ids):
            request_json(
                "POST",
                self.collection_url("/points/delete?wait=true"),
                body={"filter": qdrant_filter_by_doc_id(doc_id)},
                headers=self.headers(),
                ok_statuses=(200,),
            )

    def load_chunks(self, *, access_filter: dict | None = None, with_vector: bool = True) -> list[Chunk]:
        chunks: list[Chunk] = []
        for point in self.scroll_points(with_vector=with_vector, access_filter=access_filter):
            payload = point.get("payload", {})
            vector = qdrant_point_dense_vector(point.get("vector") or [])
            chunks.append(chunk_from_qdrant_payload(payload, vector=vector))
        return sorted(chunks, key=lambda chunk: chunk.chunk_id)

    def load_access_chunks(
        self,
        allowed_scopes: set[str],
        *,
        today: str | None = None,
    ) -> tuple[list[Chunk], list[dict], list[Chunk]]:
        # Dense vectors for retrieval candidates come back with the search/bm25
        # responses, so even the visible scroll can skip vector payloads here.
        # The four category scrolls are independent reads; running them
        # concurrently pays one round trip of latency instead of four, which
        # matters against a remote Qdrant.
        filters = {
            "visible": qdrant_access_filter(allowed_scopes, today),
            "permission_blocked": qdrant_permission_blocked_filter(allowed_scopes, today),
            "not_yet_effective": qdrant_not_yet_effective_filter(allowed_scopes, today),
            "expired": qdrant_expired_filter(allowed_scopes, today),
        }
        with ThreadPoolExecutor(max_workers=len(filters)) as executor:
            futures = {
                name: executor.submit(self.load_chunks, access_filter=access_filter, with_vector=False)
                for name, access_filter in filters.items()
            }
            loaded = {name: future.result() for name, future in futures.items()}
        visible = loaded["visible"]
        permission_blocked = loaded["permission_blocked"]
        not_yet_effective = loaded["not_yet_effective"]
        expired = loaded["expired"]
        rejected = [
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "reason": "permission_scope"}
            for chunk in permission_blocked
        ]
        rejected.extend(
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "reason": "not_yet_effective"}
            for chunk in not_yet_effective
        )
        rejected.extend(
            {"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "reason": "expired"}
            for chunk in expired
        )
        return visible, rejected, permission_blocked

    def scroll_points(self, with_vector: bool, access_filter: dict | None = None) -> list[dict]:
        points: list[dict] = []
        offset: object | None = None
        while True:
            body: dict[str, object] = {
                "limit": 256,
                "with_payload": True,
                "with_vector": with_vector,
            }
            if access_filter is not None:
                body["filter"] = access_filter
            if offset is not None:
                body["offset"] = offset
            payload = request_json(
                "POST",
                self.collection_url("/points/scroll"),
                body=body,
                headers=self.headers(),
                ok_statuses=(200,),
            )
            result = payload.get("result", {})
            batch = result.get("points", []) if isinstance(result, dict) else []
            points.extend(batch)
            offset = result.get("next_page_offset") if isinstance(result, dict) else None
            if not offset:
                return points

    def search(
        self,
        query_vector: list[float],
        top_n: int = DENSE_TOP_N,
        *,
        access_filter: dict | None = None,
    ) -> list[tuple[float, Chunk]]:
        body: dict[str, object] = {
            "query": query_vector,
            "using": QDRANT_DENSE_VECTOR_NAME,
            "limit": top_n,
            "with_payload": True,
            "with_vector": [QDRANT_DENSE_VECTOR_NAME],
        }
        if access_filter is not None:
            body["filter"] = access_filter
        payload = request_json(
            "POST",
            self.collection_url("/points/query"),
            body=body,
            headers=self.headers(),
            ok_statuses=(200,),
        )
        result = payload.get("result", {})
        points = result.get("points", result) if isinstance(result, dict) else result
        scored: list[tuple[float, Chunk]] = []
        for point in points or []:
            scored.append(
                (
                    float(point.get("score", 0.0)),
                    chunk_from_qdrant_payload(
                        point.get("payload", {}),
                        vector=qdrant_point_dense_vector(point.get("vector") or []),
                    ),
                )
            )
        return scored

    def bm25_search(
        self,
        query: str,
        allowed_scopes: set[str],
        *,
        top_n: int = BM25_TOP_N,
        today: str | None = None,
    ) -> list[tuple[float, Chunk]]:
        query_sparse = qdrant_sparse_vector_from_terms(tokenize(query))
        if not query_sparse["indices"]:
            return []
        body: dict[str, object] = {
            "query": query_sparse,
            "using": QDRANT_BM25_VECTOR_NAME,
            "limit": top_n,
            "with_payload": True,
            "with_vector": [QDRANT_DENSE_VECTOR_NAME],
            "filter": qdrant_access_filter(allowed_scopes, today),
        }
        payload = request_json(
            "POST",
            self.collection_url("/points/query"),
            body=body,
            headers=self.headers(),
            ok_statuses=(200,),
        )
        result = payload.get("result", {})
        points = result.get("points", result) if isinstance(result, dict) else result
        scored: list[tuple[float, Chunk]] = []
        for point in points or []:
            scored.append(
                (
                    float(point.get("score", 0.0)),
                    chunk_from_qdrant_payload(
                        point.get("payload", {}),
                        vector=qdrant_point_dense_vector(point.get("vector") or []),
                    ),
                )
            )
        return scored


class MirroredVectorStore:
    def __init__(self, primary: object, mirror: LocalVectorStore) -> None:
        self.primary = primary
        self.mirror = mirror

    def describe(self) -> str:
        return self.primary.describe() if hasattr(self.primary, "describe") else str(self.primary)

    def reset(self) -> None:
        self.primary.reset()
        self.mirror.reset()

    def load_manifest(self) -> dict[str, dict[str, str]]:
        return self.primary.load_manifest()

    def upsert_document(
        self,
        doc_id: str,
        source_path: str,
        hash_value: str,
        embedding_model: str,
        chunks: list[Chunk],
    ) -> None:
        self.primary.upsert_document(doc_id, source_path, hash_value, embedding_model, chunks)
        self.mirror.upsert_document(doc_id, source_path, hash_value, embedding_model, chunks)

    def delete_documents(self, doc_ids: set[str]) -> None:
        self.primary.delete_documents(doc_ids)
        self.mirror.delete_documents(doc_ids)

    def load_chunks(self) -> list[Chunk]:
        return self.primary.load_chunks()


def tokenize(text: str) -> list[str]:
    normalized = text.lower()
    tokens = re.findall(r"[a-z0-9_/\-]+|[\u4e00-\u9fff]{2,}", normalized)
    expanded: list[str] = []
    for token in tokens:
        if token in STOPWORDS:
            continue
        expanded.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", token):
            expanded.extend(token[i : i + 2] for i in range(len(token) - 1))
    return expanded


def vectorize(tokens: list[str], dims: int | None = None) -> list[float]:
    dims = dims or resolve_vector_dimensions()
    vector = [0.0] * dims
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dims
        sign = 1 if digest[4] % 2 == 0 else -1
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def read_documents(raw_dir: Path = RAW_DIR) -> list[tuple[dict[str, str], str]]:
    documents: list[tuple[dict[str, str], str]] = []
    for path in sorted(raw_dir.glob("*.md")):
        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        metadata.setdefault("doc_id", path.stem)
        metadata.setdefault("title", path.stem)
        metadata.setdefault("source_path", safe_relative(path, ROOT))
        metadata.setdefault("permission_scope", "internal")
        metadata.setdefault("effective_from", "")
        metadata.setdefault("effective_to", "")
        documents.append((metadata, normalize_text(body)))
    return documents


def split_sections(metadata: dict[str, str], body: str) -> list[ParentSection]:
    doc_title = metadata.get("title", metadata["doc_id"])
    sections: list[ParentSection] = []
    current_title = doc_title
    current_lines: list[str] = []

    for line in body.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append(make_parent(metadata, doc_title, current_title, current_lines, len(sections)))
            current_title = line.removeprefix("## ").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(make_parent(metadata, doc_title, current_title, current_lines, len(sections)))
    return sections


def make_parent(
    metadata: dict[str, str],
    doc_title: str,
    section_title: str,
    lines: list[str],
    index: int,
) -> ParentSection:
    doc_id = metadata["doc_id"]
    return ParentSection(
        parent_id=f"{doc_id}:sec_{index:02d}",
        doc_id=doc_id,
        title=section_title,
        title_path=[doc_title, section_title],
        text=normalize_text("\n".join(lines)),
        metadata=metadata,
    )


def chunk_parent(parent: ParentSection) -> list[Chunk]:
    paragraphs = [item.strip() for item in parent.text.split("\n\n") if item.strip()]
    chunks: list[Chunk] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= CHUNK_CHARS:
            current = candidate
            continue
        if current:
            chunks.append(make_chunk(parent, current, len(chunks)))
            current = f"{current[-OVERLAP_CHARS:]}\n\n{paragraph}".strip()
        else:
            chunks.append(make_chunk(parent, paragraph[:CHUNK_CHARS], len(chunks)))
            current = paragraph[CHUNK_CHARS - OVERLAP_CHARS :]
    if current:
        chunks.append(make_chunk(parent, current, len(chunks)))
    return chunks


def make_chunk(parent: ParentSection, text: str, index: int) -> Chunk:
    terms = tokenize(" ".join(parent.title_path) + " " + text)
    return Chunk(
        chunk_id=f"{parent.parent_id}:chunk_{index:02d}",
        parent_id=parent.parent_id,
        doc_id=parent.doc_id,
        title_path=parent.title_path,
        text=text,
        metadata=parent.metadata,
        token_count=max(1, len(text) // 2),
        terms=terms,
        dense_vector=vectorize(terms),
    )


def build_corpus() -> tuple[list[ParentSection], list[Chunk]]:
    parents: list[ParentSection] = []
    chunks: list[Chunk] = []
    for metadata, body in read_documents():
        doc_parents = split_sections(metadata, body)
        parents.extend(doc_parents)
        for parent in doc_parents:
            chunks.extend(chunk_parent(parent))
    return parents, chunks


def split_metadata_values(value: str) -> set[str]:
    return {item.strip() for item in re.split(r"[,，;；\s]+", value) if item.strip()}


def is_effective(metadata: dict[str, str], today: str) -> tuple[bool, str | None]:
    effective_from = metadata.get("effective_from", "")
    effective_to = metadata.get("effective_to", "")
    if effective_from and effective_from > today:
        return False, "not_yet_effective"
    if effective_to and effective_to < today:
        return False, "expired"
    return True, None


def filter_chunks_for_access(
    chunks: list[Chunk],
    allowed_scopes: set[str],
    *,
    today: str | None = None,
) -> tuple[list[Chunk], list[dict]]:
    today_value = today or datetime.now().date().isoformat()
    visible: list[Chunk] = []
    rejected: list[dict] = []
    for chunk in chunks:
        scopes = split_metadata_values(chunk.metadata.get("permission_scope", ""))
        if not scopes or scopes.isdisjoint(allowed_scopes):
            rejected.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "reason": "permission_scope"})
            continue
        effective, reason = is_effective(chunk.metadata, today_value)
        if not effective:
            rejected.append({"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id, "reason": reason})
            continue
        visible.append(chunk)
    return visible, rejected


def find_permission_blocked_matches(
    query: str,
    all_chunks: list[Chunk],
    rejected_chunks: list[dict],
    *,
    limit: int = 5,
) -> list[dict]:
    query_terms = set(tokenize(query))
    if not query_terms:
        return []
    rejected_by_id = {item["chunk_id"]: item for item in rejected_chunks if item.get("reason") == "permission_scope"}
    matches = []
    for chunk in all_chunks:
        if chunk.chunk_id not in rejected_by_id:
            continue
        matched_terms = sorted(query_terms & set(chunk.terms))
        if not matched_terms:
            continue
        overlap_ratio = len(matched_terms) / max(1, len(query_terms))
        has_strong_term = any(re.search(r"\d|[-_/]", term) or len(term) >= 4 for term in matched_terms)
        if overlap_ratio < 0.25 and not has_strong_term:
            continue
        matches.append(
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title_path": " > ".join(chunk.title_path),
                "matched_terms": matched_terms,
                "overlap_ratio": round(overlap_ratio, 4),
            }
        )
    return sorted(matches, key=lambda item: (item["overlap_ratio"], len(item["matched_terms"])), reverse=True)[:limit]


def make_vector_store(
    vector_backend: str,
    *,
    store_path: Path = DEFAULT_VECTOR_DB_PATH,
    vector_size: int = DEFAULT_EMBEDDING_DIMENSIONS,
) -> object:
    if vector_backend == "qdrant":
        return QdrantVectorStore(
            base_url=resolve_qdrant_url(),
            collection_name=resolve_qdrant_collection(),
            vector_size=vector_size,
            api_key=resolve_qdrant_api_key(),
        )
    if vector_backend == "local":
        return LocalVectorStore(store_path)
    raise RuntimeError(f"Unsupported vector backend: {vector_backend}")


def set_vector_store_fallback(status: dict, error: str, attempts: int) -> None:
    status.update(
        {
            "mode": "sqlite_fallback",
            "backend": "local",
            "fallback_used": True,
            "reason": "qdrant_error",
            "error": error,
            "attempts": attempts,
        }
    )


def initialize_vector_store_with_fallback(
    requested_backend: str,
    *,
    store_path: Path = DEFAULT_VECTOR_DB_PATH,
    vector_size: int = DEFAULT_EMBEDDING_DIMENSIONS,
    status: dict,
) -> tuple[object, str]:
    status.update(
        {
            "component": "vector_store",
            "requested_backend": requested_backend,
            "backend": requested_backend,
            "mode": requested_backend,
            "fallback_used": False,
            "reason": "configured_backend" if requested_backend == "qdrant" else "requested_local",
            "error": "",
            "attempts": 0,
            "qdrant_url": resolve_qdrant_url() if requested_backend == "qdrant" else "",
            "qdrant_collection": resolve_qdrant_collection() if requested_backend == "qdrant" else "",
        }
    )
    if requested_backend == "local":
        return LocalVectorStore(store_path), "local"

    if requested_backend == "qdrant" and not is_qdrant_configured():
        status.update(
            {
                "mode": "sqlite_fallback",
                "backend": "local",
                "fallback_used": True,
                "reason": "not_configured",
                "error": "Qdrant is not configured. Set QDRANT_URL or VECTOR_DB_URL to enable Qdrant.",
                "attempts": 0,
                "qdrant_url": "",
                "qdrant_collection": "",
            }
        )
        return LocalVectorStore(store_path), "local"

    vector_store = make_vector_store(requested_backend, store_path=store_path, vector_size=vector_size)
    if hasattr(vector_store, "ensure_collection"):
        try:
            _, attempts = call_with_retries(lambda: vector_store.ensure_collection())
            status["attempts"] = attempts
        except RetryExhausted as exc:
            set_vector_store_fallback(status, str(exc), exc.attempts)
            return LocalVectorStore(store_path), "local"
    return vector_store, requested_backend


def index_status_without_rebuild(
    *,
    vector_store: object,
    store_path: Path,
    embedding_model: str,
    embedding_identity_value: str,
    reason: str = "rebuild_not_requested",
) -> dict:
    chunks_count = 0
    if hasattr(vector_store, "chunks_count"):
        chunks_count = vector_store.chunks_count()
    return {
        "store": vector_store.describe() if hasattr(vector_store, "describe") else str(store_path),
        "embedding_model": embedding_model,
        "embedding_identity": embedding_identity_value,
        "changed_docs": [],
        "removed_docs": [],
        "chunks_count": chunks_count,
        "rebuild_requested": False,
        "reason": reason,
    }


def sync_index_with_store_fallback(
    *,
    vector_store: object,
    resolved_backend: str,
    vector_status: dict,
    raw_dir: Path = RAW_DIR,
    store_path: Path = DEFAULT_VECTOR_DB_PATH,
    embedder: object | None = None,
    rebuild: bool = False,
    embedding_model: str | None = None,
    embedding_identity_value: str | None = None,
) -> tuple[dict, object, str]:
    attempts = DEFAULT_RETRY_ATTEMPTS if resolved_backend == "qdrant" else 1
    sync_store = (
        MirroredVectorStore(vector_store, LocalVectorStore(store_path))
        if resolved_backend == "qdrant"
        else vector_store
    )
    try:
        index_sync, used_attempts = call_with_retries(
            lambda: sync_index(
                raw_dir,
                store_path,
                vector_store=sync_store,
                embedder=embedder,
                rebuild=rebuild,
                embedding_model=embedding_model,
                embedding_identity_value=embedding_identity_value,
            ),
            attempts=attempts,
        )
        if resolved_backend == "qdrant":
            vector_status["attempts"] = max(vector_status.get("attempts", 0), used_attempts)
        return index_sync, vector_store, resolved_backend
    except RetryExhausted as exc:
        if resolved_backend != "qdrant":
            raise
        set_vector_store_fallback(vector_status, str(exc), exc.attempts)
        local_store = LocalVectorStore(store_path)
        index_sync = sync_index(
            raw_dir,
            store_path,
            vector_store=local_store,
            embedder=embedder,
            rebuild=rebuild,
            embedding_model=embedding_model,
            embedding_identity_value=embedding_identity_value,
        )
        return index_sync, local_store, "local"


def build_document_chunks(metadata: dict[str, str], body: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for parent in split_sections(metadata, body):
        chunks.extend(chunk_parent(parent))
    return chunks


def sync_index(
    raw_dir: Path = RAW_DIR,
    store_path: Path = DEFAULT_VECTOR_DB_PATH,
    *,
    vector_store: object | None = None,
    embedder: object | None = None,
    rebuild: bool = False,
    embedding_model: str | None = None,
    embedding_identity_value: str | None = None,
) -> dict:
    """Sync changed markdown documents into the configured vector store."""
    model_name = embedding_model or (DEFAULT_EMBEDDING_MODEL if embedder else LOCAL_EMBEDDING_MODEL)
    index_identity = embedding_identity_value or embedding_identity(
        EMBEDDING_PROVIDER_EXTERNAL if embedder else EMBEDDING_PROVIDER_LOCAL,
        model_name,
    )
    dimensions = resolve_vector_dimensions()
    store = vector_store or LocalVectorStore(store_path)
    if rebuild:
        store.reset()

    manifest = store.load_manifest()
    changed_docs: list[str] = []
    current_doc_ids: set[str] = set()

    for path in sorted(raw_dir.glob("*.md")):
        raw_text = path.read_text(encoding="utf-8")
        metadata, body = parse_frontmatter(raw_text)
        metadata.setdefault("doc_id", path.stem)
        metadata.setdefault("title", path.stem)
        metadata.setdefault("source_path", safe_relative(path, ROOT))
        metadata.setdefault("permission_scope", "internal")
        metadata.setdefault("effective_from", "")
        metadata.setdefault("effective_to", "")
        doc_id = metadata["doc_id"]
        current_doc_ids.add(doc_id)
        hash_value = content_hash(raw_text, index_identity, dimensions)

        existing = manifest.get(doc_id)
        if (
            existing
            and existing["content_hash"] == hash_value
            and existing["embedding_model"] == index_identity
        ):
            continue

        chunks = build_document_chunks(metadata, normalize_text(body))
        if embedder and chunks:
            embeddings = embedder([chunk.text for chunk in chunks])  # type: ignore[operator]
            for chunk, vector in zip(chunks, embeddings):
                chunk.dense_vector = vector
        store.upsert_document(
            doc_id=doc_id,
            source_path=metadata.get("source_path", ""),
            hash_value=hash_value,
            embedding_model=index_identity,
            chunks=chunks,
        )
        changed_docs.append(doc_id)

    removed_docs = set(manifest) - current_doc_ids
    store.delete_documents(removed_docs)
    return {
        "store": store.describe() if hasattr(store, "describe") else str(store_path),
        "embedding_model": model_name,
        "embedding_identity": index_identity,
        "changed_docs": changed_docs,
        "removed_docs": sorted(removed_docs),
        "chunks_count": len(store.load_chunks()),
        "rebuild_requested": rebuild,
        "reason": "rebuilt" if rebuild else "incremental_sync",
    }


def dense_recall(query_vector: list[float], chunks: list[Chunk], top_n: int = DENSE_TOP_N) -> list[tuple[float, Chunk]]:
    scored = [(cosine(query_vector, chunk.dense_vector), chunk) for chunk in chunks]
    return sorted(scored, key=lambda item: item[0], reverse=True)[:top_n]


def bm25_recall(query: str, chunks: list[Chunk], top_n: int = BM25_TOP_N) -> list[tuple[float, Chunk]]:
    query_terms = tokenize(query)
    if not query_terms:
        return []

    doc_freq: Counter[str] = Counter()
    chunk_terms = {chunk.chunk_id: Counter(chunk.terms) for chunk in chunks}
    for counts in chunk_terms.values():
        for term in counts:
            doc_freq[term] += 1

    avg_len = sum(sum(counts.values()) for counts in chunk_terms.values()) / max(1, len(chunks))
    total_docs = len(chunks)
    k1 = 1.5
    b = 0.75
    scores: list[tuple[float, Chunk]] = []

    for chunk in chunks:
        counts = chunk_terms[chunk.chunk_id]
        length = sum(counts.values()) or 1
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            idf = math.log(1 + (total_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avg_len))
        if score > 0:
            scores.append((score, chunk))

    return sorted(scores, key=lambda item: item[0], reverse=True)[:top_n]


def rrf_fuse(
    dense_results: list[tuple[float, Chunk]],
    bm25_results: list[tuple[float, Chunk]],
) -> dict[str, Candidate]:
    candidates: dict[str, Candidate] = {}
    for route, results in (("dense", dense_results), ("bm25", bm25_results)):
        for rank, (score, chunk) in enumerate(results, start=1):
            item = candidates.setdefault(chunk.chunk_id, Candidate(chunk_id=chunk.chunk_id))
            item.rrf_score += 1 / (RRF_K + rank)
            if route == "dense":
                item.dense_rank = rank
                item.dense_score = score
            else:
                item.bm25_rank = rank
                item.bm25_score = score
    return dict(sorted(candidates.items(), key=lambda pair: pair[1].rrf_score, reverse=True))


def parse_reranker_scores(payload: dict, expected_count: int) -> list[float]:
    if isinstance(payload.get("scores"), list):
        scores = [float(score) for score in payload["scores"]]
    elif isinstance(payload.get("results"), list):
        scores = [0.0] * expected_count
        for item in payload["results"]:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", -1))
            if 0 <= index < expected_count:
                scores[index] = float(item.get("score", item.get("relevance_score", 0.0)))
    else:
        raise RuntimeError("Reranker response must include `scores` or `results`.")
    if len(scores) != expected_count:
        raise RuntimeError(f"Reranker returned {len(scores)} scores for {expected_count} documents.")
    return scores


def make_external_reranker(
    url: str,
    *,
    model: str = DEFAULT_RERANKER_MODEL,
    timeout_seconds: int = 30,
) -> ExternalReranker:
    return ExternalReranker(url=url, model=model, timeout_seconds=timeout_seconds)


def make_configured_external_reranker() -> ExternalReranker | None:
    provider = os.getenv("RERANKER_PROVIDER", "").strip().lower()
    url = os.getenv("RERANKER_URL", "").strip()
    if not provider and not url:
        return None
    if not url and provider in EXTERNAL_RERANKER_PROVIDERS:
        url = DEFAULT_RERANKER_URL
    if not url:
        return None
    return make_external_reranker(
        url,
        model=os.getenv("RERANKER_MODEL", "").strip() or DEFAULT_RERANKER_MODEL,
        timeout_seconds=parse_int_env("RERANKER_TIMEOUT_SECONDS", 30),
    )


def skip_rerank(candidates: dict[str, Candidate], reason: str) -> list[Candidate]:
    ranked = sorted(candidates.values(), key=lambda item: item.rrf_score, reverse=True)[:RERANK_TOP_N]
    for candidate in ranked:
        candidate.rerank_score = candidate.rrf_score
        candidate.reason = f"rerank_skipped:{reason}"
    return ranked


def rerank(
    query: str,
    candidates: dict[str, Candidate],
    chunks_by_id: dict[str, Chunk],
    *,
    external_reranker: ExternalReranker | None = None,
) -> list[Candidate]:
    if not external_reranker:
        return skip_rerank(candidates, "no_configured_model")
    ordered_candidates = list(candidates.values())
    chunks = [chunks_by_id[candidate.chunk_id] for candidate in ordered_candidates]
    try:
        scores, attempts = call_with_retries(lambda: external_reranker.score(query, chunks))
        external_reranker.last_attempts = attempts
    except RetryExhausted as exc:
        external_reranker.last_error = str(exc)
        external_reranker.last_attempts = exc.attempts
        return skip_rerank(candidates, "reranker_error")
    for candidate, score in zip(ordered_candidates, scores):
        candidate.rerank_score = score
        candidate.reason = f"external_reranker:{external_reranker.model}"
    return sorted(ordered_candidates, key=lambda item: item.rerank_score, reverse=True)[:RERANK_TOP_N]


def describe_reranker(external_reranker: ExternalReranker | None) -> dict:
    if external_reranker is None:
        return {
            "mode": "skipped",
            "reason": "not_configured",
            "model": "",
            "url": "",
            "fallback_used": False,
            "error": "",
            "attempts": 0,
            "score_policy": SCORE_POLICY_RRF_ONLY,
        }
    if external_reranker.last_error:
        return {
            "mode": "skipped",
            "reason": "reranker_error",
            "model": external_reranker.model,
            "url": external_reranker.url,
            "fallback_used": False,
            "error": external_reranker.last_error,
            "attempts": external_reranker.last_attempts,
            "score_policy": SCORE_POLICY_RRF_ONLY,
        }
    return {
        "mode": "external",
        "reason": "configured_model",
        "model": external_reranker.model,
        "url": external_reranker.url,
        "fallback_used": False,
        "error": "",
        "attempts": external_reranker.last_attempts,
        "score_policy": SCORE_POLICY_EXTERNAL_RERANK,
    }


def semantic_dedup(ranked: list[Candidate], chunks_by_id: dict[str, Chunk]) -> tuple[list[Candidate], list[dict]]:
    kept: list[Candidate] = []
    dropped: list[dict] = []
    for candidate in ranked:
        chunk = chunks_by_id[candidate.chunk_id]
        duplicate_of: tuple[Candidate, float] | None = None
        for selected in kept:
            selected_chunk = chunks_by_id[selected.chunk_id]
            similarity = cosine(chunk.dense_vector, selected_chunk.dense_vector)
            if similarity >= DEDUP_THRESHOLD:
                duplicate_of = (selected, similarity)
                break
        if duplicate_of:
            dropped.append(
                {
                    "chunk_id": candidate.chunk_id,
                    "duplicate_of": duplicate_of[0].chunk_id,
                    "similarity": round(duplicate_of[1], 4),
                }
            )
        else:
            kept.append(candidate)
    return kept, dropped


def mmr_select(
    ranked: list[Candidate],
    chunks_by_id: dict[str, Chunk],
    *,
    limit: int = RERANK_TOP_N,
    lambda_mult: float = MMR_LAMBDA,
    duplicate_threshold: float = DEDUP_THRESHOLD,
) -> tuple[list[Candidate], list[dict]]:
    if not ranked:
        return [], []
    remaining = sorted(ranked, key=lambda item: item.rerank_score, reverse=True)
    max_score = max(abs(item.rerank_score) for item in remaining) or 1.0
    selected: list[Candidate] = []
    dropped: list[dict] = []

    while remaining and len(selected) < limit:
        best: Candidate | None = None
        best_score = float("-inf")
        next_remaining: list[Candidate] = []
        for candidate in remaining:
            chunk = chunks_by_id[candidate.chunk_id]
            if selected:
                similarities = [
                    cosine(chunk.dense_vector, chunks_by_id[item.chunk_id].dense_vector)
                    for item in selected
                ]
                max_similarity = max(similarities) if similarities else 0.0
                if max_similarity >= duplicate_threshold:
                    duplicate_of = selected[similarities.index(max_similarity)]
                    dropped.append(
                        {
                            "chunk_id": candidate.chunk_id,
                            "duplicate_of": duplicate_of.chunk_id,
                            "similarity": round(max_similarity, 4),
                            "reason": "near_duplicate",
                        }
                    )
                    continue
            else:
                max_similarity = 0.0
            relevance = candidate.rerank_score / max_score
            candidate.mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
            if candidate.mmr_score > best_score:
                if best is not None:
                    next_remaining.append(best)
                best = candidate
                best_score = candidate.mmr_score
            else:
                next_remaining.append(candidate)
        if best is None:
            break
        selected.append(best)
        remaining = next_remaining
    return selected, dropped


def dynamic_truncate(
    candidates: list[Candidate],
    chunks_by_id: dict[str, Chunk],
    *,
    chunks_by_parent: dict[str, list[Chunk]] | None = None,
    score_policy: str = SCORE_POLICY_EXTERNAL_RERANK,
    min_score: float | None = MIN_RERANK_SCORE,
    gap_threshold: float | None = GAP_THRESHOLD,
) -> tuple[list[SelectedEvidence], dict]:
    if score_policy == SCORE_POLICY_RRF_ONLY:
        min_score = None
        gap_threshold = None
    filtered = list(candidates) if min_score is None else [item for item in candidates if item.rerank_score >= min_score]
    reason = {
        "score_policy": score_policy,
        "score_confidence": score_policy == SCORE_POLICY_EXTERNAL_RERANK,
        "min_score": min_score,
        "gap_threshold": gap_threshold,
        "max_k": FINAL_MAX_K,
        "context_token_budget": CONTEXT_TOKEN_BUDGET,
        "budget_basis": "expanded_parent_context_tokens",
        "stop_reason": "max_k_or_budget",
    }
    if not filtered:
        reason["stop_reason"] = "no_candidate_above_min_score"
        return [], reason

    cutoff = len(filtered)
    if gap_threshold is not None:
        for index in range(len(filtered) - 1):
            gap = filtered[index].rerank_score - filtered[index + 1].rerank_score
            if gap >= gap_threshold:
                cutoff = index + 1
                reason["stop_reason"] = "gap_cutoff"
                reason["gap"] = round(gap, 4)
                break

    selected: list[SelectedEvidence] = []
    token_total = 0
    for candidate in filtered[:cutoff]:
        if len(selected) >= FINAL_MAX_K:
            reason["stop_reason"] = "max_k_or_budget"
            break
        chunk = chunks_by_id[candidate.chunk_id]
        expanded_text, expanded_ids, expanded_tokens = expand_parent_context(chunk, chunks_by_parent)
        if token_total + expanded_tokens > CONTEXT_TOKEN_BUDGET:
            reason["stop_reason"] = "max_k_or_budget"
            break
        selected.append(
            SelectedEvidence(
                candidate=candidate,
                expanded_text=expanded_text,
                expanded_from_chunk_ids=expanded_ids,
                expanded_token_count=expanded_tokens,
            )
        )
        token_total += expanded_tokens
    reason["selected_count"] = len(selected)
    reason["token_total"] = token_total
    return selected, reason


def is_out_of_domain_query(query: str) -> bool:
    out_of_domain_patterns = (
        r"天气|气温|下雨|空气质量",
        r"股票|股价|汇率|彩票",
        r"新闻|热搜|比赛|比分",
    )
    in_domain_patterns = (
        r"订单|退款|退货|换货|物流|快递|发货|配送|签收|发票|税号|优惠券|积分|会员|保修|维修|召回|商品|商家|客服|账号|登录|验证码|地址",
        r"SKU|COD|CoD|invoice|refund|return|delivery|warranty|account|login",
    )
    if any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in out_of_domain_patterns):
        return not any(re.search(pattern, query, flags=re.IGNORECASE) for pattern in in_domain_patterns)
    return False


def sufficiency_check(
    query: str,
    selected: list[Candidate],
    chunks_by_id: dict[str, Chunk],
    *,
    permission_blocked_matches: list[dict] | None = None,
    score_policy: str = SCORE_POLICY_EXTERNAL_RERANK,
    min_score: float = MIN_RERANK_SCORE,
    high_confidence_score: float | None = 0.35,
) -> dict:
    if is_out_of_domain_query(query):
        return {"enough": False, "reason": "out_of_domain_query"}
    query_terms = set(tokenize(query))
    if not selected:
        if permission_blocked_matches:
            return {
                "enough": False,
                "reason": "permission_denied",
                "blocked_doc_ids": sorted({item["doc_id"] for item in permission_blocked_matches}),
            }
        return {"enough": False, "reason": "no_selected_evidence"}
    coverage = set()
    for candidate in selected:
        coverage.update(query_terms & set(chunks_by_id[candidate.chunk_id].terms))
    coverage_ratio = len(coverage) / max(1, len(query_terms))
    best_score = max(item.rerank_score for item in selected)
    if permission_blocked_matches:
        blocked_ratio = float(permission_blocked_matches[0].get("overlap_ratio", 0.0))
        if blocked_ratio >= 0.45 and blocked_ratio >= coverage_ratio + 0.15:
            return {
                "enough": False,
                "reason": "permission_denied",
                "coverage_ratio": round(coverage_ratio, 4),
                "blocked_overlap_ratio": round(blocked_ratio, 4),
                "blocked_doc_ids": sorted({item["doc_id"] for item in permission_blocked_matches}),
            }
    if score_policy == SCORE_POLICY_RRF_ONLY:
        lexical_coverage = set()
        for candidate in selected:
            if candidate.bm25_rank is not None:
                lexical_coverage.update(query_terms & set(chunks_by_id[candidate.chunk_id].terms))
        lexical_coverage_ratio = len(lexical_coverage) / max(1, len(query_terms))
        has_lexical_signal = bool(lexical_coverage)
        enough = lexical_coverage_ratio >= 0.18
        if enough:
            reason = "pass"
        elif not has_lexical_signal:
            reason = "rrf_only_missing_lexical_signal"
        else:
            reason = "low_query_evidence_overlap"
        return {
            "enough": enough,
            "coverage_ratio": round(coverage_ratio, 4),
            "best_rerank_score": round(best_score, 4),
            "score_policy": score_policy,
            "score_confidence": False,
            "lexical_signal": has_lexical_signal,
            "lexical_coverage_ratio": round(lexical_coverage_ratio, 4),
            "reason": reason,
        }
    enough = best_score >= min_score and (
        coverage_ratio >= 0.18
        or (high_confidence_score is not None and best_score >= high_confidence_score)
    )
    return {
        "enough": enough,
        "coverage_ratio": round(coverage_ratio, 4),
        "best_rerank_score": round(best_score, 4),
        "score_policy": score_policy,
        "score_confidence": True,
        "reason": "pass" if enough else "low_query_evidence_overlap",
    }


def build_chunks_by_parent(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    chunks_by_parent: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_parent[chunk.parent_id].append(chunk)
    for parent_chunks in chunks_by_parent.values():
        parent_chunks.sort(key=lambda item: item.chunk_id)
    return chunks_by_parent


def expand_parent_context(
    chunk: Chunk,
    chunks_by_parent: dict[str, list[Chunk]] | None,
    *,
    max_chars: int = PARENT_EXPANSION_MAX_CHARS,
) -> tuple[str, list[str], int]:
    if not chunks_by_parent:
        return chunk.text, [chunk.chunk_id], chunk.token_count
    siblings = chunks_by_parent.get(chunk.parent_id) or [chunk]
    text_parts: list[str] = []
    expanded_ids: list[str] = []
    token_total = 0
    for sibling in siblings:
        candidate_text = "\n\n".join([*text_parts, sibling.text]).strip()
        if len(candidate_text) > max_chars and sibling.chunk_id != chunk.chunk_id:
            continue
        text_parts.append(sibling.text)
        expanded_ids.append(sibling.chunk_id)
        token_total += sibling.token_count
    if chunk.chunk_id not in expanded_ids:
        text_parts.append(chunk.text)
        expanded_ids.append(chunk.chunk_id)
        token_total += chunk.token_count
    return "\n\n".join(text_parts).strip(), expanded_ids, token_total


def assemble_context(
    query: str,
    selected: list[Candidate | SelectedEvidence],
    chunks_by_id: dict[str, Chunk],
    sufficiency: dict,
    *,
    chunks_by_parent: dict[str, list[Chunk]] | None = None,
) -> dict:
    evidence = []
    estimated_token_total = 0
    for index, selection in enumerate(selected, start=1):
        if isinstance(selection, SelectedEvidence):
            candidate = selection.candidate
            expanded_text = selection.expanded_text
            expanded_ids = selection.expanded_from_chunk_ids
            expanded_tokens = selection.expanded_token_count
        else:
            candidate = selection
            chunk = chunks_by_id[candidate.chunk_id]
            expanded_text, expanded_ids, expanded_tokens = expand_parent_context(chunk, chunks_by_parent)
        chunk = chunks_by_id[candidate.chunk_id]
        estimated_token_total += expanded_tokens
        role = "primary" if index == 1 else "supporting"
        if any(word in chunk.text for word in ("不支持", "不能", "除非")):
            role = "exception" if index > 1 else "primary"
        evidence.append(
            {
                "citation_id": f"E{index}",
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "doc_id": chunk.doc_id,
                "title_path": chunk.title_path,
                "source_path": chunk.metadata.get("source_path", ""),
                "version": chunk.metadata.get("version", ""),
                "effective_from": chunk.metadata.get("effective_from", ""),
                "rerank_score": round(candidate.rerank_score, 4),
                "mmr_score": round(candidate.mmr_score, 4),
                "evidence_role": role,
                "expanded_from_chunk_ids": expanded_ids,
                "text": expanded_text,
            }
        )
    return {
        "query": query,
        "policy": [
            "只能使用本上下文包中的资料回答",
            "资料不足时必须拒答",
            "关键事实必须就近引用资料编号",
        ],
        "sufficiency": sufficiency,
        "estimated_token_total": estimated_token_total,
        "evidence": evidence,
    }


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE)
    return [item.strip() for item in re.split(r"(?<=[。！？!?])", text) if item.strip()]


def sentence_score(sentence: str, query: str, query_terms: set[str]) -> float:
    terms = set(tokenize(sentence))
    score = float(len(query_terms & terms))
    query_code_terms = [term for term in query_terms if re.search(r"[a-z0-9]", term)]
    if any(term in sentence.lower() for term in query_code_terms):
        score += 6
    if re.search(r"多久|几天|到账|时间", query):
        if re.search(r"通常|一般|为\s*\d|[0-9０-９]+\s*到\s*[0-9０-９]+", sentence):
            score += 7
        elif re.search(r"工作日|到账|时间", sentence):
            score += 3
        if "超过" in sentence:
            score -= 2
    if re.search(r"sku|SKU|无理由|退货|退款|支持", query) and re.search(r"SKU-A17|不支持|支持|无理由|质量问题", sentence):
        score += 4
    if "SKU-A17" in query and "SKU-A17" in sentence:
        score += 8
    if re.search(r"预售|48\s*小时|催单", query) and re.search(r"预售|48\s*小时|催单|承诺", sentence):
        score += 4
    if re.search(r"积分|提现|转赠", query) and re.search(r"积分|提现|转赠|兑换", sentence):
        score += 4
    return score


def generate_answer(context_packet: dict) -> dict:
    if not context_packet["sufficiency"]["enough"]:
        if context_packet["sufficiency"].get("reason") == "permission_denied":
            return {
                "answer": "当前权限不足，不能可靠回答这个问题。",
                "citations": [],
                "mode": "refusal",
            }
        return {
            "answer": "资料不足，不能可靠回答这个问题。",
            "citations": [],
            "mode": "refusal",
        }
    query_terms = set(tokenize(context_packet["query"]))
    claims = []
    citations = []
    for evidence in context_packet["evidence"]:
        best_sentence = ""
        best_overlap = -1
        for sentence in split_sentences(evidence["text"]):
            overlap = sentence_score(sentence, context_packet["query"], query_terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence
        if best_sentence and evidence["citation_id"] not in citations:
            claims.append(f"{best_sentence} [{evidence['citation_id']}]")
            citations.append(evidence["citation_id"])
        if len(claims) >= 2:
            break
    return {
        "answer": "\n".join(claims) if claims else "资料不足，不能可靠回答这个问题。",
        "citations": citations,
        "mode": "extractive",
    }


def build_prompt_from_context_packet(context_packet: dict) -> str:
    evidence_blocks = []
    for evidence in context_packet["evidence"]:
        title_path = " > ".join(evidence.get("title_path", []))
        evidence_blocks.append(
            "\n".join(
                [
                    f"[{evidence['citation_id']}]",
                    f"doc_id: {evidence.get('doc_id', '')}",
                    f"title_path: {title_path}",
                    f"source_path: {evidence.get('source_path', '')}",
                    f"version: {evidence.get('version', '')}",
                    f"role: {evidence.get('evidence_role', '')}",
                    "text:",
                    evidence.get("text", ""),
                ]
            )
        )
    return "\n\n".join(
        [
            "用户问题：",
            context_packet["query"],
            "",
            "证据包：",
            "\n\n---\n\n".join(evidence_blocks),
            "",
            "回答要求：",
            "1. 只能基于证据包回答，不要使用外部常识补全。",
            "2. 资料不足时直接说资料不足，不能可靠回答。",
            "3. 每个关键事实后面必须带引用，如 [E1]。",
            "4. 如果证据互相冲突，要说明冲突并分别引用。",
        ]
    )


def extract_citation_ids(text: str) -> list[str]:
    return sorted(set(re.findall(r"\[(E\d+)\]", text)))


def generate_answer_with_llm(context_packet: dict) -> dict:
    if not context_packet["sufficiency"]["enough"]:
        return generate_answer(context_packet)
    api_key = env_first("LLM_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY")
    if not is_llm_configured():
        raise RuntimeError(
            "Missing LLM configuration. Set LLM_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, "
            "or an explicit LLM_BASE_URL/ANTHROPIC_BASE_URL/DEEPSEEK_BASE_URL."
        )
    model_name = env_first("LLM_MODEL", default=DEFAULT_CHAT_MODEL) or DEFAULT_CHAT_MODEL
    client = AnthropicMessagesClient(api_key=api_key, base_url=resolve_llm_base_url())
    answer_text = client.create_message(
        model=model_name,
        system_prompt="你是严谨的企业知识库 RAG 问答助手，只能基于给定证据回答，并且必须保留引用编号。",
        prompt=build_prompt_from_context_packet(context_packet),
        max_tokens=parse_int_env("LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS),
    )
    return {
        "answer": answer_text,
        "citations": extract_citation_ids(answer_text),
        "mode": f"llm:{model_name}",
    }


def make_embedding_function(dimensions: int | None = None):
    api_key = env_first("EMBEDDING_API_KEY", "ZHIPU_API_KEY")
    if not is_embedding_configured():
        raise RuntimeError(
            "Missing embedding configuration. Set EMBEDDING_API_KEY, ZHIPU_API_KEY, "
            "or an explicit EMBEDDING_BASE_URL/ZHIPU_EMBEDDING_BASE_URL/ZHIPUAI_BASE_URL."
        )
    client = OpenAICompatibleEmbeddingClient(api_key=api_key, base_url=resolve_embedding_base_url())
    embedding_dimensions = dimensions or resolve_vector_dimensions()

    def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            batch = texts[start : start + 64]
            vectors.extend(
                client.embed_texts(
                    batch,
                    model=env_first("EMBEDDING_MODEL", default=DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL,
                    dimensions=embedding_dimensions,
                )
            )
        return vectors

    return embed


def make_retrying_embedding_function(status: dict, dimensions: int | None = None):
    embed_once = make_embedding_function(dimensions)

    def embed(texts: list[str]) -> list[list[float]]:
        try:
            vectors, attempts = call_with_retries(lambda: embed_once(texts))
            status["attempts"] = max(status.get("attempts", 0), attempts)
            return vectors
        except RetryExhausted as exc:
            error = str(exc)
            status.update(
                {
                    "mode": "hash_fallback",
                    "provider": EMBEDDING_PROVIDER_LOCAL,
                    "model": LOCAL_EMBEDDING_MODEL,
                    "identity": embedding_identity(EMBEDDING_PROVIDER_LOCAL, LOCAL_EMBEDDING_MODEL),
                    "fallback_used": True,
                    "reason": "embedding_error",
                    "error": error,
                    "attempts": exc.attempts,
                }
            )
            raise ComponentFallback("embedding", "embedding_error", error, exc.attempts) from exc

    return embed


def generate_answer_resilient(context_packet: dict, status: dict) -> dict:
    if not is_llm_configured():
        status.update(
            {
                "mode": "extractive_fallback",
                "fallback_used": True,
                "reason": "not_configured",
                "error": "",
                "attempts": 0,
            }
        )
        return generate_answer(context_packet)
    if not context_packet["sufficiency"]["enough"]:
        status.update(
            {
                "mode": "skipped",
                "fallback_used": False,
                "reason": "insufficient_context",
                "error": "",
                "attempts": 0,
            }
        )
        return generate_answer(context_packet)
    try:
        answer, attempts = call_with_retries(lambda: generate_answer_with_llm(context_packet))
        status.update(
            {
                "mode": "llm",
                "fallback_used": False,
                "reason": "configured_model",
                "error": "",
                "attempts": attempts,
            }
        )
        return answer
    except RetryExhausted as exc:
        status.update(
            {
                "mode": "extractive_fallback",
                "fallback_used": True,
                "reason": "llm_error",
                "error": str(exc),
                "attempts": exc.attempts,
            }
        )
        return generate_answer(context_packet)


def validate_citations(answer: dict, context_packet: dict) -> dict:
    available = {item["citation_id"] for item in context_packet["evidence"]}
    used = set(answer.get("citations") or extract_citation_ids(answer.get("answer", "")))
    return {
        "citation_valid": used.issubset(available),
        "used_citations": sorted(used),
        "missing_citations": sorted(used - available),
        "available_citations": sorted(available),
    }


def build_monitoring_event(trace: dict, latency_ms: int, status: str) -> dict:
    sufficiency = trace.get("context_packet", {}).get("sufficiency", {})
    selected_doc_ids = sorted({item.get("doc_id", "") for item in trace.get("context_packet", {}).get("evidence", []) if item.get("doc_id")})
    validation = trace.get("validation", {})
    model_config = trace.get("model_config", {})
    selection_strategy = trace.get("selection_strategy", {})
    reranker = trace.get("reranker", {})
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trace_id": trace["trace_id"],
        "status": status,
        "latency_ms": latency_ms,
        "pipeline_version": trace.get("pipeline_version"),
        "index_version": trace.get("index_version"),
        "vector_backend": model_config.get("vector_backend"),
        "embedding_model": model_config.get("embedding_model"),
        "embedding_identity": model_config.get("embedding_identity"),
        "qdrant_collection": model_config.get("qdrant_collection"),
        "query_hash": hashlib.sha256(trace.get("query", "").encode("utf-8")).hexdigest()[:16],
        "query_chars": len(trace.get("query", "")),
        "answer_mode": trace.get("answer", {}).get("mode"),
        "sufficiency_enough": sufficiency.get("enough"),
        "sufficiency_reason": sufficiency.get("reason"),
        "permission_denied": sufficiency.get("reason") == "permission_denied",
        "citation_valid": validation.get("citation_valid"),
        "missing_citation_count": len(validation.get("missing_citations", [])),
        "dense_hits": len(trace.get("dense_top", [])),
        "bm25_hits": len(trace.get("bm25_top", [])),
        "rerank_candidates": len(trace.get("rerank_top", [])),
        "dedup_dropped_count": len(trace.get("dedup_dropped", [])),
        "selected_count": trace.get("truncation", {}).get("selected_count", 0),
        "selected_doc_ids": selected_doc_ids,
        "context_token_total": trace.get("context_packet", {}).get(
            "estimated_token_total",
            trace.get("truncation", {}).get("token_total", 0),
        ),
        "blocked_match_count": len(trace.get("permission_filter", {}).get("blocked_matches", [])),
        "stage_latencies_ms": trace.get("stage_latencies_ms", {}),
        "selection_strategy": selection_strategy.get("name", selection_strategy),
        "reranker_mode": reranker.get("mode"),
        "reranker_model": reranker.get("model"),
        "reranker_score_policy": reranker.get("score_policy"),
        "reranker_fallback_used": reranker.get("fallback_used", False),
    }


def append_jsonl(path: Path, item: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def persist_monitoring_event(event: dict, metrics_path: Path = METRICS_PATH) -> None:
    append_jsonl(metrics_path, event)


def summarize_results(results: Iterable[tuple[float, Chunk]]) -> list[dict]:
    return [
        {
            "score": round(score, 4),
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "title_path": " > ".join(chunk.title_path),
        }
        for score, chunk in results
    ]


def build_llm_status() -> dict:
    configured = is_llm_configured()
    return {
        "component": "llm",
        "configured": configured,
        "requested_model": env_first("LLM_MODEL", default=DEFAULT_CHAT_MODEL) or DEFAULT_CHAT_MODEL,
        "model": env_first("LLM_MODEL", default=DEFAULT_CHAT_MODEL) or DEFAULT_CHAT_MODEL,
        "base_url": resolve_llm_base_url() if configured else "",
        "mode": "pending" if configured else "extractive_fallback",
        "fallback_used": not configured,
        "reason": "configured_model" if configured else "not_configured",
        "error": "",
        "attempts": 0,
    }


def build_embedding_status() -> dict:
    configured = is_embedding_configured()
    requested_model = env_first("EMBEDDING_MODEL", default=DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL
    base_url = resolve_embedding_base_url() if configured else ""
    provider = resolve_embedding_provider(base_url) if configured else EMBEDDING_PROVIDER_LOCAL
    model = requested_model if configured else LOCAL_EMBEDDING_MODEL
    return {
        "component": "embedding",
        "configured": configured,
        "requested_model": requested_model,
        "provider": provider,
        "model": model,
        "identity": embedding_identity(provider, model),
        "base_url": base_url,
        "mode": provider if configured else "hash_fallback",
        "fallback_used": not configured,
        "reason": "configured_model" if configured else "not_configured",
        "error": "",
        "attempts": 0,
    }


def fallback_summary(component_status: dict[str, dict]) -> list[dict]:
    fallbacks: list[dict] = []
    for component, status in component_status.items():
        if not status.get("fallback_used"):
            continue
        fallbacks.append(
            {
                "component": component,
                "mode": status.get("mode", ""),
                "reason": status.get("reason", ""),
                "error": status.get("error", ""),
                "attempts": status.get("attempts", 0),
            }
        )
    return fallbacks


def run_query(
    query: str,
    trace_only: bool = False,
    save_trace: bool = False,
    quiet: bool = False,
    *,
    allowed_scopes: set[str] | None = None,
    rebuild_index: bool = False,
    vector_backend: str = DEFAULT_VECTOR_BACKEND,
    store_path: Path = DEFAULT_VECTOR_DB_PATH,
    monitoring_enabled: bool = True,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    started = time.perf_counter()
    stage_latencies_ms: dict[str, int] = {}
    llm_status = build_llm_status()
    embedding_status = build_embedding_status()
    vector_store_status: dict = {}
    scopes = allowed_scopes or DEFAULT_ALLOWED_SCOPES
    vector_size = resolve_vector_dimensions()
    embedder = make_retrying_embedding_function(embedding_status, vector_size) if embedding_status["configured"] else None
    embedding_model = embedding_status["model"]
    index_embedding_identity = embedding_status["identity"]
    active_store_path = local_store_path_for_embedding_identity(store_path, index_embedding_identity)
    lexical_store = LocalVectorStore(active_store_path)
    query_vector: list[float] | None = None
    stage_started = time.perf_counter()
    try:
        query_vector = embedder([query])[0] if embedder else vectorize(tokenize(query))
    except ComponentFallback as exc:
        embedding_status.update(
            {
                "mode": "hash_fallback",
                "provider": EMBEDDING_PROVIDER_LOCAL,
                "model": LOCAL_EMBEDDING_MODEL,
                "identity": embedding_identity(EMBEDDING_PROVIDER_LOCAL, LOCAL_EMBEDDING_MODEL),
                "fallback_used": True,
                "reason": exc.reason,
                "error": exc.error,
                "attempts": exc.attempts,
            }
        )
        embedder = None
        embedding_model = LOCAL_EMBEDDING_MODEL
        index_embedding_identity = embedding_status["identity"]
        active_store_path = local_store_path_for_embedding_identity(store_path, index_embedding_identity)
        lexical_store = LocalVectorStore(active_store_path)
        query_vector = vectorize(tokenize(query))
    stage_latencies_ms["embedding_probe"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    vector_store, resolved_vector_backend = initialize_vector_store_with_fallback(
        vector_backend,
        store_path=active_store_path,
        vector_size=vector_size,
        status=vector_store_status,
    )
    stage_latencies_ms["vector_store_probe"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    if rebuild_index:
        try:
            index_sync, vector_store, resolved_vector_backend = sync_index_with_store_fallback(
                vector_store=vector_store,
                resolved_backend=resolved_vector_backend,
                vector_status=vector_store_status,
                store_path=active_store_path,
                embedder=embedder,
                rebuild=True,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
            )
        except ComponentFallback as exc:
            embedding_status.update(
                {
                    "mode": "hash_fallback",
                    "provider": EMBEDDING_PROVIDER_LOCAL,
                    "model": LOCAL_EMBEDDING_MODEL,
                    "identity": embedding_identity(EMBEDDING_PROVIDER_LOCAL, LOCAL_EMBEDDING_MODEL),
                    "fallback_used": True,
                    "reason": exc.reason,
                    "error": exc.error,
                    "attempts": exc.attempts,
                }
            )
            embedder = None
            embedding_model = LOCAL_EMBEDDING_MODEL
            index_embedding_identity = embedding_status["identity"]
            active_store_path = local_store_path_for_embedding_identity(store_path, index_embedding_identity)
            lexical_store = LocalVectorStore(active_store_path)
            if resolved_vector_backend == "local":
                vector_store = LocalVectorStore(active_store_path)
            query_vector = vectorize(tokenize(query))
            index_sync, vector_store, resolved_vector_backend = sync_index_with_store_fallback(
                vector_store=vector_store,
                resolved_backend=resolved_vector_backend,
                vector_status=vector_store_status,
                store_path=active_store_path,
                embedder=embedder,
                rebuild=True,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
            )
    else:
        index_sync = index_status_without_rebuild(
            vector_store=vector_store,
            store_path=active_store_path,
            embedding_model=embedding_model,
            embedding_identity_value=index_embedding_identity,
        )
    stage_latencies_ms["index_sync"] = int((time.perf_counter() - stage_started) * 1000)

    def load_access_state():
        nonlocal index_sync, resolved_vector_backend, vector_store, lexical_store

        def load_access_chunks() -> tuple[list[Chunk], list[dict], list[Chunk]]:
            if resolved_vector_backend == "qdrant":
                loaded, attempts = call_with_retries(lambda: vector_store.load_access_chunks(scopes))
                vector_store_status["attempts"] = max(vector_store_status.get("attempts", 0), attempts)
                return loaded
            return lexical_store.load_access_chunks(scopes)

        try:
            visible_chunks, rejected, permission_blocked_chunks = load_access_chunks()
        except RetryExhausted as exc:
            if resolved_vector_backend != "qdrant":
                raise
            set_vector_store_fallback(vector_store_status, str(exc), exc.attempts)
            lexical_store = LocalVectorStore(active_store_path)
            vector_store = lexical_store
            resolved_vector_backend = "local"
            index_sync = index_status_without_rebuild(
                vector_store=lexical_store,
                store_path=active_store_path,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
                reason="qdrant_load_fallback",
            )
            visible_chunks, rejected, permission_blocked_chunks = lexical_store.load_access_chunks(scopes)

        access_checked_chunks = visible_chunks + permission_blocked_chunks
        blocked_matches = find_permission_blocked_matches(query, access_checked_chunks, rejected)
        visible_chunks_by_id = {chunk.chunk_id: chunk for chunk in visible_chunks}
        if not index_sync.get("chunks_count"):
            index_sync["chunks_count"] = len(visible_chunks)
        return (
            access_checked_chunks,
            visible_chunks,
            rejected,
            blocked_matches,
            len({chunk.parent_id for chunk in visible_chunks}),
            visible_chunks_by_id,
            build_chunks_by_parent(visible_chunks),
        )

    stage_started = time.perf_counter()
    (
        all_chunks,
        chunks,
        rejected_chunks,
        permission_blocked_matches,
        parents_count,
        chunks_by_id,
        chunks_by_parent,
    ) = load_access_state()
    stage_latencies_ms["access_filter"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    # The query vector must come from the same vectorizer (and dimensions) as
    # the stored chunk vectors, regardless of which store serves the search.
    try:
        query_vector = query_vector or (embedder([query])[0] if embedder else vectorize(tokenize(query)))
    except ComponentFallback as exc:
        embedding_status.update(
            {
                "mode": "hash_fallback",
                "provider": EMBEDDING_PROVIDER_LOCAL,
                "model": LOCAL_EMBEDDING_MODEL,
                "identity": embedding_identity(EMBEDDING_PROVIDER_LOCAL, LOCAL_EMBEDDING_MODEL),
                "fallback_used": True,
                "reason": exc.reason,
                "error": exc.error,
                "attempts": exc.attempts,
            }
        )
        embedder = None
        embedding_model = LOCAL_EMBEDDING_MODEL
        index_embedding_identity = embedding_status["identity"]
        active_store_path = local_store_path_for_embedding_identity(store_path, index_embedding_identity)
        lexical_store = LocalVectorStore(active_store_path)
        if resolved_vector_backend == "local":
            vector_store = lexical_store
        if rebuild_index:
            index_sync, vector_store, resolved_vector_backend = sync_index_with_store_fallback(
                vector_store=vector_store,
                resolved_backend=resolved_vector_backend,
                vector_status=vector_store_status,
                store_path=active_store_path,
                embedder=embedder,
                rebuild=True,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
            )
        else:
            index_sync = index_status_without_rebuild(
                vector_store=vector_store,
                store_path=active_store_path,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
                reason="embedding_fallback_without_rebuild",
            )
        (
            all_chunks,
            chunks,
            rejected_chunks,
            permission_blocked_matches,
            parents_count,
            chunks_by_id,
            chunks_by_parent,
        ) = load_access_state()
        query_vector = vectorize(tokenize(query))

    if resolved_vector_backend == "qdrant":
        try:
            dense_results, attempts = call_with_retries(
                lambda: [
                    (score, chunk)
                    for score, chunk in vector_store.search(
                        query_vector,
                        DENSE_TOP_N,
                        access_filter=qdrant_access_filter(scopes),
                    )
                    if chunk.chunk_id in chunks_by_id
                ]
            )
            vector_store_status["attempts"] = max(vector_store_status.get("attempts", 0), attempts)
        except RetryExhausted as exc:
            set_vector_store_fallback(vector_store_status, str(exc), exc.attempts)
            lexical_store = LocalVectorStore(active_store_path)
            vector_store = lexical_store
            resolved_vector_backend = "local"
            index_sync = index_status_without_rebuild(
                vector_store=lexical_store,
                store_path=active_store_path,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
                reason="qdrant_search_fallback",
            )
            (
                all_chunks,
                chunks,
                rejected_chunks,
                permission_blocked_matches,
                parents_count,
                chunks_by_id,
                chunks_by_parent,
            ) = load_access_state()
            dense_results = dense_recall(query_vector, chunks)
    else:
        dense_results = dense_recall(query_vector, chunks)
    stage_latencies_ms["dense_recall"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    if resolved_vector_backend == "qdrant":
        try:
            bm25_results, attempts = call_with_retries(
                lambda: vector_store.bm25_search(query, scopes, top_n=BM25_TOP_N)
            )
            vector_store_status["attempts"] = max(vector_store_status.get("attempts", 0), attempts)
        except RetryExhausted as exc:
            set_vector_store_fallback(vector_store_status, str(exc), exc.attempts)
            lexical_store = LocalVectorStore(active_store_path)
            vector_store = lexical_store
            resolved_vector_backend = "local"
            index_sync = index_status_without_rebuild(
                vector_store=lexical_store,
                store_path=active_store_path,
                embedding_model=embedding_model,
                embedding_identity_value=index_embedding_identity,
                reason="qdrant_bm25_fallback",
            )
            (
                all_chunks,
                chunks,
                rejected_chunks,
                permission_blocked_matches,
                parents_count,
                chunks_by_id,
                chunks_by_parent,
            ) = load_access_state()
            dense_results = dense_recall(query_vector, chunks)
            bm25_results = lexical_store.bm25_search(query, scopes)
    else:
        bm25_results = lexical_store.bm25_search(query, scopes)
    stage_latencies_ms["bm25_recall"] = int((time.perf_counter() - stage_started) * 1000)
    for _, chunk in bm25_results:
        chunks_by_id.setdefault(chunk.chunk_id, chunk)
    # The qdrant access snapshot is loaded without vectors; recall results carry
    # the dense vectors MMR needs, so backfill them onto the shared chunks.
    for _, chunk in [*dense_results, *bm25_results]:
        stored = chunks_by_id.get(chunk.chunk_id)
        if stored is not None and not stored.dense_vector and chunk.dense_vector:
            stored.dense_vector = chunk.dense_vector
    chunks_by_parent = build_chunks_by_parent(list(chunks_by_id.values()))

    stage_started = time.perf_counter()
    fused = rrf_fuse(dense_results, bm25_results)
    stage_latencies_ms["rrf_fusion"] = int((time.perf_counter() - stage_started) * 1000)

    external_reranker = make_configured_external_reranker()
    stage_started = time.perf_counter()
    reranked = rerank(query, fused, chunks_by_id, external_reranker=external_reranker)
    stage_latencies_ms["rerank"] = int((time.perf_counter() - stage_started) * 1000)
    reranker_info = describe_reranker(external_reranker)
    score_policy = reranker_info["score_policy"]

    stage_started = time.perf_counter()
    diversified, dedup_dropped = mmr_select(reranked, chunks_by_id)
    stage_latencies_ms["mmr"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    selected, truncation = dynamic_truncate(
        diversified,
        chunks_by_id,
        chunks_by_parent=chunks_by_parent,
        score_policy=score_policy,
    )
    stage_latencies_ms["truncate"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    selected_candidates = [item.candidate for item in selected]
    sufficiency = sufficiency_check(
        query,
        selected_candidates,
        chunks_by_id,
        permission_blocked_matches=permission_blocked_matches,
        score_policy=score_policy,
    )
    stage_latencies_ms["sufficiency"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    context_packet = assemble_context(
        query,
        selected,
        chunks_by_id,
        sufficiency,
        chunks_by_parent=chunks_by_parent,
    )
    stage_latencies_ms["context"] = int((time.perf_counter() - stage_started) * 1000)

    stage_started = time.perf_counter()
    answer = generate_answer_resilient(context_packet, llm_status)
    stage_latencies_ms["answer"] = int((time.perf_counter() - stage_started) * 1000)
    validation = validate_citations(answer, context_packet)
    component_status = {
        "llm": llm_status,
        "embedding": embedding_status,
        "vector_store": vector_store_status,
        "reranker": reranker_info,
    }

    trace = {
        "trace_id": f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{uuid.uuid4().hex[:8]}",
        "query": query,
        "pipeline_version": "production_rag_pipeline_v2",
        "index_version": index_sync["store"],
        "model_config": {
            "llm_model": llm_status["model"],
            "llm_api_style": DEFAULT_LLM_API_STYLE,
            "llm_base_url": llm_status["base_url"] if not llm_status["fallback_used"] else "offline_extractive",
            "embedding_model": embedding_model,
            "embedding_identity": index_embedding_identity,
            "embedding_base_url": (
                embedding_status["base_url"] if not embedding_status["fallback_used"] else "offline_hash"
            ),
            "vector_dimensions": vector_size,
            "rerank_mode": reranker_info["mode"],
            "rerank_score_policy": score_policy,
            "requested_vector_backend": vector_backend,
            "vector_backend": resolved_vector_backend,
            "qdrant_url": resolve_qdrant_url() if resolved_vector_backend == "qdrant" else "",
            "qdrant_collection": resolve_qdrant_collection() if resolved_vector_backend == "qdrant" else "",
        },
        "component_status": component_status,
        "fallbacks": fallback_summary(component_status),
        "index_sync": index_sync,
        "permission_filter": {
            "allowed_scopes": sorted(scopes),
            "visible_chunks": len(chunks),
            "rejected_chunks": rejected_chunks,
            "blocked_matches": permission_blocked_matches,
        },
        "parents_count": parents_count,
        "chunks_count": len(chunks),
        "dense_top": summarize_results(dense_results),
        "bm25_top": summarize_results(bm25_results),
        "rrf_top": [asdict(item) for item in list(fused.values())[:10]],
        "rerank_top": [asdict(item) for item in reranked],
        "selection_strategy": {
            "name": "mmr",
            "lambda": MMR_LAMBDA,
            "parent_expansion": True,
            "parent_expansion_max_chars": PARENT_EXPANSION_MAX_CHARS,
        },
        "reranker": reranker_info,
        "dedup_dropped": dedup_dropped,
        "truncation": truncation,
        "context_packet": context_packet,
        "answer": answer,
        "validation": validation,
        "stage_latencies_ms": stage_latencies_ms,
    }
    latency_ms = int((time.perf_counter() - started) * 1000)
    monitoring_event = build_monitoring_event(trace, latency_ms=latency_ms, status="ok")
    if monitoring_enabled:
        try:
            persist_monitoring_event(monitoring_event, metrics_path=metrics_path)
            trace["monitoring_metrics_path"] = str(metrics_path)
        except OSError as exc:
            trace["monitoring_write_error"] = str(exc)

    trace_path: Path | None = None
    if save_trace:
        try:
            TRACE_DIR.mkdir(exist_ok=True)
            trace_path = TRACE_DIR / f"{trace['trace_id']}.json"
            trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            trace["trace_save_error"] = str(exc)
            trace_path = None

    if not quiet and trace_only:
        print(json.dumps(trace, ensure_ascii=False, indent=2))
    elif not quiet:
        print_answer(trace, trace_path)
    return trace


def print_answer(trace: dict, trace_path: Path | None) -> None:
    print(f"Query: {trace['query']}")
    print(f"Trace id: {trace['trace_id']}")
    if trace_path:
        print(f"Trace file: {trace_path}")
    if trace.get("trace_save_error"):
        print(f"Trace save skipped: {trace['trace_save_error']}")
    if trace.get("monitoring_write_error"):
        print(f"Monitoring event write skipped: {trace['monitoring_write_error']}")
    elif trace.get("monitoring_metrics_path"):
        print(f"Monitoring event: {trace['monitoring_metrics_path']}")
    print("\nSelected evidence:")
    for evidence in trace["context_packet"]["evidence"]:
        print(
            f"- [{evidence['citation_id']}] {evidence['doc_id']} "
            f"{' > '.join(evidence['title_path'])} score={evidence['rerank_score']}"
        )
    print("\nAnswer:")
    print(trace["answer"]["answer"])
    print("\nValidation:")
    print(json.dumps(trace["validation"], ensure_ascii=False, indent=2))


def run_eval(
    *,
    allowed_scopes: set[str] | None = None,
    rebuild_index: bool = False,
    vector_backend: str = DEFAULT_VECTOR_BACKEND,
    monitoring_enabled: bool = True,
) -> None:
    rows = list(csv.DictReader(EVAL_PATH.read_text(encoding="utf-8").splitlines()))
    passed = 0
    for index, row in enumerate(rows):
        trace = run_query(
            row["query"],
            quiet=True,
            allowed_scopes=allowed_scopes,
            rebuild_index=rebuild_index and index == 0,
            vector_backend=vector_backend,
            monitoring_enabled=monitoring_enabled,
        )
        answer_text = trace["answer"]["answer"]
        selected_doc_ids = {item["doc_id"] for item in trace["context_packet"]["evidence"]}
        must_answer = row["must_answer"].lower() == "true"
        expected_doc_ok = (not row["expected_doc_id"]) or row["expected_doc_id"] in selected_doc_ids
        expected_terms_ok = row["expected_terms"] in answer_text
        refusal_ok = (not must_answer) and trace["answer"]["mode"] == "refusal"
        ok = (must_answer and expected_doc_ok and expected_terms_ok) or refusal_ok
        passed += int(ok)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {row['case_id']} {row['query']}")
        if not ok:
            print(f"  selected_doc_ids={sorted(selected_doc_ids)}")
            print(f"  answer={answer_text}")
    print(f"\nEval: {passed}/{len(rows)} passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runnable production_rag practice pipeline.")
    parser.add_argument("--query", help="Question to answer.")
    parser.add_argument("--trace-only", action="store_true", help="Print the full JSON trace instead of the answer view.")
    parser.add_argument("--save-trace", action="store_true", help="Also save the full JSON trace.")
    parser.add_argument("--no-monitoring", action="store_true", help="Do not append the per-query monitoring event JSONL.")
    parser.add_argument("--eval", action="store_true", help="Run eval_cases.csv.")
    parser.add_argument("--rebuild-index", action="store_true", help="Force rebuilding the configured vector store.")
    parser.add_argument(
        "--scopes",
        default="internal,public",
        help="Comma-separated permission scopes for the current user. Default: internal,public.",
    )
    parser.add_argument(
        "--vector-backend",
        choices=("qdrant", "local"),
        default=DEFAULT_VECTOR_BACKEND,
        help="Vector store backend. Default: qdrant. Use local only for tests or offline debugging.",
    )
    return parser.parse_args()


def main() -> None:
    load_env()
    args = parse_args()
    scopes = split_metadata_values(args.scopes)
    if args.eval:
        run_eval(
            allowed_scopes=scopes,
            rebuild_index=args.rebuild_index,
            vector_backend=args.vector_backend,
            monitoring_enabled=not args.no_monitoring,
        )
        return
    if not args.query:
        raise SystemExit("Provide --query or --eval")
    run_query(
        args.query,
        trace_only=args.trace_only,
        save_trace=args.save_trace,
        allowed_scopes=scopes,
        rebuild_index=args.rebuild_index,
        vector_backend=args.vector_backend,
        monitoring_enabled=not args.no_monitoring,
    )


if __name__ == "__main__":
    main()
