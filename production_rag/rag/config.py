from __future__ import annotations

import hashlib
import os
import re
import urllib.parse
from pathlib import Path


# config.py lives one directory below the project root, so resolve two parents
# to keep ROOT pointing at production_rag/ (where .env, data/, eval_cases.csv live).
ROOT = Path(__file__).resolve().parent.parent
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


def is_qdrant_cloud_url(url: str | None = None) -> bool:
    resolved_url = (url if url is not None else resolve_qdrant_url()).strip()
    url_for_parse = resolved_url if "://" in resolved_url else f"https://{resolved_url}"
    parsed = urllib.parse.urlparse(url_for_parse)
    hostname = (parsed.hostname or "").lower()
    return hostname.endswith(".cloud.qdrant.io")


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


def embedding_identity(provider: str, model: str) -> str:
    return f"{provider.strip().lower()}:{model.strip()}"


def safe_path_slug(value: str, *, max_length: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return (slug or "embedding")[:max_length]


def local_store_path_for_embedding_identity(store_path: Path, identity: str) -> Path:
    slug = safe_path_slug(identity)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return store_path.with_name(f"{store_path.stem}.{slug}.{digest}{store_path.suffix}")
