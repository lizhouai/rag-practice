from __future__ import annotations

from dataclasses import dataclass

from rag.config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_PROVIDER_LOCAL, LOCAL_EMBEDDING_MODEL
from rag.config import embedding_identity, env_first, is_embedding_configured, resolve_embedding_base_url
from rag.config import resolve_embedding_provider, resolve_vector_dimensions
from rag.http import ComponentFallback, RetryExhausted, call_with_retries, post_json


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
