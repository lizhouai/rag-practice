from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ALIASES = {
    "": DEFAULT_MODEL,
    "bge-reranker-v2-m3": DEFAULT_MODEL,
    "BAAI/bge-reranker-v2-m3": DEFAULT_MODEL,
}


class RerankerDependencyError(RuntimeError):
    pass


class RerankerModelLoadError(RuntimeError):
    pass


def is_tokenizer_compatibility_error(exc: AttributeError) -> bool:
    return "prepare_for_model" in str(exc)


def parse_bool(value: str, default: bool = False) -> bool:
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_model_name(model: str | None) -> str:
    return MODEL_ALIASES.get((model or "").strip(), (model or DEFAULT_MODEL).strip())


def parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip().strip('"').strip("'")
    if not key:
        return None
    return key, value


def load_dotenv(env_path: Path = PROJECT_ROOT / ".env") -> dict[str, str]:
    if not env_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if parsed is None:
            continue
        key, value = parsed
        if key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


def resolve_model_reference(model: str | None, local_model_dir: str | None = None) -> str:
    local_dir = (local_model_dir or "").strip()
    if local_dir:
        return local_dir
    return resolve_model_name(model)


def format_model_load_error(model_reference: str, exc: Exception) -> str:
    if model_reference in {DEFAULT_MODEL, "bge-reranker-v2-m3"}:
        model_dir_hint = "No RERANKER_MODEL_DIR was detected, so the service tried to download the remote model."
    else:
        model_dir_hint = "RERANKER_MODEL_DIR was detected, so the service tried to load this local path."
    return "\n".join(
        [
            f"Failed to load reranker model `{model_reference}`.",
            "The service needs the BGE reranker files before it can start.",
            model_dir_hint,
            "",
            "Common causes:",
            "- Python cannot reach huggingface.co or the mirror configured by HF_ENDPOINT.",
            "- HF_ENDPOINT points to https://hf-mirror.com but that mirror is not reachable from this shell.",
            "- The model has not been downloaded into the local Hugging Face cache.",
            "",
            "Fix options:",
            "1. Use a reachable endpoint, for example unset HF_ENDPOINT or set it to a trusted mirror.",
            "2. Pre-download the model and point the service at the local directory:",
            "   huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir D:/models/bge-reranker-v2-m3",
            "   set RERANKER_MODEL_DIR=D:/models/bge-reranker-v2-m3",
            "3. Then start the service again:",
            "   set RERANKER_BACKEND=flagembedding",
            "   python scripts/serve_bge_reranker.py",
            "",
            "Original error:",
            str(exc),
        ]
    )


def normalize_documents(documents: list[Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(documents):
        if isinstance(item, str):
            doc_id = str(index)
            text = item
        elif isinstance(item, dict):
            doc_id = str(item.get("id", index))
            text = str(item.get("text", ""))
        else:
            raise ValueError("Each document must be either a string or an object with `text`.")
        if not text.strip():
            raise ValueError(f"Document at index {index} has empty text.")
        normalized.append({"id": doc_id, "text": text})
    return normalized


def build_response(query: str, documents: list[dict[str, str]], scores: list[float], model_name: str) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("`query` must not be empty.")
    if len(scores) != len(documents):
        raise ValueError(f"Backend returned {len(scores)} scores for {len(documents)} documents.")

    results = [
        {
            "index": index,
            "id": document["id"],
            "score": float(score),
        }
        for index, (document, score) in enumerate(zip(documents, scores))
    ]
    return {
        "model": model_name,
        "scores": [float(score) for score in scores],
        "results": sorted(results, key=lambda item: item["score"], reverse=True),
    }


def score_payload(payload: dict[str, Any], *, backend: Any) -> dict[str, Any]:
    query = str(payload.get("query", ""))
    documents = normalize_documents(list(payload.get("documents") or []))
    model_name = resolve_model_name(str(payload.get("model", "")))
    scores = backend.score(query, [document["text"] for document in documents])
    return build_response(query, documents, scores, model_name)


class FlagEmbeddingBackend:
    def __init__(self, model_name: str, *, use_fp16: bool = False) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RerankerDependencyError("Install FlagEmbedding or set RERANKER_BACKEND=transformers.") from exc

        try:
            self.reranker = FlagReranker(model_name, use_fp16=use_fp16)
        except OSError as exc:
            raise RerankerModelLoadError(format_model_load_error(model_name, exc)) from exc

    def score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [[query, document] for document in documents]
        scores = self.reranker.compute_score(pairs, normalize=True)
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(score) for score in scores]


class TransformersBackend:
    def __init__(self, model_name: str, *, max_length: int = 1024) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RerankerDependencyError("Install transformers and torch, or set RERANKER_BACKEND=flagembedding.") from exc

        self.torch = torch
        self.max_length = max_length
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except OSError as exc:
            raise RerankerModelLoadError(format_model_load_error(model_name, exc)) from exc
        self.model.eval()

    def score(self, query: str, documents: list[str]) -> list[float]:
        inputs = self.tokenizer(
            [query] * len(documents),
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self.torch.no_grad():
            logits = self.model(**inputs).logits.view(-1)
            scores = self.torch.sigmoid(logits).tolist()
        return [float(score) for score in scores]


class AutoBackend:
    def __init__(self, primary: Any, fallback_factory: Callable[[], Any]) -> None:
        self.primary = primary
        self.fallback_factory = fallback_factory
        self.fallback: Any | None = None

    def score(self, query: str, documents: list[str]) -> list[float]:
        try:
            return self.primary.score(query, documents)
        except AttributeError as exc:
            if not is_tokenizer_compatibility_error(exc):
                raise
            if self.fallback is None:
                self.fallback = self.fallback_factory()
            return self.fallback.score(query, documents)


def load_backend(model_name: str, backend_name: str) -> Any:
    def make_transformers_backend() -> TransformersBackend:
        return TransformersBackend(
            model_name,
            max_length=int(os.getenv("RERANKER_MAX_LENGTH", "1024")),
        )

    if backend_name == "flagembedding":
        primary = FlagEmbeddingBackend(
            model_name,
            use_fp16=parse_bool(os.getenv("RERANKER_USE_FP16", ""), default=False),
        )
        return AutoBackend(primary, make_transformers_backend)
    if backend_name == "transformers":
        return make_transformers_backend()
    if backend_name != "auto":
        raise RuntimeError("RERANKER_BACKEND must be one of: auto, flagembedding, transformers.")

    try:
        primary = FlagEmbeddingBackend(
            model_name,
            use_fp16=parse_bool(os.getenv("RERANKER_USE_FP16", ""), default=False),
        )
        return AutoBackend(primary, make_transformers_backend)
    except AttributeError as exc:
        if not is_tokenizer_compatibility_error(exc):
            raise
        return make_transformers_backend()
    except RerankerDependencyError:
        return make_transformers_backend()


def create_app() -> Any:
    load_dotenv()
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("Install fastapi and uvicorn to serve the reranker HTTP API.") from exc

    model_name = resolve_model_name(os.getenv("RERANKER_MODEL", ""))
    model_reference = resolve_model_reference(os.getenv("RERANKER_MODEL", ""), os.getenv("RERANKER_MODEL_DIR", ""))
    backend_name = os.getenv("RERANKER_BACKEND", "auto").strip().lower() or "auto"
    backend = load_backend(model_reference, backend_name)

    app = FastAPI(title="BGE reranker service", version="1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model": model_name,
            "model_reference": model_reference,
            "backend": backend_name,
        }

    @app.post("/rerank")
    def rerank_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return score_payload(payload, backend=backend)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install uvicorn to run this service directly.") from exc

    load_dotenv()
    host = os.getenv("RERANKER_HOST", "127.0.0.1")
    port = int(os.getenv("RERANKER_PORT", "8008"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
