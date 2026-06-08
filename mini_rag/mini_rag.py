from __future__ import annotations

import argparse
import json
import math
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = ROOT / "sample_docs"
DEFAULT_INDEX_PATH = ROOT / "rag_index.json"
DEFAULT_MAX_CHARS = 240
DEFAULT_OVERLAP_CHARS = 60
DEFAULT_CHAT_MODEL = "deepseek-v4-pro"
DEFAULT_EMBEDDING_MODEL = "embedding-3"
DEFAULT_LLM_API_STYLE = "anthropic"
DEFAULT_LLM_MAX_TOKENS = 1000
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"
DEFAULT_DEEPSEEK_OPENAI_BASE_URL = "https://api.deepseek.com"
DEFAULT_ZHIPU_EMBEDDING_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


@dataclass
class Chunk:
    chunk_id: str
    source: str
    index: int
    text: str
    embedding: list[float] | None = None


@dataclass
class AnthropicHTTPClient:
    api_key: str
    base_url: str

    def create_message(
        self,
        model: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
    ) -> str:
        body = json.dumps(
            {
                "model": model,
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}],
                    }
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/v1/messages",
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic API request failed: HTTP {exc.code} {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Anthropic API request failed: {exc.reason}") from exc

        return extract_anthropic_text(payload)


def load_env(path: Path) -> None:
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


def resolve_llm_base_url(api_style: str) -> str | None:
    if api_style == "anthropic":
        base_url = env_first("LLM_BASE_URL", "ANTHROPIC_BASE_URL", "DEEPSEEK_BASE_URL")
        if not base_url:
            return DEFAULT_DEEPSEEK_BASE_URL
        if base_url.rstrip("/") == DEFAULT_DEEPSEEK_OPENAI_BASE_URL:
            return DEFAULT_DEEPSEEK_BASE_URL
        return base_url
    base_url = env_first("LLM_BASE_URL", "DEEPSEEK_BASE_URL")
    return base_url or DEFAULT_DEEPSEEK_OPENAI_BASE_URL


def resolve_embedding_base_url() -> str | None:
    return env_first(
        "EMBEDDING_BASE_URL",
        "ZHIPU_EMBEDDING_BASE_URL",
        "ZHIPUAI_BASE_URL",
        default=DEFAULT_ZHIPU_EMBEDDING_BASE_URL,
    )


def make_openai_client(api_key: str, base_url: str | None = None) -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency: run `pip install -r requirements.txt` first.") from exc

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def make_llm_client(api_key: str, base_url: str | None, api_style: str) -> Any:
    if api_style == "anthropic":
        if not base_url:
            raise RuntimeError("Anthropic LLM base URL is missing.")
        return AnthropicHTTPClient(api_key=api_key, base_url=base_url)
    return make_openai_client(api_key, base_url)


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


def extract_anthropic_text(payload: dict[str, Any]) -> str:
    content = payload.get("content", [])
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    texts: list[str] = []
    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            texts.append(block["text"])
    return "".join(texts)


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def iter_markdown_files(docs_dir: Path) -> Iterable[Path]:
    yield from sorted(path for path in docs_dir.glob("*.md") if path.is_file())


def read_documents(docs_dir: Path) -> list[tuple[str, str]]:
    documents: list[tuple[str, str]] = []
    for path in iter_markdown_files(docs_dir):
        documents.append((path.name, normalize_text(path.read_text(encoding="utf-8"))))
    if not documents:
        raise FileNotFoundError(f"No markdown documents found in {docs_dir}")
    return documents


def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    sentences = re.split(r"(?<=[。！？!?])", paragraph)
    parts: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) <= max_chars:
            current += sentence
            continue
        if current:
            parts.append(current)
        current = sentence

    if current:
        parts.append(current)
    return parts or [paragraph[:max_chars]]


def chunk_document(source: str, text: str, max_chars: int, overlap_chars: int) -> list[Chunk]:
    paragraphs: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            paragraphs.extend(split_long_paragraph(paragraph, max_chars))
        else:
            paragraphs.append(paragraph)

    chunks: list[Chunk] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(Chunk("", source, len(chunks), current))
            current = f"{current[-overlap_chars:]}\n\n{paragraph}".strip()
        else:
            chunks.append(Chunk("", source, len(chunks), paragraph[:max_chars]))
            current = paragraph[max_chars - overlap_chars :]

    if current:
        chunks.append(Chunk("", source, len(chunks), current))

    for chunk in chunks:
        chunk.chunk_id = f"{source}#chunk-{chunk.index}"
    return chunks


def build_chunks(docs_dir: Path, max_chars: int, overlap_chars: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, text in read_documents(docs_dir):
        chunks.extend(chunk_document(source, text, max_chars, overlap_chars))
    return chunks


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_texts(client: Any, texts: list[str], model: str) -> list[list[float]]:
    vectors: list[list[float]] = []
    for batch in batched(texts, 64):
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


def save_index(path: Path, embedding_model: str, chunks: list[Chunk]) -> None:
    payload = {
        "embedding_model": embedding_model,
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "index": chunk.index,
                "text": chunk.text,
                "embedding": chunk.embedding,
            }
            for chunk in chunks
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(path: Path, embedding_model: str) -> list[Chunk] | None:
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("embedding_model") != embedding_model:
        return None

    chunks: list[Chunk] = []
    for item in payload["chunks"]:
        chunks.append(
            Chunk(
                chunk_id=item["chunk_id"],
                source=item["source"],
                index=item["index"],
                text=item["text"],
                embedding=item["embedding"],
            )
        )
    return chunks


def get_or_create_index(
    client: Any,
    docs_dir: Path,
    index_path: Path,
    embedding_model: str,
    max_chars: int,
    overlap_chars: int,
    rebuild: bool,
) -> list[Chunk]:
    # Teaching simplification: cache invalidation only checks the embedding model.
    # Production indexes should also verify source document hashes or versions.
    if not rebuild:
        cached = load_index(index_path, embedding_model)
        if cached:
            return cached

    chunks = build_chunks(docs_dir, max_chars, overlap_chars)
    embeddings = embed_texts(client, [chunk.text for chunk in chunks], embedding_model)

    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    save_index(index_path, embedding_model, chunks)
    return chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieve(
    client: Any,
    question: str,
    chunks: list[Chunk],
    embedding_model: str,
    top_k: int,
    min_score: float,
) -> list[tuple[float, Chunk]]:
    query_embedding = embed_texts(client, [question], embedding_model)[0]
    scored: list[tuple[float, Chunk]] = []

    for chunk in chunks:
        if chunk.embedding is None:
            continue
        score = cosine_similarity(query_embedding, chunk.embedding)
        if score >= min_score:
            scored.append((score, chunk))

    return sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]


def format_context(results: list[tuple[float, Chunk]]) -> str:
    blocks: list[str] = []
    for citation_id, (score, chunk) in enumerate(results, start=1):
        blocks.append(
            f"[{citation_id}] source={chunk.source} chunk={chunk.index} score={score:.4f}\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(
    client: Any,
    question: str,
    results: list[tuple[float, Chunk]],
    model: str,
    api_style: str,
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS,
) -> str:
    if not results:
        return "资料中没有找到足够相关的信息，不能可靠回答这个问题。"

    context = format_context(results)
    prompt = f"""问题：
{question}

检索到的资料：
{context}

请根据资料回答问题。
要求：
1. 只使用上面的资料，不要用常识补全。
2. 如果资料不足，就直接说资料不足。
3. 每个关键结论后面标注引用编号，例如 [1]。
4. 先给简短答案，再补充必要说明。"""

    system_prompt = "你是一个严谨的企业知识库问答助手。你的任务是根据检索资料回答问题，并避免编造。"

    if api_style == "anthropic":
        return client.create_message(
            model=model,
            system_prompt=system_prompt,
            prompt=prompt,
            max_tokens=max_tokens,
        )

    if api_style == "responses":
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=prompt,
        )
        return response.output_text

    if api_style == "chat_completions":
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return response.choices[0].message.content or ""

    raise ValueError(f"Unsupported LLM_API_STYLE: {api_style}")


def print_retrieval(results: list[tuple[float, Chunk]]) -> None:
    if not results:
        print("No relevant chunks found.")
        return

    for rank, (score, chunk) in enumerate(results, start=1):
        print(f"\n#{rank} score={score:.4f} source={chunk.source} chunk={chunk.index}")
        print("-" * 72)
        print(chunk.text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A minimal RAG pipeline for hands-on learning.")
    parser.add_argument("--question", required=True, help="User question to answer.")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR), help="Directory containing markdown docs.")
    parser.add_argument("--index-path", default=str(DEFAULT_INDEX_PATH), help="Path to the cached JSON index.")
    parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop chunks below this cosine score.")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help="Approximate chunk size in characters.",
    )
    parser.add_argument(
        "--overlap-chars",
        type=int,
        default=DEFAULT_OVERLAP_CHARS,
        help="Character overlap between chunks.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the embedding index.")
    parser.add_argument("--retrieve-only", action="store_true", help="Print retrieved chunks without generation.")
    return parser.parse_args()


def main() -> None:
    load_env(ROOT / ".env")
    args = parse_args()

    llm_api_key = env_first("LLM_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY")
    embedding_api_key = env_first("EMBEDDING_API_KEY", "ZHIPU_API_KEY", "ZHIPUAI_API_KEY")
    if not llm_api_key:
        raise RuntimeError("LLM API key is missing. Set LLM_API_KEY or DEEPSEEK_API_KEY.")
    if not embedding_api_key:
        raise RuntimeError("Embedding API key is missing. Set EMBEDDING_API_KEY or ZHIPU_API_KEY.")

    docs_dir = Path(args.docs_dir).resolve()
    index_path = Path(args.index_path).resolve()
    chat_model = env_first("LLM_MODEL", default=DEFAULT_CHAT_MODEL)
    configured_llm_api_style = env_first("LLM_API_STYLE")
    llm_api_style = configured_llm_api_style or DEFAULT_LLM_API_STYLE
    llm_api_style = llm_api_style.strip().lower().replace("-", "_")
    llm_base_url = resolve_llm_base_url(llm_api_style)
    llm_max_tokens = parse_int_env("LLM_MAX_TOKENS", DEFAULT_LLM_MAX_TOKENS)
    embedding_model = env_first("EMBEDDING_MODEL", default=DEFAULT_EMBEDDING_MODEL)
    embedding_base_url = resolve_embedding_base_url()

    llm_client = make_llm_client(llm_api_key, llm_base_url, llm_api_style)
    embedding_client = make_openai_client(embedding_api_key, embedding_base_url)
    chunks = get_or_create_index(
        client=embedding_client,
        docs_dir=docs_dir,
        index_path=index_path,
        embedding_model=embedding_model,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        rebuild=args.rebuild,
    )
    results = retrieve(
        client=embedding_client,
        question=args.question,
        chunks=chunks,
        embedding_model=embedding_model,
        top_k=args.top_k,
        min_score=args.min_score,
    )

    print_retrieval(results)

    if not args.retrieve_only:
        print("\n" + "=" * 72)
        print("Answer")
        print("=" * 72)
        print(answer_question(llm_client, args.question, results, chat_model, llm_api_style, llm_max_tokens))


if __name__ == "__main__":
    main()
