from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from rag.config import CHUNK_CHARS, OVERLAP_CHARS, QDRANT_SPARSE_HASH_BUCKETS, STOPWORDS
from rag.config import resolve_vector_dimensions
from rag.models import Chunk, ParentSection


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


def split_metadata_values(value: str) -> set[str]:
    return {item.strip() for item in re.split(r"[,，;；\s]+", value) if item.strip()}


def build_document_chunks(metadata: dict[str, str], body: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for parent in split_sections(metadata, body):
        chunks.extend(chunk_parent(parent))
    return chunks
