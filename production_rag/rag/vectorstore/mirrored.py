from __future__ import annotations

from rag.models import Chunk
from rag.vectorstore.sqlite import LocalVectorStore

__all__ = ["MirroredVectorStore"]


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
