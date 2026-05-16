from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

_CHROMA_SETTINGS = Settings(anonymized_telemetry=False)

_COLLECTION     = "data_nature_papers_v4"   # v4 = ONNX embedding (no transformers dep)
_MAX_PER_SOURCE = 2                          # diversity cap: at most N results from the same paper


class ChromaRetriever:
    """Semantic retriever backed by ChromaDB + ONNX all-MiniLM-L6-v2 embeddings.

    Uses ChromaDB's built-in DefaultEmbeddingFunction (ONNX runtime) so the
    heavy `transformers` / `torchvision` stack is never imported.
    """

    def __init__(self, chunks: list[dict], persist_dir: Path | None = None) -> None:
        ef = DefaultEmbeddingFunction()

        if persist_dir is not None:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_dir), settings=_CHROMA_SETTINGS
            )
        else:
            self._client = chromadb.EphemeralClient(settings=_CHROMA_SETTINGS)

        self._col = self._client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=ef,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )

        if self._col.count() == 0:
            self._index(chunks)

    def _index(self, chunks: list[dict], batch_size: int = 64) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._col.add(
                ids=[f"c{start + i}" for i in range(len(batch))],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {"source": c["source"], "chunk_id": str(c["chunk_id"])}
                    for c in batch
                ],
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        # Fetch more candidates than needed so the diversity filter has room
        n_fetch = min(top_k * 4, self._col.count())
        res = self._col.query(query_texts=[query], n_results=n_fetch)

        docs      = res["documents"] or [[]]
        metas     = res["metadatas"] or [[]]
        distances = res["distances"] or [[]]

        candidates = [
            {
                "text": doc,
                "source": str(meta.get("source", "")),
                "chunk_id": int(str(meta.get("chunk_id", 0))),
                "score": max(0.0, 1.0 - dist),
            }
            for doc, meta, dist in zip(docs[0], metas[0], distances[0])
        ]

        seen: dict[str, int] = {}
        diverse: list[dict] = []
        for hit in candidates:
            src = hit["source"]
            if seen.get(src, 0) < _MAX_PER_SOURCE:
                diverse.append(hit)
                seen[src] = seen.get(src, 0) + 1
            if len(diverse) == top_k:
                break

        return diverse
