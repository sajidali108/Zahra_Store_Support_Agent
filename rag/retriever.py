from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from agent.config import get_settings


@lru_cache
def _embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@lru_cache
def _vector_store() -> FAISS | None:
    settings = get_settings()
    index_path = Path(settings.faiss_index_path)
    if not index_path.exists():
        return None

    return FAISS.load_local(
        str(index_path),
        _embeddings(),
        allow_dangerous_deserialization=True,
    )


def search_knowledge_base(query: str, k: int = 4) -> list[dict[str, Any]]:
    store = _vector_store()
    if store is None:
        return [
            {
                "content": "Knowledge base index is not built yet. Run `uv run python rag/ingest.py` before policy retrieval.",
                "source": None,
            }
        ]

    documents = store.similarity_search(query, k=k)
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source"),
        }
        for doc in documents
    ]
