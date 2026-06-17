from __future__ import annotations

from pathlib import Path
from typing import Iterable

import httpx
from bs4 import BeautifulSoup
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.config import get_settings


STORE_PAGE_URLS = [
    "https://zahrastores.pk/pages/contact",
    "https://zahrastores.pk/pages/policies",
    "https://zahrastores.pk/pages/our-services",
    "https://zahrastores.pk/pages/shipping-policy",
    "https://zahrastores.pk/pages/term-conditions",
    "https://zahrastores.pk/pages/shipping-policy-1",
    "https://zahrastores.pk/pages/exchange-refund-policy",
    "https://zahrastores.pk/pages/terms-conditions",
    "https://zahrastores.pk/pages/privacy-policy",
]


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer"]):
        tag.decompose()

    main = soup.find("main") or soup.find("body") or soup
    text = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


async def scrape_pages(urls: Iterable[str]) -> list[Document]:
    documents: list[Document] = []
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        for url in urls:
            response = await client.get(url)
            response.raise_for_status()
            text = clean_html(response.text)
            if text:
                documents.append(Document(page_content=text, metadata={"source": url}))
    return documents


async def build_faiss_index() -> Path:
    settings = get_settings()
    documents = await scrape_pages(STORE_PAGE_URLS)
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    index_path = Path(settings.faiss_index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))
    return index_path


if __name__ == "__main__":
    import asyncio

    path = asyncio.run(build_faiss_index())
    print(f"FAISS index saved to {path}")
