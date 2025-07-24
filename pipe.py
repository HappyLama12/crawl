"""
title: CrawlRAG Pipeline with Ollama RAG
author: Your Name
date: 2025-07-13
version: 1.2
license: MIT
description: Crawl URL with depth & sitemap, chunk & optionally summarize content, store in ChromaDB, then answer questions using Ollama deepseek-r1:14b model.
requirements: sentence-transformers, chromadb, langdetect, requests
"""

import os
import re
import time
import requests
from typing import List, Union, Generator, Iterator
from urllib.parse import urlparse

from sentence_transformers import SentenceTransformer
from langdetect import detect

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, SearchRequest, Filter, FieldCondition, MatchValue
)


class Pipeline:
    def __init__(self):
        self.client = None
        self.embedder = None
        self.collection_name = "rag_data"
        self.vector_dim = 384  # Vector size for all-MiniLM-L6-v2
        self.crawl4ai_url = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235/crawl")

    async def on_startup(self):
        self._init_clients()

    async def on_shutdown(self):
        self.client = None
        self.embedder = None

    def _init_clients(self):
        if not self.client:
            self.client = QdrantClient(host="qdrant", port=6333)
            if self.collection_name not in [c.name for c in self.client.get_collections().collections]:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE)
                )
        if not self.embedder:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_urls(self, text: str) -> List[str]:
        pattern = re.compile(r"(https?://[^\s\"'<>]+)", re.IGNORECASE)
        return pattern.findall(text)

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        self._init_clients()

        # 1️⃣ Get URL
        url = body.get("url")
        if not url:
            urls = self.extract_urls(user_message)
            url = urls[0] if urls else None
        if not url:
            return "❌ Missing URL. Provide a valid http(s) URL."

        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return f"❌ Invalid URL scheme: {url}"

        # 2️⃣ Call Crawl4AI
        try:
            res = requests.post(self.crawl4ai_url, json={"urls": [url]}, timeout=60)
            res.raise_for_status()
            result = res.json()
        except Exception as e:
            return f"❌ Crawl4AI error: {e}"

        # 🟢 Text extraction
        texts = result.get("texts")
        single_text = result.get("text")

        text_content = []
        if texts and isinstance(texts, list) and any(texts):
            text_content = texts
        elif single_text and isinstance(single_text, str) and single_text.strip():
            text_content = [single_text]

        if not text_content:
            return f"❌ No text content returned from Crawl4AI for: {url}\nRaw response: {result}"

        # 3️⃣ Detect language
        try:
            language = detect(" ".join(filter(None, text_content)))
        except Exception:
            language = "unknown"

        # 4️⃣ Check for duplicates (search top-1)
        try:
            embeddings = self.embedder.encode(text_content).tolist()
            duplicate_check = self.client.search(
                collection_name=self.collection_name,
                query_vector=embeddings[0],
                limit=1
            )
            if duplicate_check and duplicate_check[0].score > 0.98:  # Threshold for near duplicate
                return f"⚠️ Duplicate content detected. Skipping storage.\nURL: {url}\nLanguage: {language}"
        except Exception:
            pass

        # 5️⃣ Store in Qdrant
        try:
            points = [
                PointStruct(
                    id=int(time.time() * 1000) + i,
                    vector=emb,
                    payload={"text": text, "url": url, "language": language}
                )
                for i, (text, emb) in enumerate(zip(text_content, embeddings))
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
        except Exception as e:
            return f"❌ Embedding/storage error: {e}"

        # ✅ 6️⃣ Return summary
        joined_text = "\n\n".join(text_content[:3])
        return (
            f"✅ Successfully crawled and stored {len(text_content)} chunks.\n"
            f"URL: {url}\n"
            f"Language: {language}\n\n"
            f"📄 **Sample content:**\n{joined_text}"
        )

