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
import uuid
import requests
from typing import List, Union, Generator, Iterator
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from langdetect import detect
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance


class Pipeline:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_size = 384
        self.qdrant_collection = "rag_data"
        self.qdrant = QdrantClient(host="qdrant", port=6333)
        self.crawl4ai_url = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434/api/generate")

        # Create collection if not exists
        if self.qdrant_collection not in [c.name for c in self.qdrant.get_collections().collections]:
            self.qdrant.recreate_collection(
                collection_name=self.qdrant_collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    async def on_startup(self):
        pass

    async def on_shutdown(self):
        pass

    def extract_urls(self, text: str) -> List[str]:
        pattern = re.compile(r"(https?://[^\s\"'<>]+)", re.IGNORECASE)
        return pattern.findall(text)

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        # 🔗 Get URL
        url = body.get("url") or (self.extract_urls(user_message)[0] if self.extract_urls(user_message) else None)
        if not url:
            return "❌ No valid URL provided."

        # 🌐 Crawl
        pages = self._crawl_site(url)
        if not pages:
            return f"❌ Failed to crawl: {url}"

        # ✂️ Chunk + Embed
        ids, chunks, vectors = self._chunk_and_embed(pages)

        # 📥 Store in Qdrant
        self._store_qdrant(ids, chunks, vectors)

        # 🧠 Get Summary using Ollama
        prompt = f"Summarize the following content in 200 words:\n\n" + "\n\n".join(chunks[:5])
        summary = self._summarize_with_ollama(prompt)

        return (
            f"✅ Content from {url} successfully embedded and stored in Qdrant.\n\n"
            f"🧠 **Summary:**\n{summary}"
        )

    def _crawl_site(self, url: str):
        payload = {
            "urls": [url],
            "depth": 2,
            "crawl_subdomains": True,
            "options": {"download_assets": False}
        }

        try:
            res = requests.post(f"{self.crawl4ai_url}/crawl", json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()

            if "id" in data:
                crawl_id = data["id"]
                while True:
                    status = requests.get(f"{self.crawl4ai_url}/crawl/{crawl_id}").json()
                    if status["status"] == "done":
                        return status["results"]
                    elif status["status"] == "error":
                        return None
                    time.sleep(5)
            elif "results" in data:
                return data["results"]
        except Exception as e:
            print(f"[ERROR] Crawl failed: {e}")
        return None

    def _chunk_and_embed(self, pages):
        docs, ids, embeddings = [], [], []

        for i, page in enumerate(pages):
            text = page.get("content") or page.get("text") or self._html_to_text(page.get("html", ""))
            if not text or len(text.strip()) < 50:
                continue

            chunks = [text[k:k+500] for k in range(0, len(text), 500)]
            for j, chunk in enumerate(chunks):
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    if detect(chunk) != "en":
                        continue
                except:
                    continue

                docs.append(chunk)
                ids.append(f"page_{i}_chunk_{j}")
                embeddings.append(self.embedder.encode(chunk).tolist())

        return ids, docs, embeddings

    def _html_to_text(self, html):
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator=" ", strip=True)

    def _store_qdrant(self, ids, docs, vectors):
        if not vectors:
            return

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, ids[i])),
                vector=vectors[i],
                payload={"text": docs[i]}
            )
            for i in range(len(ids))
        ]

        self.qdrant.upsert(collection_name=self.qdrant_collection, points=points)

    def _summarize_with_ollama(self, prompt: str):
        try:
            res = requests.post(self.ollama_url, json={"model": "deepseek-r1:14b", "prompt": prompt, "stream": False})
            res.raise_for_status()
            return res.json().get("response", "[ERROR] Empty response")
        except Exception as e:
            return f"[ERROR] Ollama request failed: {e}"
