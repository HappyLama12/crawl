"""
title: CrawlRAG Pipeline
author: Your Name
date: 2025-07-10
version: 1.0
license: MIT
description: A pipeline that crawls a URL with Crawl4AI, embeds its content, stores it in ChromaDB, and returns the content.
requirements: sentence-transformers, chromadb, langdetect, requests
"""

import os
import re
import time
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from langdetect import detect
from urllib.parse import urlparse

from typing import List, Union, Generator, Iterator


class Pipeline:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db/craweld")
        self.crawl4ai_url = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235/crawl")

    async def on_startup(self):
        """
        Called when Open WebUI starts.
        """
        self._init_clients()

    async def on_shutdown(self):
        """
        Called when Open WebUI stops.
        """
        self.client = None
        self.collection = None
        self.embedder = None

    def _init_clients(self):
        """
        Safe client init.
        """
        if not self.client:
            self.client = chromadb.HttpClient(
                host="chromadb",
                port=8000,
                settings=chromadb.Settings(
                    persist_directory=self.persist_directory
                )
            )
        if not self.collection:
            self.collection = self.client.get_or_create_collection("crawled_data")
        if not self.embedder:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_urls(self, text: str) -> List[str]:
        """
        Extract http(s) URLs from text.
        """
        pattern = re.compile(r"(https?://[^\s\"'<>]+)", re.IGNORECASE)
        return pattern.findall(text)

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Main pipeline logic.
        """

        # Fallback init if on_startup not triggered
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
        # 2️⃣ Call Crawl4AI
        try:
            res = requests.post(self.crawl4ai_url, json={"urls": [url]}, timeout=60)
            res.raise_for_status()
            result = res.json()
        except Exception as e:
            return f"❌ Crawl4AI error: {e}"
        
        # 🟢 Robust text extraction
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

        # 4️⃣ Check for duplicates
        try:
            existing = self.collection.query(query_texts=text_content, n_results=1)
            if existing.get("documents"):
                return f"⚠️ Duplicate content detected. Skipping storage.\nURL: {url}\nLanguage: {language}"
        except Exception:
            pass

        # 5️⃣ Embed & store
        try:
            embeddings = self.embedder.encode(text_content).tolist()
            ids = [f"{int(time.time())}-{i}" for i in range(len(text_content))]
            metadatas = [{"url": url, "language": language} for _ in text_content]

            self.collection.add(
                documents=text_content,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
        except Exception as e:
            return f"❌ Embedding/storage error: {e}"

        # ✅ 6️⃣ Return the crawled text back to user
        joined_text = "\n\n".join(text_content[:3])  # Send first few chunks only (optional)
        return (
            f"✅ Successfully crawled and stored {len(text_content)} chunks.\n"
            f"URL: {url}\n"
            f"Language: {language}\n\n"
            f"📄 **Sample content:**\n{joined_text}"
        )
