"""
title: CrawlRAG Pipeline
author: Your Name
date: 2025-07-10
version: 1.0
license: MIT
description: A pipeline that crawls a URL with Crawl4AI, embeds its content, and stores it in ChromaDB.
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
        self.crawl4ai_url = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235/crawl")

    async def on_startup(self):
        """
        Called when Open WebUI server starts.
        """
        self._init_clients()

    async def on_shutdown(self):
        """
        Called when Open WebUI server stops.
        """
        self.client = None
        self.collection = None
        self.embedder = None

    def _init_clients(self):
        """
        Initialize ChromaDB client and embedder. Safe to call multiple times.
        """
        if not self.client:
            self.client = chromadb.HttpClient(host="chromadb", port=8000)
        if not self.collection:
            self.collection = self.client.get_or_create_collection("crawled_data")
        if not self.embedder:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_urls(self, text: str) -> List[str]:
        """
        Extract all http(s) URLs from text using robust regex.
        """
        url_pattern = re.compile(
            r"(https?://[^\s\"'<>]+)",
            re.IGNORECASE
        )
        return url_pattern.findall(text)

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

        # ✅ Safeguard: initialize clients if needed
        self._init_clients()

        # Get URL
        url = body.get("url")
        if not url:
            urls = self.extract_urls(user_message)
            url = urls[0] if urls else None

        if not url:
            return "❌ Missing URL. Please provide a valid http(s) URL."

        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return f"❌ Invalid URL scheme: {url} - Use http or https."

        # Crawl the URL
        try:
            res = requests.post(self.crawl4ai_url, json={"urls": [url]}, timeout=60)
            res.raise_for_status()
            result = res.json()
            print(f"[DEBUG] Crawl4AI result: {result}")
        except Exception as e:
            return f"❌ Crawl4AI error: {e}"

        text_content = result.get("texts") or [result.get("text")]
        # Filter out None/empty chunks
        text_content = [t for t in text_content if t and t.strip()]

        if not text_content:
            return f"❌ No valid text content returned from Crawl4AI for: {url}"

        # Detect language
        try:
            language = detect(" ".join(text_content))
        except Exception:
            language = "unknown"

        # Check for duplicates safely
        try:
            existing = self.collection.query(query_texts=text_content, n_results=1) or {}
            print(f"[DEBUG] Chroma query result: {existing}")
            if existing.get("documents"):
                return f"⚠️ Duplicate content detected. Skipping storage.\nURL: {url}\nLanguage: {language}"
        except Exception as e:
            return f"❌ Chroma query error: {e}"

        # Embed and store
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

        return (
            f"✅ Successfully crawled and stored {len(text_content)} chunks.\n"
            f"URL: {url}\n"
            f"Language: {language}"
        )
