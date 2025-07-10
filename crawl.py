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
        self.crawl4ai_url = "http://crawl4ai:11235/crawl"

    async def on_startup(self):
        self.client = chromadb.HttpClient(host="chromadb", port=8000)
        self.collection = self.client.get_or_create_collection("crawled_data")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    async def on_shutdown(self):
        self.client = None
        self.collection = None
        self.embedder = None

    def extract_urls(self, text: str) -> List[str]:
        """
        Extracts all http(s) URLs from a text using a robust regex.
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

        # Try to get URL from body first, else extract from user_message
        url = body.get("url")
        if not url:
            urls = self.extract_urls(user_message)
            url = urls[0] if urls else None

        if not url:
            return "❌ Missing URL."

        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return "❌ Invalid URL scheme. Use http or https."

        # Crawl the URL
        try:
            res = requests.post(self.crawl4ai_url, json={"urls": [url]})
            res.raise_for_status()
            result = res.json()
        except Exception as e:
            return f"❌ Crawl4AI error: {e}"

        text_content = result.get("texts") or [result.get("text")]
        if not text_content:
            return "❌ No text content returned."

        # Detect language
        try:
            language = detect(" ".join(text_content))
        except Exception:
            language = "unknown"

        # Check for duplicates
        try:
            existing = self.collection.query(query_texts=text_content, n_results=1)
            if existing.get("documents"):
                return f"⚠️ Duplicate content detected. Skipping storage. URL: {url} Language: {language}"
        except Exception:
            pass

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

        return f"✅ Successfully crawled and stored {len(text_content)} chunks. URL: {url} Language: {language}"
