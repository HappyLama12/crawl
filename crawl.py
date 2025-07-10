"""
title: CrawlRAG Pipeline
author: Your Name
date: 2025-07-10
version: 1.1
license: MIT
description: A pipeline that crawls a URL with Crawl4AI, embeds its content, and stores it in ChromaDB.
requirements: sentence-transformers, chromadb, langdetect, requests, re
"""

import os
import time
import re
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
        self.crawl4ai_url = os.getenv("CRAWL4AI_URL", "http://localhost:11235/crawl")

    async def on_startup(self):
        # Initialize ChromaDB client & embedder once on startup
        self.client = chromadb.HttpClient(host="chromadb", port=8000)
        self.collection = self.client.get_or_create_collection("crawled_data")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    async def on_shutdown(self):
        # Cleanup if needed
        self.client = None
        self.collection = None
        self.embedder = None

    def extract_url(self, text: str) -> Union[str, None]:
        """Extract the first URL from a string using regex."""
        
        # Regex pattern to extract URLs (http(s), www, and partial domains)
        URL_REGEX = re.compile(r'\b(?:https?://|www\.)[\w\-]+(?:\.[\w\-]+)+(?:[\w\-\._~:/?#\[\]@!\$&\'\(\)\*\+,;=]*)')
        match = URL_REGEX.search(text)
        return match.group(0) if match else None

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:

        # Extract URL from body or user message
        url = body.get("url") or self.extract_url(user_message.strip())

        if not url:
            return "❌ Missing or invalid URL."

        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            url = "http://" + url  # Try to fix missing scheme
            parsed = urlparse(url)
            if not parsed.scheme.startswith("http"):
                return "❌ Invalid URL scheme. Use http or https."

        # Crawl the URL
        try:
            res = requests.post(self.crawl4ai_url, json={"url": url})
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
