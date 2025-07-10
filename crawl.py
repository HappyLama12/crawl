"""
title: HeadlessCrawlRAG Pipeline
author: Your Name
date: 2025-07-10
version: 1.0
license: MIT
description: A pipeline that uses Playwright headless browser to crawl pages, embed content, and store it in ChromaDB.
requirements: sentence-transformers, chromadb, langdetect, playwright
"""

import os
import time
import re
from urllib.parse import urlparse
from typing import List, Union, Generator, Iterator

import chromadb
from sentence_transformers import SentenceTransformer
from langdetect import detect

from playwright.sync_api import sync_playwright


class Pipeline:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None

    async def on_startup(self):
        self._init_clients()

    async def on_shutdown(self):
        self.client = None
        self.collection = None
        self.embedder = None

    def _init_clients(self):
        if not self.client:
            self.client = chromadb.HttpClient(host="chromadb", port=8000)
        if not self.collection:
            self.collection = self.client.get_or_create_collection("crawled_data")
        if not self.embedder:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_urls(self, text: str) -> List[str]:
        url_pattern = re.compile(
            r"(https?://[^\s\"'<>]+)",
            re.IGNORECASE
        )
        return url_pattern.findall(text)

    def crawl_with_playwright(self, url: str) -> str:
        """Use Playwright to load the page and extract visible text."""
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=30000)
            # Extract page's visible text content
            content = page.evaluate("() => document.body.innerText")
            browser.close()
        return content

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> Union[str, Generator, Iterator]:

        self._init_clients()

        url = body.get("url")
        if not url:
            urls = self.extract_urls(user_message)
            url = urls[0] if urls else None

        if not url:
            return "❌ Missing URL. Please provide a valid http(s) URL."

        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return f"❌ Invalid URL scheme: {url} - Use http or https."

        try:
            text = self.crawl_with_playwright(url)
            if not text or not text.strip():
                return f"❌ No text extracted from page: {url}"
            text_content = [text.strip()]
        except Exception as e:
            return f"❌ Headless crawl error: {e}"

        try:
            language = detect(text_content[0])
        except Exception:
            language = "unknown"

        # Check duplicates
        try:
            existing = self.collection.query(query_texts=text_content, n_results=1) or {}
            if existing.get("documents"):
                return f"⚠️ Duplicate content detected. Skipping storage.\nURL: {url}\nLanguage: {language}"
        except Exception as e:
            return f"❌ Chroma query error: {e}"

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

        return f"✅ Successfully crawled and stored page content.\nURL: {url}\nLanguage: {language}"
