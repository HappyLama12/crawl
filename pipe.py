"""
title: CrawlRAG Pipeline
author: Your Name
description: Crawl a URL, store chunks, then run a RAG query.
"""

import os
import re
import time
import requests
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer
from langdetect import detect
from typing import List

import chromadb
from chromadb.config import Settings


class pipelines:
    """
    Expose your methods here — Open WebUI looks for this `Tools` class.
    """

    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None

        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db/crawled")
        self.crawl4ai_url = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235/crawl")

    def _init_clients(self):
        if not self.client:
            self.client = chromadb.Client(
                Settings(persist_directory=self.persist_directory)
            )
        if not self.collection:
            self.collection = self.client.get_or_create_collection("crawled_data")
        if not self.embedder:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_urls(self, text: str) -> List[str]:
        pattern = re.compile(r"(https?://[^\s\"'<>]+)", re.IGNORECASE)
        return pattern.findall(text)

    def crawl_and_store(self, url: str) -> dict:
        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return {"error": f"❌ Invalid URL: {url}"}

        try:
            res = requests.post(self.crawl4ai_url, json={"urls": [url]}, timeout=60)
            res.raise_for_status()
            result = res.json()
        except Exception as e:
            return {"error": f"❌ Crawl4AI error: {e}"}

        texts = result.get("texts")
        single_text = result.get("text")
        text_content = []

        if texts and isinstance(texts, list) and any(texts):
            text_content = [t.strip() for t in texts if t and t.strip()]
        elif single_text and isinstance(single_text, str) and single_text.strip():
            text_content = [single_text.strip()]

        if not text_content:
            return {"error": f"❌ No text found for: {url}"}

        try:
            language = detect(" ".join(text_content))
        except Exception:
            language = "unknown"

        embeddings = self.embedder.encode(text_content).tolist()
        ids = [f"{int(time.time())}-{i}" for i in range(len(text_content))]
        metadatas = [{"url": url, "language": language} for _ in text_content]

        self.collection.add(
            documents=text_content,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        return {
            "status": "✅ Crawled & stored",
            "chunks": len(text_content),
            "language": language,
            "sample": text_content[:3]
        }

    def rag_query(self, question: str) -> str:
        results = self.collection.query(
            query_texts=[question],
            n_results=3
        )
        docs = results.get("documents", [[]])[0]
        context = "\n\n".join(docs)
        return context

    def valve(self, user_message: str, model_id: str, messages: list, body: dict) -> str:
        """
        This is your exposed 'valve'.
        """
        self._init_clients()

        url = body.get("url")
        question = body.get("question")

        if not url:
            urls = self.extract_urls(user_message)
            url = urls[0] if urls else None

        if not url:
            return "❌ Missing URL."
        if not question:
            return "❌ Missing question."

        crawl_result = self.crawl_and_store(url)
        if "error" in crawl_result:
            return crawl_result["error"]

        context = self.rag_query(question)
        answer = self._generate_answer(context, question)

        return (
            f"✅ Crawled: {url}\n"
            f"Chunks: {crawl_result['chunks']}\n"
            f"Language: {crawl_result['language']}\n\n"
            f"📄 Context:\n{context[:500]}...\n\n"
            f"💡 Answer: {answer}"
        )

    def _generate_answer(self, context: str, question: str) -> str:
        if not context.strip():
            return "No relevant context found."
        return f"(Placeholder) Based on context: '{context[:100]}...', your question is: {question}"
