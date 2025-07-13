"""
title: CrawlRAG Pipeline
author: Your Name
date: 2025-07-13
version: 1.2
license: MIT
description: A robust RAG pipeline that crawls a URL with Crawl4AI, embeds & stores content in ChromaDB, and answers a user question with retrieval.
requirements: sentence-transformers, chromadb, langdetect, requests
"""

import os
import re
import time
import requests
from sentence_transformers import SentenceTransformer
from langdetect import detect
from urllib.parse import urlparse
from typing import List, Union, Generator, Iterator

import chromadb
from chromadb.config import Settings


class Pipeline:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db/crawled")
        self.crawl4ai_url = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235/crawl")
        # Optional: your OpenAI key if you want to use it later
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

    async def on_startup(self):
        """
        Runs when your agent/plugin starts.
        """
        self._init_clients()

    async def on_shutdown(self):
        """
        Runs when your agent/plugin stops.
        """
        self.client = None
        self.collection = None
        self.embedder = None

    def _init_clients(self):
        """
        Initialize Chroma and embedder.
        """
        if not self.client:
            self.client = chromadb.Client(
                Settings(persist_directory=self.persist_directory)
            )
            # Or for remote:
            # from chromadb import HttpClient
            # self.client = HttpClient(host="chromadb", port=8000)

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

    def crawl_and_store(self, url: str) -> dict:
        """
        Crawl the URL, embed its content, and store in Chroma.
        """
        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return {"error": f"❌ Invalid URL scheme: {url}"}

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
            return {"error": f"❌ No text returned for: {url}\nRaw: {result}"}

        try:
            language = detect(" ".join(text_content))
        except Exception:
            language = "unknown"

        # Dedup check: use query_texts, not embeddings
        try:
            existing = self.collection.query(
                query_texts=[text_content[0]],
                n_results=1
            )
            if existing.get("documents") and existing["documents"][0]:
                return {
                    "warning": "⚠️ Duplicate detected. Skipping storage.",
                    "url": url,
                    "language": language
                }
        except Exception:
            pass  # Safe to skip dedup check if it fails

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
        """
        Run RAG: embed question, query Chroma, return retrieved context.
        """
        results = self.collection.query(
            query_texts=[question],  # ✅ Let Chroma embed it
            n_results=3
        )
        docs = results.get("documents", [[]])[0]
        context = "\n\n".join(docs)
        return context

    def valve(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> str:
        """
        The single RAG valve: crawl URL + store + answer question.
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

        # Step 1: Crawl + store
        crawl_result = self.crawl_and_store(url)
        if "error" in crawl_result:
            return crawl_result["error"]

        if crawl_result.get("warning"):
            return crawl_result["warning"]

        # Step 2: RAG query
        context = self.rag_query(question)

        # Step 3: Final answer (LLM or placeholder)
        answer = self._generate_answer(context, question)

        return (
            f"✅ Crawled URL: {url}\n"
            f"Stored {crawl_result['chunks']} chunk(s).\n"
            f"Language: {crawl_result['language']}\n\n"
            f"📄 **Context:**\n{context[:500]}...\n\n"
            f"💡 **Answer:** {answer}"
        )

    def _generate_answer(self, context: str, question: str) -> str:
        """
        Placeholder RAG answer logic.
        Plug your LLM call here!
        """
        if not context.strip():
            return "No relevant context found to answer your question."

        return (
            f"(Placeholder answer) Based on the retrieved context: "
            f"'{context[:100]}...', your question '{question}' is answered here."
        )

        # Example: if using OpenAI:
        # import openai
        # openai.api_key = self.openai_api_key
        # response = openai.ChatCompletion.create(
        #     model="gpt-4o",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful RAG assistant."},
        #         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        #     ]
        # )
        # return response.choices[0].message['content']
