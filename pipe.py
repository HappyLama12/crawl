"""
title: CrawlRAG Pipeline
author: Your Name
date: 2025-07-13
version: 1.5
license: MIT
description: Crawls a URL, chunks & stores it in ChromaDB, and answers a question using RAG with Ollama.
requirements: sentence-transformers, chromadb, langdetect, requests
"""

import os
import re
import time
import textwrap
import requests
from sentence_transformers import SentenceTransformer
from langdetect import detect
from urllib.parse import urlparse
from typing import List
import chromadb
from chromadb.config import Settings


class Pipeline:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db/crawled")
        self.crawl4ai_url = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235/crawl")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://ollama:11434/api/chat")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:14b")

    async def on_startup(self):
        self._init_clients()

    async def on_shutdown(self):
        self.client = None
        self.collection = None
        self.embedder = None

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
        pattern = re.compile(r"(https?://[^\s\"\'<>]+)", re.IGNORECASE)
        return pattern.findall(text)

    def chunk_text(self, text: str, max_chars: int = 1000) -> List[str]:
        """
        Split long text into smaller chunks by paragraphs and wrap long ones.
        """
        paragraphs = text.split("\n")
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            wrapped = textwrap.wrap(para, width=max_chars)
            chunks.extend(wrapped)
        return chunks

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> str:
        """
        Crawl, chunk, store & optionally answer with RAG.
        """
        try:
            self._init_clients()

            # 1️⃣ Get URL & question
            url = body.get("url")
            question = body.get("question")
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

            # 3️⃣ Extract & chunk text
            texts = result.get("texts")
            single_text = result.get("text")
            text_content = []

            if texts and isinstance(texts, list):
                for t in texts:
                    if t and t.strip():
                        text_content.extend(self.chunk_text(t.strip()))
            elif single_text and isinstance(single_text, str) and single_text.strip():
                text_content = self.chunk_text(single_text.strip())

            if not text_content:
                return f"❌ No text content returned from Crawl4AI for: {url}\nRaw response: {result}"

            # 4️⃣ Detect language
            try:
                language = detect(" ".join(text_content[:1]))
            except Exception:
                language = "unknown"

            # 5️⃣ Check for duplicates
            try:
                existing = self.collection.query(
                    query_texts=[text_content[0]], n_results=1
                )
                if existing.get("documents") and existing["documents"][0]:
                    return f"⚠️ Duplicate content detected. Skipping storage.\nURL: {url}\nLanguage: {language}"
            except Exception:
                pass

            # 6️⃣ Embed & store
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

            # 7️⃣ Run RAG if question provided
            answer = ""
            if question:
                rag_result = self.rag(question)
                answer = f"\n\n💬 **Answer to your question:**\n{rag_result}"

            return (
                f"✅ Successfully crawled & stored {len(text_content)} chunk(s).\n"
                f"URL: {url}\n"
                f"Language: {language}\n\n"
                f"📄 **Sample chunk:**\n{text_content[0][:500]}...\n"
                f"{answer}"
            )

        except Exception as e:
            return f"❌ Pipeline internal error: {e}"

    def rag(self, question: str) -> str:
        """
        Retrieve relevant chunks & answer with Ollama.
        """
        try:
            question_embedding = self.embedder.encode([question]).tolist()[0]

            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=3
            )
            retrieved_chunks = []
            for docs in results["documents"]:
                retrieved_chunks.extend(docs)

            context = "\n\n".join(retrieved_chunks)

            payload = {
                "model": self.ollama_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. "
                            "Use the provided context to answer the user's question accurately."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ]
            }

            res = requests.post(self.ollama_url, json=payload, timeout=120)
            res.raise_for_status()
            response = res.json()

            if "message" in response:
                return response["message"]["content"].strip()
            elif "choices" in response:
                return response["choices"][0]["message"]["content"].strip()
            else:
                return f"❌ Unexpected Ollama response: {response}"

        except Exception as e:
            return f"❌ RAG LLM error: {e}"
