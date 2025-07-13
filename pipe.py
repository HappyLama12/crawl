"""
title: CrawlRAG Pipeline
author: Your Name
version: 1.0.0
license: MIT
description: |
  Crawl a URL using Crawl4AI, chunk and embed the content in ChromaDB,
  then answer questions using Ollama DeepSeek-R1:14b with RAG.
requirements:
  - requests
  - chromadb
  - sentence-transformers
  - langdetect
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
            self.client = chromadb.HttpClient(host="chromadb", port=8000)
        if not self.collection:
            self.collection = self.client.get_or_create_collection("crawled_data")
        if not self.embedder:
            self.embedder = SentenceTransformer("intfloat/multilingual-e5-large")

    def extract_urls(self, text: str) -> List[str]:
        pattern = re.compile(r"(https?://[^\s\"'<>]+)", re.IGNORECASE)
        return pattern.findall(text)

    def chunk_text(self, text: str, max_words: int = 100) -> List[str]:
        words = text.split()
        return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

    def rag_answer(self, question: str) -> str:
        try:
            question_emb = self.embedder.encode([question]).tolist()
            results = self.collection.query(query_embeddings=question_emb, n_results=3)
            chunks = []
            for docs in results.get("documents", []):
                chunks.extend(docs)
            if not chunks:
                return "❌ No relevant chunks found."

            context = "\n\n".join(chunks)
            payload = {
                "model": self.ollama_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant. Use the provided context to answer accurately."
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
            return f"❌ RAG error: {e}"

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        self._init_clients()
        url = body.get("url")
        question = body.get("question")

        if not question:
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    question = msg["content"]
                    break

        if not url:
            urls = self.extract_urls(user_message)
            url = urls[0] if urls else None

        if not url:
            return "❌ Missing URL."

        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return f"❌ Invalid URL scheme: {url}"

        try:
            res = requests.post(self.crawl4ai_url, json={"urls": [url]}, timeout=90)
            res.raise_for_status()
            result = res.json()
        except Exception as e:
            return f"❌ Crawl4AI error: {e}"

        texts = result.get("texts")
        single_text = result.get("text")
        raw_texts = []
        if texts and isinstance(texts, list) and any(texts):
            raw_texts = texts
        elif single_text and isinstance(single_text, str) and single_text.strip():
            raw_texts = [single_text]

        if not raw_texts:
            return f"❌ No text from Crawl4AI for: {url}"

        text_chunks = []
        for t in raw_texts:
            text_chunks.extend(self.chunk_text(t.strip(), max_words=100))

        try:
            language = detect(" ".join(text_chunks[:1]))
        except Exception:
            language = "unknown"

        try:
            existing = self.collection.query(query_texts=[text_chunks[0]], n_results=1)
            if existing.get("documents") and existing["documents"][0]:
                return f"⚠️ Duplicate detected. Skipping.\nURL: {url}\nLanguage: {language}"
        except Exception:
            pass

        try:
            embeddings = self.embedder.encode(text_chunks).tolist()
            ids = [f"{int(time.time())}-{i}" for i in range(len(text_chunks))]
            metadatas = [{"url": url, "language": language} for _ in text_chunks]
            self.collection.add(documents=text_chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
        except Exception as e:
            return f"❌ Storage error: {e}"

        answer = ""
        if question:
            answer = self.rag_answer(question)
            answer = f"\n\n💬 **Answer:**\n{answer}"

        sample = "\n\n".join(text_chunks[:3])
        return (
            f"✅ Stored {len(text_chunks)} chunks.\n"
            f"URL: {url}\n"
            f"Language: {language}\n\n"
            f"📄 **Sample:**\n{sample}{answer}"
        )

# -- Open WebUI hooks --
pipeline = Pipeline()

async def on_startup():
    await pipeline.on_startup()

async def on_shutdown():
    await pipeline.on_shutdown()

def pipe(user_message: str, model_id: str, messages: list, body: dict) -> Union[str, Generator, Iterator]:
    return pipeline.pipe(user_message, model_id, messages, body)
