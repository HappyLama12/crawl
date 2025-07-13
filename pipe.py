"""
title: CrawlRAG Pipeline with Ollama RAG
author: Your Name
date: 2025-07-13
version: 1.1
license: MIT
description: Crawl URL, chunk & store content in ChromaDB, then answer questions using Ollama deepseek-r1:14b model.
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
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_urls(self, text: str) -> List[str]:
        pattern = re.compile(r"(https?://[^\s\"'<>]+)", re.IGNORECASE)
        return pattern.findall(text)

    def chunk_text(self, text: str, max_words: int = 100) -> List[str]:
        """
        Chunk text into smaller pieces with max_words each.
        """
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        return chunks

    def rag_answer(self, question: str) -> str:
        """
        Query ChromaDB and ask Ollama deepseek-r1:14b model.
        """
        try:
            # Embed question
            question_emb = self.embedder.encode([question]).tolist()

            # Retrieve top 3 relevant docs from ChromaDB
            results = self.collection.query(
                query_embeddings=question_emb,
                n_results=3
            )
            retrieved_chunks = []
            for docs in results.get("documents", []):
                retrieved_chunks.extend(docs)

            context = "\n\n".join(retrieved_chunks)

            payload = {
                "model": self.ollama_model,
                "messages": [
                    {"role": "system",
                     "content": "You are a helpful assistant. Use the provided context to answer the user's question accurately."},
                    {"role": "user",
                     "content": f"Context:\n{context}\n\nQuestion: {question}"}
                ]
            }

            res = requests.post(self.ollama_url, json=payload, timeout=120)
            res.raise_for_status()
            response = res.json()

            # Ollama response might vary slightly; check keys:
            if "message" in response:
                return response["message"]["content"].strip()
            elif "choices" in response:
                return response["choices"][0]["message"]["content"].strip()
            else:
                return "❌ Unexpected response format from Ollama."

        except Exception as e:
            return f"❌ RAG error: {e}"

    def pipe(
            self,
            user_message: str,
            model_id: str,
            messages: List[dict],
            body: dict
    ) -> Union[str, Generator, Iterator]:
        """
        Crawl + chunk + store + optional RAG answer.
        """
        self._init_clients()

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

        try:
            res = requests.post(self.crawl4ai_url, json={"urls": [url]}, timeout=60)
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
            return f"❌ No text content returned from Crawl4AI for: {url}\nRaw response: {result}"

        # Chunk the texts to keep chunks small
        text_content = []
        for t in raw_texts:
            text_content.extend(self.chunk_text(t.strip(), max_words=100))

        try:
            language = detect(" ".join(text_content))
        except Exception:
            language = "unknown"

        # Check duplicates (rough check with first chunk)
        try:
            existing = self.collection.query(query_texts=[text_content[0]], n_results=1)
            if existing.get("documents") and existing["documents"][0]:
                return f"⚠️ Duplicate content detected. Skipping storage.\nURL: {url}\nLanguage: {language}"
        except Exception:
            pass

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

        answer = ""
        if question:
            answer = self.rag_answer(question)
            answer = f"\n\n💬 **Answer:**\n{answer}"

        joined_text = "\n\n".join(text_content[:3])

        return (
            f"✅ Successfully crawled and stored {len(text_content)} chunks.\n"
            f"URL: {url}\n"
            f"Language: {language}\n\n"
            f"📄 **Sample content:**\n{joined_text}"
            f"{answer}"
        )
