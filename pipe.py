"""
title: CrawlRAG Pipeline
author: Your Name
date: 2025-07-13
version: 1.2
license: MIT
description: Crawls a URL, stores it in ChromaDB, and answers a question using RAG.
requirements: sentence-transformers, chromadb, langdetect, requests, openai
"""

import os
import re
import time
import requests
from sentence_transformers import SentenceTransformer
from langdetect import detect
from urllib.parse import urlparse
from typing import List, Union
import chromadb
from chromadb.config import Settings
from openai import OpenAI  # or use your preferred LLM client

class Pipeline:
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None
        self.persist_directory = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db/crawled")
        self.crawl4ai_url = os.getenv("CRAWL4AI_URL", "http://crawl4ai:11235/crawl")
        self.llm_client = OpenAI()  # Make sure you set OPENAI_API_KEY

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
            # Or use HttpClient if remote
        if not self.collection:
            self.collection = self.client.get_or_create_collection("crawled_data")
        if not self.embedder:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def extract_urls(self, text: str) -> List[str]:
        pattern = re.compile(r"(https?://[^\s\"'<>]+)", re.IGNORECASE)
        return pattern.findall(text)

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict
    ) -> str:
        """
        Crawl + store + answer.
        """
        self._init_clients()

        # 1️⃣ Get URL
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

        # Extract text
        texts = result.get("texts")
        single_text = result.get("text")
        text_content = []

        if texts and isinstance(texts, list) and any(texts):
            text_content = [t.strip() for t in texts if t and t.strip()]
        elif single_text and isinstance(single_text, str) and single_text.strip():
            text_content = [single_text.strip()]

        if not text_content:
            return f"❌ No text content returned from Crawl4AI for: {url}\nRaw response: {result}"

        # 3️⃣ Detect language
        try:
            language = detect(" ".join(text_content))
        except Exception:
            language = "unknown"

        # 4️⃣ Check for duplicates
        try:
            existing = self.collection.query(
                query_texts=[text_content[0]], n_results=1
            )
            if existing.get("documents") and existing["documents"][0]:
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

        # 6️⃣ If user gave a question, run RAG
        answer = ""
        if question:
            rag_result = self.rag(question)
            answer = f"\n\n💬 **Answer to your question:**\n{rag_result}"

        return (
            f"✅ Successfully crawled and stored {len(text_content)} chunk(s).\n"
            f"URL: {url}\n"
            f"Language: {language}\n\n"
            f"📄 **Sample content:**\n{text_content[0][:500]}...\n"
            f"{answer}"
        )

    def rag(self, question: str) -> str:
        """
        Retrieve relevant chunks and answer with LLM.
        """
        # Embed the question
        question_embedding = self.embedder.encode([question]).tolist()[0]

        # Query ChromaDB for top 3 relevant docs
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        retrieved_chunks = []
        for docs in results["documents"]:
            retrieved_chunks.extend(docs)

        context = "\n\n".join(retrieved_chunks)

        # Use OpenAI to generate the answer
        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant using the retrieved context below "
                            "to answer the question as accurately as possible."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}"
                    }
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"❌ RAG LLM error: {e}"
