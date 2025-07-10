"""
title: Data Analyst
author: User
description: Data Analyst
required_open_webui_version: 0.4.3
version: 0.1
licence: MIT
"""

from openwebui.pipelines import Pipeline, PipelineInput, PipelineOutput

import requests
import chromadb
import time
from sentence_transformers import SentenceTransformer
from langdetect import detect
from urllib.parse import urlparse

class CrawlRAGPipeline(Pipeline):
    """
    Crawl a URL with Crawl4AI, embed content, store in ChromaDB.
    """

    def __init__(self):
        super().__init__()
        self.client = chromadb.HttpClient(host="chromadb", port=8000)
        self.collection = self.client.get_or_create_collection("crawled_data")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.crawl4ai_url = "http://crawl4ai:11235/crawl"

    def run(self, input: PipelineInput) -> PipelineOutput:
        url = input.input.get("url")
        if not url:
            return PipelineOutput(error="❌ Missing URL.")

        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return PipelineOutput(error="❌ Invalid URL scheme. Use http or https.")

        try:
            res = requests.post(self.crawl4ai_url, json={"url": url})
            res.raise_for_status()
            result = res.json()
        except Exception as e:
            return PipelineOutput(error=f"❌ Crawl4AI error: {e}")

        text_content = result.get("texts") or [result.get("text")]
        if not text_content:
            return PipelineOutput(error="❌ No text content returned.")

        try:
            language = detect(" ".join(text_content))
        except Exception:
            language = "unknown"

        try:
            existing = self.collection.query(query_texts=text_content, n_results=1)
            if existing.get("documents"):
                return PipelineOutput(output={
                    "message": "⚠️ Duplicate content detected. Skipping storage.",
                    "url": url,
                    "language": language
                })
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
            return PipelineOutput(error=f"❌ Embedding/storage error: {e}")

        return PipelineOutput(output={
            "message": f"✅ Successfully crawled and stored {len(text_content)} chunks.",
            "url": url,
            "language": language
        })

def register():
    return CrawlRAGPipeline()
