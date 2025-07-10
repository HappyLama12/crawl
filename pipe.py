import os, time, json
from urllib.parse import urlparse
from typing import List

import httpx
import chromadb
from pydantic import BaseModel, Field
from langdetect import detect
from sentence_transformers import SentenceTransformer
from openwebui.plugins.pipes import PipeBase


class PipeInput(BaseModel):
    url: str = Field(..., description="URL to crawl and index")

class Pipeline(PipeBase):  # ✅ Required name
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None
        self.crawl4ai_url = os.getenv("CRAWL4AI_URL", "http://localhost:11235/crawl")

    def _init_clients(self):
        if not self.client:
            self.client = chromadb.HttpClient(host="localhost", port=8000)
        if not self.collection:
            self.collection = self.client.get_or_create_collection("auto_rag_data")
        if not self.embedder:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    async def execute(self, input: PipeInput):
        self._init_clients()
        url = input.url

        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return f"❌ Invalid URL: {url}"

        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(self.crawl4ai_url, json={"urls": [url], "use_browser": True})
                res.raise_for_status()
                result = res.json()
        except Exception as e:
            return f"❌ Crawl4AI error: {e}"

        texts = result.get("texts") or [result.get("text")]
        texts = [t for t in texts if t and t.strip()]
        if not texts:
            return f"❌ No content returned from: {url}"

        try:
            language = detect(" ".join(texts))
        except:
            language = "unknown"

        try:
            embeddings = self.embedder.encode(texts).tolist()
            ids = [f"{int(time.time())}-{i}" for i in range(len(texts))]
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=[{"url": url, "language": language} for _ in texts]
            )
        except Exception as e:
            return f"❌ Storage error: {e}"

        return {
            "status": "✅ Indexed",
            "url": url,
            "chunks": len(texts),
            "preview": texts[:2]
        }
