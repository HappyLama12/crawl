from openwebui.plugins.pipes import PipeBase
from pydantic import BaseModel, Field
from urllib.parse import urlparse
from langdetect import detect
from sentence_transformers import SentenceTransformer
import chromadb
import httpx, re, time, os, json

class PipeInput(BaseModel):
    url: str = Field(..., description="URL to crawl and embed")

class WebAutoRAGPipe(PipeBase):
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedder = None
        self.crawl4ai_url = os.getenv("CRAWL4AI_URL", "http://localhost:11235/crawl")

    def _init_clients(self):
        if not self.client:
            self.client = chromadb.HttpClient(host="localhost", port=8000)
        if not self.collection:
            self.collection = self.client.get_or_create_collection("crawled_data")
        if not self.embedder:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    async def execute(self, input: PipeInput):
        self._init_clients()
        url = input.url

        parsed = urlparse(url)
        if not parsed.scheme.startswith("http"):
            return f"❌ Invalid URL scheme: {url} - Use http or https."

        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(self.crawl4ai_url, json={"urls": [url], "use_browser": True})
                res.raise_for_status()
                result = res.json()
        except Exception as e:
            return f"❌ Crawl4AI error: {e}"

        text_content = result.get("texts") or [result.get("text")]
        text_content = [t for t in text_content if t and t.strip()]
        if not text_content:
            return f"❌ No valid text content returned from Crawl4AI for: {url}"

        try:
            language = detect(" ".join(text_content))
        except:
            language = "unknown"

        try:
            existing = self.collection.query(query_texts=text_content, n_results=1) or {}
            if existing.get("documents"):
                return f"⚠️ Duplicate content detected. Skipping storage.\nURL: {url}\nLanguage: {language}"
        except Exception as e:
            return f"❌ Chroma query error: {e}"

        try:
            embeddings = self.embedder.encode(text_content).tolist()
            timestamp = int(time.time())
            ids = [f"{timestamp}-{i}" for i in range(len(text_content))]
            metadatas = [{"url": url, "language": language} for _ in text_content]

            self.collection.add(
                documents=text_content,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
        except Exception as e:
            return f"❌ Embedding/storage error: {e}"

        return {
            "status": "✅ Successfully crawled and stored content.",
            "url": url,
            "language": language,
            "total_chunks": len(text_content),
            "chunks_preview": text_content[:2]
        }
