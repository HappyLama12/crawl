from pydantic import BaseModel, Field
import httpx

class Pipeline:
    class Valves(BaseModel):
        url: str = Field(..., description="URL to crawl and summarize")

    async def pipe(self, body):
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post("http://localhost:8000/crawl", json={"url": self.valves.url})
                res.raise_for_status()
                markdown = res.json().get("markdown", "No markdown found.")
        except Exception as e:
            markdown = f"Failed to fetch content: {str(e)}"

        return {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that summarizes webpages."},
                {"role": "user", "content": f"Summarize this webpage content:

{markdown}"}
            ]
        }