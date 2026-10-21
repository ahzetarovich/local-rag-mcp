import httpx
from fastmcp import FastMCP


RAG_API_URL = "http://rag-fastapi:8080"

mcp = FastMCP("Local RAG Tools")

@mcp.tool()
async def health_check() -> dict:
    async with httpx.AsyncClient(timeout = 10.0) as client:
        resp = await client.get(f"{RAG_API_URL}/health")
        resp.raise_for_status()
        return resp.json()

@mcp.tool()
async def search_knowledge_base(query: str, top_k: int = 5) -> dict:
    """Найти релевантные фрагменты текста в базе знаний."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{RAG_API_URL}/search",
            json={"query": query, "top_k": top_k}
        )
        resp.raise_for_status()
        return resp.json()

@mcp.tool()
async def add_to_knowledge_base(text: str, metadata_source: str = "user_upload") -> dict:
    """Добавить новый текст в базу знаний."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{RAG_API_URL}/ingest",
            json={"text": text, "metadata_source": metadata_source}
        )
        resp.raise_for_status()
        return resp.json()


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=8081)
