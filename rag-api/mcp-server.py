import os

import httpx
from fastmcp import FastMCP

RAG_API_URL = os.getenv("RAG_API_URL", "http://rag-fastapi:8080")

mcp = FastMCP("Local RAG Tools")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150


@mcp.tool()
async def health_check() -> dict:
    """Проверить, работает ли API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
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
    """Добавить текст в базу знаний. Текст режется на чанки по 1500 символов
    с перекрытием 150. Все чанки получают общий metadata_source, поэтому
    удалить их потом можно одним delete_from_knowledge_base(source=...)."""
    if not text or not text.strip():
        return {"status": "error", "message": "Текст пуст"}

    if len(text) <= CHUNK_SIZE:
        chunks = [text]
    else:
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + CHUNK_SIZE])
            start += CHUNK_SIZE - CHUNK_OVERLAP

    total_chunks = len(chunks)
    success_count = 0
    errors = []

    async with httpx.AsyncClient(timeout=600.0) as client:
        for i, chunk in enumerate(chunks):
            payload = {
                "text": chunk,
                "metadata_source": metadata_source,
                "chunk_index": i + 1,
                "chunk_total": total_chunks,
            }
            try:
                resp = await client.post(f"{RAG_API_URL}/ingest", json=payload)
                resp.raise_for_status()
                success_count += 1
            except Exception as e:
                errors.append(f"chunk {i + 1}: {e}")

    return {
        "status": "success" if success_count == total_chunks else "partial",
        "source": metadata_source,
        "chunks_loaded": success_count,
        "chunks_total": total_chunks,
        "errors": errors if errors else None,
    }


@mcp.tool()
async def delete_from_knowledge_base(
    ids: list[str] = None,
    source: str = None,
    where: dict = None,
) -> dict:
    """Удалить информацию из базы знаний. Указывается ровно один параметр:
    - ids: список ID чанков;
    - source: все чанки с данным metadata_source;
    - where: произвольный ChromaDB where-фильтр."""
    payload = {}
    if ids:
        payload["ids"] = ids
    if source:
        payload["source"] = source
    if where:
        payload["where"] = where
    if len(payload) != 1:
        return {"status": "error", "message": "Укажи ровно один из параметров: ids, source или where"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{RAG_API_URL}/delete", json=payload)
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def list_sources(limit: int = 100, offset: int = 0) -> dict:
    """Список источников в базе знаний: имя, число чанков и символов."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(
            f"{RAG_API_URL}/sources",
            params={"limit": limit, "offset": offset}
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def list_chunks(source: str, limit: int = 50, offset: int = 0) -> dict:
    """Показать чанки одного источника с их ID (пригодится для точечного удаления)."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(
            f"{RAG_API_URL}/list",
            params={"source": source, "limit": limit, "offset": offset}
        )
        resp.raise_for_status()
        return resp.json()


@mcp.tool()
async def add_file(file_path: str, metadata_source: str = None) -> dict:
    """Добавить текстовый файл в базу знаний. Путь — внутри контейнера MCP,
    то есть реально доступна только папка проекта /app (./rag-api на сервере).
    Файл с другого компьютера сначала нужно прочитать и передать через
    add_to_knowledge_base."""
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"Файл не найден: {file_path}"}

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        return {"status": "error", "message": f"Не удалось прочитать файл: {e}"}

    if not text.strip():
        return {"status": "error", "message": "Файл пуст"}

    src = metadata_source or os.path.basename(file_path)
    return await add_to_knowledge_base(text=text, metadata_source=src)


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "sse")
    port = int(os.getenv("MCP_PORT", "8081"))
    mcp.run(transport=transport, host="0.0.0.0", port=port)
