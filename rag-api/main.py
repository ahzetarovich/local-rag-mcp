import logging
import os
import time
import uuid
from typing import Optional

import chromadb
import ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-api")

app = FastAPI(title="Local RAG API")

COLLECTION_NAME = "my_database"
CHROMA_HOST = os.getenv("CHROMA_HOST", "rag-chroma-db")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "100000"))

try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=120.0)
except Exception as e:
    logger.error(f"Error initializing clients: {e}")
    raise


class EmbedRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class IngestRequest(BaseModel):
    text: str
    metadata_source: str = "user_upload"
    chunk_index: Optional[int] = None
    chunk_total: Optional[int] = None

class DeleteRequest(BaseModel):
    ids: Optional[list[str]] = None
    source: Optional[str] = None
    where: Optional[dict] = None


@app.get("/health", operation_id="health_check")
def health():
    return {"status": "ok", "service": "rag-fastapi"}


@app.post("/embed", operation_id="create_embedding")
def embed_text(req: EmbedRequest):
    try:
        response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=req.text)
        return {"embedding": response["embedding"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", operation_id="search_knowledge_base")
def search(req: SearchRequest):
    try:
        response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=req.query)
        query_embedding = response["embedding"]
        results = collection.query(query_embeddings=[query_embedding], n_results=req.top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        return {"results": [
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index"),
                "chunk_total": meta.get("chunk_total"),
                "id": id_,
            }
            for doc, meta, id_ in zip(documents, metadatas, ids)
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", operation_id="add_to_knowledge_base")
def ingest_text(req: IngestRequest):
    logger.info(
        f"[INGEST] Start, text length={len(req.text)}, source={req.metadata_source}, "
        f"chunk={req.chunk_index}/{req.chunk_total}"
    )
    try:
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="Text is empty")
        if len(req.text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=413, detail=f"Text too long: {len(req.text)} > {MAX_TEXT_LENGTH}")

        metadata = {
            "source": req.metadata_source,
            "text_len": len(req.text),
        }
        if req.chunk_index is not None:
            metadata["chunk_index"] = req.chunk_index
        if req.chunk_total is not None:
            metadata["chunk_total"] = req.chunk_total

        t0 = time.time()
        response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=req.text)
        embedding = response["embedding"]
        doc_id = str(uuid.uuid4())
        collection.add(
            embeddings=[embedding],
            documents=[req.text],
            metadatas=[metadata],
            ids=[doc_id],
        )
        logger.info(f"[INGEST] Done in {time.time()-t0:.1f}s, id={doc_id}")
        return {"status": "success", "message": "Text added to knowledge base", "id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[INGEST] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete", operation_id="delete_from_knowledge_base")
def delete_knowledge(req: DeleteRequest):
    logger.info(f"[DELETE] ids={req.ids}, source={req.source}, where={req.where}")
    try:
        given = [x for x in (req.ids, req.source, req.where) if x]
        if not given:
            raise HTTPException(status_code=400, detail="Provide ids, source, or where filter")
        if len(given) > 1:
            raise HTTPException(status_code=400, detail="Provide only one of ids, source, where")

        if req.ids:
            kwargs = {"ids": req.ids}
        elif req.source:
            kwargs = {"where": {"source": req.source}}
        else:
            kwargs = {"where": req.where}

        before = collection.count()
        collection.delete(**kwargs)
        after = collection.count()
        deleted = before - after
        logger.info(f"[DELETE] Deleted {deleted} entries (before={before}, after={after})")
        return {"status": "success", "deleted": deleted, "before": before, "after": after}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[DELETE] ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sources", operation_id="list_sources")
def list_sources(limit: int = 100, offset: int = 0):
    """Список источников: сколько чанков и символов в каждом."""
    try:
        agg = {}
        batch = 1000
        pos = 0
        while True:
            result = collection.get(limit=batch, offset=pos, include=["metadatas"])
            ids = result["ids"]
            if not ids:
                break
            for meta in result["metadatas"]:
                meta = meta or {}
                src = meta.get("source", "unknown")
                item = agg.setdefault(src, {
                    "source": src,
                    "chunks": 0,
                    "chars": 0,
                    "chunk_total": 0,
                })
                item["chunks"] += 1
                item["chars"] += int(meta.get("text_len") or 0)
                item["chunk_total"] = max(item["chunk_total"], int(meta.get("chunk_total") or 0))
            if len(ids) < batch:
                break
            pos += batch

        items = sorted(agg.values(), key=lambda x: x["source"].lower())
        for item in items:
            item["chunk_total"] = max(item["chunk_total"], item["chunks"])
        return {
            "total": len(items),
            "limit": limit,
            "offset": offset,
            "items": items[offset:offset + limit],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list", operation_id="list_chunks")
def list_chunks(limit: int = 100, offset: int = 0, source: Optional[str] = None):
    """Чанки базы знаний, при желании — только одного источника."""
    try:
        kwargs = {"limit": limit, "offset": offset, "include": ["metadatas", "documents"]}
        if source:
            kwargs["where"] = {"source": source}
        result = collection.get(**kwargs)
        if source:
            total = len(collection.get(where={"source": source}, include=["metadatas"])["ids"])
        else:
            total = collection.count()
        items = [
            {
                "id": id_,
                "source": (meta or {}).get("source", "unknown"),
                "chunk_index": (meta or {}).get("chunk_index"),
                "chunk_total": (meta or {}).get("chunk_total"),
                "text_len": len(doc),
            }
            for id_, meta, doc in zip(result["ids"], result["metadatas"], result["documents"])
        ]
        return {"total": total, "limit": limit, "offset": offset, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
