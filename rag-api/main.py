from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import ollama
import uuid
import os

app = FastAPI(title="Local RAG API")


COLLECTION_NAME = "my_database"
CHROMA_HOST = os.getenv("CHROMA_HOST", "rag-chroma-db")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")


try:
    chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=120.0)
except Exception as e:
    print(f"Ошибка при инициализации клиентов: {e}")

class EmbedRequest(BaseModel):
    text: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class IngestRequest(BaseModel):
    text: str
    metadata_source: str = "user_upload"

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
        return {"results": [{"text": doc, "source": meta.get("source", "unknown")} for doc, meta in zip(documents, metadatas)]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest", operation_id="add_to_knowledge_base")
def ingest_text(req: IngestRequest):
    """Добавить новый текст в базу знаний."""
    print("⏳ [INGEST] Начало обработки запроса...", flush=True)
    try:
        print(f"🤖 [INGEST] Запрос эмбеддинга у Ollama (модель: {EMBEDDING_MODEL})...", flush=True)
        response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=req.text)
        print("✅ [INGEST] Эмбеддинг получен!", flush=True)
        
        embedding = response["embedding"]
        doc_id = str(uuid.uuid4())
        
        print("🗄️ [INGEST] Сохранение в ChromaDB...", flush=True)
        collection.add(
            embeddings=[embedding],
            documents=[req.text],
            metadatas=[{"source": req.metadata_source}],
            ids=[doc_id]
        )
        print("🎉 [INGEST] Успешно сохранено!", flush=True)
        return {"status": "success", "message": f"Текст добавлен в базу знаний (ID: {doc_id})", "id": doc_id}
    except Exception as e:
        print(f"❌ [INGEST] ОШИБКА: {e}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))
