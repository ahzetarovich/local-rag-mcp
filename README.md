# Local RAG System for AI Agents

A fully local Retrieval-Augmented Generation (RAG) backend designed to give AI agents long-term memory and access to local documents.

## Architecture
* **FastAPI**: REST API for document ingestion and semantic search.
* **ChromaDB**: Vector database for storing document embeddings.
* **Ollama**: Local LLM inference for generating embeddings.
* **FastMCP**: Model Context Protocol server to connect AI agents to the RAG backend.
* **Docker Compose**: For easy deployment and isolation.

## Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/ahzetarovich/local-rag-mcp.git
   cd local-rag-mcp
2. Create your environment file:
   ```bash
   cp .env.example .env
3. Edit .env if needed (defaults are set for Docker internal network).
4. Start the services:
   ```bash
   docker compose up -d --build

## MCP Tools
This server exposes three tools via FastMCP:

    add_to_knowledge_base(text, metadata_source): Chunks and saves text to ChromaDB.
    
    search_knowledge_base(query, top_k): Finds relevant context based on semantic similarity.
    
    health_check(): Checks if the API is running.

## Configuration
All settings are managed via .env:

    OLLAMA_HOST: URL to Ollama API (default: http://ollama:11434).

    CHROMA_HOST: Hostname of ChromaDB (default: rag-chroma-db).

    EMBEDDING_MODEL: Ollama model for embeddings (default: nomic-embed-text:latest).

