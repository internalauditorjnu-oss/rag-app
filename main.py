from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from qdrant_client import QdrantClient

app = FastAPI()

# Load environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
OPENROUTER_EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "text-embedding-3-small")

# Initialize Qdrant client
qdrant = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)


# Request schema
class QueryRequest(BaseModel):
    query: str


def get_embedding(text: str):
    """Get embeddings from OpenRouter (instead of Hugging Face)."""
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    data = {"model": OPENROUTER_EMBED_MODEL, "input": text}

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="OpenRouter embedding error")

    return response.json()["data"][0]["embedding"]


def query_llm(context: str, question: str):
    """Send query + context to OpenRouter LLM."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for Q&A."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="OpenRouter LLM error")

    return response.json()["choices"][0]["message"]["content"]


@app.post("/ask")
def ask(request: QueryRequest):
    # Step 1: Embed query
    query_vector = get_embedding(request.query)

    # Step 2: Search in Qdrant
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=3,
    )

    if not hits:
        return {"answer": "No relevant documents found."}

    # Step 3: Build context
    context = "\n".join([hit.payload.get("text", "") for hit in hits])

    # Step 4: Ask LLM with retrieved context
    answer = query_llm(context, request.query)

    return {"query": request.query, "answer": answer}


@app.get("/")
def home():
    return {"message": "RAG pipeline live. Use /docs to test."}
