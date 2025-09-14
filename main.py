from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, requests, numpy as np
from qdrant_client import QdrantClient

app = FastAPI()

# --- ENV VARS ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_MODEL}"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

class AskReq(BaseModel):
    query: str

def get_embedding(text: str):
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    r = requests.post(HF_URL, headers=headers, json={"inputs": text})
    if r.status_code != 200:
        raise HTTPException(status_code=500, detail="HF error")
    arr = np.array(r.json())
    if arr.ndim == 2:
        return arr.mean(axis=0).tolist()
    return arr.tolist()

@app.post("/ask")
def ask(req: AskReq):
    emb = get_embedding(req.query)
    hits = qdrant.search(QDRANT_COLLECTION, query_vector=emb, limit=5)
    context = "\n".join([str(h.payload) for h in hits])
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful RAG assistant."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.query}"}
        ]
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
        json=payload
    )
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="OpenRouter error")
    return resp.json()
