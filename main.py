import os
import json
import base64
import numpy as np

from fastapi import FastAPI
from dotenv import load_dotenv

from google import genai
from google.oauth2 import service_account
from googleapiclient.discovery import build

import faiss

app = FastAPI()
load_dotenv()

# =========================
# GEMINI
# =========================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =========================
# DRIVE CONFIG
# =========================
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# =========================
# VECTOR STORE (FAISS)
# =========================
EMBED_DIM = 768
index = faiss.IndexFlatL2(EMBED_DIM)

chunks = []   # store text chunks
meta = []     # store metadata (source file)

# =========================
# DRIVE AUTH
# =========================
def get_drive_service():
    cred_json = base64.b64decode(os.getenv("GOOGLE_CREDENTIALS_B64"))
    creds_info = json.loads(cred_json)

    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=SCOPES
    )

    return build("drive", "v3", credentials=creds)

# =========================
# EMBEDDING
# =========================
def embed(text: str):
    res = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return np.array(res.embeddings[0].values, dtype=np.float32)

# =========================
# CHUNKING
# =========================
def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# =========================
# SYNC DRIVE → VECTOR DB
# =========================
def sync_drive():
    global chunks, meta, index

    service = get_drive_service()

    files = service.files().list(
        pageSize=20,
        fields="files(id, name, modifiedTime)"
    ).execute().get("files", [])

    for f in files:
        try:
            content = service.files().get_media(fileId=f["id"]).execute()
            text = content.decode("utf-8", errors="ignore")

            for c in chunk_text(text):
                vec = embed(c)

                index.add(np.array([vec]))
                chunks.append(c)
                meta.append({
                    "file": f["name"],
                    "id": f["id"]
                })

        except Exception as e:
            print(f"Skip {f['name']} -> {e}")

# =========================
# SEARCH (VECTOR)
# =========================
def search(query, k=3):
    q_vec = embed(query)

    D, I = index.search(np.array([q_vec]), k)

    results = []
    for i in I[0]:
        if i < len(chunks):
            results.append({
                "text": chunks[i],
                "meta": meta[i]
            })

    return results

# =========================
# INIT KB (manual trigger)
# =========================
@app.get("/sync")
def sync():
    sync_drive()
    return {
        "status": "synced",
        "chunks": len(chunks)
    }

# =========================
# CHAT (PRODUCTION RAG)
# =========================
@app.get("/chat")
def chat(q: str):

    docs = search(q)

    context = "\n\n".join(
        [f"{d['meta']['file']}:\n{d['text']}" for d in docs]
    )

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"""
Kamu adalah AI knowledge base internal perusahaan.

Gunakan konteks berikut:

{context}

Jika tidak ada jawaban di konteks, jawab: tidak ditemukan.

Pertanyaan:
{q}
"""
    )

    return {
        "question": q,
        "answer": response.text,
        "sources": [d["meta"]["file"] for d in docs]
    }

# =========================
# HEALTH
# =========================
@app.get("/")
def home():
    return {"status": "Production RAG running"}
