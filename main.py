import os
import json
import base64
import numpy as np

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from google import genai
from google.oauth2 import service_account
from googleapiclient.discovery import build

import faiss
from io import BytesIO

# ==============================
# INIT
# ==============================
app = FastAPI()
load_dotenv()

# ==============================
# GEMINI
# ==============================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 🔥 FIX: model
MODEL = "models/gemini-1.5-flash"

# ==============================
# STORAGE
# ==============================
BASE = "/data"
INDEX_PATH = f"{BASE}/index.faiss"
CHUNKS_PATH = f"{BASE}/chunks.json"
META_PATH = f"{BASE}/meta.json"

EMBED_DIM = 768

# ==============================
# LOAD INDEX
# ==============================
def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatL2(EMBED_DIM)

index = load_index()

chunks = json.load(open(CHUNKS_PATH)) if os.path.exists(CHUNKS_PATH) else []
meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else []

# ==============================
# DRIVE AUTH
# ==============================
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_drive_service():
    cred = base64.b64decode(os.getenv("GOOGLE_CREDENTIALS_B64"))
    creds_info = json.loads(cred)

    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=SCOPES
    )

    return build("drive", "v3", credentials=creds)

# ==============================
# EMBEDDING (SAFE)
# ==============================
def embed(text):
    try:
        res = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return np.array(res.embeddings[0].values, dtype=np.float32)
    except:
        return np.zeros((EMBED_DIM,), dtype=np.float32)

# ==============================
# SEARCH
# ==============================
def search(query, k=5):
    if len(chunks) == 0 or index.ntotal == 0:
        return []

    try:
        q = embed(query)
        D, I = index.search(np.array([q]), k)

        results = []
        for i in I[0]:
            if i != -1 and i < len(chunks):
                results.append({
                    "text": chunks[i],
                    "meta": meta[i]
                })

        return results
    except:
        return []

# ==============================
# CHAT ENGINE (FIXED RAG LOGIC)
# ==============================
@app.get("/chat")
def chat(q: str):

    try:
        docs = search(q)

        context = "\n\n".join(
            [f"{d['meta']['file']}: {d['text']}" for d in docs]
        ).strip()

        # 🔥 FIX: lebih realistis
        use_rag = len(docs) > 0

        if use_rag:
            prompt = f"""
You are an enterprise internal AI assistant.

Rules:
- Use ONLY internal documents
- If unsure, say "not found in documents"

Answer format:
Answer:
Reasoning:
Sources:

DOCUMENTS:
{context}

QUESTION:
{q}
"""
        else:
            prompt = f"""
You are a helpful AI assistant.

Answer clearly and concisely.

Question:
{q}
"""

        res = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        return {
            "mode": "internal-rag" if use_rag else "external-ai",
            "answer": res.text,
            "sources": list(set([d["meta"]["file"] for d in docs]))
        }

    except Exception as e:
        # 🔥 FIX: tampilkan error asli biar tidak buta
        return {
            "mode": "error",
            "answer": str(e)
        }

# ==============================
# ROOT
# ==============================
@app.get("/")
def home():
    return {"status": "KB AI running (stable mode)"}

# ==============================
# UI FIXED (ENTER + CLEAN)
# ==============================
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
<title>KB Team East AI</title>
<style>
body{margin:0;font-family:Arial;background:#0b1220;color:#fff;}
.container{max-width:800px;margin:auto;padding:20px;}
.chat{height:70vh;overflow-y:auto;background:#111827;padding:15px;border-radius:10px;}
.msg{margin:10px 0;padding:10px;border-radius:10px;white-space:pre-wrap;}
.user{background:#2563eb;text-align:right;}
.ai{background:#1f2937;}
.input{display:flex;margin-top:10px;}
input{flex:1;padding:12px;border:none;border-radius:8px;outline:none;}
button{margin-left:10px;padding:12px;background:#22c55e;border:none;border-radius:8px;cursor:pointer;}
</style>
</head>

<body>

<div class="container">
<h2>🧠 KB Team East AI</h2>

<div id="chat" class="chat"></div>

<div class="input">
<input id="q" placeholder="Ask anything..." />
<button onclick="send()">Send</button>
</div>

</div>

<script>

const input = document.getElementById("q");
const chat = document.getElementById("chat");

function add(text, cls){
    const div = document.createElement("div");
    div.className = "msg " + cls;
    div.innerText = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

async function send(){

    const q = input.value.trim();
    if(!q) return;

    add(q,"user");
    input.value="";

    const res = await fetch("/chat?q=" + encodeURIComponent(q));
    const data = await res.json();

    add(data.answer + "\\n\\nmode: " + data.mode, "ai");
}

/* ENTER FIX */
input.addEventListener("keydown", e=>{
    if(e.key==="Enter"){
        e.preventDefault();
        send();
    }
});

</script>

</body>
</html>
"""

# ==============================
# RUN
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
