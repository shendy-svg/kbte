import os
import json
import base64
import numpy as np
import threading
import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from google import genai
from google.oauth2 import service_account
from googleapiclient.discovery import build

import faiss
from pypdf import PdfReader
from io import BytesIO

# ==============================
# 🔷 INIT APP
# ==============================
app = FastAPI()
load_dotenv()

# ==============================
# 🔷 GEMINI CLIENT (FIXED MODEL)
# ==============================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-flash-latest"

# ==============================
# 🔷 STORAGE (RAILWAY PERSISTENT)
# ==============================
BASE = "/data"

INDEX_PATH = f"{BASE}/index.faiss"
CHUNKS_PATH = f"{BASE}/chunks.json"
META_PATH = f"{BASE}/meta.json"
STATE_PATH = f"{BASE}/state.json"

EMBED_DIM = 768

# ==============================
# 🔷 FAISS INIT
# ==============================
def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatL2(EMBED_DIM)

index = load_index()

chunks = json.load(open(CHUNKS_PATH)) if os.path.exists(CHUNKS_PATH) else []
meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else []

# ==============================
# 🔷 DRIVE AUTH
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
# 🔷 EMBEDDING
# ==============================
def embed(text):
    res = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return np.array(res.embeddings[0].values, dtype=np.float32)

# ==============================
# 🔷 CHUNKING
# ==============================
def chunk_text(text, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]

# ==============================
# 🔷 PDF PARSER
# ==============================
def extract_pdf(file_bytes):
    reader = PdfReader(BytesIO(file_bytes))
    return "\n".join([p.extract_text() or "" for p in reader.pages])

# ==============================
# 🔷 DOWNLOAD FILE FROM DRIVE
# ==============================
def download_file(service, file_id, mime):

    if mime == "application/vnd.google-apps.document":
        return service.files().export_media(fileId=file_id, mimeType="text/plain").execute()

    if mime == "application/vnd.google-apps.spreadsheet":
        return service.files().export_media(fileId=file_id, mimeType="text/csv").execute()

    if mime == "application/vnd.google-apps.presentation":
        return service.files().export_media(fileId=file_id, mimeType="text/plain").execute()

    return service.files().get_media(fileId=file_id).execute()

# ==============================
# 🔷 SEARCH (RAG)
# ==============================
def search(query, k=5):
    if len(chunks) == 0:
        return []

    q = embed(query)
    D, I = index.search(np.array([q]), k)

    results = []
    for i in I[0]:
        if i < len(chunks):
            results.append({
                "text": chunks[i],
                "meta": meta[i]
            })

    return results

# ==============================
# 🔥 SYNC DRIVE (MANUAL / AUTO READY)
# ==============================
def sync_drive():
    global index, chunks, meta

    service = get_drive_service()
    state = {}

    files = service.files().list(
        pageSize=100,
        fields="files(id,name,mimeType,modifiedTime)"
    ).execute().get("files", [])

    for f in files:
        try:
            raw = download_file(service, f["id"], f["mimeType"])

            if f["mimeType"] == "application/pdf":
                text = extract_pdf(raw)
            else:
                text = raw.decode("utf-8", errors="ignore")

            if not text.strip():
                continue

            for c in chunk_text(text):
                vec = embed(c)

                index.add(np.array([vec]))
                chunks.append(c)
                meta.append({
                    "file": f["name"],
                    "type": f["mimeType"]
                })

        except Exception as e:
            print("skip:", f["name"], e)

    print("sync done:", len(chunks))

# ==============================
# 🔥 CHAT ENGINE (RAG + FALLBACK)
# ==============================
@app.get("/chat")
def chat(q: str):

    docs = search(q)

    context = "\n\n".join(
        [f"{d['meta']['file']}:\n{d['text']}" for d in docs]
    )

    # ======================
    # INTERNAL RAG MODE
    # ======================
    if context.strip():

        prompt = f"""
You are an enterprise AI assistant.

TASK:
- Answer using internal documents only
- Provide reasoning
- Provide sources (file names)
- Provide suggestions (2-3)

FORMAT:
Answer:
Reasoning:
Sources:
Suggestions:

DOCUMENTS:
{context}

QUESTION:
{q}
"""

        res = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return {
            "mode": "internal-rag",
            "answer": res.text,
            "sources": list(set([d["meta"]["file"] for d in docs]))
        }

    # ======================
    # EXTERNAL FALLBACK
    # ======================
    prompt = f"""
You are a senior AI assistant.

The question is NOT found in internal documents.

TASK:
- Answer clearly using general knowledge
- Provide reasoning
- Provide suggestions

QUESTION:
{q}
"""

    res = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return {
        "mode": "external-ai",
        "answer": res.text,
        "sources": []
    }

# ==============================
# 🔥 ROOT
# ==============================
@app.get("/")
def home():
    return {"status": "KB Team East AI running"}

# ==============================
# 🔥 UI (CHATGPT STYLE FIXED)
# ==============================
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html>
<head>
<title>KB Team East AI</title>

<style>
body {
    margin:0;
    font-family: Arial;
    background:#0b1220;
    color:white;
}

.container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}

.chat {
    height: 70vh;
    overflow-y: auto;
    background: #111827;
    padding: 15px;
    border-radius: 10px;
}

.msg {
    margin: 10px 0;
    padding: 10px;
    border-radius: 10px;
}

.user {
    background:#2563eb;
    text-align:right;
}

.ai {
    background:#1f2937;
}

.input {
    display:flex;
    margin-top:10px;
}

input {
    flex:1;
    padding:12px;
    border:none;
    border-radius:8px;
}

button {
    margin-left:10px;
    padding:12px;
    background:#22c55e;
    border:none;
    border-radius:8px;
    cursor:pointer;
}
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

async function send(){

let q = document.getElementById("q").value;
if(!q) return;

let chat = document.getElementById("chat");

chat.innerHTML += `<div class='msg user'>${q}</div>`;

let res = await fetch("/chat?q=" + encodeURIComponent(q));
let data = await res.json();

chat.innerHTML += `
<div class='msg ai'>
${data.answer}
<br><small>mode: ${data.mode}</small>
</div>
`;

document.getElementById("q").value = "";
chat.scrollTop = chat.scrollHeight;
}

</script>

</body>
</html>
"""

# ==============================
# 🔥 STARTUP OPTION
# ==============================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
