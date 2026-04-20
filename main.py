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
# 🔷 GEMINI
# ==============================
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ==============================
# 🔷 STORAGE PATH (RAILWAY VOLUME)
# ==============================
BASE_PATH = "/data"

INDEX_PATH = f"{BASE_PATH}/index.faiss"
CHUNKS_PATH = f"{BASE_PATH}/chunks.json"
META_PATH = f"{BASE_PATH}/meta.json"
STATE_PATH = f"{BASE_PATH}/state.json"

EMBED_DIM = 768

# ==============================
# 🔷 LOAD OR INIT FAISS
# ==============================
def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return faiss.IndexFlatL2(EMBED_DIM)

def save_index(index):
    os.makedirs(BASE_PATH, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)

# ==============================
# 🔷 STATE
# ==============================
def load_state():
    if os.path.exists(STATE_PATH):
        return json.load(open(STATE_PATH))
    return {}

def save_state(state):
    json.dump(state, open(STATE_PATH, "w"))

# ==============================
# 🔷 DRIVE AUTH
# ==============================
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

def get_drive_service():
    cred_json = base64.b64decode(os.getenv("GOOGLE_CREDENTIALS_B64"))
    creds_info = json.loads(cred_json)

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
# 🔷 CHUNK
# ==============================
def chunk_text(text, size=800):
    return [text[i:i+size] for i in range(0, len(text), size)]

# ==============================
# 🔷 TEXT EXTRACTOR
# ==============================
def extract_text(file_bytes, mime_type):
    try:
        # PDF
        if mime_type == "application/pdf":
            reader = PdfReader(BytesIO(file_bytes))
            return "\n".join([p.extract_text() or "" for p in reader.pages])

        # CSV / Docs export / TXT
        return file_bytes.decode("utf-8", errors="ignore")

    except Exception as e:
        print("extract error:", e)
        return ""

# ==============================
# 🔷 DOWNLOAD FILE
# ==============================
def download_file(service, file_id, mime_type):

    if mime_type == "application/vnd.google-apps.document":
        return service.files().export_media(
            fileId=file_id,
            mimeType="text/plain"
        ).execute()

    elif mime_type == "application/vnd.google-apps.spreadsheet":
        return service.files().export_media(
            fileId=file_id,
            mimeType="text/csv"
        ).execute()

    elif mime_type == "application/vnd.google-apps.presentation":
        return service.files().export_media(
            fileId=file_id,
            mimeType="text/plain"
        ).execute()

    else:
        return service.files().get_media(fileId=file_id).execute()

# ==============================
# 🔷 GLOBAL STORE (IN MEMORY CACHE)
# ==============================
index = load_index()
chunks = json.load(open(CHUNKS_PATH)) if os.path.exists(CHUNKS_PATH) else []
meta = json.load(open(META_PATH)) if os.path.exists(META_PATH) else []

# ==============================
# 🔷 SAVE STORE
# ==============================
def save_store():
    os.makedirs(BASE_PATH, exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    json.dump(chunks, open(CHUNKS_PATH, "w"))
    json.dump(meta, open(META_PATH, "w"))

# ==============================
# 🔷 SYNC DRIVE (ENTERPRISE)
# ==============================
def sync_drive():
    global index, chunks, meta

    service = get_drive_service()
    state = load_state()

    files = service.files().list(
        pageSize=100,
        fields="files(id,name,mimeType,modifiedTime)"
    ).execute().get("files", [])

    for f in files:
        file_id = f["id"]

        # skip unchanged file
        if file_id in state and state[file_id] == f["modifiedTime"]:
            continue

        try:
            file_bytes = download_file(service, file_id, f["mimeType"])
            text = extract_text(file_bytes, f["mimeType"])

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

            state[file_id] = f["modifiedTime"]

        except Exception as e:
            print("skip:", f["name"], e)

    save_store()
    save_state(state)

    print("✅ SYNC DONE:", len(chunks))

# ==============================
# 🔷 SEARCH
# ==============================
def search(query, k=5):
    if len(chunks) == 0:
        return []

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

# ==============================
# 🔷 BACKGROUND SYNC (OPTIONAL)
# ==============================
def background_sync():
    while True:
        try:
            sync_drive()
        except Exception as e:
            print("sync error:", e)

        time.sleep(300)  # 5 min

# ==============================
# 🔷 STARTUP
# ==============================
@app.on_event("startup")
def startup():
    threading.Thread(target=background_sync, daemon=True).start()

# ==============================
# 🔷 ROOT
# ==============================
@app.get("/")
def home():
    return {"status": "Enterprise RAG running"}

# ==============================
# 🔷 MANUAL SYNC
# ==============================
@app.post("/sync")
def manual_sync():
    sync_drive()
    return {"status": "synced"}

# ==============================
# 🔷 CHAT (RAG)
# ==============================
@app.get("/chat")
def chat(q: str):

    docs = search(q)

    context = "\n\n".join(
        [f"{d['meta']['file']}:\n{d['text']}" for d in docs]
    )

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"""
You are an enterprise internal knowledge assistant.

Use ONLY this context:

{context}

If not found, answer: not found.

Question:
{q}
"""
    )

    return {
        "answer": response.text,
        "sources": list(set([d["meta"]["file"] for d in docs]))
    }

# ==============================
# 🔷 UI
# ==============================
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <html>
    <body>
        <h2>Enterprise RAG Chat</h2>
        <input id="q" placeholder="Ask..." />
        <button onclick="send()">Send</button>
        <div id="chat"></div>

        <script>
        async function send(){
            let q = document.getElementById("q").value;
            let res = await fetch("/chat?q=" + q);
            let data = await res.json();

            document.getElementById("chat").innerHTML +=
                "<p><b>You:</b> " + q + "</p>" +
                "<p><b>AI:</b> " + data.answer + "</p><hr/>";
        }
        </script>
    </body>
    </html>
    """
