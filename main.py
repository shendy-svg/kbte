import os
import json
import base64
import time
import threading
import numpy as np

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from google import genai
from google.oauth2 import service_account
from googleapiclient.discovery import build

import faiss

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
# 🔷 DRIVE CONFIG
# ==============================
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# ==============================
# 🔷 VECTOR STORE (FAISS)
# ==============================
EMBED_DIM = 768
index = faiss.IndexFlatL2(EMBED_DIM)

chunks = []
meta = []

lock = threading.Lock()

# ==============================
# 🔷 AUTO SYNC CONFIG
# ==============================
SYNC_INTERVAL = 300  # 5 menit

# ==============================
# 🔷 DRIVE AUTH
# ==============================
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
def embed(text: str):
    res = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return np.array(res.embeddings[0].values, dtype=np.float32)

# ==============================
# 🔷 CHUNKING
# ==============================
def chunk_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

# ==============================
# 🔷 SYNC DRIVE → FAISS (FULL REBUILD)
# ==============================
def sync_drive():
    global chunks, meta, index

    print("🔄 Syncing Google Drive...")

    service = get_drive_service()

    files = service.files().list(
        pageSize=20,
        fields="files(id, name)"
    ).execute().get("files", [])

    new_chunks = []
    new_meta = []
    new_index = faiss.IndexFlatL2(EMBED_DIM)

    for f in files:
        try:
            content = service.files().get_media(fileId=f["id"]).execute()
            text = content.decode("utf-8", errors="ignore")

            for c in chunk_text(text):
                vec = embed(c)

                new_index.add(np.array([vec]))
                new_chunks.append(c)
                new_meta.append({"file": f["name"]})

        except Exception as e:
            print(f"Skip {f['name']}: {e}")

    # atomic swap (IMPORTANT)
    with lock:
        chunks = new_chunks
        meta = new_meta
        index = new_index

    print(f"✅ Sync complete. chunks={len(chunks)}")

# ==============================
# 🔁 AUTO SYNC LOOP
# ==============================
def auto_sync_loop():
    while True:
        try:
            sync_drive()
        except Exception as e:
            print("❌ Sync error:", e)

        time.sleep(SYNC_INTERVAL)

# ==============================
# 🔷 SEARCH (VECTOR)
# ==============================
def search(query, k=3):
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
# 🔷 STARTUP EVENT (AUTO SYNC)
# ==============================
@app.on_event("startup")
def startup_event():
    print("🚀 Starting KB AI...")

    # initial sync
    sync_drive()

    # background thread
    thread = threading.Thread(target=auto_sync_loop, daemon=True)
    thread.start()

    print("🔁 Auto-sync enabled")

# ==============================
# 🔷 ROOT
# ==============================
@app.get("/")
def home():
    return {"status": "RAG KB AI AUTO-SYNC RUNNING"}

# ==============================
# 🔷 CHAT API (RAG)
# ==============================
@app.get("/chat")
def chat(q: str):

    docs = search(q)

    context = "\n\n".join(
        [f"{d['meta']['file']}:\n{d['text']}" for d in docs]
    )

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=f"""
You are an internal knowledge base AI.

Use ONLY the following context:

{context}

If the answer is not found in the context, respond with: "not found".

Question:
{q}
"""
    )

    return {
        "question": q,
        "answer": response.text,
        "sources": [d["meta"]["file"] for d in docs]
    }

# ==============================
# 🔷 CHAT UI (STEP 9)
# ==============================
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>KB Team East AI Chat</title>
        <style>
            body {
                font-family: Arial;
                margin: 40px;
                background: #f5f5f5;
            }
            #chat {
                border: 1px solid #ccc;
                background: white;
                padding: 10px;
                height: 450px;
                overflow-y: auto;
                margin-bottom: 10px;
            }
            input {
                width: 75%;
                padding: 10px;
            }
            button {
                padding: 10px 15px;
            }
            .user { color: blue; margin: 5px 0; }
            .ai { color: green; margin: 5px 0; }
        </style>
    </head>
    <body>

        <h2>📚 KB Team East AI Chat (Auto-Sync)</h2>

        <div id="chat"></div>

        <input id="msg" placeholder="Ask me anything..." />
        <button onclick="send()">Send</button>

        <script>
            async function send() {
                let msg = document.getElementById("msg").value;

                if (!msg) return;

                document.getElementById("chat").innerHTML +=
                    "<div class='user'><b>You:</b> " + msg + "</div>";

                let res = await fetch("/chat?q=" + encodeURIComponent(msg));
                let data = await res.json();

                document.getElementById("chat").innerHTML +=
                    "<div class='ai'><b>AI:</b> " + data.answer + "</div><br>";

                document.getElementById("msg").value = "";

                document.getElementById("chat").scrollTop =
                    document.getElementById("chat").scrollHeight;
            }
        </script>

    </body>
    </html>
    """

# ==============================
# 🔷 RAILWAY ENTRY POINT
# ==============================
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(app, host="0.0.0.0", port=port)
