from fastapi import FastAPI
from dotenv import load_dotenv
import os
import base64
# Gemini (NEW SDK)
from google import genai

# Google Drive API
from google.oauth2 import service_account
from googleapiclient.discovery import build

app = FastAPI()

# ==============================
# 🔷 LOAD ENV
# ==============================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ==============================
# 🔷 GEMINI CLIENT
# ==============================
client = genai.Client(api_key=api_key)

# ==============================
# 🔷 GOOGLE DRIVE SETUP
# ==============================
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_drive_files():
    cred_json = base64.b64decode(os.getenv("GOOGLE_CREDENTIALS_B64"))
    credentials_info = json.loads(cred_json)

    creds = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=SCOPES
    )

    service = build('drive', 'v3', credentials=creds)

    results = service.files().list(
        pageSize=10,
        fields="files(id, name)"
    ).execute()

    return results.get('files', [])

# ==============================
# 🔷 ROOT
# ==============================
@app.get("/")
def home():
    return {"status": "KB AI running"}

# ==============================
# 🔷 TEST DRIVE FILES
# ==============================
@app.get("/files")
def list_files():
    files = get_drive_files()
    return {"files": files}

# ==============================
# 🔷 CHAT (BASIC)
# ==============================
@app.get("/chat")
def chat(q: str):

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=f"""
        Jawab berdasarkan knowledge base internal.
        Jika tidak ada, jawab: tidak ditemukan.

        Pertanyaan:
        {q}
        """
    )

    return {"answer": response.text}
