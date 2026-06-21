from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
import chromadb
import tempfile
from dotenv import load_dotenv
from groq import Groq
import os
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="pdf_chunks"
)

def chunk_text(text, size=500):
    chunks = []

    for i in range(0, len(text), size):
        chunks.append(text[i:i + size])

    return chunks

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    return {
        "message": "Deployment test successful"
    }

@app.get("/search")
def search(query: str):
    return {
        "query": query,
        "results": ["Deployment test successful"]
    }

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask_question(data: Question):

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": data.question
            }
        ]
    )

    return {
        "answer": response.choices[0].message.content
    }