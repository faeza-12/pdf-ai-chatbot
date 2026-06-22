from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
import re
import math
import hashlib
from chromadb.api.types import EmbeddingFunction
import chromadb
import tempfile
from dotenv import load_dotenv
from groq import Groq
import os
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


class SimpleHashEmbeddingFunction(EmbeddingFunction):

    def __call__(self, input):

        embeddings = []

        for text in input:

            vector = [0.0] * 384

            words = re.findall(r"\w+", text.lower())

            for word in words:
                index = int(
                    hashlib.md5(word.encode()).hexdigest(),
                    16
                ) % 384

                vector[index] += 1.0

            norm = math.sqrt(
                sum(x * x for x in vector)
            )

            if norm > 0:
                vector = [
                    x / norm
                    for x in vector
                ]

            embeddings.append(vector)

        return embeddings


client = chromadb.Client()

collection = client.get_or_create_collection(
    name="pdf_chunks",
    embedding_function=SimpleHashEmbeddingFunction()
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

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)

    text = ""

    for page in reader.pages:

        extracted = page.extract_text()

        if extracted:
            text += extracted

    chunks = chunk_text(text)

    try:
        existing = collection.get()

        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    except Exception:
        pass

    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))]
    )

    return {
        "message": "PDF uploaded successfully",
        "chunks": len(chunks)
    }


@app.get("/search")
def search(query: str):

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    return results["documents"][0]


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_question(data: Question):

    results = collection.query(
        query_texts=[data.question],
        n_results=3
    )

    context = "\n".join(
        results["documents"][0]
    )

    prompt = f"""
Use the PDF context below to answer the question.

Context:
{context}

Question:
{data.question}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "answer": response.choices[0].message.content
    }