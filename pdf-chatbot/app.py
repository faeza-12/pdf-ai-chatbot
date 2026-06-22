from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
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

model = SentenceTransformer("all-MiniLM-L6-v2")

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

    embeddings = model.encode(chunks).tolist()

    try:
        existing = collection.get()

        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    except Exception:
        pass

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )

    return {
        "message": "PDF uploaded successfully",
        "chunks": len(chunks)
    }


@app.get("/search")
def search(query: str):

    query_embedding = model.encode(
        [query]
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return results["documents"][0]


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_question(data: Question):

    query_embedding = model.encode(
        [data.question]
    ).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
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