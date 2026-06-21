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
# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ChromaDB
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

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    # Read PDF
    reader = PdfReader(temp_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Chunk text
    chunks = chunk_text(text)

    # Create embeddings
    embeddings = model.encode(chunks).tolist()

    # Store in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(chunks))]
    )

    return {
        "message": "PDF stored successfully",
        "chunks": len(chunks)
    }


@app.get("/search")
def search(query: str):

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return {
        "query": query,
        "results": results["documents"][0]
    }

class Question(BaseModel):
    question: str


@app.post("/ask")
def ask_question(data: Question):

    query_embedding = model.encode(data.question).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
Answer the question using only the context below.

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
# model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = model.encode(chunks).tolist()
# query_embedding = model.encode(query).tolist()
# query_embedding = model.encode(data.question).tolist()
    return {"message": "test deploy"}