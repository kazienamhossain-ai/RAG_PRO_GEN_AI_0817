import os
import shutil

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


# --------------------------------------------------
# Constants
# --------------------------------------------------

UPLOAD_DIR = "./document_loaders"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="KAZI PDF AI Assistant API"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)


# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)


# --------------------------------------------------
# Retriever
# --------------------------------------------------

retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10
    }
)


# --------------------------------------------------
# Gemini
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    google_api_key=api_key
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class QuestionRequest(BaseModel):

    question: str


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():
    return {"message": "PDF AI Assistant API is running"}


# --------------------------------------------------
# List PDFs
# --------------------------------------------------

@app.get("/pdfs")
def list_pdfs():
    files = [
        f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")
    ]
    result = []
    for f in files:
        path = os.path.join(UPLOAD_DIR, f)
        size = os.path.getsize(path)
        result.append({
            "name": f,
            "size": f"{round(size / 1024, 1)} KB"
        })
    return {"pdfs": result}


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Load and split
    loader = PyPDFLoader(save_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # Add to ChromaDB
    vector_db.add_documents(chunks)

    return {
        "message": f"{file.filename} uploaded and indexed successfully.",
        "chunks": len(chunks)
    }


# --------------------------------------------------
# Delete PDF
# --------------------------------------------------

@app.delete("/pdfs/{filename}")
def delete_pdf(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    os.remove(path)
    return {"message": f"{filename} deleted."}


# --------------------------------------------------
# Ask endpoint
# --------------------------------------------------

@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question

    # Retrieve relevant chunks
    retrieved_docs = retriever.invoke(question)

    # Create context
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Prompt
    prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not available in the context, say:

"I could not find the answer in the PDF."

Do not make up information.

Context:
{context}

User Question:
{question}
"""

    # Generate answer
    response = llm.invoke(prompt)

    answer = response.content

    if isinstance(answer, list):

        answer = "".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict)
        )

    return {
        "question": question,
        "answer": answer
    }