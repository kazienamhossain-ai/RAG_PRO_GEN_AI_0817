import os
from dotenv import load_dotenv  # Fix: Removed trailing 'from'
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

loader = PyPDFLoader("document_loaders/GRU.pdf")
docs = loader.load()
print("Number of documents/pages:", len(docs))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)
print("Number of chunks:", len(chunks))

# Fix: Changed to the standard stable Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="text-embedding-004",
    google_api_key=api_key
)

vector = embeddings.embed_query(chunks[0].page_content)
print(f"Vector length: {len(vector)}")
