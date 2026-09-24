import os
import time
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_exponential
from google.genai.errors import ClientError
from langchain_google_genai.chat_models import GoogleRateLimitError

# --------------------------------------------------
# Config
# --------------------------------------------------

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
UPLOAD_DIR = "./document_loaders"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(
    page_title="Kazi & Anurag PDF Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Black & Gold CSS
# --------------------------------------------------

st.markdown("""
<style>

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
[data-testid="stMain"], .main, .block-container {
    background-color: #0a0a0a !important;
    color: #d4b896 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d0d 0%, #111111 100%) !important;
    border-right: 1.5px solid #c9a227 !important;
    box-shadow: 6px 0 40px rgba(201,162,39,0.28), 2px 0 12px rgba(0,0,0,0.9) !important;
}
[data-testid="stSidebar"] * { color: #d4b896 !important; }

/* ── Gold Header ── */
.gold-header {
    background: linear-gradient(135deg, #1a1200 0%, #2a1f00 50%, #1a1200 100%);
    border: 1.5px solid #c9a227;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 16px;
    box-shadow:
        0 0 60px rgba(201,162,39,0.30),
        0 8px 40px rgba(0,0,0,0.85),
        inset 0 1px 0 rgba(201,162,39,0.15),
        inset 0 -1px 0 rgba(0,0,0,0.5);
    text-align: center;
}
.gold-header h1 {
    font-size: 32px;
    font-weight: 800;
    color: #c9a227 !important;
    text-shadow: 0 0 32px rgba(201,162,39,0.70), 0 2px 8px rgba(0,0,0,0.8);
    margin: 0 0 6px 0;
    letter-spacing: 0.5px;
}
.gold-header p {
    font-size: 13px;
    color: #7a6520 !important;
    margin: 0;
    letter-spacing: 1px;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0d0d0d !important;
    border-bottom: 1.5px solid #2a1f00 !important;
    gap: 12px !important;
    padding: 12px 20px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.7) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: #161616 !important;
    border: 1.5px solid #2a1f00 !important;
    border-radius: 999px !important;
    color: #5a4a1a !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 9px 28px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.03) !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.3px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    border-color: #c9a227 !important;
    color: #c9a227 !important;
    box-shadow: 0 4px 22px rgba(201,162,39,0.25) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #c9a227 0%, #e8c040 50%, #a07d1a 100%) !important;
    border-color: #c9a227 !important;
    color: #0a0a0a !important;
    box-shadow:
        0 6px 28px rgba(201,162,39,0.65),
        0 2px 8px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(255,255,255,0.25) !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background: transparent !important;
    display: none !important;
}
[data-testid="stTabContent"] {
    background: #0a0a0a !important;
    border: 1px solid #1a1500 !important;
    border-top: none !important;
    border-radius: 0 0 14px 14px !important;
    padding: 20px !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.7), inset 0 1px 0 rgba(201,162,39,0.05) !important;
}

/* ── Chat Bubbles ── */
.bubble-user {
    background: linear-gradient(135deg, #c9a227, #e8c040, #a07d1a);
    color: #0a0a0a;
    font-weight: 600;
    padding: 13px 20px;
    border-radius: 16px 16px 4px 16px;
    margin: 8px 0 8px 18%;
    box-shadow: 0 6px 28px rgba(201,162,39,0.50), 0 2px 8px rgba(0,0,0,0.6);
    font-size: 14px;
    line-height: 1.65;
}
.bubble-bot {
    background: #141414;
    color: #d4b896;
    border: 1px solid #2a2200;
    padding: 13px 20px;
    border-radius: 16px 16px 16px 4px;
    margin: 8px 18% 8px 0;
    box-shadow: 0 6px 28px rgba(0,0,0,0.65), inset 0 1px 0 rgba(201,162,39,0.06);
    font-size: 14px;
    line-height: 1.65;
}

/* ── PDF Cards ── */
.pdf-card {
    background: #141414;
    border: 1px solid #2a1f00;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 12px;
    box-shadow:
        0 6px 32px rgba(0,0,0,0.65),
        0 2px 8px rgba(0,0,0,0.5),
        inset 0 1px 0 rgba(201,162,39,0.10);
    transition: all 0.25s ease;
}
.pdf-card:hover {
    border-color: #c9a227;
    box-shadow: 0 8px 40px rgba(201,162,39,0.22), 0 2px 8px rgba(0,0,0,0.6);
    transform: translateY(-1px);
}
.pdf-item {
    background: #141414;
    border: 1px solid #2a1f00;
    border-radius: 12px;
    padding: 13px 18px;
    margin-bottom: 10px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.55), inset 0 1px 0 rgba(201,162,39,0.05);
    transition: all 0.2s ease;
}
.pdf-item:hover { border-color: #5a4a1a; box-shadow: 0 4px 24px rgba(0,0,0,0.7); }
.pdf-item-active {
    border-color: #c9a227 !important;
    background: rgba(201,162,39,0.07) !important;
    box-shadow: 0 0 30px rgba(201,162,39,0.28), inset 0 1px 0 rgba(201,162,39,0.12) !important;
}

/* ── Badges ── */
.badge-active {
    background: rgba(201,162,39,0.18);
    color: #c9a227;
    border: 1px solid rgba(201,162,39,0.4);
    border-radius: 999px;
    padding: 3px 11px;
    font-size: 11px;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(201,162,39,0.2);
}
.badge-ready {
    background: rgba(255,255,255,0.03);
    color: #4a3a10;
    border: 1px solid #2a1f00;
    border-radius: 999px;
    padding: 3px 11px;
    font-size: 11px;
    font-weight: 600;
}

/* ── Section Titles ── */
.section-title {
    font-size: 11px;
    font-weight: 800;
    color: #c9a227;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 16px 0 10px 0;
    text-shadow: 0 0 12px rgba(201,162,39,0.35);
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #c9a227, #e8c040, #a07d1a) !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 18px rgba(201,162,39,0.40), 0 2px 6px rgba(0,0,0,0.5) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    box-shadow: 0 8px 28px rgba(201,162,39,0.65), 0 2px 8px rgba(0,0,0,0.6) !important;
    transform: translateY(-2px) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: #111111 !important;
    border: 2px dashed #2a1f00 !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #c9a227 !important;
    box-shadow: 0 0 28px rgba(201,162,39,0.22) !important;
}
[data-testid="stFileUploader"] * { color: #c9a227 !important; }

/* ── Chat Input ── */
[data-testid="stChatInput"] {
    background: #0d0d0d !important;
    border-top: 1.5px solid #2a1f00 !important;
    box-shadow: 0 -4px 20px rgba(0,0,0,0.6) !important;
}
[data-testid="stChatInput"] textarea {
    background: #161616 !important;
    color: #d4b896 !important;
    border: 1.5px solid #2a1f00 !important;
    border-radius: 12px !important;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.4) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #c9a227 !important;
    box-shadow: 0 0 18px rgba(201,162,39,0.28), inset 0 2px 8px rgba(0,0,0,0.4) !important;
}

/* ── Misc ── */
[data-testid="stSpinner"] * { color: #c9a227 !important; }
hr { border-color: #2a1f00 !important; }
[data-testid="stSuccess"] {
    background: rgba(201,162,39,0.10) !important;
    border: 1px solid rgba(201,162,39,0.35) !important;
    color: #c9a227 !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(201,162,39,0.15) !important;
}
[data-testid="stError"] {
    background: rgba(220,50,50,0.10) !important;
    border: 1px solid rgba(220,50,50,0.35) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(220,50,50,0.12) !important;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #2a1f00; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #c9a227; }

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Cache resources
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=api_key,
        temperature=0.2
    )

def get_vector_db():
    """Always returns a fresh Chroma instance (not cached) so new docs are visible."""
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=load_embedding_model()
    )

def add_chunks_with_rate_limit(vdb, chunks, batch_size=50):
    """Add chunks in small batches with delay to avoid 429 rate limit (100 req/min free tier)."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for attempt in range(3):
            try:
                vdb.add_documents(batch)
                break
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait = 65 if attempt == 0 else 120
                    time.sleep(wait)
                else:
                    raise e
        if i + batch_size < len(chunks):
            time.sleep(62)  # stay under 100 req/min free tier limit

def is_pdf_already_indexed(vdb, pdf_path):
    """Check if this PDF's source path is already in chroma to avoid duplicate indexing."""
    try:
        results = vdb._collection.get(where={"source": pdf_path}, limit=1)
        return len(results["ids"]) > 0
    except Exception:
        return False

llm = load_llm()

# --------------------------------------------------
# Retry
# --------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=10),
    reraise=True
)
def call_llm_with_retry(llm_model, prompt_input):
    try:
        return llm_model.invoke(prompt_input)
    except (GoogleRateLimitError, ClientError) as e:
        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            print("Rate limit hit. Backing off...")
        raise e

# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": "Hi! Ask me anything about the PDF."}
    ]

if "active_pdf" not in st.session_state:
    st.session_state.active_pdf = None

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.markdown("### ✨ Kazi & Anurag")
    st.markdown("**PDF AI Assistant**")
    st.divider()

    st.markdown('<div class="section-title">📚 PDF Library</div>', unsafe_allow_html=True)
    pdfs = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]

    if not pdfs:
        st.caption("No PDFs uploaded yet.")
    else:
        for pdf in pdfs:
            size = round(os.path.getsize(os.path.join(UPLOAD_DIR, pdf)) / 1024, 1)
            is_active = st.session_state.active_pdf == pdf
            badge = '<span class="badge-active">● Active</span>' if is_active else '<span class="badge-ready">○ Ready</span>'
            cls = "pdf-item pdf-item-active" if is_active else "pdf-item"
            st.markdown(
                f'<div class="{cls}">📄 <b>{pdf}</b> <span style="color:#4a3a10;font-size:11px">{size} KB</span> {badge}</div>',
                unsafe_allow_html=True
            )
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Use", key=f"sb_use_{pdf}"):
                    st.session_state.active_pdf = pdf
                    st.rerun()
            with c2:
                if st.button("🗑️", key=f"sb_del_{pdf}"):
                    os.remove(os.path.join(UPLOAD_DIR, pdf))
                    if st.session_state.active_pdf == pdf:
                        st.session_state.active_pdf = None
                    st.rerun()

    st.divider()
    st.markdown('<div class="section-title">⚡ Model Info</div>', unsafe_allow_html=True)
    st.caption("LLM: gemini-3.8-flash")
    st.caption("Embeddings: gemini-embedding-001")
    st.caption("Search: similarity (k=6)")
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {"role": "bot", "content": "Hi! Ask me anything about the PDF."}
        ]
        st.rerun()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown("""
<div class="gold-header">
    <h1>✨ Kazi & Anurag PDF Assistant</h1>
    <p>Powered by Gemini + ChromaDB</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab1, tab2 = st.tabs(["💬 Chat", "📄 PDF Assistant"])

# ==================================================
# TAB 1 — Chat
# ==================================================

with tab1:

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)

    if not st.session_state.active_pdf:
        st.info("⚠️ No PDF selected. Go to the **PDF Assistant** tab to upload and activate a PDF first.")

    question = st.chat_input("Ask a question about the PDF...")

    if question:
        if not st.session_state.active_pdf:
            st.warning("Please upload and select a PDF first from the PDF Assistant tab.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            st.markdown(f'<div class="bubble-user">{question}</div>', unsafe_allow_html=True)

            response = None
            with st.spinner("Searching PDF and generating answer..."):
                try:
                    # Fresh DB + retriever every query so newly uploaded PDFs are always included
                    vdb = get_vector_db()

                    active_path = os.path.normpath(os.path.join(UPLOAD_DIR, st.session_state.active_pdf))
                    retriever = vdb.as_retriever(
                        search_type="similarity",
                        search_kwargs={
                            "k": 6,
                            "filter": {"source": active_path}
                        }
                    )

                    retrieved_docs = retriever.invoke(question)

                    # Fallback: if filter returns nothing try without filter
                    if not retrieved_docs:
                        retriever_fallback = vdb.as_retriever(
                            search_type="similarity",
                            search_kwargs={"k": 6}
                        )
                        retrieved_docs = retriever_fallback.invoke(question)

                    if not retrieved_docs:
                        answer = "I could not find any relevant content in the PDF. Please make sure the PDF is uploaded and indexed."
                        st.session_state.messages.append({"role": "bot", "content": answer})
                        st.markdown(f'<div class="bubble-bot">{answer}</div>', unsafe_allow_html=True)
                        st.rerun()
                    else:
                        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                        prompt = f"""You are a helpful assistant that answers questions based on PDF content.
Use the context below to answer the question thoroughly and accurately.
If the answer is not in the context, say: "I could not find the answer in the PDF."

Context from PDF:
{context}

Question: {question}

Answer:"""
                        response = call_llm_with_retry(llm, prompt)

                except Exception as e:
                    err = str(e)
                    if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                        st.error("❌ Gemini API quota exhausted. Please wait and try again.")
                    else:
                        st.error(f"❌ Error: {err}")
                    print(f"Failed: {e}")

            if response is not None:
                answer = response.content
                if isinstance(answer, list):
                    answer = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in answer
                    )
                st.session_state.messages.append({"role": "bot", "content": answer})
                st.markdown(f'<div class="bubble-bot">{answer}</div>', unsafe_allow_html=True)
                st.rerun()

# ==================================================
# TAB 2 — PDF Assistant
# ==================================================

with tab2:

    st.markdown('<div class="section-title">⬆️ Upload PDF</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop or browse a PDF from your PC",
        type=["pdf"],
        label_visibility="visible"
    )

    if uploaded_file:
        save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)

        with st.spinner(f"Indexing {uploaded_file.name}... (may take a minute for large PDFs)"):
            loader = PyPDFLoader(save_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            vdb = get_vector_db()
            norm_path = os.path.normpath(save_path)
            for c in chunks:
                c.metadata["source"] = norm_path
            if is_pdf_already_indexed(vdb, norm_path):
                st.info(f"ℹ️ {uploaded_file.name} is already indexed. Using existing index.")
            else:
                add_chunks_with_rate_limit(vdb, chunks, batch_size=80)

        st.success(f"✅ {uploaded_file.name} ready! ({len(chunks)} chunks)")
        st.session_state.active_pdf = uploaded_file.name
        st.rerun()

    st.divider()

    st.markdown('<div class="section-title">📚 PDF Library</div>', unsafe_allow_html=True)
    pdfs = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]

    if not pdfs:
        st.markdown(
            '<div class="pdf-card" style="text-align:center;color:#3a3020;">No PDFs uploaded yet.</div>',
            unsafe_allow_html=True
        )
    else:
        for pdf in pdfs:
            size = round(os.path.getsize(os.path.join(UPLOAD_DIR, pdf)) / 1024, 1)
            is_active = st.session_state.active_pdf == pdf
            badge = '<span class="badge-active">● Active</span>' if is_active else '<span class="badge-ready">○ Ready</span>'
            cls = "pdf-item pdf-item-active" if is_active else "pdf-item"
            col1, col2, col3 = st.columns([5, 1, 1])
            with col1:
                st.markdown(
                    f'<div class="{cls}">📄 <b style="color:#d4b896">{pdf}</b>&nbsp;&nbsp;<span style="color:#4a3a10;font-size:11px">{size} KB</span>&nbsp;&nbsp;{badge}</div>',
                    unsafe_allow_html=True
                )
            with col2:
                if st.button("✅ Use", key=f"tab_use_{pdf}"):
                    st.session_state.active_pdf = pdf
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"tab_del_{pdf}"):
                    os.remove(os.path.join(UPLOAD_DIR, pdf))
                    if st.session_state.active_pdf == pdf:
                        st.session_state.active_pdf = None
                    st.rerun()

    st.divider()

    st.markdown('<div class="section-title">🔍 How It Works</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="pdf-card">
    <ul style="color:#8a7040;font-size:13px;line-height:2;padding-left:18px;">
        <li>Upload any PDF from your Windows PC</li>
        <li>PDF is split into chunks and indexed in ChromaDB</li>
        <li>Your question is embedded using Gemini Embeddings</li>
        <li>Relevant chunks are retrieved via MMR search</li>
        <li>Gemini generates an answer from the context</li>
    </ul>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">⚡ Model Info</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="pdf-card">
    <ul style="color:#8a7040;font-size:13px;line-height:2;padding-left:18px;">
        <li>LLM: gemini-3.8-flash</li>
        <li>Embeddings: gemini-embedding-001</li>
        <li>Search: similarity (k=6)</li>
        <li>Chunk size: 1000 · Overlap: 200</li>
    </ul>
</div>
""", unsafe_allow_html=True)