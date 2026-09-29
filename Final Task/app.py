import os
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st

# Ensure project directory is in sys.path for robust import resolution
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunker import TextChunker
from app.config import (
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    HYBRID_ALPHA,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from app.decision_layer import AgentResult, DecisionLayer
from app.document_loader import Document, DocumentLoader
from app.embeddings import get_active_model_name
from app.rag_pipeline import RAGPipeline
from app.retriever import HybridRetriever
from app.vector_store import VectorStore

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS matching clean, readable high-visibility UI style
st.markdown(
    """
    <style>
    /* Global typography enhancement */
    html, body, [class*="css"], .stMarkdown {
        font-size: 18px;
    }

    /* Chat message bubble styling & spacing */
    [data-testid="stChatMessage"] {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
        border: 1px solid #edf0f2;
    }

    /* Assistant and user response text */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span,
    [data-testid="stChatMessageContent"] div {
        font-size: 20px !important;
        line-height: 1.65 !important;
        color: #1e293b !important;
    }

    /* Tables inside assistant answers */
    [data-testid="stChatMessage"] table,
    [data-testid="stChatMessageContent"] table {
        font-size: 19px !important;
        width: 100% !important;
        margin: 14px 0 !important;
    }
    [data-testid="stChatMessage"] th,
    [data-testid="stChatMessageContent"] th {
        font-size: 20px !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
        background-color: #f1f5f9 !important;
    }
    [data-testid="stChatMessage"] td,
    [data-testid="stChatMessageContent"] td {
        font-size: 19px !important;
        padding: 10px 16px !important;
    }

    /* Tool used caption */
    .tool-caption {
        color: #475569;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin-bottom: 6px;
        margin-top: 18px;
        font-family: inherit;
        letter-spacing: 0.2px;
    }

    /* Document ready card in sidebar */
    .doc-ready-card {
        background-color: #e8f5e9;
        border: 1px solid #c8e6c9;
        border-radius: 8px;
        padding: 14px 16px;
        color: #1b5e20;
        font-weight: 600;
        font-size: 18px !important;
        margin: 14px 0 10px 0;
    }

    /* Chat input box (st.chat_input) */
    [data-testid="stChatInput"] textarea {
        font-size: 20px !important;
        line-height: 1.5 !important;
        font-family: inherit !important;
        min-height: 58px !important;
        padding: 12px 16px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        font-size: 20px !important;
        color: #94a3b8 !important;
    }

    .stTextInput input, div[data-testid="stTextInput"] input {
        font-size: 20px !important;
    }

    /* User question text inside chat feed */
    .user-question-text {
        font-size: 21px !important;
        font-weight: 600 !important;
        line-height: 1.5 !important;
        color: #0f172a !important;
    }

    /* Sidebar controls & labels */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] .stHeading {
        font-size: 26px !important;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        font-size: 17px !important;
    }
    [data-testid="stSidebar"] .stCaption {
        font-size: 15px !important;
        color: #64748b !important;
    }

    /* Buttons */
    .stButton button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"] {
        font-size: 17px !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
    }

    /* Expanders (Sources) */
    [data-testid="stExpander"] details summary p {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    [data-testid="stExpander"] details div p, 
    [data-testid="stExpander"] details div span,
    [data-testid="stExpander"] details div .stCaption {
        font-size: 16px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# APPLICATION SINGLETONS
# -----------------------------------------------------------------------------
@st.cache_resource
def get_shared_vector_store() -> VectorStore:
    vs = VectorStore()
    vs.load()
    return vs


@st.cache_resource
def get_agent_pipeline():
    vs = get_shared_vector_store()
    retriever = HybridRetriever(
        vs,
        top_k=TOP_K,
        similarity_threshold=SIMILARITY_THRESHOLD,
        alpha=HYBRID_ALPHA,
    )
    rag_pipe = RAGPipeline(retriever)
    decision_agent = DecisionLayer(rag_pipe)
    return vs, retriever, rag_pipe, decision_agent


vector_store, retriever, rag_pipeline, agent = get_agent_pipeline()

# Session State for Document & Chat Search History
if "active_doc_name" not in st.session_state:
    if vector_store.chunks:
        st.session_state.active_doc_name = vector_store.chunks[0].source
    else:
        st.session_state.active_doc_name = None

if "history" not in st.session_state:
    st.session_state.history = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("AI Study Assistant")
    st.caption("Upload a TXT or PDF document and ask questions about it.")

    # Upload document widget
    uploaded_file = st.file_uploader(
        "Upload document",
        type=["pdf", "txt", "docx"],
        help="Upload a TXT, PDF, or DOCX document",
    )

    # Optional sample files
    sample_files = []
    if DOCUMENTS_DIR.exists():
        sample_files = [
            f.name for f in DOCUMENTS_DIR.iterdir()
            if f.suffix.lower() in [".txt", ".pdf", ".docx"]
        ]
    selected_sample = None
    if sample_files:
        with st.expander("Or choose a sample document", expanded=False):
            selected_sample = st.selectbox(
                "Sample file:",
                options=["None"] + sample_files,
                index=0,
            )

    # Process Document Button
    if st.button("Process Document", type="primary", use_container_width=True):
        doc: Optional[Document] = None
        try:
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                logger.info("[1] Document uploaded: '%s' (%d bytes)", uploaded_file.name, len(file_bytes))
                logger.info("[2] Text extraction started: '%s'", uploaded_file.name)
                doc = DocumentLoader.load_from_bytes(file_bytes, uploaded_file.name)
                logger.info("[3] Text extraction completed: '%s' (%d chars)", uploaded_file.name, len(doc.content))
                st.session_state.active_doc_name = uploaded_file.name
            elif selected_sample and selected_sample != "None":
                sample_path = DOCUMENTS_DIR / selected_sample
                logger.info("[1] Document uploaded (sample): '%s'", selected_sample)
                logger.info("[2] Text extraction started: '%s'", selected_sample)
                doc = DocumentLoader.load_from_path(sample_path)
                logger.info("[3] Text extraction completed: '%s' (%d chars)", selected_sample, len(doc.content))
                st.session_state.active_doc_name = selected_sample
            else:
                st.warning("Please upload a file or choose a sample document.")

            if doc:
                with st.spinner("Indexing document..."):
                    logger.info("[4] Chunking started for '%s'", doc.source)
                    chunks = TextChunker().chunk_document(doc)
                    logger.info("[5] Number of chunks created: %d chunks", len(chunks))

                    # Stages [6]-[11] execute inside build_from_chunks
                    vector_store.build_from_chunks(chunks)
                    vector_store.save()

                    st.session_state.history = []  # fresh search history for new doc
                    st.rerun()
        except Exception as exc:
            logger.error("Error processing document: %s", exc, exc_info=True)
            st.error(f"Error processing document: {exc}")

    # Document ready status box
    if not vector_store.is_empty():
        doc_display_name = st.session_state.active_doc_name or (
            vector_store.chunks[0].source if vector_store.chunks else "document.pdf"
        )
        st.markdown(
            f"""
            <div class="doc-ready-card">
                Document ready
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"**File:** {doc_display_name}")
        st.write(f"**Chunks:** {vector_store.total_chunks}")
        st.write(f"**Embedding Model:** `{get_active_model_name()}`")

        if st.button("Clear Document", use_container_width=True):
            vector_store.clear()
            st.session_state.active_doc_name = None
            st.session_state.history = []
            st.rerun()

    # Clear chat history button if messages exist
    if st.session_state.history:
        if st.button("Clear Search History", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# -----------------------------------------------------------------------------
# MAIN CHAT & QUERY SEARCH HISTORY FEED
# -----------------------------------------------------------------------------

# If no search history yet, show friendly suggestions
if not st.session_state.history:
    st.markdown("### Ask questions about your document")
    st.caption("Type any question below, or click a suggestion to start:")

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("📝 Explain simple", use_container_width=True):
        st.session_state.pending_query = "Explain this in simple words"
        st.rerun()
    if col2.button("📌 Key points", use_container_width=True):
        st.session_state.pending_query = "Give key points"
        st.rerun()
    if col3.button("🧮 25 * 8", use_container_width=True):
        st.session_state.pending_query = "25 * 8"
        st.rerun()
    if col4.button("📊 Remote math", use_container_width=True):
        st.session_state.pending_query = "The document says there are 500 employees. If 15% work remotely, how many employees work remotely?"
        st.rerun()

# Render all previous query search history on the screen
for turn in st.session_state.history:
    # 1. Tool used label
    tool_label = turn.get("tool_used", "document_search")
    st.markdown(f'<div class="tool-caption">Tool used: {tool_label}</div>', unsafe_allow_html=True)

    # 2. User query bubble
    with st.chat_message("user", avatar="🔴"):
        st.markdown(f'<div class="user-question-text">{turn["query"]}</div>', unsafe_allow_html=True)

    # 3. Assistant response bubble
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(turn["answer"])

        # Collapsible Sources expander
        with st.expander("Sources", expanded=False):
            if turn.get("retrieval_results"):
                for idx, r in enumerate(turn["retrieval_results"], start=1):
                    st.write(f"- **{r.source}** (Chunk {r.chunk_id}) &mdash; *Similarity: {r.hybrid_score:.2f}*")
                    st.caption(f'"{r.text[:280]}..."')
            elif turn.get("sources"):
                for s in turn["sources"]:
                    st.write(f"- {s}")
            else:
                st.caption("No document context used.")

        # Show calculation formula if applicable
        if turn.get("expression") and turn.get("calculation_result") is not None:
            st.caption(f"**Calculation Formula:** `{turn['expression']} = {turn['calculation_result']}`")


# -----------------------------------------------------------------------------
# CHAT INPUT & EXECUTION
# -----------------------------------------------------------------------------
user_query = st.chat_input("Ask something...")

# Check if a suggestion chip was clicked
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query and user_query.strip():
    clean_q = user_query.strip()

    # Determine tool and generate answer
    with st.spinner("Searching..."):
        res: AgentResult = agent.execute(clean_q)

    # Append to query search history
    st.session_state.history.append({
        "query": clean_q,
        "answer": res.answer,
        "tool_used": res.tool_used,
        "sources": res.sources,
        "retrieval_results": res.retrieval_results,
        "expression": res.expression,
        "calculation_result": res.calculation_result,
        "route": res.route,
    })

    # Rerun to cleanly update the chronological history feed
    st.rerun()
