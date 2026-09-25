# Day 7: Week 1 Integration Demo (Mini RAG System)

Welcome to **Day 7 of the AI/RAG Internship Task**. Day 7 marks the culmination and integration of the entire Week 1 curriculum into a production-grade, robust, end-to-end Retrieval-Augmented Generation (RAG) system.

---

## 🎯 Day 7 Objective

Day 7 focuses on **Stability, Cleanup, Edge Case Resilience, and Pipeline Integration**. 

Instead of maintaining fragmented scripts across chunking, embeddings, vector indexing, retrieval, and LLM generation, Day 7 unifies all components into one cohesive, end-to-end RAG architecture with source citations and strict grounding.

---

## 🏗️ Architecture & Pipeline Flow

The unified RAG pipeline follows this deterministic lifecycle:

```text
Upload/Add Text Document (.txt)
              │
              ▼
    ┌───────────────────┐
    │  DocumentLoader   │  Reads files, skips empty files with warnings
    └─────────┬─────────┘
              │
              ▼
    ┌───────────────────┐
    │   TextChunker     │  Recursive character splitting (500 chars, 100 overlap)
    └─────────┬─────────┘  Preserves metadata: {source, chunk_id, text}
              │
              ▼
    ┌───────────────────┐
    │ EmbeddingService  │  Dense 384-d vector embeddings (all-MiniLM-L6-v2)
    └─────────┬─────────┘  L2-normalized for cosine similarity
              │
              ▼
    ┌───────────────────┐
    │ FAISSVectorStore  │  IndexFlatIP index with disk caching & auto-rebuild
    └─────────┬─────────┘
              │
   User Query │
              ▼
    ┌───────────────────┐
    │    Retriever      │  Embeds query, searches top_k chunks,
    └─────────┬─────────┘  filters by similarity threshold (>= 0.35), deduplicates
              │
              ▼
    ┌───────────────────┐
    │   GroqService     │  Calls Groq LLM (e.g., openai/gpt-oss-120b)
    └─────────┬─────────┘  Strict grounding prompt preventing hallucinations
              │
              ▼
    ┌───────────────────┐
    │  Citation Engine  │  Formats Grounded Answer + Source Attributions
    └───────────────────┘
```

---

## 💡 How RAG Works (Interview-Friendly Explanation)

**Retrieval-Augmented Generation (RAG)** solves two critical limitations of Large Language Models: knowledge cutoffs and factual hallucinations.

1. **Ingestion & Indexing (Offline Phase)**:
   - **Load**: Documents are read from disk.
   - **Chunk**: Long text is sliced into smaller, semantically coherent segments (e.g., 500 characters with 100 character overlap) so relevant details aren't lost in long documents.
   - **Embed**: A sentence transformer model converts each text chunk into a high-dimensional mathematical vector (dense embedding) capturing semantic meaning.
   - **Index**: Vectors are indexed in FAISS (Facebook AI Similarity Search) along with chunk metadata (`source`, `chunk_id`, `text`).

2. **Retrieval & Grounded Generation (Online Phase)**:
   - When a user asks a question, the question is converted into an embedding in the exact same vector space.
   - FAISS calculates vector similarity (cosine distance via dot product) to find the closest chunks.
   - Chunks below a relevance threshold are discarded to avoid confusing the LLM with irrelevant content.
   - The retrieved chunks are formatted into a prompt as explicit context.
   - The LLM is instructed: *"Answer using ONLY the provided context. If the answer is not present, say 'I don't have enough information in the provided documents.'"*
   - Finally, the response is delivered along with exact citations derived directly from vector store metadata, NOT hallucinated by the LLM.

---

## 🛡️ Edge Cases Handled

Day 7 includes defensive programming covering 10 critical edge cases:

1. **No documents found**:
   - If the documents directory is empty or missing, the system outputs `No documents found. Please add a text document first.` without crashing.
2. **Empty documents**:
   - If a file is empty or contains only whitespace, the loader outputs `Warning: document '<name>' is empty and was skipped.` and seamlessly continues processing other files.
3. **Empty user question**:
   - If the user presses Enter without input, the system prompts `Please enter a valid question.` without querying the vector store or LLM.
4. **Question unrelated to documents**:
   - If asked out-of-domain questions (e.g., *"What is the capital of France?"*), the system either filters out irrelevant chunks or the LLM answers: *"I don't have enough information in the provided documents."*
5. **No relevant chunks found**:
   - If FAISS similarity scores fall below `SIMILARITY_THRESHOLD` (0.35), the retriever returns an empty set, outputting `No relevant information was found in the documents.` without wasting LLM tokens.
6. **Missing GROQ_API_KEY**:
   - Validates that `GROQ_API_KEY` is present. If missing, displays:
     ```text
     GROQ_API_KEY is not configured.
     Please add it to your .env file.
     ```
     No unhandled stack trace and no secrets exposed.
7. **Groq API failure**:
   - Catches `AuthenticationError`, `RateLimitError`, `APIConnectionError`, and `APIStatusError`. If the service is unreachable or rate-limited, prints a graceful message:
     ```text
     Unable to generate the answer because the LLM service is currently unavailable.
     ```
8. **FAISS index failure / Corruption**:
   - The vector store checks for corrupted or missing index files. If corrupted or missing, it automatically falls back to re-indexing documents and saving a fresh index.
9. **Invalid TOP_K input**:
   - Safeguards against `top_k <= 0`, negative numbers, or non-integer values by automatically falling back to a safe default (`DEFAULT_TOP_K = 3`).
10. **User exits**:
    - Gracefully handles `exit`, `quit`, `q`, or `Ctrl+C`, printing `Goodbye!` and exiting with status 0.
11. **Windows Console Encoding**:
    - Automatically reconfigures `sys.stdout` and `sys.stderr` to `utf-8` on Windows, preventing `UnicodeEncodeError` from special dashes or unicode quotes.

---

## 📁 Project Directory Structure

```text
Week1/ (and Day7/)
│
├── app/
│   ├── __init__.py           # Exports all core classes & parameters
│   ├── main.py               # Interactive CLI entrypoint
│   ├── config.py             # Environment variables, paths, and defaults
│   ├── models.py             # Document, Chunk, and RetrievedChunk models
│   ├── loader.py             # DocumentLoader reading .txt files
│   ├── chunker.py            # TextChunker with chunk_id & source metadata
│   ├── vector_store.py       # FAISSVectorStore with disk persistence & auto-recovery
│   ├── retriever.py          # Retriever with similarity threshold & deduplication
│   ├── prompts.py            # Strict grounding system prompt
│   ├── citation.py           # Deterministic source formatting from chunk metadata
│   ├── rag_pipeline.py       # Unified RAGPipeline facade
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── embedding_service.py # SentenceTransformer (all-MiniLM-L6-v2)
│   ├── llm/
│   │   ├── __init__.py
│   │   └── groq_service.py   # Groq SDK client (openai/gpt-oss-120b)
│   ├── loaders/
│   │   ├── __init__.py
│   │   └── document_loader.py # Forwarding module
│   ├── chunking/
│   │   ├── __init__.py
│   │   └── text_splitter.py  # Forwarding module
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── faiss_store.py    # Forwarding module
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retriever.py      # Forwarding module
│   └── rag/
│       ├── __init__.py
│       ├── prompt.py         # Forwarding module
│       ├── citation.py       # Forwarding module
│       └── rag_pipeline.py   # Forwarding module
│
├── documents/                # Corpus documents (.txt)
│   ├── machine_learning.txt
│   ├── deep_learning.txt
│   ├── artificial_intelligence.txt
│   ├── machine_learning_fundamentals.txt
│   ├── rag_fundamentals.txt
│   └── ...
│
├── data/
│   ├── documents/            # Mirrored documents directory
│   └── vector_store/         # Cached FAISS index and chunk metadata
│
├── .env                      # Local configuration (never committed)
├── .env.example              # Template configuration
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🚀 How to Run

### 1. Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure `.env`

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` and set your Groq API key:

```env
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
EMBEDDING_MODEL=all-MiniLM-L6-v2
TOP_K=3
SIMILARITY_THRESHOLD=0.35
```

### 4. Run the Application

```bash
python -m app.main
```

Or pass a direct question flag:

```bash
python -m app.main --question "What is machine learning?"
```

---

## 🧪 Sample Demo Output

```text
============================================================
                     DAY 7 - RAG SYSTEM                     
============================================================

Loading RAG pipeline...

Documents loaded: 8
Chunks created: 270
Vector index ready.

============================================================
                      RAG SYSTEM READY                      
============================================================

Ask a question.
Type 'exit' or 'quit' to stop.

Question: What is machine learning?

Retrieving relevant information...
Generating grounded answer...

------------------------------------------------------------
ANSWER
------------------------------------------------------------

Machine learning is a branch (or subset) of artificial intelligence in which computer systems learn patterns from data—without being explicitly programmed—and use those learned patterns to make predictions, decisions, or improve their performance over time.

------------------------------------------------------------
SOURCES
------------------------------------------------------------

[1] machine_learning_fundamentals.txt - chunk 1
[2] artificial_intelligence.txt - chunk 2
[3] machine_learning.txt - chunk 1
```
