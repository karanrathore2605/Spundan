# Day 5 - Mini RAG

A production-style Retrieval-Augmented Generation system using:

- Python
- Sentence Transformers
- FAISS
- Gemini
- PDF/TXT document loading

## Architecture

Documents
    ↓
Document Loader
    ↓
Text Chunking
    ↓
Embeddings
    ↓
FAISS Vector Store
    ↓
Retriever
    ↓
Prompt Construction
    ↓
Gemini
    ↓
Final Answer

## Project Structure

```text
Day5/
├── documents/
├── data/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── models.py
│   ├── loaders/
│   ├── chunking/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── retrieval/
│   ├── llm/
│   └── rag/
├── .env
├── .gitignore
└── requirements.txt