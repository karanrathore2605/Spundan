# Day 9 - Rule-Based AI Agent

A small production-oriented agent that routes requests deterministically:

- Math -> Calculator
- Theory/knowledge -> RAG
- Other -> Direct Groq LLM
- Empty/invalid input -> controlled error

## Structure

```text
Day9/
├── app/
│   ├── main.py
│   ├── agent.py
│   ├── tools.py
│   ├── rag.py
│   ├── llm.py
│   └── config.py
├── documents/
├── vector_store/
├── .env
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Add your Groq API key to `.env`.

## Run

```bash
python -m app.main
```

## Demo

```text
Question: What is 25 * 8?
Route: CALCULATOR
Answer: 200

Question: Explain transformers
Route: RAG
Answer: ...
Sources:
- transformers.txt
```

## Design

Routing is intentionally rule-based. The LLM does not decide which tool to call. This makes the behavior deterministic and easy to test.

The RAG tool uses Sentence Transformers embeddings + FAISS retrieval, then Groq generates an answer grounded in retrieved context.

## Limitations

The router uses simple regex/keyword rules, so ambiguous or multi-intent questions can require more sophisticated parsing. For a larger production system, routing rules could be expanded and covered by automated tests.
