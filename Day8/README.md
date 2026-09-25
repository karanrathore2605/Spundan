# Day 8: LLM Tools Architecture — Production Python Project

Welcome to **Day 8** of the AI / RAG Internship Project.

This project focuses on the core concept of **Tools for Large Language Models (LLMs)** — understanding that **Tools are Functions for LLMs**.

---

## Table of Contents

1. [What are LLM Tools?](#what-are-llm-tools)
2. [Core Concepts Explained](#core-concepts-explained)
   - [1. What is a Tool?](#1-what-is-a-tool)
   - [2. What is Function Calling?](#2-what-is-function-calling)
   - [3. Difference Between an LLM and a Tool](#3-difference-between-an-llm-and-a-tool)
   - [4. Why Tools are Useful](#4-why-tools-are-useful)
   - [5. Tool Name](#5-tool-name)
   - [6. Tool Description](#6-tool-description)
   - [7. Tool Input](#7-tool-input)
   - [8. Tool Output](#8-tool-output)
   - [9. Tool Schema](#9-tool-schema)
   - [10. Tool Registry](#10-tool-registry)
   - [11. Manual Tool Execution](#11-manual-tool-execution)
   - [12. Automatic Tool Calling](#12-automatic-tool-calling)
   - [13. How This Project Can Later Support Automatic Tool Calling](#13-how-this-project-can-later-support-automatic-tool-calling)
3. [Expected End-to-End Architecture](#expected-end-to-end-architecture)
4. [Project Structure](#project-structure)
5. [Detailed File Walkthrough](#detailed-file-walkthrough)
6. [Deep Dive: Safe Calculator Tool](#deep-dive-safe-calculator-tool)
7. [Deep Dive: Document Search / RAG Tool](#deep-dive-document-search--rag-tool)
8. [Groq Integration & Evolution to Automatic Tool Calling](#groq-integration--evolution-to-automatic-tool-calling)
9. [Installation & Setup](#installation--setup)
10. [Running the Application](#running-the-application)
11. [Running the Test Suite](#running-the-test-suite)
12. [Interview Preparation Guide](#interview-preparation-guide)

---

## What are LLM Tools?

Large Language Models (LLMs) are extraordinary next-token predictors. They can draft essays, write code, and synthesize summaries. However, **LLMs have fundamental limitations**:

1. **They cannot do deterministic arithmetic reliably**: A model does not contain a CPU ALU; it guesses the next tokens based on probability, which often leads to subtle math hallucinations (e.g., `87234 * 91823`).
2. **They do not possess access to private or local data**: An LLM's weights are frozen at training time. It cannot natively query your local filesystem or vector database.
3. **They cannot take real-world actions**: An LLM cannot execute system commands, call external REST APIs, or modify databases on its own.

**Tools solve this problem.**

A **tool** is a deterministic, executable piece of software (a Python function, class, or API client) that an application exposes to an LLM or a user.

```text
                        ┌──────────────────────────────┐
                        │             LLM              │
                        │ (Reasoning & Decision Engine)│
                        └──────────────┬───────────────┘
                                       │ 1. Inspects query & tool schemas
                                       │ 2. Selects appropriate tool
                                       ▼
                        ┌──────────────────────────────┐
                        │    Structured Tool Input     │
                        │    (e.g., {"expr": "5*12"})  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │      Tool Execution          │
                        │ (Safe Calculator / FAISS RAG)│
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │    Structured Tool Output    │
                        │    (e.g., 60 or RAG Chunks)  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │      Final Answer to User    │
                        │  (Grounded, Accurate, Clean) │
                        └──────────────────────────────┘
```

The LLM acts as the **brain** (determining *intent* and *arguments*), while tools act as the **hands and eyes** (performing deterministic calculation and data retrieval).

---

## Core Concepts Explained

### 1. What is a Tool?
A **tool** is an encapsulated software function or component designed with a strict interface (defined inputs, deterministic execution logic, and predictable outputs). In this project:
- `CalculatorTool`: Performs exact arithmetic via Python Abstract Syntax Trees (AST).
- `DocumentSearchTool`: Executes dense semantic vector search over documents using FAISS.

### 2. What is Function Calling?
**Function Calling** is a protocol supported by modern LLMs (such as Groq's Llama 3 models) where:
1. The developer provides a list of available tool definitions (JSON schemas).
2. The LLM evaluates a user prompt. If the prompt requires external capability, the LLM outputs a **structured JSON payload** specifying the tool name and arguments (rather than answering directly).
3. The host application executes the function with those arguments and feeds the result back to the LLM.

### 3. Difference Between an LLM and a Tool

| Aspect | LLM (e.g., Groq LLaMA 3) | Tool (e.g., Calculator, Vector Search) |
| :--- | :--- | :--- |
| **Nature** | Probabilistic neural network | Deterministic software code |
| **Computation** | Token probability estimation (prone to math errors) | Exact ALU arithmetic / deterministic algorithms |
| **Knowledge** | Static weights up to training cutoff | Real-time, dynamic, local documents |
| **Execution** | Inference only (read-only token generation) | Real actions (computations, DB queries, file I/O) |
| **Security Risk** | Hallucination, prompt injection | Code injection if not safely built (e.g., raw `eval`) |

### 4. Why Tools are Useful
- **Accuracy**: Eliminates math errors and data hallucinations.
- **Privacy & Grounding**: Fetches proprietary enterprise knowledge on demand.
- **Modularity**: New capabilities can be plugged into an agent without retraining the LLM.
- **Auditing & Safety**: Every tool call can be logged, validated, rate-limited, and authorized.

### 5. Tool Name
A unique, descriptive identifier (e.g., `"calculator"`, `"document_search"`). The LLM uses this name in its function call payload to designate which tool to execute.

### 6. Tool Description
A plain-English summary of what the tool does, when to use it, and what it cannot do. The LLM reads this description to decide whether the tool matches the user's intent.
*Example:* `"Performs safe mathematical calculations. Supports arithmetic operations (+, -, *, /, //, %, **) and parentheses."*

### 7. Tool Input
The parameters required by the tool. For manual execution, this is the string passed via the terminal (e.g., `"5 * 12"` or `"Explain AI"`). For automated LLM calling, this is a validated JSON dictionary (e.g., `{"expression": "5 * 12"}`).

### 8. Tool Output
The data returned after tool execution. In this project, outputs are standardized into a `ToolResult` model containing:
- `success`: Boolean flag
- `data`: Structured payload (e.g., `60` or a list of chunk dicts)
- `error`: Explanatory error message if failed
- `raw_input`: Original input string for auditing

### 9. Tool Schema
A standardized JSON Schema definition specifying:
- Function name
- Description
- Argument names, types (`string`, `number`, `boolean`, etc.), descriptions, and required fields.

Schemas allow the LLM to know exactly what argument formats the code expects.

### 10. Tool Registry
A centralized catalog that manages all available tools. Instead of hardcoding conditional `if/else` checks across the codebase, a `ToolRegistry` allows tools to register dynamically. The registry handles tool discovery, execution dispatching, and schema export.

### 11. Manual Tool Execution
Executing tools directly under human control (e.g., via a CLI menu). The user explicitly selects which tool to run and provides the input.
*Why start with Manual Execution in Day 8?*
- Allows developers to isolate, test, and verify tool stability, edge cases, and security before connecting complex autonomous agent loops.

### 12. Automatic Tool Calling
The LLM autonomously parses user requests, determines if a tool is needed, selects the tool, creates the arguments, and synthesizes the returned output into a natural conversational reply without manual human intervention for each step.

### 13. How This Project Can Later Support Automatic Tool Calling
The Day 8 architecture was specifically constructed to make the transition to automatic tool calling trivial:
1. `BaseTool.schema.to_groq_format()` generates the exact JSON schema required by Groq's `client.chat.completions.create(tools=...)` API.
2. `ToolRegistry.execute(tool_name, input_data)` executes the tool using the tool call returned by the model.
3. The tool's output is packaged into a `tool` role message and returned to Groq to generate the final grounded response.

---

## Expected End-to-End Architecture

### Day 8 Manual Execution Architecture

```text
                           USER
                            │
                            ▼
                     ┌─────────────┐
                     │  CLI Menu   │
                     │ (app/main)  │
                     └──────┬──────┘
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
   Option 1: Calculator               Option 2: Document Search
          │                                   │
          ▼                                   ▼
   CalculatorTool                      DocumentSearchTool
          │                                   │
          ▼                                   ▼
   Safe AST Evaluator                  RAG Retriever
   (No eval, DoS guarded)                     │
          │                                   ▼
          ▼                            EmbeddingEngine
   Computed Result                      (all-MiniLM-L6-v2)
   (e.g., 60)                                 │
          │                                   ▼
          │                            FAISS Vector Store
          │                            (IndexFlatIP Cosine)
          │                                   │
          │                                   ▼
          │                            Relevant Chunks
          │                            + Source Citations
          │                                   │
          └─────────────────┬─────────────────┘
                            ▼
                     Terminal Output
```

### Future Evolution: Automatic Tool Calling Flow

```text
                           USER
                            │ "What is 5 * 12 and what does our AI doc say?"
                            ▼
                     ┌─────────────┐
                     │ Groq LLM    │ ◄── (Receives Tool Schemas from ToolRegistry)
                     └──────┬──────┘
                            │
                            ▼
                     Model Decides Tools:
                     1. calculator(expression="5 * 12")
                     2. document_search(query="AI doc overview")
                            │
                            ▼
                     ┌─────────────┐
                     │ ToolRegistry│
                     │  Execution  │
                     └──────┬──────┘
                            │
                            ▼
                     Tool Outputs Returned to LLM
                            │
                            ▼
                     ┌─────────────┐
                     │ Groq LLM    │
                     └──────┬──────┘
                            │
                            ▼
                     Final Grounded Response with Citations & Math
```

---

## Project Structure

```text
Day8/
│
├── app/
│   ├── __init__.py                # Package initialization
│   ├── config.py                  # Environment config and logging setup
│   ├── main.py                    # Terminal CLI with interactive & flag modes
│   │
│   ├── schemas/                   # Pydantic data contracts
│   │   ├── __init__.py
│   │   └── tool_schema.py         # ToolResult, ToolDefinition, SearchResponse
│   │
│   ├── tools/                     # Core Tools Layer
│   │   ├── __init__.py
│   │   ├── base.py                # BaseTool abstract base class
│   │   ├── calculator.py          # Safe AST calculator tool
│   │   ├── document_search.py     # RAG document search tool
│   │   └── registry.py            # Centralized ToolRegistry
│   │
│   ├── rag/                       # Modular Day 7 RAG Pipeline
│   │   ├── __init__.py
│   │   ├── loader.py              # DocumentLoader with source metadata
│   │   ├── chunker.py             # TextChunker with overlapping windows
│   │   ├── embeddings.py          # SentenceTransformers dense vector engine
│   │   ├── vector_store.py        # FAISS in-memory index (IndexFlatIP)
│   │   └── retriever.py           # RAGRetriever orchestrator
│   │
│   └── llm/                       # LLM Provider Layer
│       ├── __init__.py
│       └── groq_client.py         # Groq API client with grounded generation
│
├── documents/                     # Indexed knowledge base
│   ├── ai.txt                     # Artificial Intelligence fundamentals
│   ├── machine_learning.txt       # Machine Learning concepts
│   └── sample.txt                 # RAG architecture overview
│
├── tests/                         # Production test suites
│   ├── __init__.py
│   ├── test_calculator.py         # Arithmetic, injection security, error tests
│   └── test_document_search.py    # Vector retrieval, sources, schema tests
│
├── requirements.txt               # Locked dependencies
├── .env.example                   # Environment variable template
├── .env                           # Local environment configuration
├── .gitignore                     # Git ignore rules
└── README.md                      # Comprehensive documentation
```

---

## Detailed File Walkthrough

| File | Purpose | Key Responsibilities |
| :--- | :--- | :--- |
| `app/config.py` | Configuration management | Loads `.env` via `python-dotenv`, defines `AppConfig`, configures structured logging. |
| `app/schemas/tool_schema.py` | Data contracts | Defines Pydantic models for `ToolResult`, `ToolDefinition` (with `.to_groq_format()`), and `SearchResponse`. |
| `app/tools/base.py` | Tool abstraction | Defines `BaseTool` requiring `name`, `description`, `schema`, `execute()`, and `format_output()`. |
| `app/tools/calculator.py` | Safe arithmetic tool | Evaluates arithmetic via `ast.NodeVisitor`. Strictly forbids `eval`, code execution, and DoS exponents. |
| `app/tools/document_search.py` | RAG retrieval tool | Wraps `RAGRetriever`. Returns structured context and formats terminal citations cleanly. |
| `app/tools/registry.py` | Tool catalog | Central `ToolRegistry` supporting registration, lookup, execution, and dictionary mappings (`TOOLS`). |
| `app/rag/loader.py` | File reader | Scans `documents/` for `.txt`/`.md` files, attaches file paths, sizes, and sources. |
| `app/rag/chunker.py` | Text segmentation | Splits document text into overlapping sliding-window chunks (`chunk_size=300`, `chunk_overlap=50`). |
| `app/rag/embeddings.py` | Vectorization | Wraps `sentence-transformers/all-MiniLM-L6-v2`. Computes unit-normalized embeddings. |
| `app/rag/vector_store.py` | Vector index | Uses `faiss.IndexFlatIP` for inner product / cosine similarity search. Maps indices back to chunks. |
| `app/rag/retriever.py` | Retrieval pipeline | Orchestrates loader, chunker, embeddings, and FAISS. Returns ranked `SearchResponse`. |
| `app/llm/groq_client.py` | Groq integration | Connects to Groq cloud API for chat completions and grounded synthesis using RAG context. |
| `app/main.py` | User interface | Interactive CLI menu (1. Calculator, 2. Search, 3. Exit) and CLI flag runner (`--tool`, `--input`). |

---

## Deep Dive: Safe Calculator Tool

### Why NOT `eval(expression)`?
In Python, `eval()` executes arbitrary Python code in the interpreter. If a user or an adversarial prompt submits:
```python
__import__('os').system('rmdir /s /q C:\\')
# or
open('.env').read()
```
`eval()` will execute it immediately, leading to complete Remote Code Execution (RCE) and data exfiltration.

### The AST Solution
`app/tools/calculator.py` uses Python's built-in `ast` (Abstract Syntax Tree) module with a strict whitelist:
1. **Parser**: `ast.parse(expression, mode="eval")` parses the string into a syntax tree.
2. **Visitor**: `SafeExpressionEvaluator(ast.NodeVisitor)` walks the tree.
3. **Whitelist**: Only the following node types are permitted:
   - `ast.Expression`
   - `ast.Constant` (strictly `int` or `float` numbers; strings and objects are rejected)
   - `ast.UnaryOp` (`+`, `-`)
   - `ast.BinOp` (`+`, `-`, `*`, `/`, `//`, `%`, `**`)
4. **Blacklist / Default Deny**: ANY other AST node (e.g., `ast.Call`, `ast.Name`, `ast.Attribute`, `ast.Import`) triggers an immediate `ValueError: Dangerous or unsupported syntax detected`.
5. **Denial-of-Service Protection**:
   - Max string length: 200 characters.
   - Max exponent limit: rejects exponents greater than 1,000 to prevent CPU/memory locking from operations like `99999 ** 99999`.
6. **Graceful Errors**:
   - `5 / 0` -> returns `"Error: Division by zero"` without crashing the application.

---

## Deep Dive: Document Search / RAG Tool

### Modular RAG Design
The Document Search Tool reuses the complete Day 7 RAG pipeline:
1. **Document Loading**: Text files from `documents/` are loaded with metadata (filename, character count).
2. **Chunking**: Chunks are generated with configurable sliding window (`chunk_size=300`, `chunk_overlap=50`) so context is not truncated across sentence boundaries.
3. **Embedding Model**: `all-MiniLM-L6-v2` produces 384-dimensional dense vectors. All embeddings are L2 normalized.
4. **FAISS Vector Store**: Uses `faiss.IndexFlatIP` (Inner Product). Because vectors are unit normalized, inner product is mathematically identical to cosine similarity:
   $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|} = u \cdot v \quad (\text{when } \|u\| = \|v\| = 1)$$
5. **Separation of Concerns**:
   - `retriever.retrieve(query)` returns pure data (`SearchResponse`).
   - `tool.format_output(result)` handles terminal styling and citation presentation.

---

## Groq Integration & Evolution to Automatic Tool Calling

`app/llm/groq_client.py` provides a production client for Groq using the official Python SDK:

```python
from app.llm.groq_client import GroqClient

client = GroqClient()
# Generates grounded answer using retrieved RAG context
answer = client.generate_grounded_answer(query="Explain AI", context=retrieved_text)
```

### How to Enable Automatic Tool Calling
Groq supports native tool calling. To connect our `ToolRegistry` to Groq automatically:

```python
# 1. Export tool schemas to Groq format
tools = registry.get_groq_schemas()

# 2. Call Groq with tool definitions
response = groq_client.client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "What is 15 * 24?"}],
    tools=tools,
    tool_choice="auto",
)

# 3. Check if LLM requested a tool call
tool_call = response.choices[0].message.tool_calls[0]
tool_name = tool_call.function.name
tool_args = json.loads(tool_call.function.arguments)

# 4. Execute tool from our registry
result = registry.execute(tool_name, tool_args.get("expression") or tool_args.get("query"))

# 5. Send tool result back to Groq for final answer synthesis
```

This clean decoupling allows the exact same tools to run both in Day 8's manual terminal demo and in Day 9's fully autonomous agent loop!

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.14)
- Git

### 2. Clone and Navigate to Directory
```powershell
cd c:\Users\Admin\Desktop\Python\Day8
```

### 3. Create and Activate Virtual Environment (Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Open `.env` and set your `GROQ_API_KEY`:
```text
GROQ_API_KEY=gsk_your_actual_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
TOP_K_RESULTS=3
CHUNK_SIZE=300
CHUNK_OVERLAP=50
```

---

## Running the Application

### 1. Interactive Terminal Demo (Primary Demo Target)

Run the application:
```powershell
python -m app.main
```

Expected Terminal Interface:
```text
==================================================
           DAY 8 - LLM TOOLS DEMO
==================================================

Available tools:
1. Calculator
2. Document Search
3. Exit

Choose tool: 1

Enter expression: 5 * 12

Calculator Result:
60

Available tools:
1. Calculator
2. Document Search
3. Exit

Choose tool: 2

Enter query: Explain AI

Searching documents...

Retrieved Context:

[1] Artificial Intelligence (AI) is the simulation of human intelligence processes by computer systems. These processes include learning (the acquisition of information and rules for using the information), reasoning (using rules to reach approximate or definite conclusions), and self-correction.
    Source: ai.txt (Similarity: 0.78)

[2] and trained for a particular task, such as virtual personal assistants, autonomous vehicles, or recommendation engines. Artificial General Intelligence (AGI), or strong AI, is hypothetical artificial intelligence that possesses the ability to understand, learn, and apply knowledge across a wide variety of tasks.
    Source: ai.txt (Similarity: 0.68)

Sources:
- ai.txt

Available tools:
1. Calculator
2. Document Search
3. Exit

Choose tool: 3

Exiting...
```

### 2. Direct CLI Execution (Automated Scripts / Headless)

You can also run any tool directly via command-line flags:

#### Calculator Direct Execution:
```powershell
python -m app.main --tool calculator --input "5 * 12"
```
Output:
```text
Tool selected: calculator
Tool input: 5 * 12

Calculator Result:
60
```

#### Document Search Direct Execution:
```powershell
python -m app.main --tool document_search --input "Explain AI"
```
Output:
```text
Tool selected: document_search
Tool input: Explain AI
Searching documents...

Retrieved Context:

[1] Artificial Intelligence (AI) is the simulation of human intelligence processes by computer systems...
    Source: ai.txt (Similarity: 0.78)
...
Sources:
- ai.txt
```

---

## Running the Test Suite

We use `pytest` for comprehensive automated testing across arithmetic correctness, injection security, vector retrieval, and schemas.

Run all tests:
```powershell
pytest tests/ -v
```

Expected Test Output:
```text
tests/test_calculator.py::TestSafeCalculator::test_basic_arithmetic PASSED
tests/test_calculator.py::TestSafeCalculator::test_unary_and_floating_point PASSED
tests/test_calculator.py::TestSafeCalculator::test_division_by_zero PASSED
tests/test_calculator.py::TestSafeCalculator::test_empty_and_whitespace_input PASSED
tests/test_calculator.py::TestSafeCalculator::test_invalid_syntax PASSED
tests/test_calculator.py::TestSafeCalculator::test_security_rejection_of_dangerous_expressions PASSED
tests/test_calculator.py::TestSafeCalculator::test_dos_protection_large_exponents PASSED
tests/test_calculator.py::TestCalculatorStandaloneFunction::test_calculator_function_success PASSED
tests/test_calculator.py::TestCalculatorStandaloneFunction::test_calculator_function_division_by_zero PASSED
tests/test_calculator.py::TestCalculatorStandaloneFunction::test_calculator_function_dangerous_input PASSED
tests/test_calculator.py::TestCalculatorToolClass::test_tool_metadata PASSED
tests/test_calculator.py::TestCalculatorToolClass::test_tool_execute_success PASSED
tests/test_calculator.py::TestCalculatorToolClass::test_tool_execute_division_by_zero PASSED
tests/test_calculator.py::TestCalculatorToolClass::test_tool_execute_empty PASSED
tests/test_document_search.py::TestRAGComponents::test_document_loader_loads_sample_files PASSED
tests/test_document_search.py::TestRAGComponents::test_chunker_creates_overlapping_chunks PASSED
tests/test_document_search.py::TestDocumentSearchTool::test_tool_metadata PASSED
tests/test_document_search.py::TestDocumentSearchTool::test_search_valid_query_explain_ai PASSED
tests/test_document_search.py::TestDocumentSearchTool::test_search_machine_learning_query PASSED
tests/test_document_search.py::TestDocumentSearchTool::test_search_empty_query PASSED
tests/test_document_search.py::TestDocumentSearchTool::test_standalone_document_search_function PASSED

============================= 21 passed in 4.32s ==============================
```

---

## Interview Preparation Guide

When discussing this Day 8 project in an AI / Backend engineering interview, highlight these key design decisions:

1. **Why Manual Tool Execution first?**
   - *"In production AI systems, you never expose untested functions directly to an autonomous LLM. We first built manual execution and unit test suites to guarantee tool determinism, boundary validation, and security before hooking them to agent loops."*

2. **Security & AST vs. `eval`:**
   - *"We avoided `eval()` completely because it permits arbitrary Python code execution (RCE). Instead, we wrote a recursive AST Node Visitor that enforces an explicit whitelist: only numbers and arithmetic operators are traversed. Any function calls, variable lookups, or imports are blocked at parse time."*

3. **Separation of Retrieval and Presentation:**
   - *"The `DocumentSearchTool` returns a structured data contract (`SearchResponse`) containing raw chunk text, source filenames, and similarity scores. Presentation logic (formatting bullet points, citations, and terminal colors) is isolated in `format_output()`. This enables the same tool to output terminal text today and structured JSON to an LLM tomorrow."*

4. **Extensibility via Tool Registry:**
   - *"Our `ToolRegistry` decouples tool consumers from tool implementations. Adding a new tool (e.g., weather or SQL database search) requires zero changes to the CLI or the LLM handler — just subclass `BaseTool` and call `registry.register()`."*

5. **Groq Integration & Tool Schemas:**
   - *"Every tool defines a JSON Schema matching the OpenAI / Groq tool-calling specification. When we evolve to automated agent calling, Groq can inspect these exact schemas to select tools autonomously."*
