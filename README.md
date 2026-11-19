<<<<<<< HEAD
# Project6
=======
# Project5: Production-Grade RAG Application

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

A high-performance, enterprise-level Retrieval-Augmented Generation (RAG) system designed for general public use. Project5 allows users to ingest thousands of documents (PDFs, DOCX, URLs) and retrieve grounded, accurate answers with full source citations and performance telemetry.

## Key Features

- **Hexagonal Architecture:** Strict separation of business logic (Domain) from external services (Infrastructure) using Ports & Adapters.
- **Multi-Provider LLM Routing:** Automatic failover between LLM providers (e.g., OpenAI to Anthropic) to ensure reliability and handle rate limits.
- **Advanced Observability:** Tracks latency, token usage, and provider selection for every interaction.
- **Strict Type Safety:** Enforced via `pyright` in strict mode and comprehensive type hints.
- **Async Ingestion Pipeline:** Non-blocking document processing using LangGraph for efficient indexing into Qdrant.
- **Grounded Responses:** Guardrails ensure the AI only answers based on provided evidence, preventing hallucinations.

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.12+ |
| **Package Manager** | `uv` (Ultra-fast Python installer/resolver) |
| **AI Orchestration** | LangGraph |
| **Vector Database** | Qdrant |
| **LLM Providers** | OpenAI, Anthropic, AWS Bedrock (via Adapters) |
| **Type Checking** | Pyright (Strict Mode) |
| **Linting/Formatting** | Ruff |

## Quick Start

### Prerequisites
- macOS or Linux with `make` installed.
- No prior Python or `uv` installation required—`make setup` handles everything.

### 1. Clone and Bootstrap
```bash
git clone https://github.com/imranshahadat/Project5.git
cd Project5
make setup
make verify    # Ensure everything is configured correctly
make test      # Run the test suite
make pyright   # Run strict type checking
make run       # Start the application
>>>>>>> c9ee331 (chore: init hexagonal layout, uv/Python 3.12, pyright, Makefile, Pydantic config, core ports, and docs)
