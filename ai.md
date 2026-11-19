# Project6: Enterprise-Grade LLM & RAG Application

**Project6** is a high-performance, production-ready AI application built with Python 3.12+. It features a strict **Hexagonal Architecture** (Ports & Adapters) to ensure modularity, testability, and ease of maintenance. The project integrates **LangChain** for robust LLM interactions and is structured to support **LangGraph** for complex agentic workflows.

Designed for enterprise use, it prioritizes **strict type safety**, **observability**, and **provider independence**, allowing seamless switching between LLM providers like OpenRouter, OpenAI, and Anthropic.

## 🌟 Key Features

- **Hexagonal Architecture:** Strict separation of business logic (Domain) from external services (Infrastructure) using Ports & Adapters.
- **LangChain Integration:** Leverages LangChain’s ecosystem for reliable LLM communication and prompt management.
- **LangGraph Ready:** Structured to support stateful, multi-step agentic workflows via LangGraph.
- **Multi-Provider LLM Routing:** Swap providers (e.g., OpenRouter ↔ OpenAI) via configuration without changing code.
- **Strict Type Safety:** Enforced via `pyright` in **strict mode** and comprehensive type hints on all dataclasses and protocols.
- **Modular Dataclasses:** Core models are defined in separate, reusable files for maximum maintainability.
- **Advanced Observability:** Tracks token usage, latency, and provider selection for every interaction.
- **Robust Tooling:** Automated setup, linting, and testing via `uv` and `Makefile`.

## 🛠 Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.12+ |
| **Package Manager** | `uv` (Ultra-fast Python installer/resolver) |
| **AI Orchestration** | LangChain / LangGraph |
| **LLM Providers** | OpenRouter, OpenAI, Anthropic (via Adapters) |
| **Type Checking** | Pyright (**Strict Mode**) |
| **Linting/Formatting** | Ruff |
| **Configuration** | Pydantic Settings |
| **Testing** | Pytest + Asyncio |

## 📂 Project Structure

```text
src/project6_1/
├── __init__.py
├── domain/                  # Pure business logic (No dependencies)
│   ├── models/              # Reusable dataclasses in separate files
│   │   ├── chat.py          # ChatMessage, TokenUsage
│   │   └── ...
│   └── errors.py            # Exception taxonomy
├── application/             # Use cases and Ports (Interfaces)
│   ├── ports/               # Contracts (e.g., ChatModel Protocol)
│   └── use_cases/           # Business orchestration
├── adapters/                # Implementation details
│   └── driven/
│       └── llm/
│           ├── openrouter_chat.py  # LangChain-based adapter
│           └── fake_chat.py        # For testing
└── infrastructure/          # Composition root & Config
    ├── config/
    ├── container.py         # Wires everything together
    └── providers.py         # Provider registry