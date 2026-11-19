# Project5 — AI Engineering Architecture & Engineering Guide

| | |
|---|---|
| **Product** | AI-powered document chat (RAG): ingest → chunk → embed → retrieve → grounded answer with citations |
| **Stack** | Python 3.12+, uv, pyright strict, ruff, FastAPI, pydantic v2 |
| **Architecture** | Ports & Adapters (Hexagonal) |
| **Status** | Living document — any architectural change updates this file in the same PR |
| **Audience** | Every contributor, including future-you |

> **How to use this document:** Read §1–§3 once. Keep §9 (patterns), §10 (SOLID), and §14 (checklists) open while coding. CI enforces the tooling rules automatically (§4).

---

## Table of Contents

1. Purpose and Scope
2. Architecture Overview — Ports and Adapters
3. Project Structure
4. Toolchain and Configuration
5. Domain Layer (pure business core)
6. Application Layer (ports and use cases)
7. Adapters (driving and driven)
8. Infrastructure (config, container, resilience, observability)
9. Design Pattern Catalog
10. SOLID Principles Mapping
11. AI-Engineering Practices
12. Testing Strategy
13. Error Handling and HTTP Mapping
14. Code Standards and Checklists
15. Anti-Patterns (never do these)
16. Roadmap and Build Order
17. Glossary
- Appendix A: `.env.example`
- Appendix B: `Makefile`
- Appendix C: CI Workflow

---

## 1. Purpose and Scope

### 1.1 Why this document exists

Code is read 10× more than it is written. This file is the **single source of truth** for *why* the codebase is shaped the way it is, so that:

- new code gets reviewed against explicit standards, not reviewer taste,
- architectural decisions survive staff turnover (including future-you at 2 a.m.),
- the "AI engineering part" — prompts, providers, vectors, costs — is treated as **engineering**, not glue code.

### 1.2 What this document governs

Everything under `src/` and `tests/`. If a change alters a layer boundary, a port signature, or a pattern, this document is updated **in the same PR** — docs and code drift together or not at all.

### 1.3 Goals and non-goals

| Goal | Non-goal (deliberately deferred — YAGNI) |
|---|---|
| Production-grade single RAG service | Microservices |
| Swap any LLM / vector store via config | A plugin marketplace |
| Provider-independent domain and use cases | Supporting 10 providers on day one |
| Contract-tested adapters | Live-API tests on every commit |
| Full cost/usage accounting | FinOps dashboards |
| Evaluation harness hooks (prompt versions) | A full eval pipeline |

---

## 2. Architecture Overview — Ports and Adapters

### 2.1 What it is

Ports & Adapters (Hexagonal Architecture, Alistair Cockburn) splits the system into three zones:

```
        ┌───────────── DRIVING adapters (left side) ─────────────┐
        │   FastAPI routes · CLI · background workers            │
        │   "what triggers the app"                              │
        └───────────────────────┬────────────────────────────────┘
                                ▼
                ┌─────── APPLICATION ────────┐
                │  use cases + PORTS (ifs)   │
                │  "what the app does"       │
                └────────────────────────────┘
                                ▲
        ┌───────────── DRIVEN adapters (right side) ─────────────┐
        │  OpenAI/Anthropic/Ollama · pgvector/Qdrant · Redis     │
        │  PDF loaders · "what the app talks to"                 │
        └───────────────────────┬────────────────────────────────┘
                                ▲
                        ┌── DOMAIN ──┐
                        │ pure logic │
                        └────────────┘
```

- A **port** is an interface *owned by the application* (`application/ports/*`) that says what the app *needs*, not how it's done.
- A **driven adapter** implements a port for one concrete technology (OpenAI, pgvector…).
- A **driving adapter** calls use cases (HTTP API, CLI).
- The **domain** holds the business vocabulary and rules — pure Python, zero dependencies.

### 2.2 Why it matters

Each LLM/vector provider violates expectations differently. Without a port, every one of these differences leaks into business code:

| Difference | OpenAI | Anthropic | Ollama/local |
|---|---|---|---|
| System prompt | message with role `system` | separate top-level `system` param | a message |
| Structured output | `response_format: json_schema` | tool-use trick | varies |
| Usage fields | `prompt_tokens` / `completion_tokens` | `input_tokens` / `output_tokens` | often missing |
| Finish reason | `finish_reason: "stop"` | `stop_reason: "end_turn"` | varies |
| Rate-limit errors | 429 + headers | 429, different shape | none |
| Embedding batch cap | 2048 inputs | differs | model-dependent |

The port normalizes all of this into one canonical shape. Consequences:

- **Swap providers via config**, not refactors — one adapter file added, zero files edited.
- **Test the whole brain of the app** (use cases) with fakes, offline, in milliseconds.
- **Team parallelism**: one person builds the pgvector adapter while another builds retrieval, against a frozen port.
- **Upgrade safety**: when OpenAI ships a breaking SDK change, the blast radius is one file.

> 💡 **Litmus test:** if deleting a third-party SDK from `requirements` would force edits in `application/` or `domain/`, the architecture is broken.

### 2.3 How we enforce it — the dependency rule

| Layer | May import | Must never import |
|---|---|---|
| `domain/` | stdlib only | anything from this project, any SDK |
| `application/` | `domain/` | fastapi, openai, asyncpg, redis — *any* SDK |
| `adapters/driven/` | `application/ports`, `domain/` | other adapters |
| `adapters/driving/` | `application/` use cases | driven adapter internals |
| `infrastructure/` | everything | — (composition root; the *only* place concretes meet) |

Concrete SDK imports (`openai`, `anthropic`, `asyncpg`, `redis`) are allowed **only** in `adapters/` and `infrastructure/`. Ruff is configured so a misplaced import fails CI.

> ⚠️ **Rule:** dependencies point inward. The domain knows nothing; everything knows the domain.

---

## 3. Project Structure

### 3.1 The tree

```
project5/
├── pyproject.toml              # single source of truth for deps + tooling
├── uv.lock                     # committed — reproducible builds
├── .env.example                # committed template (never real secrets)
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── Makefile
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml      # app + pgvector + redis (local stack)
├── src/
│   └── project5/
│       ├── __init__.py
│       ├── py.typed            # PEP 561 marker
│       ├── domain/                     # PURE. No SDKs. No frameworks. No I/O.
│       │   ├── models/
│       │   │   ├── document.py         # Document, Chunk, content_hash
│       │   │   ├── embedding.py        # EmbeddingVector, EmbeddedChunk
│       │   │   ├── chat.py             # ChatMessage, TokenUsage
│       │   │   └── answer.py           # RagAnswer, Citation, RetrievedChunk
│       │   ├── errors.py               # exception taxonomy
│       │   └── policies/
│       │       └── chunking.py         # chunking strategies (pure logic)
│       ├── application/
│       │   ├── ports/                  # the contracts this app depends on
│       │   │   ├── llm.py              # ChatModel, StreamingChatModel
│       │   │   ├── embeddings.py       # EmbeddingModel
│       │   │   ├── vector_store.py     # VectorStore
│       │   │   ├── document_loader.py  # DocumentLoader
│       │   │   ├── cache.py            # Cache
│       │   │   └── prompts.py          # PromptCatalog
│       │   └── use_cases/
│       │       ├── ingest_document.py
│       │       ├── ask_question.py     # the RAG pipeline
│       │       └── retrieval.py        # SimpleRetriever (+ optional reranker)
│       ├── adapters/
│       │   ├── driving/
│       │   │   ├── api/                # FastAPI — thin HTTP shell
│       │   │   │   ├── app.py
│       │   │   │   ├── routes/         # ask.py, ingest.py, health.py
│       │   │   │   ├── schemas.py      # pydantic request/response
│       │   │   │   ├── error_handlers.py
│       │   │   │   └── middleware.py   # request-id, access logs
│       │   │   └── cli/                # typer CLI for local ops
│       │   └── driven/
│       │       ├── llm/
│       │       │   ├── openai_chat.py
│       │       │   ├── anthropic_chat.py
│       │       │   ├── ollama_chat.py
│       │       │   └── fake_chat.py    # deterministic — tests + local dev
│       │       ├── embeddings/
│       │       ├── vector_store/
│       │       │   ├── pgvector_store.py
│       │       │   └── in_memory_store.py
│       │       ├── loaders/
│       │       └── cache/
│       ├── infrastructure/
│       │   ├── config/settings.py      # pydantic-settings, validated at boot
│       │   ├── container.py            # COMPOSITION ROOT
│       │   ├── providers.py            # provider registry / factory
│       │   ├── resilience.py           # retry / timeout / circuit breaker / cache
│       │   └── observability/logging.py
│       └── prompts/                    # versioned prompt templates
│           └── rag_answer.py
└── tests/
    ├── unit/                           # domain + use cases (fakes, no I/O)
    ├── contract/                       # all LLM adapters pass the SAME suite
    ├── integration/                    # real pgvector via docker compose
    └── e2e/                            # FastAPI app wired with fakes
```


after some modification. using dataclasess separtly

src/project6_1/
├── __init__.py
├── domain/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── chat.py              # ChatMessage, TokenUsage, Role
│   │   ├── document.py          # Document, Chunk, content_hash
│   │   ├── embedding.py         # EmbeddingVector, EmbeddedChunk
│   │   └── answer.py            # RagAnswer, Citation, RetrievedChunk
│   ├── errors.py                # Exception taxonomy
│   └── policies/
│       ├── __init__.py
│       └── chunking.py          # Chunking strategies (pure logic)
├── application/
│   ├── __init__.py
│   ├── ports/
│   │   ├── __init__.py
│   │   ├── llm.py               # ChatModel, StreamingChatModel
│   │   ├── embeddings.py        # EmbeddingModel
│   │   ├── vector_store.py      # VectorStore
│   │   ├── document_loader.py   # DocumentLoader
│   │   └── cache.py             # Cache
│   └── use_cases/
│       ├── __init__.py
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── ask_question.py  # AskCommand
│       │   └── ingest_document.py # IngestCommand
│       ├── ask_question.py      # Use case orchestrator
│       ├── ingest_document.py   # Ingestion pipeline
│       └── retrieval.py         # SimpleRetriever
├── adapters/
│   ├── __init__.py
│   ├── driving/
│   │   ├── __init__.py
│   │   └── api/
│   │       ├── __init__.py
│   │       ├── app.py           # FastAPI app
│   │       └── routes/
│   │           ├── __init__.py
│   │           ├── ask.py
│   │           └── ingest.py
│   └── driven/
│       ├── __init__.py
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── openrouter_chat.py    # LangChain-based adapter
│       │   ├── anthropic_chat.py
│       │   └── fake_chat.py          # For testing
│       ├── embeddings/
│       │   ├── __init__.py
│       │   └── openai_embeddings.py
│       ├── vector_store/
│       │   ├── __init__.py
│       │   └── qdrant_store.py
│       └── loaders/
│           ├── __init__.py
│           └── pdf_loader.py
├── infrastructure/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── container.py             # Composition root
│   ├── providers.py             # Provider registry
│   ├── resilience.py            # Retry/timeout/circuit breaker
│   └── workflows/
│       ├── __init__.py
│       └── rag_graph.py         # LangGraph workflow definition
└── prompts/
    ├── __init__.py
    └── rag_answer.py            # Versioned prompt templates

### 3.2 Folder responsibilities

| Folder | One-sentence purpose | Allowed imports |
|---|---|---|
| `domain/models/` | Define the business vocabulary with self-validating value objects | stdlib |
| `domain/policies/` | Pure decision logic (chunking, scoring) | stdlib |
| `application/ports/` | Declare what the app needs from the outside world | `domain/` |
| `application/use_cases/` | Orchestrate ports into business operations | `domain/`, `ports/` |
| `adapters/driven/` | Implement ports for concrete technologies | ports, domain, SDKs |
| `adapters/driving/` | Expose use cases to the world (HTTP, CLI) | use cases, domain |
| `infrastructure/` | Wire concretes together; config; cross-cutting concerns | everything |
| `tests/` | Prove all of the above, fastest-first | `src/` |

**Why `src/` layout?** Tests run against the *installed* package, not the working directory — packaging mistakes surface in CI, not in production.

---

## 4. Toolchain and Configuration

### 4.1 uv — dependency management

**Why:** deterministic lockfile (`uv.lock`), 10–100× faster than pip, PEP 735 dependency groups, one tool for venv + deps + running.

```bash
uv init --package          # creates src/ layout + hatchling backend
uv sync                    # build .venv from lockfile  (CI: uv sync --frozen)
uv add openai              # runtime dependency
uv add --group dev pytest  # dev dependency
uv lock --upgrade          # refresh lockfile deliberately, review the diff
uv run <cmd>               # run anything inside the project venv
```

> ⚠️ **Rule:** `uv.lock` is committed. CI uses `--frozen`. "It works on my machine" starts with an identical lockfile.

### 4.2 `pyproject.toml` — annotated

```toml
[project]
name = "project5"
version = "0.1.0"
description = "Production-grade RAG service"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.8",
    "pydantic-settings>=2.4",
    "openai>=1.40",
    "anthropic>=0.34",
    "tenacity>=9.0",
    "structlog>=24.1",
]

[dependency-groups]                     # PEP 735 — uv-native dev deps
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=5.0",
    "respx>=0.21",                      # mock httpx → contract-test LLM adapters offline
    "ruff>=0.6",
    "pyright>=1.1.385",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/project5"]

[tool.pyright]
include = ["src", "tests"]
typeCheckingMode = "strict"
pythonVersion = "3.12"
venvPath = "."
venv = ".venv"

[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests"]

[tool.ruff.lint]
select = [
    "E", "F", "W",   # pycodestyle + pyflakes
    "I",             # import order (also enforces layer hygiene visually)
    "UP",            # modern syntax: X | None, built-in generics
    "B",             # bugbear: mutable defaults, except mistakes
    "SIM", "C4",     # simplifications
    "ASYNC",         # blocking calls inside async functions
    "S",             # bandit: hardcoded secrets, unsafe calls
    "TRY",           # exception hygiene: no bare raise, no silent swallow
    "PTH",           # pathlib over os.path (portability)
    "RUF",
]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["S101", "S106"]           # asserts + literal test creds are fine

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--strict-markers --cov=src/project5 --cov-report=term-missing"
markers = ["integration: requires real infrastructure"]
```

### 4.3 pyright in strict mode

**Why:** in an AI codebase, half the bugs are shape bugs — `usage` being `None`, a provider returning `str | None`, an optional field assumed present. Strict mode turns every one of them into a red squiggle before runtime.

**How:**

- Every public function fully annotated, including `-> None`.
- `Optional` provider payloads narrowed explicitly (`if usage is None: raise ProviderError(...)`) — never `assert` in production paths (`assert` disappears under `python -O`).
- `src/project5/py.typed` committed so consumers of your package get types too.
- Untyped third-party libraries: add `types-*` stub packages, or wrap them behind a port (which you should be doing anyway) and type only the wrapper.
- `reveal_type(x)` is your friend while writing; delete before committing.

> ⚠️ **Rule:** `uv run pyright` must exit clean on every commit. Zero suppressions without a comment explaining why.

### 4.4 Ruff, pytest, pre-commit

- **Ruff** lints + formats (`uv run ruff check --fix && uv run ruff format .`). The selected rule groups map directly to this document's principles: `PTH` → portability, `S` → defensibility, `ASYNC` → correctness under concurrency, `TRY` → fail-loud.
- **pytest** with `asyncio_mode = "auto"` — async tests written as plain `async def`, no decorators needed.
- **pre-commit** runs ruff + pyright locally so CI failures are caught before push (Appendix B/C).

---

## 5. Domain Layer (pure business core)

### 5.1 What

The vocabulary and rules of the business: what a `Chunk` is, what a valid `ChatMessage` is, which failures exist. Value objects that **cannot exist in an invalid state**.

### 5.2 Why

- Business rules are the most stable part of the system; providers and frameworks churn. Isolating them means provider migrations never touch core logic.
- Self-validating value objects implement **fail-fast**: invalid data is impossible to construct, so downstream code never re-checks.
- Pure functions/classes = trivially unit-testable with zero mocks.

### 5.3 How

```python
# src/project5/domain/models/chat.py
"""Canonical chat vocabulary — provider-independent by design."""
from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Role
    content: str

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("ChatMessage.content must not be empty")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
```

```python
# src/project5/domain/errors.py
"""One exception hierarchy the whole application speaks.
Adapters TRANSLATE provider errors into this taxonomy."""
class AppError(Exception):
    """Base for every error this application is expected to handle."""

class DomainError(AppError):
    """Invalid input or violated business rule → maps to 4xx."""

class ExternalError(AppError):
    """A downstream system failed → maps to 5xx."""

class ProviderError(ExternalError):
    def __init__(self, provider: str, detail: str) -> None:
        super().__init__(f"{provider}: {detail}")
        self.provider = provider

class RateLimitError(ProviderError): ...
class ProviderTimeoutError(ProviderError): ...
class ProviderUnavailableError(ProviderError): ...
class ConfigurationError(AppError):
    """Startup fail-fast: bad config must kill the process immediately."""
```

**Domain rules for every file:**

1. `@dataclass(frozen=True, slots=True)` — immutable, memory-lean.
2. Validate in `__post_init__`; raise `ValueError`/`DomainError` immediately.
3. No SDK, framework, HTTP, or logging imports. Ever.
4. No raw user content at module level; no globals.
5. If a domain file needs `import openai`, the design is broken — stop and re-scope.

---

## 6. Application Layer (ports and use cases)

### 6.1 Ports — the contracts

**What:** interfaces owned by the application that declare *what* it needs (`ChatModel`, `VectorStore`) without saying *who* provides it.

**Why:** this is Dependency Inversion (the "D" in SOLID) made concrete. High-level policy (use cases) never depends on low-level detail (SDKs). It is also what makes the "swap any LLM" requirement a config change instead of a refactor.

**Protocol vs ABC — decision rule:**

| Situation | Use |
|---|---|
| Port = pure contract, adapters independent | `typing.Protocol` (structural — adapters don't even import the port; fakes are trivial) |
| Adapters share real implementation (e.g., common loader plumbing) | `abc.ABC` (Template Method) |
| Need `isinstance` checks | `@runtime_checkable Protocol` (rare — usually a design smell) |

**How — the LLM port:**

```python
# src/project5/application/ports/llm.py
"""Port: chat completion. Adapters translate this ↔ provider SDKs."""
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from project5.domain.models.chat import ChatMessage, TokenUsage


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Capabilities — so use cases never write `if provider == "openai"`."""
    provider: str
    default_model: str
    supports_structured_output: bool
    supports_streaming: bool
    context_window: int


@dataclass(frozen=True, slots=True)
class ChatRequest:
    messages: tuple[ChatMessage, ...]
    model: str
    temperature: float = 0.2
    max_tokens: int = 1024
    json_schema: dict[str, object] | None = None   # honored only if capability allows


@dataclass(frozen=True, slots=True)
class ChatResponse:
    text: str
    model: str
    usage: TokenUsage          # mandatory → cost accounting works app-wide
    finish_reason: str         # canonical: "stop" | "length" | "content_filter"


class ChatModel(Protocol):
    """Non-streaming chat. ISP: simple adapters implement only this."""
    info: ModelInfo

    async def complete(self, request: ChatRequest) -> ChatResponse: ...


class StreamingChatModel(Protocol):
    """Separate capability (Interface Segregation) — not all providers stream well."""
    def stream(self, request: ChatRequest) -> AsyncIterator[str]: ...
```

**Other ports (signatures):**

```python
# application/ports/embeddings.py
@dataclass(frozen=True, slots=True)
class EmbeddingInfo:
    provider: str
    model: str
    dimensions: int            # vector store collection must match!
    max_batch_size: int        # adapters enforce their own caps

class EmbeddingModel(Protocol):
    info: EmbeddingInfo
    async def embed(self, texts: Sequence[str]) -> list[EmbeddingVector]: ...

# application/ports/vector_store.py
class VectorStore(Protocol):
    async def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None: ...
    async def search(self, query_vector: EmbeddingVector, *, top_k: int) -> list[RetrievedChunk]: ...
    async def delete_document(self, document_id: UUID) -> None: ...

# application/ports/document_loader.py
class DocumentLoader(Protocol):
    def can_handle(self, source: DocumentSource) -> bool: ...
    def load(self, source: DocumentSource) -> LoadedDocument: ...   # sync (CPU-bound)

# application/ports/cache.py
class Cache(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, *, ttl_s: int) -> None: ...
```

> ⚠️ **YAGNI guard:** only create ports where real variance exists (LLM, embeddings, vector store, loaders, cache, prompts). No port for your own use cases. No generic `CrudRepository`. Every port must pay rent.

### 6.2 Use cases — orchestration

**What:** one class per business operation (`AskQuestion`, `IngestDocument`), composed exclusively of ports and domain types.

**Why:** this is where "the app's brain" lives — and because it touches only ports, the *entire* brain is unit-testable offline.

**How:**

```python
# src/project5/application/use_cases/ask_question.py
from dataclasses import dataclass

from project5.application.ports.llm import ChatModel, ChatRequest
from project5.application.ports.prompts import PromptCatalog
from project5.domain.errors import DomainError
from project5.domain.models.answer import Citation, RagAnswer


@dataclass(frozen=True, slots=True)
class AskCommand:
    question: str
    top_k: int = 5

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise DomainError("question must not be empty")      # fail fast at entry
        if not 1 <= self.top_k <= 20:
            raise DomainError("top_k must be between 1 and 20")


class AskQuestion:
    """Orchestrates retrieval → prompt construction → generation → citations."""
    def __init__(self, retriever: Retriever, chat: ChatModel, prompts: PromptCatalog) -> None:
        self._retriever = retriever
        self._chat = chat
        self._prompts = prompts

    async def execute(self, command: AskCommand) -> RagAnswer:
        hits = await self._retriever.retrieve(command.question, top_k=command.top_k)
        if not hits:
            return RagAnswer.no_context(command.question)        # explicit, tested path

        template = self._prompts.get("rag_answer")
        request = ChatRequest(
            messages=template.render(
                question=command.question,
                context="\n\n".join(f"[{h.chunk_id}] {h.text}" for h in hits),
            ),
            model=self._chat.info.default_model,
        )
        response = await self._chat.complete(request)
        return RagAnswer(
            text=response.text,
            citations=[Citation(chunk_id=h.chunk_id, document_id=h.document_id, score=h.score)
                       for h in hits],
            usage=response.usage,
        )
```

Retrieval is an application service (not a port — it composes ports and has no external variance of its own):

```python
# application/use_cases/retrieval.py
class SimpleRetriever:
    def __init__(
        self,
        embeddings: EmbeddingModel,
        store: VectorStore,
        reranker: Reranker | None = None,      # Null-Object-style optional hook
        *,
        candidate_k: int = 20,
    ) -> None: ...

    async def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        vectors = await self._embeddings.embed([query])
        candidates = await self._store.search(vectors[0], top_k=self._candidate_k)
        if self._reranker is not None:
            candidates = await self._reranker.rerank(query, candidates)
        return candidates[:top_k]
```

**Use-case rules:**

1. Constructor receives dependencies; never constructs adapters.
2. Validate the command at entry (fail fast), then trust the types.
3. No SDK imports, no HTTP, no `if provider ==`.
4. Every branch (including "no context found") is explicit and tested.

---

## 7. Adapters (driving and driven)

### 7.1 Driving adapters — the thin shell

**What:** FastAPI app + CLI. **Why:** transport details (status codes, JSON, headers) must be swappable and must never invade business logic. **How:** routes are 3–5 lines each.

```python
# adapters/driving/api/app.py
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()                    # pydantic-settings: fail fast at boot
    container = build_container(settings)    # ConfigurationError here = crash at startup (good)
    await container.startup()
    app.state.container = container
    yield
    await container.shutdown()


def get_ask(request: Request) -> AskQuestion:
    return request.app.state.container.ask_question


@router.post("/ask", response_model=AnswerResponse)
async def ask(body: AskRequest, use_case: AskQuestion = Depends(get_ask)) -> AnswerResponse:
    answer = await use_case.execute(body.to_command())    # pydantic → domain DTO
    return AnswerResponse.from_domain(answer)             # domain → HTTP contract
```

Pydantic models live **only** here (boundary validation). Domain stays framework-free.

### 7.2 Driven adapters — where translation happens

**What:** one class per provider implementing a port. **Why:** every SDK quirk dies inside its adapter. **How:**

```python
# src/project5/adapters/driven/llm/openai_chat.py
"""Adapter: OpenAI ChatModel. Translates SDK ↔ canonical types & errors."""
import openai
from openai import AsyncOpenAI

from project5.application.ports.llm import ChatModel, ChatRequest, ChatResponse, ModelInfo
from project5.domain.errors import ProviderError, ProviderTimeoutError, RateLimitError
from project5.domain.models.chat import ChatMessage, TokenUsage


class OpenAiChatModel:
    def __init__(self, client: AsyncOpenAI, info: ModelInfo) -> None:
        self._client = client          # injected — never constructed internally
        self.info = info

    async def complete(self, request: ChatRequest) -> ChatResponse:
        try:
            response = await self._client.chat.completions.create(
                model=request.model,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format=(
                    {"type": "json_schema", "json_schema": {"schema": request.json_schema}}
                    if request.json_schema else openai.NOT_GIVEN
                ),
            )
        except openai.RateLimitError as exc:            # most specific first!
            raise RateLimitError("openai", str(exc)) from exc
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError("openai", str(exc)) from exc
        except openai.APIError as exc:
            raise ProviderError("openai", str(exc)) from exc

        choice = response.choices[0]
        usage = response.usage
        if usage is None:                               # strict-mode narrowing, no assert
            raise ProviderError("openai", "response missing usage data")
        return ChatResponse(
            text=choice.message.content or "",
            model=response.model,
            usage=TokenUsage(usage.prompt_tokens, usage.completion_tokens),
            finish_reason=_normalize_finish_reason(choice.finish_reason),
        )


def _normalize_finish_reason(raw: str | None) -> str:
    return {"stop": "stop", "length": "length", "content_filter": "content_filter"}.get(
        raw or "", "stop"
    )
```

The Anthropic adapter implements the **same port** with its own translation (`system` → top-level param, `stop_reason` mapping, different usage field names). **Adding a provider = one new file. Zero existing files change.**

The fake (used in ~95% of tests — fakes beat mocks for readability):

```python
# adapters/driven/llm/fake_chat.py
class FakeChatModel:
    """Deterministic ChatModel for unit tests and local dev."""
    info = ModelInfo("fake", "fake-1", True, False, 8192)

    def __init__(self, reply: str = "ok") -> None:
        self._reply = reply
        self.calls: list[ChatRequest] = []     # tests assert on recorded calls

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        return ChatResponse(self._reply, "fake-1", TokenUsage(10, 5), "stop")
```

### 7.3 Checklist: adding a new provider

1. Create `adapters/driven/llm/<provider>_chat.py` implementing `ChatModel`.
2. Translate **every** SDK exception into the domain taxonomy — no SDK error escapes the adapter.
3. Map finish reasons and usage fields to canonical types.
4. Register a factory: `@register_chat("<provider>")` in `infrastructure/providers.py`.
5. Extend the `Literal` in `LlmSettings.provider` (fail-fast for typos, for free).
6. Add entries to `.env.example`.
7. Add the adapter to the **contract test** parametrize list (§12) — it must pass the identical suite.
8. Update this document.

### 7.4 Checklist: adding a new use case

1. Command DTO with `__post_init__` validation in `application/use_cases/`.
2. Depend only on ports + domain.
3. Unit test with fakes — including the unhappy paths.
4. HTTP schema + route + any new error → HTTP mappings.
5. Wire in `container.py`.
6. One e2e test through FastAPI with fakes.

---

## 8. Infrastructure (config, container, resilience, observability)

### 8.1 Configuration — validated once, safe defaults, no secrets in code

**Why:** configuration errors must crash at boot (fail fast), not at 3 a.m. on request 50,000. Secrets must be structurally incapable of leaking into logs.

**How:**

```python
# src/project5/infrastructure/config/settings.py
from typing import Literal
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LlmSettings(BaseModel):
    provider: Literal["openai", "anthropic", "ollama"]   # invalid value = instant boot crash
    api_key: SecretStr                                    # repr → "**********" — leak-proof
    timeout_s: float = Field(default=30, gt=0)            # safe default, never None
    max_retries: int = Field(default=3, ge=0, le=10)


class RagSettings(BaseModel):
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=100, ge=0)
    top_k: int = Field(default=5, ge=1, le=20)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_nested_delimiter="__", extra="ignore")
    debug: bool = False                                   # debug OFF by default
    llm: LlmSettings
    rag: RagSettings = RagSettings()
```

Env format (see Appendix A): `LLM__PROVIDER=openai`, `LLM__API_KEY=sk-...`. `.env` is gitignored; `.env.example` is committed.

### 8.2 The composition root — the only place concretes meet

**What:** `container.py` — the single file that knows `OpenAiChatModel` exists. **Why:** elsewhere, everything depends on abstractions; here (and only here), the object graph is assembled. **How:**

```python
# src/project5/infrastructure/container.py
@dataclass(slots=True)
class Container:
    settings: Settings
    chat: ChatModel
    embeddings: EmbeddingModel
    store: VectorStore
    ask_question: AskQuestion
    ingest: IngestDocument

    async def startup(self) -> None: ...    # open pools — resource lifecycle lives here
    async def shutdown(self) -> None: ...   # close them (context-manager discipline)


def build_container(settings: Settings) -> Container:
    chat = CachedChatModel(                            # decorators wrap any ChatModel
        ResilientChatModel(
            build_chat_model(settings.llm),
            timeout_s=settings.llm.timeout_s,
            max_attempts=settings.llm.max_retries,
        ),
        cache=build_cache(settings),
        ttl_s=settings.cache.ttl_s,
    )
    store = build_vector_store(settings)
    embeddings = build_embeddings(settings)
    retriever = SimpleRetriever(embeddings, store, candidate_k=settings.rag.top_k * 4)
    prompts = InMemoryPromptCatalog.default()
    return Container(
        settings=settings, chat=chat, embeddings=embeddings, store=store,
        ask_question=AskQuestion(retriever, chat, prompts),
        ingest=IngestDocument(...),
    )
```

### 8.3 Provider registry (Factory + Registry pattern)

**Why:** Open/Closed — new providers added without touching selection logic; unknown providers crash at boot with a clear message.

```python
# src/project5/infrastructure/providers.py
from collections.abc import Callable

_CHAT_FACTORIES: dict[str, Callable[[LlmSettings], ChatModel]] = {}


def register_chat(name: str) -> Callable[[ChatFactory], ChatFactory]:
    def decorator(factory: ChatFactory) -> ChatFactory:
        if name in _CHAT_FACTORIES:                    # fail fast on duplicate
            raise ConfigurationError(f"duplicate chat provider: {name}")
        _CHAT_FACTORIES[name] = factory
        return factory
    return decorator


@register_chat("openai")
def _openai(settings: LlmSettings) -> ChatModel:
    client = AsyncOpenAI(api_key=settings.api_key.get_secret_value(), timeout=settings.timeout_s)
    return OpenAiChatModel(client, _openai_info(settings))


def build_chat_model(settings: LlmSettings) -> ChatModel:
    try:
        return _CHAT_FACTORIES[settings.provider](settings)
    except KeyError as exc:
        raise ConfigurationError(f"unknown llm provider: {settings.provider!r}") from exc
```

Factories create **raw** adapters; the container applies decorators. Separation keeps both simple.

### 8.4 Resilience — Decorators around the port, never inside adapters

**Why:** retries, timeouts, circuit breaking, caching are *policies about calling any external brain*. Writing them once as decorators protects all providers forever (DRY + Open/Closed), and adapters stay pure translators.

```python
# src/project5/infrastructure/resilience.py
import asyncio
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

_RETRYABLE = (RateLimitError, ProviderTimeoutError, ProviderUnavailableError)


class ResilientChatModel:
    """Timeout + exponential backoff around ANY ChatModel."""
    def __init__(self, inner: ChatModel, *, timeout_s: float = 30, max_attempts: int = 3) -> None:
        self._inner = inner
        self._timeout_s = timeout_s
        self._max_attempts = max_attempts
        self.info = inner.info                        # capabilities pass through

    async def complete(self, request: ChatRequest) -> ChatResponse:
        async with asyncio.timeout(self._timeout_s):
            retrying = AsyncRetrying(
                stop=stop_after_attempt(self._max_attempts),
                wait=wait_exponential(multiplier=0.5, max=8),
                retry=retry_if_exception_type(_RETRYABLE),
                reraise=True,
            )
            return await retrying(self._inner.complete, request)


class CachedChatModel:
    """Caches deterministic requests only — temperature > 0 must NEVER be cached."""
    def __init__(self, inner: ChatModel, cache: Cache, *, ttl_s: int = 3600) -> None: ...
    async def complete(self, request: ChatRequest) -> ChatResponse:
        if request.temperature > 0:
            return await self._inner.complete(request)
        key = _cache_key(request)                     # sha256 over canonical request
        ...                                           # get → miss → call → set
```

**Circuit breaker** (same decorator shape — sketch; harden or use `aiobreaker` in prod):

```
closed ──(N consecutive failures)──► open ──(cooldown)──► half-open ──(probe ok)──► closed
   ▲                                                                      │
   └────────────────────────────(probe fails)─────────────────────────────┘
```

While `open`, calls raise `ProviderUnavailableError` **instantly** — your API returns a fast 503 instead of hanging 30 seconds per request during an outage.

> ⚠️ **Rule:** every external call has a timeout, bounded retries on retryable errors only, and a circuit breaker. "Retry on everything" turns a 429 storm into a self-inflicted DDoS.

### 8.5 Observability

- **structlog** JSON logs in production; pretty console in dev. One logger per module: `logger = structlog.get_logger(__name__)`.
- **Request-ID middleware** stamps every log line, so one request is traceable across retrieval → generation.
- **Log what:** model name, token usage, latency, error taxonomy, provider. **Never log (at INFO):** raw prompts, raw user documents, API keys (`SecretStr` enforces the last one).
- Every `ChatResponse` carries usage → per-endpoint/per-day cost aggregation is a group-by, not a retrofit.

---

## 9. Design Pattern Catalog

| # | Pattern | Where | Why here |
|---|---|---|---|
| 1 | Ports & Adapters | whole architecture | provider/storage independence (§2) |
| 2 | Dependency Injection | constructor injection everywhere; `container.py` | testability, loose coupling |
| 3 | Strategy | `ChunkingStrategy`, optional `Reranker` | swap algorithms without editing callers |
| 4 | Decorator | `ResilientChatModel`, `CachedChatModel`, circuit breaker | add cross-cutting behavior without touching adapters |
| 5 | Factory + Registry | `providers.py` | Open/Closed provider onboarding |
| 6 | Repository | `VectorStore` port + pgvector/in-memory adapters | persistence swappable, fake-able |
| 7 | Facade | `Container` (and route-level response builders) | one simple entry point over a complex graph |
| 8 | Null Object | `None` reranker / `NullCache` | optional behavior without `if` checks in use cases |
| 9 | Builder | `PromptTemplate.render(**vars) -> tuple[ChatMessage, ...]` | prompt assembly validated in one place |
| 10 | Template Method | *future:* `BaseFileLoader` ABC when ≥3 loaders share plumbing | don't abstract 2 implementations |
| 11 | Chain of Responsibility | *future:* guardrail pipeline (PII → length → jailbreak checks) | staged request filtering |
| 12 | Observer / Events | *future:* ingestion events → metrics/webhooks | YAGNI until a real subscriber exists |

**Detail — the four you'll use daily:**

- **Decorator (§8.4):** `CachedChatModel(ResilientChatModel(OpenAiChatModel(...)))` — behavior composes like layers, and each layer is unit-tested alone. This is "open for extension, closed for modification" as a one-liner.
- **Strategy (domain):** chunking is pure logic with swappable implementations:

  ```python
  class ChunkingStrategy(Protocol):
      def chunk(self, document: Document) -> list[Chunk]: ...

  class FixedWindowChunker:      # size + overlap — deterministic
  class RecursiveChunker:        # respects sentence/paragraph boundaries
  ```
  Selected by config in the container; use cases never know which one runs.
- **Factory + Registry (§8.3):** dispatch by config value; unknown value = boot-time `ConfigurationError`.
- **Facade (§8.2):** `Container.ask_question` hides ~10 collaborating objects behind one attribute.

---

## 10. SOLID Principles Mapping

| Principle | Where it lives in this codebase | How you can verify it |
|---|---|---|
| **S** — Single Responsibility | adapters translate; use cases orchestrate; domain decides; container wires | Describe any class in one sentence without "and" |
| **O** — Open/Closed | new provider = 1 adapter + 1 registry entry; new behavior = 1 decorator | "Can I add this without editing existing files?" |
| **L** — Liskov | every `ChatModel` adapter honors the port contract — enforced by shared contract tests | Run the contract suite against the new adapter |
| **I** — Interface Segregation | `ChatModel` vs `StreamingChatModel`; capabilities in `ModelInfo` | "Does any adapter implement a method it can't honor?" |
| **D** — Dependency Inversion | use cases → ports; concretes only in `infrastructure/` | `grep -r "import openai" src/project5/application/` → empty |

> 📌 **The one-sentence test (SRP):** if describing a class requires "and", split it. `OpenAiChatModel` → "Translates OpenAI SDK calls into canonical chat types and errors." ✅

---

## 11. AI-Engineering Practices

These separate a demo from production. Each is cheap now, expensive to retrofit.

1. **Prompts are versioned code artifacts.** `PromptTemplate(name, version, system, user, required_vars)` behind a `PromptCatalog` port. A prompt buried as a string literal in a function cannot be evaluated, rolled back, or diffed.
2. **Structured output is validated, never trusted.** If `info.supports_structured_output`, send the JSON schema, parse with pydantic, retry once with a repair instruction on validation failure. LLMs hallucinate keys; your parser must expect it.
3. **Token/cost accounting is built into the port.** `ChatResponse.usage` is mandatory; log it every call; aggregate per endpoint/day. Retrofitted cost tracking is guesswork.
4. **Idempotent ingestion.** `content_hash = sha256(normalized_text)` per document; upsert skips unchanged docs. Re-uploading the same PDF must not double your embedding bill.
5. **Timeout every external call** (`asyncio.timeout`), cap retries, circuit-break failing providers (§8.4).
6. **Streaming is a capability from day one** (`StreamingChatModel` port), so SSE endpoints never force an architecture change later.
7. **Log hygiene / least privilege:** no raw prompts at INFO; `SecretStr` for keys; request-ID on every line; full detail only in logs, generic messages only to clients.
8. **Don't put LangChain/LlamaIndex inside the architecture.** Use direct SDKs behind your ports. Frameworks abstract the wrong things (your prompts, costs, error handling), churn APIs, and fight pyright strict. If ever needed, they enter as an adapter — never through the domain.

> 📌 **YAGNI list — do NOT build yet:** multi-tenancy, hybrid search, full eval harness, agent loops, reranker ensemble. Each becomes a single-file addition later precisely because the ports exist.

---

## 12. Testing Strategy

### 12.1 The pyramid

| Level | Count | Speed | Tests what | Uses |
|---|---|---|---|---|
| **Unit** | hundreds | ms | domain rules + use cases | `FakeChatModel`, `InMemoryVectorStore` |
| **Contract** | per adapter | ms (respx-mocked HTTP) | adapter honors the port contract | every `ChatModel` adapter, parametrized |
| **Integration** | handful | s | real pgvector SQL | docker compose |
| **E2E** | few | s | full HTTP round-trip | FastAPI app + fakes |

### 12.2 Contract tests — the mechanism that makes provider-swapping safe

```python
# tests/contract/test_chat_model_contract.py
@pytest.mark.parametrize("factory", [_make_openai, _make_anthropic, _make_ollama])
class TestChatModelContract:
    async def test_returns_canonical_usage(self, factory) -> None: ...
    async def test_maps_rate_limit_to_domain_error(self, factory, respx_mock) -> None: ...
    async def test_maps_timeout_to_domain_error(self, factory, respx_mock) -> None: ...
    async def test_empty_message_rejected(self, factory) -> None: ...
    async def test_honors_json_schema_capability(self, factory) -> None: ...
```

CI runs these against `respx`-mocked HTTP (deterministic, no keys). A nightly job runs the same suite against live APIs. **A new provider isn't "supported" until its adapter passes the identical suite** — that's Liskov Substitution, mechanically enforced.

### 12.3 Fakes over mocks

```python
class FakeChatModel:
    ...  # records self.calls — tests assert on real interactions
```

`assert fake.calls[0].messages[0].content == "..."` reads better than mock frameworks, works with pyright, and doubles as dev tooling (wire fakes via config for local runs without API keys).

### 12.4 Coverage policy

- `domain/` + `application/`: ~100% (pure logic — no excuse).
- Adapters: covered by contract tests.
- Global gate: `--cov-fail-under=80`.
- Unhappy paths are mandatory: empty retrieval, provider timeout, malformed JSON output, rate limit.

---

## 13. Error Handling and HTTP Mapping

Adapters translate provider failures into the domain taxonomy (§5.3); the API maps the taxonomy to HTTP **in one place**:

| Exception | HTTP | Client sees | Logs contain |
|---|---|---|---|
| `DomainError` | 400 / 422 | specific validation message | context |
| `RateLimitError` | 429 | "upstream rate limited" + `Retry-After` | provider, retry count |
| `ProviderTimeoutError` | 504 | "upstream timeout" | provider, elapsed |
| `ProviderError` | 502 | "upstream failure" | full provider detail |
| anything else | 500 | generic message | **full traceback** |

> ⚠️ **Defensibility rule:** fail loud internally, fail safe externally. Internal detail never reaches clients; client-visible messages never hide the failure from logs.

Always `raise X(...) from exc` — the causal chain is your debugging lifeline. Never bare `except:`, never `except Exception: pass` (ruff `TRY` enforces both).

---

## 14. Code Standards and Checklists

### 14.1 The ten rules for every file

1. Every public function fully annotated, including `-> None`; pyright strict clean.
2. Constructors receive dependencies; never instantiate adapters inside use cases.
3. Concrete SDK imports exist **only** in `adapters/` and `infrastructure/`.
4. Validate at every boundary (HTTP schema, `__post_init__`, `Settings` boot); trust types afterwards.
5. `raise SpecificError(...) from exc`; no bare except; no swallowed errors.
6. `logger = structlog.get_logger(__name__)`; no `print`, ever.
7. Resources via `async with` / `with` — no manual open/close pairs.
8. No mutable default arguments (`def f(x=[])` is banned — ruff `B` catches it).
9. `pathlib` for paths; env vars for config; nothing platform-specific (ruff `PTH`).
10. One test = one behavior; Arrange-Act-Assert; fakes over mocks.

### 14.2 Per-file self-review checklist

- [ ] Can I describe this class in one sentence without "and"? (SRP)
- [ ] Can clients corrupt internal state? → privatize (`_field`, frozen) (Encapsulation)
- [ ] Does this file construct its own dependencies? → inject them (DIP)
- [ ] Adding this feature required editing existing files? → decorator/strategy instead (OCP)
- [ ] Any hardcoded path, key, or model name? → config + `ModelInfo` (Portability)
- [ ] Any unvalidated input crossing a boundary? → fail fast (Defensibility)
- [ ] Would testing this require mocking >2 things? → split logic from I/O (Testability)
- [ ] Am I abstracting with only one implementation and no real variance? → delete it (YAGNI)
- [ ] Did I write this logic before? → extract (DRY)
- [ ] Does any provider type (`openai.*`) appear outside its adapter? → translate (Encapsulation)

### 14.3 Code smells → fixes in this codebase

| Smell | Violates | Fix |
|---|---|---|
| SDK import inside a use case | DIP | move to adapter |
| `if provider == "openai"` in business code | OCP | `ModelInfo` capabilities + registry |
| Constructor instantiating `AsyncOpenAI()` | DIP | inject client via container |
| Use case doing retrieve + embed + chunk + generate | SRP | split: `SimpleRetriever`, ingestion pipeline |
| Returning an SDK response object past the adapter | Encapsulation | return canonical `ChatResponse` |
| `except Exception: pass` | Defensibility | domain taxonomy + `raise ... from` |
| Hardcoded `"gpt-4o"` string | Portability | `ModelInfo.default_model` from settings |
| Logging raw prompts / keys | Least privilege | `SecretStr`, log hygiene rules (§8.5) |
| Retry logic duplicated per adapter | DRY | shared `ResilientChatModel` decorator |
| Port with one impl and no plausible second | YAGNI | delete the port |

### 14.4 PR checklist

- [ ] `uv run ruff check && uv run ruff format --check && uv run pyright && uv run pytest` — all green
- [ ] New behavior has tests, including one unhappy path
- [ ] No new concrete import outside `adapters/` / `infrastructure/`
- [ ] Port changed? → all adapters + contract tests updated
- [ ] Docs changed? → this file updated in the same PR

---

## 15. Anti-Patterns (never do these)

1. **LangChain as the architecture.** Direct SDKs behind ports; frameworks (if ever) as adapters.
2. **The God Service.** One `RagService` doing ingest+retrieve+generate+cache+retry. Split by responsibility (§9).
3. **Microservices now.** One well-boundaried service splits cleanly later; premature distribution splits nothing but your velocity.
4. **Port-per-class over-abstraction.** A port needs real variance or two plausible implementations.
5. **Live API calls in unit tests.** Fakes and respx; live providers = nightly contract job only.
6. **Secrets in pyproject, code, or committed `.env`.** Env vars + `SecretStr` + `.env.example` only.
7. **Blocking calls inside async functions** (e.g., sync `requests` in a route). Ruff `ASYNC` fails CI; use async clients or `asyncio.to_thread`.
8. **Unguarded streaming.** Streams get timeouts and cancellation handling too.
9. **`assert` for production validation.** Vanishes under `python -O`; raise domain errors instead.
10. **"Temporary" TODOs in `domain/`.** The purest layer accepts no debt.

---

## 16. Roadmap and Build Order

| # | Milestone | Deliverable | Done when |
|---|---|---|---|
| 0 | Skeleton | uv + pyproject + ruff/pyright strict green + pre-commit + CI | empty-but-annotated tree passes all checks |
| 1 | Domain + ports | models, error taxonomy, all ports, `FakeChatModel`, `InMemoryVectorStore` | domain unit tests green, offline |
| 2 | Vertical slice | `AskQuestion` + FastAPI route + e2e with fakes | full request round-trip works with zero real providers |
| 3 | Real LLM | OpenAI adapter + contract tests + registry + resilience decorators | provider swap = one env var |
| 4 | Persistence | pgvector adapter + docker-compose + integration tests | vectors survive restarts |
| 5 | Ingestion | loaders → chunking strategies → idempotent upsert | re-upload of same file is a no-op |
| 6 | Hardening | observability, Dockerfile, rate limiting at the edge | staging deploy green |
| 7+ | Growth (YAGNI-gated) | streaming SSE, reranker, eval harness | only when a real requirement arrives |

**The discipline that makes all of this work:** at every commit, `uv run pyright && uv run pytest` is green. The structure is just the enforcement mechanism for the principles above.

---

## 17. Glossary

| Term | Meaning |
|---|---|
| **Port** | Interface owned by the application declaring what it needs from outside (`ChatModel`) |
| **Adapter (driven)** | Implements a port for one concrete technology (`OpenAiChatModel`) |
| **Adapter (driving)** | Calls the application's use cases from outside (HTTP, CLI) |
| **Use case** | One application-level business operation (`AskQuestion.execute`) |
| **Composition root** | The single place (`container.py`) where concrete implementations are assembled |
| **Value object** | Immutable, self-validating domain type (`ChatMessage`) |
| **DTO / Command** | Data carrier crossing a boundary (pydantic at HTTP; dataclass at use case) |
| **Contract test** | Same behavioral suite run against every implementation of a port |
| **Fake** | Working lightweight implementation for tests (`FakeChatModel`) vs **mock** (asserts on call machinery) |
| **Null Object** | Do-nothing implementation used as a safe default (`None` reranker, `NullCache`) |
| **Composition over inheritance** | Wrapping objects (decorators) instead of deep class hierarchies |
| **Fail fast** | Reject invalid state at the earliest boundary with a loud, specific error |
| **Structured output** | LLM response constrained to a JSON schema and validated on receipt |
| **Idempotency** | Repeating an operation produces the same result (content-hash ingestion) |

---

## Appendix A: `.env.example`

```bash
# --- Application ---
DEBUG=false

# --- LLM provider ---
LLM__PROVIDER=openai            # openai | anthropic | ollama
LLM__API_KEY=sk-your-key-here   # SecretStr — never committed
LLM__TIMEOUT_S=30
LLM__MAX_RETRIES=3

# --- RAG pipeline ---
RAG__CHUNK_SIZE=800
RAG__CHUNK_OVERLAP=100
RAG__TOP_K=5
```

## Appendix B: `Makefile`

```make
.PHONY: install lint typecheck test run
install:
	uv sync
lint:
	uv run ruff check --fix .
	uv run ruff format .
typecheck:
	uv run pyright
test:
	uv run pytest
run:
	uv run uvicorn project5.adapters.driving.api.app:app --reload
```

## Appendix C: CI Workflow (`.github/workflows/ci.yml`)

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with:
          python-version: "3.12"
      - run: uv sync --frozen
      - run: uv run ruff check .
      - run: uv run ruff format --check .
      - run: uv run pyright
      - run: uv run pytest --cov-fail-under=80
```

---

*Maintainer's note: this document is part of the codebase. If a PR changes the architecture and not this file, the PR is incomplete.*
