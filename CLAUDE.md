# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A local RAG (Retrieval-Augmented Generation) system built with Python, using Ollama for local LLM inference, ChromaDB for vector storage, and LangChain for orchestration. The system enables question-answering over custom documents entirely on the local machine.

## Technology Stack

- **Language**: Python 3.9+
- **Package Manager**: uv (fast Python package installer)
- **LLM**: Ollama (local inference)
- **Vector Database**: ChromaDB
- **Framework**: LangChain
- **Testing**: pytest with coverage
- **Code Quality**: Black (formatter), Ruff (linter), mypy (type checker)

## Common Commands

### Setup and Installation

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --extra dev

# Install Ollama models (external requirement)
ollama pull llama2
ollama pull nomic-embed-text
```

### Running the Application

```bash
# Run example usage script
uv run python scripts/example_usage.py

# Ingest documents from a directory
uv run python scripts/ingest_documents.py data/documents/

# Ingest with vector store reset
uv run python scripts/ingest_documents.py data/documents/ --reset
```

### Development Commands

```bash
# Run all tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/local_rag --cov-report=html

# Run a single test file
uv run pytest tests/test_pipeline.py

# Run a specific test
uv run pytest tests/test_pipeline.py::test_pipeline_initialization

# Format code
uv run black src/ tests/ scripts/

# Lint code
uv run ruff check src/ tests/ scripts/

# Type check
uv run mypy src/
```

## Code Architecture

### Component Structure

The RAG system is organized into modular components:

1. **config.py**: Centralized configuration using Pydantic settings
   - Loads from environment variables or `.env` file
   - Type-safe configuration with validation
   - Settings for Ollama, ChromaDB, and RAG parameters

2. **embeddings.py**: Embedding management
   - Wraps Ollama embeddings via LangChain
   - Handles both document and query embeddings
   - Uses `nomic-embed-text` model by default

3. **vectorstore.py**: Vector storage and retrieval
   - ChromaDB integration with persistent storage
   - Document chunking using RecursiveCharacterTextSplitter
   - Similarity search with optional scoring
   - Collection management (add, delete, count)

4. **generator.py**: Text generation
   - Ollama LLM integration for response generation
   - Customizable prompt templates
   - Streaming support for real-time responses
   - Context-aware generation using retrieved documents

5. **pipeline.py**: Main orchestrator
   - High-level API combining all components
   - Methods: `add_documents()`, `query()`, `query_stream()`, `get_stats()`, `reset()`
   - Manages component lifecycle and data flow

### Data Flow

```
User Query
    ↓
[Embedding Manager] - Convert query to vector
    ↓
[Vector Store] - Similarity search for relevant documents
    ↓
[Generator] - Generate response using LLM + context
    ↓
Response to User
```

### Configuration System

The project uses a layered configuration approach:
- Default values in `config.py`
- Override via `.env` file (copy from `.env.example`)
- Override via environment variables
- Override programmatically via `Settings` object

Key settings:
- `OLLAMA_MODEL`: LLM for generation (default: llama2)
- `OLLAMA_EMBEDDING_MODEL`: Model for embeddings (default: nomic-embed-text)
- `CHUNK_SIZE`: Text chunk size (default: 1000)
- `TOP_K`: Number of documents to retrieve (default: 4)

### Testing Strategy

Tests are located in `tests/` and use pytest:
- Fixtures for test settings and pipeline initialization
- Cleanup after each test to avoid state pollution
- Test ChromaDB uses separate collection (`test_documents`)
- Coverage tracking with pytest-cov

## Project Structure

```
src/local_rag/          # Main package
├── __init__.py         # Exports RAGPipeline
├── config.py           # Settings management
├── embeddings.py       # Ollama embeddings wrapper
├── vectorstore.py      # ChromaDB integration
├── generator.py        # Ollama LLM generation
└── pipeline.py         # Main RAG orchestrator

tests/                  # Test suite
├── __init__.py
└── test_pipeline.py    # Pipeline integration tests

scripts/                # Utility scripts
├── example_usage.py    # Basic usage example
└── ingest_documents.py # Document ingestion tool

data/documents/         # Place documents to ingest here
```

## Development Notes

### Adding New Features

When extending the system:
- Keep components modular and single-purpose
- Update configuration in `config.py` if new settings are needed
- Add tests in `tests/` for new functionality
- Update example scripts if the API changes
- Follow the existing patterns for error handling

### Working with Ollama

Ensure Ollama is running locally:
```bash
ollama serve  # If not running as service
```

List available models:
```bash
ollama list
```

Change models by updating `.env`:
```env
OLLAMA_MODEL=mistral
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

### ChromaDB Persistence

- Vector data persists in `./chroma_db/` (configurable)
- Use `pipeline.reset()` to clear all data
- Each collection is independent (useful for multiple projects)
