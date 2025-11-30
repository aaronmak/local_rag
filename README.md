# Local RAG System

A local Retrieval-Augmented Generation (RAG) system built with Ollama, ChromaDB, and LangChain. This system allows you to build a question-answering system over your own documents, running entirely on your local machine.

## Features

- **Local-first**: Runs completely on your machine with Ollama
- **Vector Storage**: Uses ChromaDB for efficient document retrieval
- **Multiple Document Formats**: Supports PDF and text files
- **Flexible**: Built on LangChain for easy customization
- **Simple API**: Easy-to-use Python interface
- **Streaming Support**: Stream responses for better UX

## Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- [Ollama](https://ollama.ai) installed and running

### Installing Ollama

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the required models:

   ```bash
   ollama pull llama2
   ollama pull nomic-embed-text
   ```

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd local_rag
   ```

2. Install dependencies using uv:

   ```bash
   uv sync
   ```

3. Create a `.env` file (optional):

   ```bash
   cp .env.example .env
   ```

## Quick Start

### Basic Usage

```python
from local_rag import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline()

# Add documents
documents = [
    "Python is a high-level programming language.",
    "Machine learning is a subset of AI.",
]
rag.add_documents(documents)

# Query the system
result = rag.query("What is Python?")
print(result["answer"])
```

### Run the Example Script

```bash
uv run python scripts/example_usage.py
```

### Ingest Documents

Place your documents (`.txt` or `.pdf` files) in the `data/documents/` directory, then run:

```bash
uv run python scripts/ingest_documents.py data/documents/
```

The script automatically detects and processes both text and PDF files, extracting text content and storing it with metadata including file type, source path, and page count (for PDFs).

Options:

- `--reset`: Clear the vector store before ingesting

## Development

### Setup Development Environment

Install development dependencies:

```bash
uv sync --extra dev
```

### Running Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src/local_rag --cov-report=html
```

### Code Formatting

Format code with Black:

```bash
uv run black src/ tests/ scripts/
```

Lint with Ruff:

```bash
uv run ruff check src/ tests/ scripts/
```

Type checking with mypy:

```bash
uv run mypy src/
```

## Configuration

Configuration can be set via environment variables or a `.env` file:

| Variable                   | Default                  | Description                     |
| -------------------------- | ------------------------ | ------------------------------- |
| `OLLAMA_BASE_URL`          | `http://localhost:11434` | Ollama API URL                  |
| `OLLAMA_MODEL`             | `llama2`                 | Model for text generation       |
| `OLLAMA_EMBEDDING_MODEL`   | `nomic-embed-text`       | Model for embeddings            |
| `CHROMA_PERSIST_DIRECTORY` | `./chroma_db`            | ChromaDB storage path           |
| `CHROMA_COLLECTION_NAME`   | `documents`              | Collection name                 |
| `CHUNK_SIZE`               | `1000`                   | Text chunk size                 |
| `CHUNK_OVERLAP`            | `200`                    | Chunk overlap size              |
| `TOP_K`                    | `4`                      | Number of documents to retrieve |
| `TEMPERATURE`              | `0.7`                    | Generation temperature          |

## Project Structure

```
local_rag/
├── src/local_rag/          # Main package
│   ├── __init__.py         # Package exports
│   ├── config.py           # Configuration management
│   ├── embeddings.py       # Embedding functionality
│   ├── vectorstore.py      # ChromaDB vector store
│   ├── generator.py        # Ollama text generation
│   └── pipeline.py         # Main RAG pipeline
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── data/                   # Data directory
│   └── documents/          # Place documents here
├── config/                 # Configuration files
└── pyproject.toml          # Project configuration
```

## API Reference

### RAGPipeline

Main interface for the RAG system.

#### Methods

- `add_documents(documents, metadatas=None)`: Add documents to the knowledge base
- `query(question, k=None)`: Query the system and get an answer
- `query_with_scores(question, k=None)`: Query with similarity scores
- `query_stream(question, k=None)`: Query with streaming response
- `get_stats()`: Get system statistics
- `reset()`: Clear the vector store

## License

MIT
