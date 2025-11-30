"""Tests for the RAG pipeline."""

import pytest
from local_rag import RAGPipeline
from local_rag.config import Settings


@pytest.fixture
def test_settings():
    """Create test settings."""
    return Settings(
        chroma_persist_directory="./test_chroma_db",
        chroma_collection_name="test_documents",
    )


@pytest.fixture
def rag_pipeline(test_settings):
    """Create a RAG pipeline for testing."""
    pipeline = RAGPipeline(settings=test_settings)
    yield pipeline
    # Cleanup
    try:
        pipeline.reset()
    except Exception:
        pass


def test_pipeline_initialization(rag_pipeline):
    """Test that the pipeline initializes correctly."""
    assert rag_pipeline is not None
    assert rag_pipeline.settings is not None
    assert rag_pipeline.embedding_manager is not None
    assert rag_pipeline.vectorstore is not None
    assert rag_pipeline.generator is not None


def test_add_documents(rag_pipeline):
    """Test adding documents to the pipeline."""
    documents = [
        "This is a test document.",
        "This is another test document.",
    ]

    ids = rag_pipeline.add_documents(documents)
    assert len(ids) > 0

    stats = rag_pipeline.get_stats()
    assert stats["num_documents"] > 0


def test_query(rag_pipeline):
    """Test querying the pipeline."""
    documents = [
        "Python is a programming language.",
        "JavaScript is also a programming language.",
    ]

    rag_pipeline.add_documents(documents)

    result = rag_pipeline.query("What is Python?")

    assert "answer" in result
    assert "context" in result
    assert "question" in result
    assert result["question"] == "What is Python?"
    assert result["num_context_docs"] > 0


def test_get_stats(rag_pipeline):
    """Test getting pipeline statistics."""
    stats = rag_pipeline.get_stats()

    assert "num_documents" in stats
    assert "collection_name" in stats
    assert "ollama_model" in stats
    assert "embedding_model" in stats
    assert stats["collection_name"] == "test_documents"
