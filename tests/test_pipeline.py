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


def test_query_with_scores(rag_pipeline):
    """Test querying with similarity scores."""
    documents = [
        "Python is a programming language.",
        "JavaScript is also a programming language.",
    ]

    rag_pipeline.add_documents(documents)

    result = rag_pipeline.query_with_scores("What is Python?")

    assert "answer" in result
    assert "context" in result
    assert "question" in result
    assert result["question"] == "What is Python?"
    assert result["num_context_docs"] > 0

    # Check that context includes scores
    assert len(result["context"]) > 0
    doc, score = result["context"][0]
    assert isinstance(score, float)


def test_query_stream(rag_pipeline):
    """Test streaming query responses."""
    documents = [
        "Python is a programming language.",
        "JavaScript is also a programming language.",
    ]

    rag_pipeline.add_documents(documents)

    chunks = list(rag_pipeline.query_stream("What is Python?"))

    assert len(chunks) > 0
    # Verify we can join chunks into a complete response
    full_response = "".join(chunks)
    assert len(full_response) > 0


def test_reset(rag_pipeline):
    """Test resetting the pipeline."""
    documents = [
        "Python is a programming language.",
        "JavaScript is also a programming language.",
    ]

    # Add documents
    rag_pipeline.add_documents(documents)
    stats = rag_pipeline.get_stats()
    assert stats["num_documents"] > 0

    # Reset
    rag_pipeline.reset()
    stats = rag_pipeline.get_stats()
    assert stats["num_documents"] == 0


def test_add_documents_with_metadata(rag_pipeline):
    """Test adding documents with metadata."""
    documents = [
        "This is document one.",
        "This is document two.",
    ]
    metadatas = [
        {"source": "test1.txt", "author": "Alice"},
        {"source": "test2.txt", "author": "Bob"},
    ]

    ids = rag_pipeline.add_documents(documents, metadatas)
    assert len(ids) == 2

    # Query and verify we can retrieve documents
    result = rag_pipeline.query("document one")
    assert result["num_context_docs"] > 0


def test_custom_prompt_template(test_settings):
    """Test using a custom prompt template."""
    custom_template = """Custom template: {context}

Question: {question}
Answer:"""

    pipeline = RAGPipeline(settings=test_settings, prompt_template=custom_template)

    documents = ["Python is a programming language."]
    pipeline.add_documents(documents)

    result = pipeline.query("What is Python?")
    assert "answer" in result

    # Cleanup
    pipeline.reset()


def test_pipeline_with_k_parameter(rag_pipeline):
    """Test querying with custom k parameter."""
    documents = [
        "Document one about Python.",
        "Document two about JavaScript.",
        "Document three about Ruby.",
        "Document four about Go.",
    ]

    rag_pipeline.add_documents(documents)

    # Query with k=2
    result = rag_pipeline.query("programming languages", k=2)
    assert result["num_context_docs"] == 2

    # Query with k=3
    result_with_scores = rag_pipeline.query_with_scores("programming languages", k=3)
    assert result_with_scores["num_context_docs"] == 3
