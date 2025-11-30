"""Main RAG pipeline that orchestrates all components."""

from typing import List, Optional

from langchain_core.documents import Document

from .config import Settings, get_settings
from .embeddings import EmbeddingManager
from .generator import Generator
from .vectorstore import VectorStoreManager


class RAGPipeline:
    """Complete RAG pipeline integrating embeddings, retrieval, and generation."""

    def __init__(self, settings: Optional[Settings] = None, prompt_template: Optional[str] = None):
        """Initialize the RAG pipeline.

        Args:
            settings: Optional application settings (uses defaults if not provided)
            prompt_template: Optional custom prompt template for generation
        """
        self.settings = settings or get_settings()

        # Initialize components
        self.embedding_manager = EmbeddingManager(self.settings)
        self.vectorstore = VectorStoreManager(self.settings, self.embedding_manager)
        self.generator = Generator(self.settings, prompt_template)

    def add_documents(self, documents: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
        """Add documents to the knowledge base.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document

        Returns:
            List of document IDs
        """
        return self.vectorstore.add_documents(documents, metadatas)

    def query(self, question: str, k: Optional[int] = None) -> dict:
        """Query the RAG system.

        Args:
            question: User's question
            k: Number of context documents to retrieve (defaults to settings.top_k)

        Returns:
            Dictionary containing answer, context documents, and metadata
        """
        # Retrieve relevant documents
        context_docs = self.vectorstore.similarity_search(question, k=k)

        # Generate answer
        answer = self.generator.generate(question, context_docs)

        return {
            "answer": answer,
            "context": context_docs,
            "question": question,
            "num_context_docs": len(context_docs),
        }

    def query_with_scores(self, question: str, k: Optional[int] = None) -> dict:
        """Query the RAG system with similarity scores.

        Args:
            question: User's question
            k: Number of context documents to retrieve (defaults to settings.top_k)

        Returns:
            Dictionary containing answer, context documents with scores, and metadata
        """
        # Retrieve relevant documents with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=k)
        context_docs = [doc for doc, _ in docs_with_scores]

        # Generate answer
        answer = self.generator.generate(question, context_docs)

        return {
            "answer": answer,
            "context": docs_with_scores,
            "question": question,
            "num_context_docs": len(context_docs),
        }

    def query_stream(self, question: str, k: Optional[int] = None):
        """Query the RAG system with streaming response.

        Args:
            question: User's question
            k: Number of context documents to retrieve (defaults to settings.top_k)

        Yields:
            Chunks of the generated answer
        """
        # Retrieve relevant documents
        context_docs = self.vectorstore.similarity_search(question, k=k)

        # Stream the answer
        yield from self.generator.generate_stream(question, context_docs)

    def get_stats(self) -> dict:
        """Get statistics about the RAG system.

        Returns:
            Dictionary with system statistics
        """
        return {
            "num_documents": self.vectorstore.get_collection_count(),
            "collection_name": self.settings.chroma_collection_name,
            "ollama_model": self.settings.ollama_model,
            "embedding_model": self.settings.ollama_embedding_model,
        }

    def reset(self):
        """Reset the vector store by deleting all documents."""
        self.vectorstore.delete_collection()
        # Reinitialize vector store
        self.vectorstore = VectorStoreManager(self.settings, self.embedding_manager)
