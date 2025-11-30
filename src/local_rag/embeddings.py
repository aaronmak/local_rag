"""Embedding functionality using Ollama."""

from typing import List

from langchain_community.embeddings import OllamaEmbeddings

from .config import Settings


class EmbeddingManager:
    """Manages document embeddings using Ollama."""

    def __init__(self, settings: Settings):
        """Initialize the embedding manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        self.embeddings = OllamaEmbeddings(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(text)
