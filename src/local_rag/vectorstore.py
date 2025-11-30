"""Vector store management using ChromaDB."""

from typing import List, Optional

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .embeddings import EmbeddingManager


class VectorStoreManager:
    """Manages the ChromaDB vector store."""

    def __init__(self, settings: Settings, embedding_manager: EmbeddingManager):
        """Initialize the vector store manager.

        Args:
            settings: Application settings
            embedding_manager: Embedding manager instance
        """
        self.settings = settings
        self.embedding_manager = embedding_manager

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))

        # Initialize vector store
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=settings.chroma_collection_name,
            embedding_function=embedding_manager.embeddings,
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )

    def add_documents(
        self, documents: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document

        Returns:
            List of document IDs
        """
        # Split documents into chunks
        docs = [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents, metadatas or [{}] * len(documents))
        ]
        split_docs = self.text_splitter.split_documents(docs)

        # Add to vector store
        ids = self.vectorstore.add_documents(split_docs)
        return ids

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return (defaults to settings.top_k)

        Returns:
            List of similar documents
        """
        k = k or self.settings.top_k
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: Optional[int] = None) -> List[tuple]:
        """Search for similar documents with similarity scores.

        Args:
            query: Query text
            k: Number of results to return (defaults to settings.top_k)

        Returns:
            List of tuples (document, score)
        """
        k = k or self.settings.top_k
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.settings.chroma_collection_name)

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            Number of documents
        """
        collection = self.client.get_collection(self.settings.chroma_collection_name)
        return collection.count()
