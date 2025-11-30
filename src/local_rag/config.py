"""Configuration management for the RAG system."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Ollama settings
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    ollama_model: str = Field(default="llama2", description="Ollama model to use for generation")
    ollama_embedding_model: str = Field(
        default="nomic-embed-text", description="Ollama model to use for embeddings"
    )

    # ChromaDB settings
    chroma_persist_directory: Path = Field(
        default=Path("./chroma_db"), description="Directory to persist ChromaDB data"
    )
    chroma_collection_name: str = Field(
        default="documents", description="ChromaDB collection name"
    )

    # RAG settings
    chunk_size: int = Field(default=1000, description="Size of text chunks for embedding")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    top_k: int = Field(default=4, description="Number of documents to retrieve")

    # Generation settings
    temperature: float = Field(default=0.7, description="Temperature for text generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")

    # PDF processing settings
    pdf_preserve_layout: bool = Field(
        default=True, description="Preserve layout and bounding box information from PDFs"
    )


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
