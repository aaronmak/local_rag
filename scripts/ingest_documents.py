#!/usr/bin/env python3
"""Script to ingest documents from a directory into the RAG system."""

import argparse
from pathlib import Path
from typing import List

from local_rag import RAGPipeline


def load_text_files(directory: Path) -> List[tuple[str, dict]]:
    """Load all text files from a directory.

    Args:
        directory: Directory containing text files

    Returns:
        List of tuples (content, metadata)
    """
    documents = []

    for file_path in directory.rglob("*.txt"):
        try:
            content = file_path.read_text(encoding="utf-8")
            metadata = {
                "source": str(file_path),
                "filename": file_path.name,
            }
            documents.append((content, metadata))
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return documents


def main():
    """Ingest documents into the RAG system."""
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG system")
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing documents to ingest",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the vector store before ingesting",
    )

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory {args.directory} does not exist")
        return

    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline()

    # Reset if requested
    if args.reset:
        print("Resetting vector store...")
        rag.reset()

    # Load documents
    print(f"\nLoading documents from {args.directory}...")
    docs_with_metadata = load_text_files(args.directory)

    if not docs_with_metadata:
        print("No documents found!")
        return

    # Separate documents and metadata
    documents = [doc for doc, _ in docs_with_metadata]
    metadatas = [meta for _, meta in docs_with_metadata]

    # Add to RAG system
    print(f"\nAdding {len(documents)} documents to the RAG system...")
    ids = rag.add_documents(documents, metadatas)

    # Print stats
    stats = rag.get_stats()
    print(f"\nIngestion complete!")
    print(f"  Total documents in system: {stats['num_documents']}")
    print(f"  Documents added: {len(ids)}")


if __name__ == "__main__":
    main()
