#!/usr/bin/env python3
"""Example usage of the RAG pipeline."""

from local_rag import RAGPipeline


def main():
    """Run a simple example."""
    # Initialize the RAG pipeline
    print("Initializing RAG pipeline...")
    rag = RAGPipeline()

    # Add some sample documents
    print("\nAdding sample documents...")
    documents = [
        "Python is a high-level programming language known for its simplicity and readability.",
        "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
        "ChromaDB is a vector database designed for AI applications and embeddings.",
        "Ollama allows you to run large language models locally on your machine.",
        "LangChain is a framework for developing applications powered by language models.",
    ]
    rag.add_documents(documents)

    # Get stats
    stats = rag.get_stats()
    print("\nRAG System Stats:")
    print(f"  Documents: {stats['num_documents']}")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  LLM Model: {stats['ollama_model']}")
    print(f"  Embedding Model: {stats['embedding_model']}")

    # Query the system
    print("\n" + "=" * 80)
    question = "What is Python?"
    print(f"Question: {question}")
    print("=" * 80)

    result = rag.query(question)
    print(f"\nAnswer:\n{result['answer']}")

    print(f"\n\nContext used ({result['num_context_docs']} documents):")
    for i, doc in enumerate(result["context"], 1):
        print(f"\n{i}. {doc.page_content}")

    # Query with streaming
    print("\n" + "=" * 80)
    question = "How does machine learning relate to AI?"
    print(f"Question: {question}")
    print("=" * 80)
    print("\nStreaming Answer:")

    for chunk in rag.query_stream(question):
        print(chunk, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    main()
