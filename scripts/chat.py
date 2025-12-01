#!/usr/bin/env python3
"""Interactive terminal chat interface for the RAG system."""

import sys

from local_rag import RAGPipeline


def print_header():
    """Print the welcome header."""
    print("\n" + "=" * 80)
    print("Local RAG Chat Interface")
    print("=" * 80)
    print("\nWelcome! Ask questions about your documents.")
    print("\nCommands:")
    print("  /stats  - Show RAG system statistics")
    print("  /help   - Show this help message")
    print("  /quit   - Exit the chat")
    print("=" * 80 + "\n")


def print_stats(rag: RAGPipeline):
    """Print RAG system statistics."""
    stats = rag.get_stats()
    print("\n" + "-" * 80)
    print("RAG System Statistics")
    print("-" * 80)
    print(f"Documents in collection: {stats['num_documents']}")
    print(f"Collection name:         {stats['collection_name']}")
    print(f"LLM model:               {stats['ollama_model']}")
    print(f"Embedding model:         {stats['embedding_model']}")
    print("-" * 80 + "\n")


def print_help():
    """Print help information."""
    print("\n" + "-" * 80)
    print("Help")
    print("-" * 80)
    print("Ask any question about your documents. The system will:")
    print("  1. Search for relevant document chunks")
    print("  2. Use those chunks as context")
    print("  3. Generate an answer using the local LLM")
    print("\nCommands:")
    print("  /stats  - Show system statistics")
    print("  /help   - Show this help message")
    print("  /quit   - Exit the chat")
    print("-" * 80 + "\n")


def main():
    """Run the interactive chat interface."""
    # Initialize RAG pipeline
    print("Initializing RAG system...")
    try:
        rag = RAGPipeline()
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running (ollama serve)")
        print("  2. Required models are installed:")
        print("     - ollama pull llama3:8b")
        print("     - ollama pull nomic-embed-text")
        sys.exit(1)

    # Check if there are documents
    stats = rag.get_stats()
    if stats["num_documents"] == 0:
        print("\n⚠️  Warning: No documents found in the vector store!")
        print("Please ingest documents first using:")
        print("  uv run python scripts/ingest_documents.py data/documents/\n")

        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != "y":
            print("Exiting...")
            sys.exit(0)

    print_header()

    # Main chat loop
    while True:
        try:
            # Get user input
            question = input("You: ").strip()

            # Handle empty input
            if not question:
                continue

            # Handle commands
            if question.startswith("/"):
                command = question.lower()

                if command == "/quit" or command == "/exit" or command == "/q":
                    print("\nGoodbye!")
                    break

                elif command == "/stats":
                    print_stats(rag)
                    continue

                elif command == "/help" or command == "/h":
                    print_help()
                    continue

                else:
                    print(f"Unknown command: {question}")
                    print("Type /help for available commands.\n")
                    continue

            # Query the RAG system
            print("", flush=True)

            try:
                # Stream the response
                for chunk in rag.query_stream(question):
                    print(chunk, end="", flush=True)
                print("\n")

            except Exception as e:
                print(f"\n\nError generating response: {e}")
                print("Please check that Ollama is running and the models are available.\n")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            print("Type /quit to exit or press Enter to continue.\n")
            continue

        except EOFError:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
