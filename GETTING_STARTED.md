# Getting Started with Local RAG

A simple guide to start chatting with your documents in 3 easy steps.

## What You'll Need

- A Mac computer (works best with M1, M2, M3, or M4 chips)
- About 15 minutes for setup
- Your documents (PDF or text files)

## Step 1: Install the Required Software

### Install Ollama (the AI engine)

1. Visit [ollama.ai](https://ollama.ai)
2. Click "Download for Mac"
3. Open the downloaded file and drag Ollama to your Applications folder
4. Open Ollama - you should see a llama icon in your menu bar

### Download the AI models

Open Terminal (find it in Applications → Utilities) and run these two commands:

```bash
ollama pull gpt-oss:20b
ollama pull nomic-embed-text
```

This downloads the AI models. It may take a few minutes depending on your internet speed.

### Install uv (the Python package manager)

In Terminal, run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart Terminal.

## Step 2: Set Up the Chat System

### Get the code

In Terminal, navigate to where you want to install this (like your Documents folder), then run:

```bash
git clone https://github.com/aaronmak/local_rag.git
cd local_rag
```

### Install dependencies

```bash
uv sync
```

This installs all the required software packages.

## Step 3: Add Your Documents and Start Chatting

### Add your documents

1. Copy your PDF or text files into the `data/documents/` folder
2. In Terminal, run this command to process them:

```bash
uv run python scripts/ingest_documents.py data/documents/
```

You'll see messages showing each file being loaded. This may take a minute or two.

### Start the chat

```bash
uv run python scripts/chat.py
```

You should see:

```
================================================================================
Local RAG Chat Interface
================================================================================

Welcome! Ask questions about your documents.

Commands:
  /stats  - Show RAG system statistics
  /help   - Show this help message
  /quit   - Exit the chat
================================================================================

You:
```

Now just type your questions and press Enter!

## Using the Chat

### Ask questions

Simply type your question and press Enter. The AI will search through your documents and answer based on what it finds.

Example:

```
You: What are the main topics in these documents?
```

The AI will stream its response in real-time.

### Helpful commands

- Type `/stats` to see how many documents are loaded
- Type `/help` if you need a reminder of the commands
- Type `/quit` when you're done chatting

### Tips

- Be specific with your questions for better answers
- The AI can only answer based on what's in your documents
- If you add more documents later, run the ingest command again:

  ```bash
  uv run python scripts/ingest_documents.py data/documents/
  ```

## Troubleshooting

### "Error initializing RAG system"

Make sure Ollama is running (look for the llama icon in your menu bar). If not:

1. Open the Ollama application
2. Wait for the menu bar icon to appear
3. Try running the chat again

### "No documents found in the vector store!"

You need to add documents first:

1. Put PDF or text files in `data/documents/`
2. Run: `uv run python scripts/ingest_documents.py data/documents/`
3. Then start the chat again

### The AI is slow

The first response may be slower as the model loads. Subsequent responses should be faster. If you have an older Mac, responses will naturally be slower than on newer M-series chips.

## Next Steps

Once you're comfortable with the basics:

- Add more documents to expand the knowledge base
- Try different types of questions
- Check out `README.md` for advanced features and customization options

Enjoy chatting with your documents! 🚀
