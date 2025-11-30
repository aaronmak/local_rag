"""Text generation using Ollama."""

from typing import List, Optional

from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from .config import Settings


DEFAULT_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""


class Generator:
    """Generates responses using Ollama LLM."""

    def __init__(self, settings: Settings, prompt_template: Optional[str] = None):
        """Initialize the generator.

        Args:
            settings: Application settings
            prompt_template: Optional custom prompt template
        """
        self.settings = settings
        self.llm = Ollama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.temperature,
        )

        template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def generate(self, question: str, context_documents: List[Document]) -> str:
        """Generate an answer based on context documents.

        Args:
            question: User's question
            context_documents: Retrieved context documents

        Returns:
            Generated answer
        """
        # Combine context documents
        context = "\n\n".join([doc.page_content for doc in context_documents])

        # Format prompt
        formatted_prompt = self.prompt.format(context=context, question=question)

        # Generate response
        response = self.llm.invoke(formatted_prompt)
        return response

    def generate_stream(self, question: str, context_documents: List[Document]):
        """Generate an answer with streaming output.

        Args:
            question: User's question
            context_documents: Retrieved context documents

        Yields:
            Chunks of the generated answer
        """
        # Combine context documents
        context = "\n\n".join([doc.page_content for doc in context_documents])

        # Format prompt
        formatted_prompt = self.prompt.format(context=context, question=question)

        # Generate response with streaming
        for chunk in self.llm.stream(formatted_prompt):
            yield chunk
