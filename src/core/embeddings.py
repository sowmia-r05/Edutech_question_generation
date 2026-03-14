"""Embedding generation for questions using Google Gemini embeddings."""

import logging
import os
from typing import Any

from google import genai
from google.genai import types

from src.core.models import Question

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for generating embeddings using Google Gemini."""

    # Dimension of text-embedding-004 model
    EMBEDDING_DIMENSION = 3072

    def __init__(self, api_key: str | None = None, model_name: str = "models/gemini-embedding-001"):

        """
        Initialize the embedding client.

        Args:
            api_key: Google Gemini API key. If None, reads from GEMINI_API_KEY env var.
            model_name: Name of the embedding model to use.

        Raises:
            ValueError: If no API key is provided.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            msg = (
                "Gemini API key not provided. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
            raise ValueError(msg)

        self.client = genai.Client(api_key=self.api_key)

        self.model_name = model_name
        logger.info(f"Initialized embedding client with model: {model_name}")

    def _embed_text(self, text: str, task_type: str) -> list[float]:
        """
        Core embedding method using the new google-genai SDK.

        Args:
            text: Text to embed.
            task_type: Task type for embedding (retrieval_document or retrieval_query).

        Returns:
            Embedding vector as a list of floats.
        """
        result = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return result.embeddings[0].values

    def embed_question(self, question: Question) -> list[float]:
        """
        Generate an embedding for a question.

        Args:
            question: Question object to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        text_to_embed = self._prepare_text(question)

        try:
            embedding = self._embed_text(text_to_embed, "retrieval_document")
            logger.debug(
                f"Generated embedding for question {question.question_number} "
                f"(dim: {len(embedding)})"
            )
            return embedding

        except Exception as e:
            logger.exception(
                f"Error generating embedding for question {question.question_number}: {e}"
            )
            raise

    def embed_questions_batch(
        self, questions: list[Question], batch_size: int = 100
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple questions.

        Args:
            questions: List of Question objects to embed.
            batch_size: Number of questions to process at once.

        Returns:
            List of embedding vectors.
        """
        embeddings = []

        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            logger.info(f"Embedding batch {i // batch_size + 1} ({len(batch)} questions)")

            try:
                for question in batch:
                    text = self._prepare_text(question)
                    embedding = self._embed_text(text, "retrieval_document")
                    embeddings.append(embedding)

            except Exception as e:
                logger.exception(f"Error in batch embedding: {e}")
                raise

        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings

    def embed_query(self, query_text: str) -> list[float]:
        """
        Generate an embedding for a search query.

        Args:
            query_text: Query text to embed.

        Returns:
            Embedding vector as a list of floats.
        """
        try:
            embedding = self._embed_text(query_text, "retrieval_query")
            return embedding

        except Exception as e:
            logger.exception(f"Error generating query embedding: {e}")
            raise

    def embed_content_chunks(
        self,
        chunks: list[Any],
        batch_size: int = 100,
    ) -> list[list[float]]:
        """
        Generate embeddings for content chunks.

        Args:
            chunks: List of ContentChunk objects to embed.
            batch_size: Number of chunks to process at once.

        Returns:
            List of embedding vectors.
        """
        embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            logger.info(f"Embedding content batch {i // batch_size + 1} ({len(batch)} chunks)")

            try:
                for chunk in batch:
                    text = self._prepare_chunk_text(chunk)
                    embedding = self._embed_text(text, "retrieval_document")
                    embeddings.append(embedding)

            except Exception as e:
                logger.exception(f"Error in batch chunk embedding: {e}")
                raise

        logger.info(f"Successfully generated {len(embeddings)} content chunk embeddings")
        return embeddings

    @staticmethod
    def _prepare_text(question: Question) -> str:
        """Prepare the text to be embedded from a question."""
        options_text = " | ".join(question.options)
        return (
            f"Subject: {question.subject} ({question.sub_subject}). "
            f"Question: {question.question_text} "
            f"Options: {options_text}"
        )

    @staticmethod
    def _prepare_chunk_text(chunk: Any) -> str:
        """Prepare text from a content chunk for embedding."""
        topics_text = ", ".join(chunk.topics) if chunk.topics else ""
        concepts_text = ", ".join(chunk.concepts) if chunk.concepts else ""
        image_desc_text = " ".join(chunk.image_descriptions) if chunk.image_descriptions else ""

        grade = chunk.metadata.get("grade", "")
        subject = chunk.metadata.get("subject", "")

        parts = [
            f"Grade: {grade}",
            f"Subject: {subject}",
            f"Topics: {topics_text}" if topics_text else "",
            f"Concepts: {concepts_text}" if concepts_text else "",
            f"Content: {chunk.content}",
            f"Images: {image_desc_text}" if image_desc_text else "",
            f"Difficulty: {chunk.difficulty}",
        ]

        return " | ".join(part for part in parts if part and not part.endswith(": "))

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.EMBEDDING_DIMENSION