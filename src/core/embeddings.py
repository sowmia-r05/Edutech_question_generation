"""Embedding generation for questions using Google Gemini embeddings."""

import logging
import os
from typing import Any

import google.generativeai as genai

from src.core.models import Question

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for generating embeddings using Google Gemini."""

    # Dimension of text-embedding-004 model
    EMBEDDING_DIMENSION = 768

    def __init__(self, api_key: str | None = None, model_name: str = "models/text-embedding-004"):
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

        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        logger.info(f"Initialized embedding client with model: {model_name}")

    def embed_question(self, question: Question) -> list[float]:
        """
        Generate an embedding for a question.

        The embedding is generated from the question text and options combined.

        Args:
            question: Question object to embed.

        Returns:
            Embedding vector as a list of floats.

        Raises:
            Exception: If embedding generation fails.
        """
        # Combine question text and options for richer embeddings
        text_to_embed = self._prepare_text(question)

        try:
            result = genai.embed_content(
                model=self.model_name, content=text_to_embed, task_type="retrieval_document"
            )

            embedding = result["embedding"]
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

        Raises:
            Exception: If embedding generation fails.
        """
        embeddings = []

        for i in range(0, len(questions), batch_size):
            batch = questions[i : i + batch_size]
            logger.info(f"Embedding batch {i // batch_size + 1} ({len(batch)} questions)")

            batch_texts = [self._prepare_text(q) for q in batch]

            try:
                # Gemini API doesn't support batch embedding in a single call,
                # so we process them individually
                for text in batch_texts:
                    result = genai.embed_content(
                        model=self.model_name, content=text, task_type="retrieval_document"
                    )
                    embeddings.append(result["embedding"])

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

        Raises:
            Exception: If embedding generation fails.
        """
        try:
            result = genai.embed_content(
                model=self.model_name, content=query_text, task_type="retrieval_query"
            )

            return result["embedding"]

        except Exception as e:
            logger.exception(f"Error generating query embedding: {e}")
            raise

    @staticmethod
    def _prepare_text(question: Question) -> str:
        """
        Prepare the text to be embedded from a question.

        Combines question text, options, and subject information for richer embeddings.

        Args:
            question: Question object.

        Returns:
            Formatted text string for embedding.
        """
        options_text = " | ".join(question.options)
        return (
            f"Subject: {question.subject} ({question.sub_subject}). "
            f"Question: {question.question_text} "
            f"Options: {options_text}"
        )

    def embed_content_chunks(
        self,
        chunks: list[Any],  # ContentChunk objects
        batch_size: int = 100,
    ) -> list[list[float]]:
        """
        Generate embeddings for content chunks.

        Args:
            chunks: List of ContentChunk objects to embed.
            batch_size: Number of chunks to process at once.

        Returns:
            List of embedding vectors.

        Raises:
            Exception: If embedding generation fails.
        """
        embeddings = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            logger.info(f"Embedding content batch {i // batch_size + 1} ({len(batch)} chunks)")

            batch_texts = [self._prepare_chunk_text(chunk) for chunk in batch]

            try:
                for text in batch_texts:
                    result = genai.embed_content(
                        model=self.model_name, content=text, task_type="retrieval_document"
                    )
                    embeddings.append(result["embedding"])

            except Exception as e:
                logger.exception(f"Error in batch chunk embedding: {e}")
                raise

        logger.info(f"Successfully generated {len(embeddings)} content chunk embeddings")
        return embeddings

    @staticmethod
    def _prepare_chunk_text(chunk: Any) -> str:
        """
        Prepare text from a content chunk for embedding.

        Args:
            chunk: ContentChunk object.

        Returns:
            Formatted text string for embedding.
        """
        # Combine various chunk attributes for rich semantic representation
        topics_text = ", ".join(chunk.topics) if chunk.topics else ""
        concepts_text = ", ".join(chunk.concepts) if chunk.concepts else ""
        image_desc_text = " ".join(chunk.image_descriptions) if chunk.image_descriptions else ""

        grade = chunk.metadata.get("grade", "")
        subject = chunk.metadata.get("subject", "")

        # Create comprehensive text for embedding
        parts = [
            f"Grade: {grade}",
            f"Subject: {subject}",
            f"Topics: {topics_text}" if topics_text else "",
            f"Concepts: {concepts_text}" if concepts_text else "",
            f"Content: {chunk.content}",
            f"Images: {image_desc_text}" if image_desc_text else "",
            f"Difficulty: {chunk.difficulty}",
        ]

        # Filter out empty parts and join
        return " | ".join(part for part in parts if part and not part.endswith(": "))

    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            Embedding dimension size.
        """
        return self.EMBEDDING_DIMENSION
