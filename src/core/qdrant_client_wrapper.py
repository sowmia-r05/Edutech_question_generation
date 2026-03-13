"""Qdrant client wrapper for managing question collections."""

import hashlib
import logging
import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    PointStruct,
    VectorParams,
)

from src.core.models import Question

logger = logging.getLogger(__name__)


class QdrantManager:
    """Manager for Qdrant operations."""

    def __init__(
        self, url: str | None = None, api_key: str | None = None, embedding_dimension: int = 768
    ):
        """
        Initialize the Qdrant manager.

        Args:
            url: Qdrant URL. If None, reads from QDRANT_URL env var.
            api_key: Qdrant API key. If None, reads from QDRANT_API_KEY env var.
            embedding_dimension: Dimension of the embedding vectors.

        Raises:
            ValueError: If no URL is provided.
        """
        self.url = url or os.getenv("QDRANT_URL", "http://100.98.192.36:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.embedding_dimension = embedding_dimension

        # Initialize client
        if self.api_key:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = QdrantClient(url=self.url)

        logger.info(f"Initialized Qdrant client connecting to: {self.url}")
        self._verify_connection()

    def _verify_connection(self) -> None:
        """
        Verify connection to Qdrant.

        Raises:
            Exception: If connection fails.
        """
        try:
            collections = self.client.get_collections()
            logger.info(
                f"Successfully connected to Qdrant. "
                f"Found {len(collections.collections)} collections."
            )
        except Exception as e:
            logger.exception(f"Failed to connect to Qdrant at {self.url}: {e}")
            raise

    def ensure_collection_exists(self, collection_name: str) -> None:
        """
        Ensure a collection exists, creating it if necessary.

        Args:
            collection_name: Name of the collection.

        Raises:
            Exception: If collection creation fails.
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]

            if collection_name in existing_names:
                logger.info(f"Collection '{collection_name}' already exists")
                return

            # Create collection
            logger.info(
                f"Creating collection '{collection_name}' with dimension {self.embedding_dimension}"
            )

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension, distance=Distance.COSINE
                ),
            )

            logger.info(f"Successfully created collection '{collection_name}'")

        except Exception as e:
            logger.exception(f"Error ensuring collection exists: {e}")
            raise

    def upsert_questions(
        self, collection_name: str, questions: list[Question], embeddings: list[list[float]]
    ) -> None:
        """
        Upsert questions into a Qdrant collection.

        Args:
            collection_name: Name of the collection.
            questions: List of Question objects.
            embeddings: List of embedding vectors corresponding to questions.

        Raises:
            ValueError: If questions and embeddings lists have different lengths.
            Exception: If upsert operation fails.
        """
        if len(questions) != len(embeddings):
            msg = (
                f"Mismatch between questions ({len(questions)}) and embeddings ({len(embeddings)})"
            )
            raise ValueError(msg)

        if not questions:
            logger.warning("No questions to upsert")
            return

        # Ensure collection exists
        self.ensure_collection_exists(collection_name)

        # Prepare points
        points = []
        for question, embedding in zip(questions, embeddings, strict=False):
            point_id = self._generate_point_id(question)
            payload = question.to_dict()

            point = PointStruct(id=point_id, vector=embedding, payload=payload)
            points.append(point)

        # Upsert points
        try:
            logger.info(f"Upserting {len(points)} points to collection '{collection_name}'")

            self.client.upsert(collection_name=collection_name, points=points)

            logger.info(f"Successfully upserted {len(points)} points")

        except Exception as e:
            logger.exception(f"Error upserting points: {e}")
            raise

    def upsert_question(
        self, collection_name: str, question: Question, embedding: list[float]
    ) -> None:
        """
        Upsert a single question into a Qdrant collection.

        Args:
            collection_name: Name of the collection.
            question: Question object.
            embedding: Embedding vector for the question.

        Raises:
            Exception: If upsert operation fails.
        """
        self.upsert_questions(collection_name, [question], [embedding])

    def search_questions(
        self,
        collection_name: str,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: Filter | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar questions in a collection.

        Args:
            collection_name: Name of the collection.
            query_embedding: Query embedding vector.
            limit: Maximum number of results to return.
            score_threshold: Minimum score threshold for results.
            filter_conditions: Optional filter conditions for the search.

        Returns:
            List of search results with scores and payloads.

        Raises:
            Exception: If search operation fails.
        """
        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions,
            )

            return [
                {"id": point.id, "score": point.score, "payload": point.payload}
                for point in results.points
            ]

        except Exception as e:
            # Handle case where collection doesn't exist yet (fresh setup)
            if "doesn't exist" in str(e) or "Not found" in str(e):
                logger.warning(
                    f"Collection '{collection_name}' does not exist yet. Returning empty results."
                )
                return []
            # For other errors, log and raise
            logger.exception(f"Error searching collection '{collection_name}': {e}")
            raise

    def get_collection_info(self, collection_name: str) -> dict[str, Any]:
        """
        Get information about a collection.

        Args:
            collection_name: Name of the collection.

        Returns:
            Dictionary containing collection information.

        Raises:
            Exception: If operation fails.
        """
        try:
            collection_info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }
        except Exception as e:
            logger.exception(f"Error getting collection info: {e}")
            raise

    def list_collections(self) -> list[str]:
        """
        List all collection names.

        Returns:
            List of collection names.

        Raises:
            Exception: If operation fails.
        """
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.exception(f"Error listing collections: {e}")
            raise

    @staticmethod
    def _generate_point_id(question: Question) -> str:
        """
        Generate a deterministic ID for a question.

        Uses a hash of pdf_source + question_number to ensure consistent IDs.

        Args:
            question: Question object.

        Returns:
            Deterministic UUID string.
        """
        # Create a deterministic string from question metadata
        id_string = f"{question.pdf_source}_{question.question_number}"

        # Generate a hash
        hash_object = hashlib.md5(id_string.encode())
        return hash_object.hexdigest()

        # Convert to UUID format (though it's deterministic)

    def upsert_content_chunks(
        self,
        collection_name: str,
        chunks: list[Any],  # ContentChunk objects
        embeddings: list[list[float]],
    ) -> None:
        """
        Upsert content chunks into a Qdrant collection.

        Args:
            collection_name: Name of the collection.
            chunks: List of ContentChunk objects.
            embeddings: List of embedding vectors corresponding to chunks.

        Raises:
            ValueError: If chunks and embeddings lists have different lengths.
            Exception: If upsert operation fails.
        """
        if len(chunks) != len(embeddings):
            msg = f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})"
            raise ValueError(msg)

        if not chunks:
            logger.warning("No content chunks to upsert")
            return

        # Ensure collection exists
        self.ensure_collection_exists(collection_name)

        # Prepare points
        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            point_id = self._generate_chunk_id(chunk)
            payload = chunk.to_dict()

            point = PointStruct(id=point_id, vector=embedding, payload=payload)
            points.append(point)

        # Upsert points
        try:
            logger.info(f"Upserting {len(points)} content chunks to collection '{collection_name}'")

            self.client.upsert(collection_name=collection_name, points=points)

            logger.info(f"Successfully upserted {len(points)} content chunks")

        except Exception as e:
            logger.exception(f"Error upserting content chunks: {e}")
            raise

    @staticmethod
    def _generate_chunk_id(chunk: Any) -> str:
        """
        Generate a deterministic ID for a content chunk.

        Args:
            chunk: ContentChunk object.

        Returns:
            Deterministic hash string.
        """
        # Create a deterministic string from chunk metadata
        id_string = f"{chunk.metadata.get('pdf_source', '')}_{chunk.chunk_id}"

        # Generate a hash
        hash_object = hashlib.md5(id_string.encode())
        return hash_object.hexdigest()

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete.

        Raises:
            Exception: If deletion fails.
        """
        try:
            logger.warning(f"Deleting collection '{collection_name}'")
            self.client.delete_collection(collection_name)
            logger.info(f"Successfully deleted collection '{collection_name}'")
        except Exception as e:
            logger.exception(f"Error deleting collection: {e}")
            raise
