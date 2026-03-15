"""Qdrant client wrapper for managing question collections."""

import hashlib
import logging
import os
import time
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

# ── Timeout / retry constants ─────────────────────────────────────────────────
# Increase these if your Qdrant server is on a slow/remote network.
DEFAULT_TIMEOUT: int = 60          # seconds per single HTTP request
UPSERT_BATCH_SIZE: int = 5         # points per upsert call (keep small for remote servers)
MAX_RETRIES: int = 3               # retry attempts on timeout/connection errors
RETRY_BASE_DELAY: float = 2.0      # seconds — doubles each attempt (exponential back-off)


class QdrantManager:
    """Manager for Qdrant operations."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        embedding_dimension: int = 768,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the Qdrant manager.

        Args:
            url: Qdrant URL. If None, reads from QDRANT_URL env var.
            api_key: Qdrant API key. If None, reads from QDRANT_API_KEY env var.
            embedding_dimension: Dimension of the embedding vectors.
            timeout: HTTP read/write timeout in seconds (default: 60).
        """
        # Priority: explicit arg → QDRANT_URL env var → localhost fallback
        self.url = url or os.getenv("QDRANT_URL") or "http://localhost:6333"
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.embedding_dimension = embedding_dimension
        self.timeout = timeout

        # ── Build client ─────────────────────────────────────────────────────
        # QdrantClient only accepts a plain number for timeout (not httpx.Timeout).
        # Pass it directly as an int.
        client_kwargs: dict[str, Any] = {"url": self.url, "timeout": self.timeout}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key

        self.client = QdrantClient(**client_kwargs)

        logger.info(
            f"Qdrant client → {self.url}  "
            f"(connect_timeout=10s, read_timeout={self.timeout}s, "
            f"upsert_batch={UPSERT_BATCH_SIZE})"
        )
        self._verify_connection()

    # ── Connection ────────────────────────────────────────────────────────────

    def _verify_connection(self) -> None:
        """Verify connection to Qdrant with a lightweight collections list call."""
        try:
            collections = self.client.get_collections()
            logger.info(
                f"✅ Connected to Qdrant at {self.url} — "
                f"{len(collections.collections)} collections found."
            )
        except Exception as e:
            logger.error(
                f"\n{'='*65}\n"
                f"  ❌ QDRANT CONNECTION FAILED\n"
                f"  URL tried : {self.url}\n"
                f"  Error     : {e}\n"
                f"\n"
                f"  Quick fixes:\n"
                f"  1. Is Docker running?\n"
                f"     docker ps   ← should show qdrant container\n"
                f"     docker start qdrant   ← restart if stopped\n"
                f"\n"
                f"  2. Wrong IP? Set correct URL in .env:\n"
                f"     QDRANT_URL=http://localhost:6333      (same machine)\n"
                f"     QDRANT_URL=http://<tailscale-ip>:6333 (remote)\n"
                f"     Run: tailscale ip   to get current Tailscale IP\n"
                f"\n"
                f"  3. Test manually:\n"
                f"     curl {self.url}/collections\n"
                f"{'='*65}"
            )
            raise

    # ── Collection management ─────────────────────────────────────────────────

    def ensure_collection_exists(self, collection_name: str) -> None:
        """
        Ensure a collection exists, creating it if necessary.

        Args:
            collection_name: Name of the collection.
        """
        try:
            collections = self.client.get_collections()
            existing_names = [col.name for col in collections.collections]

            if collection_name in existing_names:
                logger.info(f"Collection '{collection_name}' already exists")
                return

            logger.info(
                f"Creating collection '{collection_name}' "
                f"(dim={self.embedding_dimension})"
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

    def list_collections(self) -> list[str]:
        """Return names of all existing collections."""
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.exception(f"Error listing collections: {e}")
            raise

    # ── Core upsert helper (batched + retried) ────────────────────────────────

    def _upsert_points_with_retry(
        self, collection_name: str, points: list[PointStruct]
    ) -> None:
        """
        Upsert a list of PointStructs in small batches with exponential-back-off
        retry on timeout / connection errors.

        Why batching?
          A single upsert call carrying 10+ large payloads can exceed the server's
          read timeout on a remote/slow network. Sending 5 points at a time keeps
          each individual HTTP round-trip short.

        Why retry?
          Transient network blips should not abort a long generation run.
        """
        for batch_start in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[batch_start: batch_start + UPSERT_BATCH_SIZE]
            batch_label = (
                f"points {batch_start + 1}–{batch_start + len(batch)}"
                f"/{len(points)}"
            )

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    self.client.upsert(
                        collection_name=collection_name, points=batch
                    )
                    logger.debug(f"Upserted {batch_label} (attempt {attempt})")
                    break  # success — move to next batch

                except Exception as e:
                    is_last = attempt == MAX_RETRIES
                    wait = RETRY_BASE_DELAY * (2 ** (attempt - 1))  # 2s, 4s, 8s
                    if is_last:
                        logger.error(
                            f"Upsert failed for {batch_label} after "
                            f"{MAX_RETRIES} attempts: {e}"
                        )
                        raise
                    logger.warning(
                        f"Upsert timeout for {batch_label} "
                        f"(attempt {attempt}/{MAX_RETRIES}) — "
                        f"retrying in {wait:.0f}s…  error: {e}"
                    )
                    time.sleep(wait)

    # ── Questions ─────────────────────────────────────────────────────────────

    def upsert_questions(
        self,
        collection_name: str,
        questions: list[Question],
        embeddings: list[list[float]],
    ) -> None:
        """
        Upsert questions into a Qdrant collection (batched + retried).

        Args:
            collection_name: Target collection name.
            questions: List of Question objects.
            embeddings: Corresponding embedding vectors.
        """
        if len(questions) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(questions)} questions vs {len(embeddings)} embeddings"
            )
        if not questions:
            logger.warning("No questions to upsert")
            return

        self.ensure_collection_exists(collection_name)

        points = [
            PointStruct(
                id=self._generate_point_id(q),
                vector=emb,
                payload=q.to_dict(),
            )
            for q, emb in zip(questions, embeddings, strict=False)
        ]

        logger.info(
            f"Upserting {len(points)} question(s) to '{collection_name}' "
            f"in batches of {UPSERT_BATCH_SIZE}…"
        )
        self._upsert_points_with_retry(collection_name, points)
        logger.info(f"Successfully upserted {len(points)} question(s)")

    def upsert_question(
        self,
        collection_name: str,
        question: Question,
        embedding: list[float],
    ) -> None:
        """Upsert a single question (convenience wrapper)."""
        self.upsert_questions(collection_name, [question], [embedding])

    # ── Content chunks ────────────────────────────────────────────────────────

    def upsert_content_chunks(
        self,
        collection_name: str,
        chunks: list[Any],
        embeddings: list[list[float]],
    ) -> None:
        """
        Upsert content chunks into a Qdrant collection (batched + retried).

        Args:
            collection_name: Target collection name.
            chunks: List of ContentChunk objects.
            embeddings: Corresponding embedding vectors.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )
        if not chunks:
            logger.warning("No content chunks to upsert")
            return

        self.ensure_collection_exists(collection_name)

        points = [
            PointStruct(
                id=self._generate_chunk_id(chunk),
                vector=emb,
                payload=chunk.to_dict(),
            )
            for chunk, emb in zip(chunks, embeddings, strict=False)
        ]

        logger.info(
            f"Upserting {len(points)} chunk(s) to '{collection_name}' "
            f"in batches of {UPSERT_BATCH_SIZE}…"
        )
        self._upsert_points_with_retry(collection_name, points)
        logger.info(f"Successfully upserted {len(points)} chunk(s)")

    # ── Search ────────────────────────────────────────────────────────────────

    def search_questions(
        self,
        collection_name: str,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: Filter | None = None,
    ) -> list[Any]:
        """
        Search for similar questions in a collection.

        Args:
            collection_name: Name of the collection.
            query_embedding: Query embedding vector.
            limit: Maximum number of results to return.
            score_threshold: Minimum cosine similarity score.
            filter_conditions: Optional Qdrant filter.

        Returns:
            List of ScoredPoint results.
        """
        try:
            # qdrant-client >= 1.7: .search() was removed, use .query_points()
            # qdrant-client <  1.7: .search() still works
            # We try the new API first, fall back to old if needed.
            from qdrant_client.models import NamedVector

            if hasattr(self.client, "query_points"):
                # New API (qdrant-client >= 1.7)
                from qdrant_client.models import QueryRequest
                query_kwargs: dict[str, Any] = {
                    "collection_name": collection_name,
                    "query": query_embedding,
                    "limit": limit,
                }
                if score_threshold is not None:
                    query_kwargs["score_threshold"] = score_threshold
                if filter_conditions is not None:
                    query_kwargs["query_filter"] = filter_conditions

                result = self.client.query_points(**query_kwargs)
                # query_points returns a QueryResponse with .points attribute
                return result.points if hasattr(result, "points") else result

            else:
                # Old API (qdrant-client < 1.7)
                search_kwargs: dict[str, Any] = {
                    "collection_name": collection_name,
                    "query_vector": query_embedding,
                    "limit": limit,
                }
                if score_threshold is not None:
                    search_kwargs["score_threshold"] = score_threshold
                if filter_conditions is not None:
                    search_kwargs["query_filter"] = filter_conditions

                return self.client.search(**search_kwargs)

        except Exception as e:
            logger.exception(f"Error searching collection '{collection_name}': {e}")
            raise

    def search_content(
        self,
        collection_name: str,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float | None = None,
        filter_conditions: Filter | None = None,
    ) -> list[Any]:
        """Search content chunks — same interface as search_questions."""
        return self.search_questions(
            collection_name, query_embedding, limit, score_threshold, filter_conditions
        )

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_collection(self, collection_name: str) -> None:
        """Delete a collection entirely."""
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
        except Exception as e:
            logger.exception(f"Error deleting collection '{collection_name}': {e}")
            raise

    # ── ID helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _generate_point_id(question: Question) -> str:
        """Generate a deterministic MD5-based ID for a Question."""
        id_string = f"{question.pdf_source}_{question.question_number}"
        return hashlib.md5(id_string.encode()).hexdigest()

    @staticmethod
    def _generate_chunk_id(chunk: Any) -> str:
        """Generate a deterministic MD5-based ID for a ContentChunk."""
        chunk_id = getattr(chunk, "chunk_id", None) or str(id(chunk))
        return hashlib.md5(chunk_id.encode()).hexdigest()