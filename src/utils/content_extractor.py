"""Content extraction and understanding from PDFs using Google Gemini."""

import json
import logging
import time
from pathlib import Path
from typing import Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from src.core.models import PDFMetadata

logger = logging.getLogger(__name__)


class ContentChunk:
    """Represents a chunk of content from a PDF."""

    def __init__(
        self,
        chunk_id: str,
        content: str,
        content_type: str,
        page_number: int,
        topics: list[str],
        concepts: list[str],
        difficulty: str,
        has_images: bool,
        image_descriptions: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize content chunk."""
        self.chunk_id = chunk_id
        self.content = content
        self.content_type = content_type
        self.page_number = page_number
        self.topics = topics
        self.concepts = concepts
        self.difficulty = difficulty
        self.has_images = has_images
        self.image_descriptions = image_descriptions or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "content_type": self.content_type,
            "page_number": self.page_number,
            "topics": self.topics,
            "concepts": self.concepts,
            "difficulty": self.difficulty,
            "has_images": self.has_images,
            "image_descriptions": self.image_descriptions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContentChunk":
        """Create from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            content=data["content"],
            content_type=data["content_type"],
            page_number=data["page_number"],
            topics=data["topics"],
            concepts=data["concepts"],
            difficulty=data["difficulty"],
            has_images=data["has_images"],
            image_descriptions=data.get("image_descriptions", []),
            metadata=data.get("metadata", {}),
        )


class ContentExtractor:
    """Extract and understand content from PDFs using Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-pro",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        request_delay: float = 1.0,
    ):
        """
        Initialize the content extractor.

        Args:
            api_key: Google Gemini API key.
            model_name: Gemini model to use (default: gemini-2.5-pro - ideal for comprehensive PDF analysis).
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Initial delay between retries (exponential backoff).
            request_delay: Delay between requests to avoid rate limits.
        """
        import os

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            msg = "Gemini API key not provided. Set GEMINI_API_KEY environment variable."
            raise ValueError(msg)

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        logger.info(f"Initialized ContentExtractor with model: {model_name}")
        logger.info(f"Rate limiting: {request_delay}s delay, {max_retries} max retries")

    def extract_content(self, pdf_path: Path, pdf_metadata: PDFMetadata) -> list[ContentChunk]:
        """
        Extract and understand content from a PDF.

        Args:
            pdf_path: Path to the PDF file.
            pdf_metadata: Metadata about the PDF.

        Returns:
            List of ContentChunk objects with rich metadata.
        """
        # Convert to Path if string
        from pathlib import Path

        pdf_path = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path

        if not pdf_path.exists():
            msg = f"PDF file not found: {pdf_path}"
            raise FileNotFoundError(msg)

        logger.info(f"Extracting content from: {pdf_path}")
        logger.info(f"Processing {pdf_metadata.grade}/{pdf_metadata.subject}")

        # Upload PDF
        uploaded_file = self._upload_file_with_retry(pdf_path)
        logger.info(f"Uploaded PDF: {uploaded_file.name}")

        # Wait to avoid rate limits
        time.sleep(self.request_delay)

        # Create analysis prompt
        prompt = self._create_content_analysis_prompt(pdf_metadata)

        # Analyze content with retry logic
        response = self._generate_content_with_retry(uploaded_file, prompt)

        # Parse response into content chunks
        chunks = self._parse_content_response(response.text, pdf_metadata)

        logger.info(f"Extracted {len(chunks)} content chunks from {pdf_path.name}")
        return chunks

    def _create_content_analysis_prompt(self, pdf_metadata: PDFMetadata) -> str:
        """Create prompt for comprehensive content analysis."""
        return f"""Analyze this {pdf_metadata.subject} PDF for {pdf_metadata.grade} and extract educational content as JSON chunks.

For each meaningful section/topic, create a chunk with:
- chunk_id: unique ID (e.g., "p1_topic1")
- content: actual educational content (text, formulas, concepts)
- content_type: text|diagram|image|equation|table|mixed
- page_number: integer
- topics: list of topics (e.g., ["Addition", "Fractions"])
- concepts: key concepts (e.g., ["Place Value", "Regrouping"])
- difficulty: easy|medium|hard
- has_images: boolean
- image_descriptions: list of image descriptions if any
- metadata: {{learning_objectives: [], prerequisites: [], grade_appropriate: boolean, visual_type: string (if has_images)}}

REQUIREMENTS:
1. Output ONLY valid JSON array - no markdown, no ```json markers
2. Analyze ALL pages, text, images, diagrams, equations
3. Describe visual elements in detail
4. Create focused chunks (5-15 chunks total)
5. Each chunk covers one main topic/concept

EXAMPLE:
[
  {{"chunk_id": "p1_addition", "content": "Two-digit addition with carrying: 25+37=62", "content_type": "text", "page_number": 1, "topics": ["Addition"], "concepts": ["Carrying", "Place Value"], "difficulty": "medium", "has_images": false, "image_descriptions": [], "metadata": {{"learning_objectives": ["Master carrying"], "prerequisites": ["Single-digit addition"], "grade_appropriate": true}}}},
  {{"chunk_id": "p2_diagram", "content": "Base-10 blocks showing addition", "content_type": "diagram", "page_number": 2, "topics": ["Visual Math"], "concepts": ["Base-10"], "difficulty": "easy", "has_images": true, "image_descriptions": ["Blocks arranged for 25+37 with regrouping"], "metadata": {{"learning_objectives": ["Visualize addition"], "prerequisites": ["Know place value"], "grade_appropriate": true, "visual_type": "diagram"}}}}
]

Extract content:
"""

    def _upload_file_with_retry(self, pdf_path: Path) -> Any:
        """Upload file with retry logic."""
        for attempt in range(self.max_retries):
            try:
                return genai.upload_file(str(pdf_path))
            except google_exceptions.ResourceExhausted:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Rate limit during upload. Retrying in {wait_time}s... "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.exception("Max retries reached for file upload")
                    raise
            except Exception as e:
                logger.exception(f"Error uploading file: {e}")
                raise
        return None

    def _generate_content_with_retry(self, uploaded_file: Any, prompt: str) -> Any:
        """Generate content with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    [uploaded_file, prompt],
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,  # Lower temperature for factual extraction
                        max_output_tokens=16384,  # Sufficient for comprehensive PDF analysis
                    ),
                )

                # Check if response is valid
                if not response.parts:
                    logger.warning(
                        f"Empty response received. Finish reason: {response.candidates[0].finish_reason if response.candidates else 'unknown'}"
                    )
                    if attempt < self.max_retries - 1:
                        logger.info("Retrying with simplified prompt...")
                        time.sleep(2)
                        continue
                    msg = f"Failed to get valid response after {self.max_retries} attempts"
                    raise ValueError(msg)

                return response
            except google_exceptions.ResourceExhausted:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(
                        f"Rate limit hit. Retrying in {wait_time}s... "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(wait_time)
                else:
                    logger.exception("Max retries reached due to rate limiting")
                    raise
            except Exception as e:
                logger.exception(f"Error generating content: {e}")
                raise
        return None

    def _parse_content_response(
        self, response_text: str, pdf_metadata: PDFMetadata
    ) -> list[ContentChunk]:
        """Parse Gemini response into ContentChunk objects."""
        try:
            # Clean response
            cleaned_text = response_text.strip()

            # Remove markdown code blocks if present
            if cleaned_text.startswith("```"):
                first_newline = cleaned_text.find("\n")
                last_backticks = cleaned_text.rfind("```")
                if first_newline != -1 and last_backticks != -1:
                    cleaned_text = cleaned_text[first_newline + 1 : last_backticks].strip()

            # Parse JSON
            chunks_data = json.loads(cleaned_text)

            if not isinstance(chunks_data, list):
                msg = "Expected a JSON array of content chunks"
                raise ValueError(msg)

            chunks = []
            for idx, chunk_data in enumerate(chunks_data):
                try:
                    # Add PDF metadata
                    chunk_data["metadata"]["grade"] = pdf_metadata.grade
                    chunk_data["metadata"]["subject"] = pdf_metadata.subject
                    chunk_data["metadata"]["pdf_source"] = pdf_metadata.pdf_filename

                    chunk = ContentChunk.from_dict(chunk_data)
                    chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Failed to parse chunk {idx}: {e}")
                    continue

            if not chunks:
                msg = "No valid content chunks were parsed"
                raise ValueError(msg)

            return chunks

        except json.JSONDecodeError as e:
            logger.exception(f"Failed to parse JSON response: {e}")
            logger.exception(f"Response text: {response_text[:1000]}...")
            msg = f"Invalid JSON response from Gemini: {e}"
            raise ValueError(msg)
        except Exception as e:
            logger.exception(f"Error parsing response: {e}")
            raise
