"""Enhanced question generator with anti-hallucination and duplicate prevention."""

import json
import logging
import time
from datetime import datetime
from typing import Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from src.core.embeddings import EmbeddingClient
from src.core.models import Question
from src.core.qdrant_client_wrapper import QdrantManager

logger = logging.getLogger(__name__)


class QuestionGeneratorV2:
    """
    Generate questions with safeguards:
    - No hallucinations (strict content adherence)
    - Duplicate prevention
    - Content exhaustion detection
    - Question storage for tracking
    """

    SIMILARITY_THRESHOLD = 0.85  # For duplicate detection

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        qdrant_manager: QdrantManager,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize the enhanced question generator."""
        import os

        self.embedding_client = embedding_client
        self.qdrant_manager = qdrant_manager

        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            msg = "Gemini API key not provided."
            raise ValueError(msg)

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"Initialized QuestionGeneratorV2 with model: {model_name}")
        logger.info("Safeguards: Anti-hallucination, Duplicate detection, Content tracking")

    def _generate_in_batches(
        self,
        grade: str,
        subject: str | None,
        topics: list[str] | None,
        difficulty: str | None,
        num_questions: int,
        content_chunks_limit: int,
        batch_size: int,
        allow_similar: bool,
    ) -> dict[str, Any]:
        """Generate questions in batches to avoid token limits."""
        all_questions: list[Question] = []
        total_duplicates = 0
        batches_needed = (num_questions + batch_size - 1) // batch_size

        for batch_num in range(batches_needed):
            batch_start = batch_num * batch_size
            batch_count = min(batch_size, num_questions - batch_start)

            logger.info(
                f"\nBatch {batch_num + 1}/{batches_needed}: Generating {batch_count} questions..."
            )

            # Generate this batch (will call the single batch logic below)
            result = self._generate_single_batch(
                grade=grade,
                subject=subject,
                topics=topics,
                difficulty=difficulty,
                num_questions=batch_count,
                content_chunks_limit=content_chunks_limit,
                allow_similar=allow_similar,
                existing_questions=all_questions,  # Pass previously generated
            )

            if result["questions"]:
                # Renumber questions sequentially
                for i, q in enumerate(result["questions"], start=len(all_questions) + 1):
                    q.question_number = i
                all_questions.extend(result["questions"])
                total_duplicates += result.get("duplicates_avoided", 0)

            # Handle different result statuses
            if result["status"] == "exhausted":
                logger.warning(f"Content exhausted after {len(all_questions)} questions")
                break
            if result["status"] == "error":
                logger.warning(
                    f"Batch {batch_num + 1} failed with error: {result['message']}. "
                    "Continuing to next batch..."
                )
                # Continue to next batch - don't break on JSON errors
                continue

        status = "success" if len(all_questions) >= num_questions else "partial"
        message = f"Generated {len(all_questions)} questions in {batch_num + 1} batch(es)"

        return {
            "questions": all_questions,
            "status": status,
            "message": message,
            "duplicates_avoided": total_duplicates,
            "content_exhausted": False,
            "new_questions_count": len(all_questions),
            "existing_questions_used": 0,
        }

    def _generate_single_batch(
        self,
        grade: str,
        subject: str | None,
        topics: list[str] | None,
        difficulty: str | None,
        num_questions: int,
        content_chunks_limit: int,
        allow_similar: bool,
        existing_questions: list[Question],
    ) -> dict[str, Any]:
        """Generate a single batch of questions."""
        # This contains the original logic from generate_questions
        content_collection = f"{grade}_content"
        chunks = self._retrieve_content_chunks(
            collection_name=content_collection,
            subject=subject,
            topics=topics,
            difficulty=difficulty,
            limit=content_chunks_limit,
        )

        if not chunks:
            return {
                "questions": [],
                "status": "exhausted",
                "message": "No content found",
                "duplicates_avoided": 0,
                "content_exhausted": True,
            }

        # Check exhaustion
        exhaustion_check = self._check_content_exhaustion(
            chunks=chunks, existing_questions=existing_questions, num_requested=num_questions
        )

        if exhaustion_check["exhausted"]:
            return {
                "questions": [],
                "status": "exhausted",
                "message": exhaustion_check["message"],
                "duplicates_avoided": 0,
                "content_exhausted": True,
            }

        # Generate questions
        prompt = self._create_strict_generation_prompt(
            chunks=chunks,
            grade=grade,
            subject=subject,
            num_questions=num_questions,
            difficulty=difficulty,
            existing_questions=existing_questions,
        )

        try:
            response = self._generate_with_retry(prompt)
            new_questions = self._parse_and_validate_questions(
                response.text, grade, chunks, existing_questions
            )

            if not new_questions:
                # Empty result could be JSON error or actual failure
                # Don't treat as exhaustion - return error status to allow retry
                return {
                    "questions": [],
                    "status": "error",
                    "message": "Failed to parse or validate questions (possible JSON error)",
                    "duplicates_avoided": 0,
                    "content_exhausted": False,
                }

            # Filter duplicates
            unique_questions, duplicates_count = self._filter_duplicates(
                new_questions, existing_questions, allow_similar
            )

            # Store questions
            if unique_questions:
                questions_collection = f"{grade}_questions"
                self._store_questions_to_qdrant(
                    questions=unique_questions,
                    collection_name=questions_collection,
                    chunks_used=chunks,
                )

            return {
                "questions": unique_questions,
                "status": "success",
                "message": f"Generated {len(unique_questions)} questions",
                "duplicates_avoided": duplicates_count,
                "content_exhausted": False,
            }

        except Exception as e:
            logger.exception(f"Batch generation error: {e}")
            return {
                "questions": [],
                "status": "error",
                "message": str(e),
                "duplicates_avoided": 0,
                "content_exhausted": False,
            }

    def generate_questions(
        self,
        grade: str,
        subject: str | None = None,
        topics: list[str] | None = None,
        difficulty: str | None = None,
        num_questions: int = 10,
        content_chunks_limit: int = 10,
        allow_similar: bool = False,
    ) -> dict[str, Any]:
        """
        Generate questions with comprehensive safeguards.

        For large requests (>15 questions), automatically splits into batches.

        Returns:
            Dictionary with:
            - questions: List of Question objects
            - status: "success" | "partial" | "exhausted"
            - message: Human-readable status message
            - duplicates_avoided: Count of similar questions skipped
            - content_exhausted: Boolean indicating if content is depleted
        """
        logger.info(f"Generating {num_questions} questions for {grade}")
        logger.info(f"Filters - Subject: {subject}, Topics: {topics}, Difficulty: {difficulty}")

        # For large requests, generate in batches to avoid token limits
        BATCH_SIZE = 15
        if num_questions > BATCH_SIZE:
            logger.info(f"Large request detected. Generating in batches of {BATCH_SIZE}...")
            return self._generate_in_batches(
                grade=grade,
                subject=subject,
                topics=topics,
                difficulty=difficulty,
                num_questions=num_questions,
                content_chunks_limit=content_chunks_limit,
                batch_size=BATCH_SIZE,
                allow_similar=allow_similar,
            )

        # Step 1: Retrieve content chunks
        content_collection = f"{grade}_content"
        chunks = self._retrieve_content_chunks(
            collection_name=content_collection,
            subject=subject,
            topics=topics,
            difficulty=difficulty,
            limit=content_chunks_limit,
        )

        if not chunks:
            return {
                "questions": [],
                "status": "exhausted",
                "message": f"No content found for {grade} with given filters. Please ingest more PDFs.",
                "duplicates_avoided": 0,
                "content_exhausted": True,
            }

        logger.info(f"Retrieved {len(chunks)} content chunks")

        # Step 2: Check existing questions to avoid duplicates
        questions_collection = f"{grade}_questions"
        existing_questions = self._get_existing_questions(questions_collection, subject, topics)
        logger.info(f"Found {len(existing_questions)} existing questions")

        # Step 3: Check for content exhaustion
        exhaustion_check = self._check_content_exhaustion(
            chunks=chunks, existing_questions=existing_questions, num_requested=num_questions
        )

        if exhaustion_check["exhausted"]:
            return {
                "questions": existing_questions[:num_questions] if existing_questions else [],
                "status": "exhausted",
                "message": exhaustion_check["message"],
                "duplicates_avoided": 0,
                "content_exhausted": True,
                "suggestion": "Ingest new PDFs or expand search criteria",
            }

        # Step 4: Generate questions with anti-hallucination prompt
        prompt = self._create_strict_generation_prompt(
            chunks=chunks,
            grade=grade,
            subject=subject,
            num_questions=num_questions,
            difficulty=difficulty,
            existing_questions=existing_questions,
        )

        try:
            response = self._generate_with_retry(prompt)
            new_questions = self._parse_and_validate_questions(
                response.text, grade, chunks, existing_questions
            )

            if not new_questions:
                return {
                    "questions": existing_questions[:num_questions] if existing_questions else [],
                    "status": "exhausted",
                    "message": "Failed to generate new unique questions from available content. Content may be exhausted.",
                    "duplicates_avoided": 0,
                    "content_exhausted": True,
                }

            # Step 5: Filter duplicates
            unique_questions, duplicates_count = self._filter_duplicates(
                new_questions, existing_questions, allow_similar
            )

            logger.info(
                f"Generated {len(unique_questions)} unique questions ({duplicates_count} duplicates filtered)"
            )

            # Step 6: Store questions in Qdrant
            if unique_questions:
                self._store_questions_to_qdrant(
                    questions=unique_questions,
                    collection_name=questions_collection,
                    chunks_used=chunks,
                )

            # Combine with existing if needed
            all_questions = unique_questions + existing_questions
            final_questions = all_questions[:num_questions]

            status = "success" if len(unique_questions) >= num_questions else "partial"
            message = f"Generated {len(unique_questions)} new questions"
            if duplicates_count > 0:
                message += f" ({duplicates_count} duplicates avoided)"

            return {
                "questions": final_questions,
                "status": status,
                "message": message,
                "duplicates_avoided": duplicates_count,
                "content_exhausted": False,
                "new_questions_count": len(unique_questions),
                "existing_questions_used": min(
                    num_questions - len(unique_questions), len(existing_questions)
                ),
            }

        except Exception as e:
            logger.exception(f"Error generating questions: {e}")
            return {
                "questions": existing_questions[:num_questions] if existing_questions else [],
                "status": "error",
                "message": f"Error: {e!s}",
                "duplicates_avoided": 0,
                "content_exhausted": False,
            }

    def _retrieve_content_chunks(
        self,
        collection_name: str,
        subject: str | None,
        topics: list[str] | None,
        difficulty: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant unused or lightly-used content chunks."""
        try:
            query_parts = []
            if subject:
                query_parts.append(f"Subject: {subject}")
            if topics:
                query_parts.append(f"Topics: {', '.join(topics)}")
            if difficulty:
                query_parts.append(f"Difficulty: {difficulty}")

            query = " | ".join(query_parts) if query_parts else "educational content"

            query_embedding = self.embedding_client.embed_query(query)

            results = self.qdrant_manager.search_questions(
                collection_name=collection_name, query_embedding=query_embedding, limit=limit
            )

            return [result["payload"] for result in results]

        except Exception as e:
            logger.exception(f"Error retrieving content: {e}")
            return []

    def _get_existing_questions(
        self, collection_name: str, subject: str | None, topics: list[str] | None
    ) -> list[Question]:
        """Retrieve existing questions to check for duplicates."""
        try:
            # Try to get existing questions collection
            query = f"Subject: {subject}" if subject else "questions"
            query_embedding = self.embedding_client.embed_query(query)

            results = self.qdrant_manager.search_questions(
                collection_name=collection_name,
                query_embedding=query_embedding,
                limit=100,  # Get many to check for duplicates
            )

            questions = []
            for result in results:
                try:
                    q_data = result["payload"]
                    if q_data.get("type") == "question":  # Only get questions, not content
                        questions.append(Question.from_dict(q_data))
                except Exception as e:
                    logger.debug(f"Skipping invalid question: {e}")
                    continue

            return questions

        except Exception:
            # Collection doesn't exist yet
            return []

    def _check_content_exhaustion(
        self, chunks: list[dict[str, Any]], existing_questions: list[Question], num_requested: int
    ) -> dict[str, Any]:
        """Check if content has been exhausted."""
        # Estimate: each chunk can generate ~3-5 unique questions
        estimated_capacity = len(chunks) * 4
        already_generated = len(existing_questions)

        if already_generated >= estimated_capacity:
            return {
                "exhausted": True,
                "message": (
                    f"Content exhausted! Already generated {already_generated} questions "
                    f"from {len(chunks)} content chunks. "
                    f"Estimated capacity: ~{estimated_capacity} unique questions. "
                    "Please ingest new PDF materials or expand search criteria."
                ),
            }

        remaining_capacity = estimated_capacity - already_generated
        if remaining_capacity < num_requested:
            return {
                "exhausted": False,
                "message": (
                    f"Limited capacity: Can generate ~{remaining_capacity} more unique questions "
                    f"from current content (requested: {num_requested})"
                ),
                "warning": True,
            }

        return {"exhausted": False, "message": "Sufficient content available"}

    def _create_strict_generation_prompt(
        self,
        chunks: list[dict[str, Any]],
        grade: str,
        subject: str | None,
        num_questions: int,
        difficulty: str | None,
        existing_questions: list[Question],
    ) -> str:
        """Create STRICT anti-hallucination prompt."""
        # Prepare content context
        content_context = []
        for i, chunk in enumerate(chunks, 1):
            context = f"[CONTENT CHUNK {i}]\n"
            context += f"Source: {chunk.get('metadata', {}).get('pdf_source', 'unknown')}\n"
            context += f"Page: {chunk.get('page_number', 'N/A')}\n"
            context += f"Topics: {', '.join(chunk.get('topics', []))}\n"
            context += f"Concepts: {', '.join(chunk.get('concepts', []))}\n"
            context += f"Content: {chunk.get('content', '')}\n"
            if chunk.get("has_images") and chunk.get("image_descriptions"):
                context += (
                    f"ACTUAL IMAGES IN PDF: {json.dumps(chunk.get('image_descriptions', []))}\n"
                )
            else:
                context += "IMAGES: None\n"
            context += f"[END CHUNK {i}]\n"
            content_context.append(context)

        context_text = "\n".join(content_context)

        # Existing questions (to avoid)
        existing_q_text = ""
        if existing_questions:
            existing_q_text = "\nEXISTING QUESTIONS TO AVOID:\n"
            for i, eq in enumerate(existing_questions[:10], 1):
                existing_q_text += f"{i}. {eq.question_text}\n"

        difficulty_inst = (
            f"All questions MUST be {difficulty} difficulty." if difficulty else "Mix difficulties."
        )

        return f"""CRITICAL ANTI-HALLUCINATION RULES:
1. Use ONLY content from the chunks below - NO external knowledge
2. Reference ONLY images explicitly listed in "ACTUAL IMAGES IN PDF"
3. Do NOT invent images, diagrams, or visual content
4. Do NOT create questions about content not in the chunks
5. If chunks lack sufficient content, generate FEWER questions
6. Each question MUST cite its source chunk number
7. Artifacts field: ONLY include actual image descriptions from chunks (or empty if none)

CONTENT FROM INGESTED PDFs (YOUR ONLY SOURCE):
{context_text}
{existing_q_text}

YOUR TASK:
Generate UP TO {num_questions} NEW, UNIQUE multiple-choice questions for {grade} students.
- Use ONLY the content chunks above
- {difficulty_inst}
- DO NOT duplicate existing questions
- DO NOT invent facts, images, or content
- If insufficient unique content remains, generate FEWER questions

OUTPUT FORMAT (JSON only, no markdown):
[
  {{{{
    "serial_number": <sequential starting from 1>,
    "year": 2025,
    "class": "Grade {grade[-1] if grade else "5"}",
    "grade": "{grade}",
    "subject": "{subject or "from_content"}",
    "sub_subject": "<from chunk topics>",
    "question_text": "<question text based on chunk>",
    "content": "<p>Question text here with HTML formatting</p>",
    "type": "MCQ_SA",
    "difficulty": <0-5 based on content complexity>,
    "option1": "<first option>",
    "option2": "<second option>",
    "option3": "<third option>",
    "option4": "<fourth option>",
    "option5": "",
    "option6": "",
    "options": ["<opt1>", "<opt2>", "<opt3>", "<opt4>"],
    "answer": "<must match one option>",
    "answer_index": <0-3>,
    "explanation": "<p>Detailed explanation with HTML. Can include <b>bold</b>, <i>italic</i>, <u>underline</u>, <math>equations</math>, etc.</p>",
    "hints": "First hint with <b>formatting</b>, Second hint",
    "passage": "",
    "citation": "<pdf_source> p.<page_number>",
    "settings": "",
    "question_tags": "<class>,<subject>,<sub_subject>",
    "question_image": "2025_{grade}_{subject or "subject"}_q<num>.png",
    "file_path": "<path to source PDF file>",
    "pdf_source": "<actual PDF filename from chunk metadata>",
    "page_number": <actual page from chunk>,
    "question_number": <sequential>,
    "artifacts": [],
    "source_chunk": <chunk number 1-{len(chunks)} this question is from>
  }}}}
]

FIELD REQUIREMENTS:
- serial_number: Sequential integer starting from 1 (same as question_number)
- content: MUST be HTML-formatted with <p> tags, can include <b>, <i>, <u>, <span style="...">, <math>, <mathinline>, <img />
- type: Use "MCQ_SA" for single-answer MCQs (default)
- difficulty: Integer 0-5 based on question complexity
- explanation: HTML-formatted explanation (can be empty if not needed)
- hints: Comma-separated hints, can include inline HTML formatting
- passage: HTML-formatted passage/context (empty if not needed)
- citation: Source reference like "filename.pdf p.123"
- question_tags: Comma-separated tags combining class, subject, sub_subject
- artifacts: MUST be empty [] (will be populated later with generated image paths)
- pdf_source: MUST match chunk metadata
- page_number: MUST match chunk
- file_path: MUST be the path to source PDF
- All content MUST be traceable to a specific chunk

Generate questions now (JSON only):
"""

    def _generate_with_retry(self, prompt: str) -> Any:
        """Generate with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.7,  # Lower for factual accuracy
                        max_output_tokens=8192,
                    ),
                )

                if not response.parts:
                    logger.warning("Empty response")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2**attempt))
                        continue
                    msg = "Failed to generate"
                    raise ValueError(msg)

                return response

            except google_exceptions.ResourceExhausted:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(f"Rate limit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                logger.exception(f"Generation error: {e}")
                raise
        return None

    def _parse_and_validate_questions(
        self,
        response_text: str,
        grade: str,
        chunks: list[dict[str, Any]],
        existing_questions: list[Question],
    ) -> list[Question]:
        """Parse and validate questions against source content."""
        try:
            cleaned = response_text.strip()

            # Remove markdown code blocks more robustly
            if cleaned.startswith("```"):
                # Remove opening ```json or ```
                lines = cleaned.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # Remove closing ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            # Try to find JSON array if text contains other content
            if not cleaned.startswith("["):
                # Find first [ and last ]
                start_idx = cleaned.find("[")
                end_idx = cleaned.rfind("]")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned = cleaned[start_idx : end_idx + 1]

            # Attempt to parse JSON
            try:
                questions_data = json.loads(cleaned)
            except json.JSONDecodeError as json_err:
                # Check if response was truncated (missing closing bracket)
                if not cleaned.endswith("]"):
                    logger.warning(
                        "JSON appears truncated (missing closing bracket). "
                        "Attempting to salvage valid questions..."
                    )
                    # Try to add closing bracket and parse
                    try:
                        questions_data = json.loads(cleaned + "]")
                        logger.info("Successfully recovered truncated JSON")
                    except json.JSONDecodeError:
                        # If that doesn't work, try to find last complete question
                        last_complete = cleaned.rfind("},")
                        if last_complete != -1:
                            try:
                                questions_data = json.loads(cleaned[: last_complete + 1] + "]")
                                logger.info(
                                    f"Recovered {len(questions_data)} complete questions from truncated response"
                                )
                            except json.JSONDecodeError:
                                raise json_err from None
                        else:
                            raise json_err from None
                else:
                    raise
            if not isinstance(questions_data, list):
                msg = "Expected JSON array"
                raise ValueError(msg)

            # Estimate token count from response
            token_count_estimate = len(response_text.split()) * 1.3  # Rough estimate

            validated_questions = []
            for q_data in questions_data:
                try:
                    # Add timestamp
                    q_data["last_generated"] = datetime.utcnow().isoformat()
                    q_data["type"] = "question"  # Mark as question

                    # Validate against chunks
                    validation_result = self._validate_question_content(q_data, chunks)
                    if not validation_result["valid"]:
                        logger.warning(
                            f"Skipping potentially hallucinated question: {q_data.get('question_text', '')[:50]}..."
                        )
                        continue

                    # Enrich metadata
                    q_data = self._enrich_question_metadata(
                        q_data,
                        chunks,
                        validation_result,
                        int(token_count_estimate // len(questions_data)),
                    )

                    question = Question.from_dict(q_data)
                    validated_questions.append(question)

                except Exception as e:
                    logger.warning(f"Failed to parse question: {e}")
                    continue

            return validated_questions

        except json.JSONDecodeError as e:
            logger.exception(f"Invalid JSON: {e}")
            logger.exception(f"Response: {response_text[:500]}...")
            return []
        except Exception as e:
            logger.exception(f"Parse error: {e}")
            return []

    def _enrich_question_metadata(
        self,
        q_data: dict[str, Any],
        chunks: list[dict[str, Any]],
        validation_result: dict[str, Any],
        token_count: int,
    ) -> dict[str, Any]:
        """
        Enrich question metadata with additional tracking fields and LMS defaults.

        Args:
            q_data: Question data dictionary
            chunks: Content chunks used for generation
            validation_result: Result from validation
            token_count: Estimated token count for this question

        Returns:
            Enriched question data dictionary with all LMS fields
        """
        # Sync serial_number with question_number
        if "question_number" in q_data and "serial_number" not in q_data:
            q_data["serial_number"] = q_data["question_number"]
        elif "serial_number" in q_data and "question_number" not in q_data:
            q_data["question_number"] = q_data["serial_number"]

        # Ensure LMS fields have defaults if not set by generator
        q_data.setdefault("type", "MCQ_SA")
        q_data.setdefault("difficulty", 0)
        q_data.setdefault("option1", "")
        q_data.setdefault("option2", "")
        q_data.setdefault("option3", "")
        q_data.setdefault("option4", "")
        q_data.setdefault("option5", "")
        q_data.setdefault("option6", "")
        q_data.setdefault("explanation", "")
        q_data.setdefault("hints", "")
        q_data.setdefault("passage", "")
        q_data.setdefault("citation", "")
        q_data.setdefault("settings", "")
        q_data.setdefault("question_tags", "")
        q_data.setdefault("content", "")

        # Populate individual option fields from options list if not already set
        if "options" in q_data and isinstance(q_data["options"], list):
            if not q_data["option1"] and len(q_data["options"]) > 0:
                q_data["option1"] = q_data["options"][0]
            if not q_data["option2"] and len(q_data["options"]) > 1:
                q_data["option2"] = q_data["options"][1]
            if not q_data["option3"] and len(q_data["options"]) > 2:
                q_data["option3"] = q_data["options"][2]
            if not q_data["option4"] and len(q_data["options"]) > 3:
                q_data["option4"] = q_data["options"][3]
            if not q_data["option5"] and len(q_data["options"]) > 4:
                q_data["option5"] = q_data["options"][4]
            if not q_data["option6"] and len(q_data["options"]) > 5:
                q_data["option6"] = q_data["options"][5]

        # Generate HTML content from question_text if not set
        if not q_data["content"] and "question_text" in q_data:
            q_data["content"] = f"<p>{q_data['question_text']}</p>"

        # Quality metrics
        q_data["ocr_confidence"] = validation_result["confidence"]
        q_data["validation_status"] = "validated" if validation_result["valid"] else "flagged"

        # Generation metadata
        q_data["token_count"] = int(token_count)
        q_data["reasoning_steps"] = [
            "Retrieved relevant content chunks",
            "Validated against source PDF",
            "Generated MCQ with strict anti-hallucination",
            "Validated source references",
        ]

        # Content analysis
        source_chunk = validation_result.get("chunk")
        if source_chunk:
            topics = source_chunk.get("topics", [])
            concepts = source_chunk.get("concepts", [])
            difficulty = source_chunk.get("difficulty", "medium")

            # Set file_path to source PDF path
            pdf_metadata = source_chunk.get("metadata", {})
            q_data["file_path"] = pdf_metadata.get("pdf_path", q_data.get("pdf_source", ""))

            q_data["question_context"] = (
                f"Based on {source_chunk.get('content_type', 'text')} content from page {q_data.get('page_number', 'N/A')}"
            )
            q_data["question_complexity"] = difficulty
            q_data["tags"] = list(
                {*topics[:3], q_data.get("subject", ""), q_data.get("sub_subject", "")}
            )
            q_data["references"] = concepts[:5] if concepts else []

            # Auto-generate citation if not set
            if not q_data["citation"] and q_data.get("pdf_source"):
                q_data["citation"] = f"{q_data['pdf_source']} p.{q_data.get('page_number', 'N/A')}"

            # Auto-generate question_tags if not set
            if not q_data["question_tags"]:
                tag_parts = []
                if q_data.get("class"):
                    tag_parts.append(q_data["class"])
                if q_data.get("subject"):
                    tag_parts.append(q_data["subject"])
                if q_data.get("sub_subject"):
                    tag_parts.append(q_data["sub_subject"])
                q_data["question_tags"] = ",".join(tag_parts)
        else:
            # Fallback: set file_path to pdf_source
            q_data["file_path"] = q_data.get("pdf_source", "")
            q_data["question_context"] = (
                f"Generated for {q_data.get('grade', '')} {q_data.get('subject', '')}"
            )
            q_data["question_complexity"] = "medium"
            q_data["tags"] = [q_data.get("subject", ""), q_data.get("sub_subject", "")]
            q_data["references"] = []

            # Auto-generate citation if not set
            if not q_data["citation"] and q_data.get("pdf_source"):
                q_data["citation"] = f"{q_data['pdf_source']} p.{q_data.get('page_number', 'N/A')}"

            # Auto-generate question_tags if not set
            if not q_data["question_tags"]:
                tag_parts = []
                if q_data.get("class"):
                    tag_parts.append(q_data["class"])
                if q_data.get("subject"):
                    tag_parts.append(q_data["subject"])
                if q_data.get("sub_subject"):
                    tag_parts.append(q_data["sub_subject"])
                q_data["question_tags"] = ",".join(tag_parts)

        # Image tracking (artifacts will be populated later with generated image paths)
        q_data["artifacts"] = []  # Clear any PDF image descriptions - only for generated images
        q_data["images_tagged_count"] = 0  # Will be updated when images are generated
        q_data["images_path"] = []  # Will be populated when images are generated
        q_data["artifacts_path"] = []  # Will be populated when images are generated

        # Classification & tracking
        q_data["processing_status"] = "success" if validation_result["valid"] else "partial"
        q_data["issues_found"] = validation_result.get("issues", [])

        # Related questions (empty for now, can be populated later)
        q_data["related_questions"] = []

        return q_data

    def _validate_question_content(
        self, q_data: dict[str, Any], chunks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Validate that question content matches source chunks.

        Returns:
            Dictionary with 'valid' (bool), 'issues' (list), 'chunk' (dict), 'confidence' (float)
        """
        issues = []
        confidence = 0.85  # Start with slightly lower confidence
        source_chunk = None

        # Check if source_chunk is valid
        source_chunk_idx = q_data.get("source_chunk")
        if (
            source_chunk_idx
            and isinstance(source_chunk_idx, int)
            and 1 <= source_chunk_idx <= len(chunks)
        ):
            chunk = chunks[source_chunk_idx - 1]
            source_chunk = chunk
            confidence = 0.95  # Higher confidence when source is validated

            # Validate PDF source
            expected_pdf = chunk.get("metadata", {}).get("pdf_source")
            if expected_pdf and q_data.get("pdf_source") != expected_pdf:
                issues.append(f"PDF source mismatch: {q_data.get('pdf_source')} vs {expected_pdf}")
                confidence = 0.70
                return {"valid": False, "issues": issues, "chunk": None, "confidence": confidence}
        else:
            # Missing source_chunk - still allow but log warning
            issues.append("source_chunk not specified or invalid")
            logger.debug(f"Question missing source_chunk: {q_data.get('question_text', '')[:50]}")
            # Try to infer source chunk from PDF name
            if chunks and q_data.get("pdf_source"):
                for i, chunk in enumerate(chunks):
                    if chunk.get("metadata", {}).get("pdf_source") == q_data.get("pdf_source"):
                        source_chunk = chunk
                        q_data["source_chunk"] = i + 1  # Auto-assign source chunk
                        confidence = 0.80
                        break

        # Clear artifacts - will be populated later with generated image paths only
        # (Artifacts are for GENERATED images, not PDF source images)
        q_data["artifacts"] = []

        return {
            "valid": True,  # Always valid unless critical mismatch
            "issues": issues,
            "chunk": source_chunk,
            "confidence": confidence,
        }

    def _filter_duplicates(
        self, new_questions: list[Question], existing_questions: list[Question], allow_similar: bool
    ) -> tuple[list[Question], int]:
        """Filter out duplicate questions."""
        if allow_similar or not existing_questions:
            return new_questions, 0

        unique = []
        duplicates = 0

        for new_q in new_questions:
            is_duplicate = False

            # Embed new question
            new_embedding = self.embedding_client.embed_question(new_q)

            for existing_q in existing_questions:
                # Embed existing question
                existing_embedding = self.embedding_client.embed_question(existing_q)

                # Calculate cosine similarity
                similarity = self._cosine_similarity(new_embedding, existing_embedding)

                if similarity > self.SIMILARITY_THRESHOLD:
                    logger.info(
                        f"Duplicate detected (similarity: {similarity:.2f}): {new_q.question_text[:50]}..."
                    )
                    is_duplicate = True
                    duplicates += 1
                    break

            if not is_duplicate:
                unique.append(new_q)

        return unique, duplicates

    @staticmethod
    def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def _store_questions_to_qdrant(
        self, questions: list[Question], collection_name: str, chunks_used: list[dict[str, Any]]
    ) -> None:
        """Store generated questions back to Qdrant for tracking."""
        try:
            # Generate embeddings for questions
            embeddings = self.embedding_client.embed_questions_batch(questions)

            # Add tracking metadata
            for q in questions:
                q_dict = q.to_dict()
                q_dict["type"] = "question"
                q_dict["content_chunks_used"] = len(chunks_used)

            # Store in Qdrant
            self.qdrant_manager.upsert_questions(
                collection_name=collection_name, questions=questions, embeddings=embeddings
            )

            logger.info(f"Stored {len(questions)} questions to {collection_name}")

        except Exception as e:
            logger.exception(f"Error storing questions: {e}")
            # Don't fail if storage fails
