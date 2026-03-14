"""
Question generator with:
- NAPLAN alignment validation using past papers as reference
- Non-repetitive generation across all runs (Qdrant-backed deduplication)
- Full exam set mode (Numeracy: 34q, Language Convention: 50q)
- Sub-topic mode (variable count based on difficulty)
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any

from google import genai
from google.genai import types

from src.core.embeddings import EmbeddingClient
from src.core.models import Question
from src.core.qdrant_client_wrapper import QdrantManager

logger = logging.getLogger(__name__)


# ============================================================
# NAPLAN EXAM SET CONFIGURATION
# ============================================================
EXAM_SET_CONFIG = {
    "numeracy": {
        "total_questions": 34,
        "distribution": {"easy": 10, "medium": 16, "hard": 8},
        "topics": [
            "number and place value", "fractions and decimals",
            "money and financial mathematics", "patterns and algebra",
            "measurement", "geometry", "data and statistics",
            "probability", "multiplication", "division",
        ],
    },
    "language": {
        "total_questions": 50,
        "distribution": {"easy": 15, "medium": 25, "hard": 10},
        "topics": [
            "spelling", "grammar", "punctuation",
            "vocabulary", "sentence structure", "word knowledge",
        ],
    },
    "language_convention": {
        "total_questions": 50,
        "distribution": {"easy": 15, "medium": 25, "hard": 10},
        "topics": [
            "spelling", "grammar", "punctuation",
            "vocabulary", "sentence structure", "word knowledge",
        ],
    },
    "reading": {
        "total_questions": 40,
        "distribution": {"easy": 12, "medium": 20, "hard": 8},
        "topics": [
            "comprehension", "inference", "vocabulary in context",
            "text structure", "author purpose", "literal meaning",
        ],
    },
}

SUBTOPIC_QUESTION_COUNT = {
    "easy": 5,
    "medium": 8,
    "hard": 12,
    "mixed": 15,
}


def _clean_json_response(text: str) -> str:
    """Remove all markdown fences from a response string."""
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")
    return text.strip()


def _make_question(fields: dict, correct_letter: str, correct_idx: int) -> Question | None:
    """Try to create a Question object, falling back for old models.py versions."""
    # Build list of field combinations to try, from most complete to minimal
    attempts = [
        fields,  # Full fields
        {k: v for k, v in fields.items()
         if k not in ("categories",)},  # Without categories
        {k: v for k, v in fields.items()
         if k not in ("categories", "correct_answer", "correct_option_index",
                      "pdf_source", "page_number", "file_path", "content", "artifacts")},
        # Minimal
        {k: v for k, v in fields.items()
         if k in ("question_number", "year", "grade", "subject", "sub_subject",
                  "question_text", "options", "explanation", "difficulty")},
    ]

    for attempt in attempts:
        try:
            q = Question(**attempt)
            # Patch missing attributes so exporter always works
            for attr, val in [
                ("correct_answer", correct_letter),
                ("correct_option_index", correct_idx),
                ("categories", f"{fields.get('grade', '')},{fields.get('subject', '')}"),
                ("artifacts", []),
                ("images_path", []),
                ("artifacts_path", []),
                ("question_image", ""),
                ("images_tagged_count", 0),
            ]:
                if not hasattr(q, attr):
                    setattr(q, attr, val)
            return q
        except TypeError:
            continue
    return None


class QuestionGeneratorV2:
    """NAPLAN-aligned question generator."""

    SIMILARITY_THRESHOLD = 0.82

    def __init__(self, embedding_client: EmbeddingClient, qdrant_manager: QdrantManager,
                 api_key: str | None = None, model_name: str = "gemini-2.5-flash",
                 max_retries: int = 3, retry_delay: float = 2.0):
        self.embedding_client = embedding_client
        self.qdrant_manager = qdrant_manager
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in .env file.")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"Initialized QuestionGeneratorV2 with model: {model_name}")
        logger.info("Modes: exam_set | subtopic | standard")
        logger.info("Safeguards: NAPLAN validation, cross-run dedup, anti-hallucination")

    # ============================================================
    # PUBLIC API
    # ============================================================

    def generate_exam_set(self, grade: str, subject: str) -> dict[str, Any]:
        subject_key = subject.lower().replace(" ", "_")
        config = EXAM_SET_CONFIG.get(subject_key) or EXAM_SET_CONFIG.get(subject.lower())
        if not config:
            config = EXAM_SET_CONFIG["numeracy"]

        total = config["total_questions"]
        distribution = config["distribution"]
        topics = config["topics"]

        logger.info(f"Generating full exam set: {subject} ({total} questions)")
        logger.info(f"Difficulty distribution: {distribution}")

        all_questions = []
        for difficulty, count in distribution.items():
            logger.info(f"  Generating {count} {difficulty} questions...")
            result = self._generate_in_batches(
                grade=grade, subject=subject, topics=topics,
                difficulty=difficulty, num_questions=count,
                content_chunks_limit=25, batch_size=count,
                allow_similar=False, existing_questions=all_questions,
                validate_naplan=True,
            )
            all_questions.extend(result.get("questions", []))
            logger.info(f"  Running total: {len(all_questions)}/{total}")

        for i, q in enumerate(all_questions):
            q.question_number = i + 1

        logger.info(f"Exam set complete: {len(all_questions)} questions")
        return {
            "questions": all_questions,
            "mode": "exam_set",
            "subject": subject,
            "target": total,
            "generated": len(all_questions),
            "status": "complete" if len(all_questions) >= total else "partial",
        }

    def generate_subtopic_questions(self, grade: str, subject: str,
                                     subtopic: str, difficulty: str = "mixed") -> dict[str, Any]:
        num_questions = SUBTOPIC_QUESTION_COUNT.get(difficulty, 10)
        logger.info(f"Generating {num_questions} {difficulty} questions for: {subtopic}")

        result = self._generate_in_batches(
            grade=grade, subject=subject, topics=[subtopic],
            difficulty=difficulty if difficulty != "mixed" else None,
            num_questions=num_questions, content_chunks_limit=20,
            batch_size=num_questions, allow_similar=False,
            existing_questions=[], validate_naplan=True,
        )
        return {
            "questions": result.get("questions", []),
            "mode": "subtopic",
            "subtopic": subtopic,
            "difficulty": difficulty,
            "target": num_questions,
            "generated": len(result.get("questions", [])),
            "status": result.get("status", "partial"),
        }

    def generate_questions(self, grade: str, subject: str | None = None,
                           topics: list[str] | None = None, difficulty: str | None = None,
                           num_questions: int = 10, allow_similar: bool = False,
                           content_chunks_limit: int = 20) -> dict[str, Any]:
        return self._generate_in_batches(
            grade=grade, subject=subject, topics=topics, difficulty=difficulty,
            num_questions=num_questions, content_chunks_limit=content_chunks_limit,
            batch_size=15, allow_similar=allow_similar,
            existing_questions=[], validate_naplan=True,
        )

    # ============================================================
    # CORE PIPELINE
    # ============================================================

    def _generate_in_batches(self, grade, subject, topics, difficulty,
                              num_questions, content_chunks_limit, batch_size,
                              allow_similar, existing_questions, validate_naplan=True):
        stored_questions = self._load_all_stored_questions(grade, subject)
        logger.info(f"Loaded {len(stored_questions)} previously stored questions (cross-run dedup)")

        combined_existing = stored_questions + existing_questions
        all_new_questions = []
        duplicates_avoided = 0
        naplan_rejected = 0
        batch_num = 0
        max_batches = 15

        while len(all_new_questions) < num_questions and batch_num < max_batches:
            remaining = num_questions - len(all_new_questions)
            request_count = min(batch_size, remaining) + 5
            batch_num += 1

            logger.info(f"Batch {batch_num}: Requesting {request_count} questions "
                        f"({len(all_new_questions)}/{num_questions} done)...")

            chunks = self._retrieve_content_chunks(
                grade=grade, subject=subject, topics=topics,
                difficulty=difficulty, limit=content_chunks_limit,
            )
            if not chunks:
                logger.warning("No content chunks found. Please ingest PDFs first.")
                break

            raw_questions = self._generate_batch_raw(
                chunks=chunks, grade=grade, subject=subject,
                topics=topics, difficulty=difficulty,
                num_questions=request_count,
                existing_questions=combined_existing + all_new_questions,
            )

            if not raw_questions:
                logger.warning("No questions returned from model.")
                break

            all_so_far = combined_existing + all_new_questions
            filtered, dupes = self._filter_duplicates(raw_questions, all_so_far, allow_similar)
            duplicates_avoided += dupes

            if validate_naplan:
                validated, rejected = self._validate_batch_naplan(filtered, grade)
                naplan_rejected += rejected
            else:
                validated = filtered

            if not validated:
                logger.warning(f"Batch {batch_num}: All filtered. Retrying...")
                continue

            take = min(len(validated), remaining)
            all_new_questions.extend(validated[:take])
            logger.info(f"Batch {batch_num}: Kept {take}. Total: {len(all_new_questions)}/{num_questions}")

        offset = len(stored_questions)
        for i, q in enumerate(all_new_questions):
            q.question_number = offset + i + 1

        self._store_questions(all_new_questions, grade)
        logger.info(f"Done: {len(all_new_questions)} new | "
                    f"{duplicates_avoided} dupes avoided | {naplan_rejected} NAPLAN rejected")

        return {
            "questions": all_new_questions,
            "status": "complete" if len(all_new_questions) >= num_questions else "partial",
            "duplicates_avoided": duplicates_avoided,
            "naplan_rejected": naplan_rejected,
            "new_questions_count": len(all_new_questions),
        }

    def _generate_batch_raw(self, chunks, grade, subject, topics, difficulty,
                             num_questions, existing_questions) -> list[Question]:
        prompt = self._build_prompt(
            chunks=chunks, grade=grade, subject=subject, topics=topics,
            difficulty=difficulty, num_questions=num_questions,
            existing_questions=existing_questions,
        )
        response_text = self._call_gemini(prompt)
        if not response_text:
            return []
        return self._parse_questions(response_text, grade)

    # ============================================================
    # PROMPT
    # ============================================================

    def _build_prompt(self, chunks, grade, subject, topics, difficulty,
                       num_questions, existing_questions) -> str:
        chunks_text = ""
        for i, chunk in enumerate(chunks):
            payload = chunk.payload if hasattr(chunk, "payload") else {}
            content = payload.get("content", str(chunk)) if isinstance(payload, dict) else str(chunk)
            page = payload.get("page_number", "?") if isinstance(payload, dict) else "?"
            pdf = payload.get("pdf_source", "unknown") if isinstance(payload, dict) else "unknown"
            chunks_text += f"\n[Source {i+1} | {pdf} p.{page}]\n{content}\n"

        naplan_examples = self._get_naplan_examples(grade, subject)

        existing_text = ""
        if existing_questions:
            existing_text = "\nDO NOT GENERATE QUESTIONS SIMILAR TO THESE:\n"
            for q in existing_questions[-40:]:
                existing_text += f"- {q.question_text[:100]}\n"

        diff_str = f"Difficulty: {difficulty}" if difficulty else "Mix of easy, medium, hard"
        subj_str = f"Subject: {subject}" if subject else "Subject: as per content"
        topic_str = f"Focus topics: {', '.join(topics)}" if topics else "Topics: varied"

        return f"""You are an expert NAPLAN exam question writer for Australian {grade} students.

Generate exactly {num_questions} NAPLAN-style multiple-choice questions from the BOOK CONTENT.
Use NAPLAN EXAMPLES as your quality and style standard.

NAPLAN STYLE EXAMPLES:
{naplan_examples}

BOOK CONTENT:
{chunks_text}

REQUIREMENTS:
- Grade: {grade} (Australian Year level)
- {subj_str}
- {topic_str}
- {diff_str}
- Each question tests ONE specific skill only
- Exactly 4 options, only ONE correct answer
- Wrong options must be plausible
- Clear simple Australian English
- Vary topics and question formats
{existing_text}

IMPORTANT: Return ONLY a raw JSON array. NO markdown. NO backticks. NO explanation. Just the JSON:
[
  {{
    "question_text": "Full question text?",
    "option_1": "First option",
    "option_2": "Second option",
    "option_3": "Third option",
    "option_4": "Fourth option",
    "correct_option": 1,
    "explanation": "Why this answer is correct",
    "subject": "subject name",
    "sub_subject": "specific topic",
    "difficulty": "easy|medium|hard",
    "skill_tested": "skill description",
    "naplan_strand": "strand name",
    "pdf_source": "filename.pdf",
    "page_number": 1,
    "categories": "{grade},{subject}"
  }}
]"""

    # ============================================================
    # GEMINI CALL
    # ============================================================

    def _call_gemini(self, prompt: str) -> str | None:
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.8,
                        max_output_tokens=8192,
                    ),
                )
                text = response.text
                logger.debug(f"Gemini raw (first 200): {text[:200]}")
                return text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Retry in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Gemini call failed: {e}")
                    return None
        return None

    # ============================================================
    # PARSER
    # ============================================================

    def _parse_questions(self, response_text: str, grade: str) -> list[Question]:
        try:
            # Strip ALL markdown fences
            text = _clean_json_response(response_text)

            start = text.find("[")
            end = text.rfind("]") + 1
            if start == -1 or end == 0:
                logger.error(f"No JSON array found. Preview: {response_text[:200]}")
                return []

            data = json.loads(text[start:end])
            questions = []
            year = datetime.now().year

            for i, d in enumerate(data):
                if not str(d.get("question_text", "")).strip():
                    continue
                options = [d.get(f"option_{j}", "") for j in range(1, 5)]
                correct_idx = max(0, min(3, int(d.get("correct_option", 1)) - 1))
                correct_letter = ["A", "B", "C", "D"][correct_idx]
                skill = d.get("skill_tested", "")

                fields = dict(
                    question_number=i + 1,
                    year=year,
                    grade=grade,
                    subject=d.get("subject", ""),
                    sub_subject=d.get("sub_subject", skill),
                    question_text=d.get("question_text", ""),
                    options=options,
                    correct_answer=correct_letter,
                    correct_option_index=correct_idx,
                    explanation=d.get("explanation", ""),
                    difficulty=d.get("difficulty", "medium"),
                    pdf_source=d.get("pdf_source", ""),
                    page_number=d.get("page_number", 0),
                    file_path=d.get("file_path", ""),
                    categories=f"{grade},{d.get('subject', '')}",
                    content=f"<p>{d.get('question_text', '')}</p>",
                    artifacts=[],
                )

                q = _make_question(fields, correct_letter, correct_idx)
                if q is None:
                    logger.warning(f"Could not create Question for item {i}, skipping")
                    continue
                questions.append(q)

            logger.info(f"Parsed {len(questions)} questions")
            return questions

        except Exception as e:
            logger.error(f"Error parsing questions: {e}")
            return []

    # ============================================================
    # NAPLAN VALIDATOR
    # ============================================================

    def _validate_batch_naplan(self, questions: list[Question],
                                grade: str) -> tuple[list[Question], int]:
        if not questions:
            return [], 0
        try:
            questions_json = [
                {
                    "index": i,
                    "question": q.question_text,
                    "options": q.options,
                    "subject": q.subject,
                    "difficulty": q.difficulty,
                }
                for i, q in enumerate(questions)
            ]

            prompt = f"""NAPLAN quality reviewer for Australian {grade} students.
Review each question. Score 1-10. valid=true if score>=6.
Return ONLY raw JSON array, no markdown, no backticks:
[{{"index": 0, "valid": true, "score": 8, "reason": "clear"}}]

Questions:
{json.dumps(questions_json, indent=2)}"""

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=2000),
            )
            text = _clean_json_response(response.text)
            start, end = text.find("["), text.rfind("]") + 1
            results = json.loads(text[start:end])

            valid_indices = {r["index"] for r in results if r.get("valid", True)}
            validated, rejected = [], 0
            for i, q in enumerate(questions):
                if i in valid_indices:
                    validated.append(q)
                else:
                    r = next((x for x in results if x["index"] == i), {})
                    logger.info(f"NAPLAN rejected Q{i}: score={r.get('score','?')} {r.get('reason','')}")
                    rejected += 1
            return validated, rejected
        except Exception as e:
            logger.warning(f"NAPLAN validation skipped: {e}")
            return questions, 0

    # ============================================================
    # NAPLAN EXAMPLES
    # ============================================================

    def _get_naplan_examples(self, grade: str, subject: str | None) -> str:
        try:
            subject_key = subject.lower().replace(" ", "_") if subject else "general"
            ref_col = f"{grade}_{subject_key}_naplan_reference"
            query_embedding = self.embedding_client.embed_query(
                f"NAPLAN {grade} {subject or ''} example question"
            )
            results = self.qdrant_manager.search_questions(
                collection_name=ref_col,
                query_embedding=query_embedding,
                limit=5,
            )
            if not results:
                return self._default_naplan_style(subject)
            examples = ""
            for i, r in enumerate(results):
                payload = r.payload if hasattr(r, "payload") else {}
                content = payload.get("content", "") if isinstance(payload, dict) else ""
                if content:
                    examples += f"\n[NAPLAN Example {i+1}]\n{content}\n"
            return examples or self._default_naplan_style(subject)
        except Exception:
            return self._default_naplan_style(subject)

    def _default_naplan_style(self, subject: str | None) -> str:
        if subject and "numer" in subject.lower():
            return (
                "[NAPLAN Numeracy Style]\n"
                "- Direct: 'What is 24 x 5?'\n"
                "- Word problem: 'Mia has 3 bags with 8 apples each. How many in total?'\n"
                "- Data reading: 'The table shows... How many more...?'\n"
                "- Options close in value: e.g. 38, 40, 42, 44\n"
            )
        elif subject and ("lang" in subject.lower() or "convention" in subject.lower()):
            return (
                "[NAPLAN Language Conventions Style]\n"
                "- Spelling: 'Which word is spelled correctly?'\n"
                "- Grammar: 'Choose the correct word: She ___ to school every day.'\n"
                "- Punctuation: 'Which sentence uses punctuation correctly?'\n"
            )
        return (
            "[NAPLAN Style]\n"
            "- Single concept per question\n"
            "- Clear Australian English\n"
            "- 4 plausible options\n"
        )

    # ============================================================
    # DEDUPLICATION
    # ============================================================

    def _load_all_stored_questions(self, grade: str, subject: str | None) -> list[Question]:
        try:
            query_embedding = self.embedding_client.embed_query(
                f"{grade} {subject or ''} questions"
            )
            results = self.qdrant_manager.search_questions(
                collection_name=f"{grade}_questions",
                query_embedding=query_embedding,
                limit=2000,
            )
            questions = []
            for r in results:
                payload = r.payload if hasattr(r, "payload") else {}
                if isinstance(payload, dict) and payload.get("question_text"):
                    fields = dict(
                        question_number=payload.get("question_number", 0),
                        year=payload.get("year", 2026),
                        grade=payload.get("grade", grade),
                        subject=payload.get("subject", ""),
                        sub_subject=payload.get("sub_subject", ""),
                        question_text=payload.get("question_text", ""),
                        options=payload.get("options", []),
                        correct_answer=payload.get("correct_answer", "A"),
                        correct_option_index=payload.get("correct_option_index", 0),
                        explanation=payload.get("explanation", ""),
                        difficulty=payload.get("difficulty", "medium"),
                    )
                    q = _make_question(fields, fields["correct_answer"],
                                       fields["correct_option_index"])
                    if q:
                        questions.append(q)
            return questions
        except Exception as e:
            logger.warning(f"Could not load stored questions (first run?): {e}")
            return []

    def _filter_duplicates(self, questions, existing, allow_similar):
        if allow_similar or not questions or not existing:
            return questions, 0
        filtered, dupes = [], 0
        for q in questions:
            if any(self._are_similar(q.question_text, e.question_text) for e in existing):
                dupes += 1
            else:
                filtered.append(q)
        return filtered, dupes

    def _are_similar(self, t1: str, t2: str) -> bool:
        w1, w2 = set(t1.lower().split()), set(t2.lower().split())
        if not w1 or not w2:
            return False
        return len(w1 & w2) / len(w1 | w2) > self.SIMILARITY_THRESHOLD

    # ============================================================
    # CONTENT RETRIEVAL
    # ============================================================

    def _retrieve_content_chunks(self, grade, subject=None, topics=None,
                                  difficulty=None, limit=20, collection_name=None):
        try:
            query = f"educational content grade {grade}"
            if subject:
                query += f" {subject}"
            if topics:
                query += f" {' '.join(topics)}"
            query_embedding = self.embedding_client.embed_query(query)
            col = collection_name or f"{grade}_content"
            return self.qdrant_manager.search_questions(
                collection_name=col,
                query_embedding=query_embedding,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Error retrieving content: {e}")
            return []

    # ============================================================
    # STORAGE
    # ============================================================

    def _store_questions(self, questions: list[Question], grade: str):
        if not questions:
            return
        try:
            embeddings = self.embedding_client.embed_questions_batch(questions)
            self.qdrant_manager.upsert_questions(
                collection_name=f"{grade}_questions",
                questions=questions,
                embeddings=embeddings,
            )
            logger.info(f"Stored {len(questions)} questions in Qdrant")
        except Exception as e:
            logger.warning(f"Could not store questions: {e}")

    def check_capacity(self, grade, subject=None, topics=None, difficulty=None) -> dict:
        chunks = self._retrieve_content_chunks(grade, subject, topics, difficulty, limit=100)
        stored = self._load_all_stored_questions(grade, subject)
        estimated = len(chunks) * 4
        available = max(0, estimated - len(stored))
        return {
            "available_chunks": len(chunks),
            "estimated_total_capacity": estimated,
            "already_generated": len(stored),
            "estimated_remaining": available,
            "recommended_max": min(available, 200),
        }