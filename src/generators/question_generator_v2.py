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

# Maps difficulty string → int (0-5 scale used by Question dataclass)
DIFFICULTY_STR_TO_INT = {
    "easy": 1,
    "medium": 2,
    "hard": 4,
}

# Maps difficulty int → display string
DIFFICULTY_INT_TO_STR = {v: k for k, v in DIFFICULTY_STR_TO_INT.items()}

# Valid specific sub-subjects per domain — used to reject generic fallback values
NUMERACY_SUBTOPICS = {
    "number and place value", "fractions", "decimals", "fractions and decimals",
    "money", "financial mathematics", "money and financial mathematics",
    "patterns", "algebra", "patterns and algebra",
    "measurement", "length", "area", "volume", "mass", "time",
    "geometry", "shapes", "angles", "symmetry",
    "data", "statistics", "data and statistics", "graphs", "tables",
    "probability", "chance",
    "multiplication", "division", "addition", "subtraction",
    "ratios", "percentages", "estimation",
}

LANGUAGE_SUBTOPICS = {
    "spelling", "grammar", "punctuation", "vocabulary",
    "sentence structure", "word knowledge", "nouns", "verbs",
    "adjectives", "adverbs", "pronouns", "conjunctions",
    "apostrophes", "commas", "capital letters", "homophones",
    "prefixes", "suffixes", "compound words",
}


def _is_generic_sub_subject(value: str, subject: str) -> bool:
    """Return True if sub_subject is just a copy of the subject name (not specific enough)."""
    v = value.strip().lower()
    s = (subject or "").strip().lower()
    generic = {"numeracy", "maths", "mathematics", "language", "language convention",
               "reading", "english", "general", ""}
    return v == s or v in generic


# ============================================================
# HELPERS
# ============================================================

def _clean_json_response(text: str) -> str:
    """Remove all markdown fences from a response string."""
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")
    return text.strip()


def _grade_to_class_name(grade: str) -> str:
    """Convert grade string like 'grade5' → 'Grade 5'."""
    num = re.sub(r"[^0-9]", "", grade)
    return f"Grade {num}" if num else grade.capitalize()


def _salvage_partial_json(text: str) -> list[dict]:
    """
    Robustly parse a JSON array — handles both complete and truncated responses.

    Gemini 2.5 Flash can return very long responses. If max_output_tokens is
    hit mid-array the closing ] is missing. This function:
      1. Tries a normal full parse first.
      2. If that fails, extracts every fully-formed {...} object individually
         using a bracket-depth counter, so partial responses still yield
         the questions that were fully written.
    """
    start = text.find("[")
    if start == -1:
        return []

    # Attempt 1: normal full parse
    end = text.rfind("]") + 1
    if end > 0:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Attempt 2: bracket-depth salvage for truncated arrays
    salvaged = []
    depth = 0
    obj_start = None
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                fragment = text[obj_start: i + 1]
                try:
                    obj = json.loads(fragment)
                    salvaged.append(obj)
                except json.JSONDecodeError:
                    pass
                obj_start = None

    if salvaged:
        logger.warning(
            f"JSON was truncated — salvaged {len(salvaged)} complete objects from partial response."
        )
    return salvaged


def _make_question(fields: dict, correct_letter: str, correct_idx: int) -> "Question | None":
    """Try to create a Question object, falling back through simpler field sets."""
    attempts = [
        fields,
        {k: v for k, v in fields.items() if k not in ("categories",)},
        {k: v for k, v in fields.items()
         if k not in ("categories", "correct_answer", "correct_option_index",
                      "pdf_source", "page_number", "file_path", "content", "artifacts")},
        # Minimal — must still include the two required non-default fields
        {k: v for k, v in fields.items()
         if k in ("serial_number", "question_number", "year", "class_name",
                  "grade", "subject", "sub_subject",
                  "question_text", "options", "explanation", "difficulty")},
    ]

    for attempt in attempts:
        try:
            q = Question(**attempt)
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


# ============================================================
# GENERATOR
# ============================================================

class QuestionGeneratorV2:
    """NAPLAN-aligned question generator."""

    SIMILARITY_THRESHOLD = 0.82

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        qdrant_manager: QdrantManager,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.embedding_client = embedding_client
        self.qdrant_manager = qdrant_manager
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in .env file.")
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client = genai.Client(api_key=self.api_key)

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

        logger.info(f"Generating exam set: {total} questions for {grade} {subject}")
        logger.info(f"Distribution: {distribution}")

        all_questions: list[Question] = []
        for difficulty, count in distribution.items():
            logger.info(f"  Generating {count} {difficulty} questions...")
            result = self._generate_in_batches(
                grade=grade, subject=subject, topics=topics,
                difficulty=difficulty, num_questions=count,
                content_chunks_limit=20, batch_size=15,
                allow_similar=False, existing_questions=all_questions,
                validate_naplan=True,
            )
            all_questions.extend(result["questions"])
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

    def generate_subtopic_questions(
        self, grade: str, subject: str, subtopic: str, difficulty: str = "mixed"
    ) -> dict[str, Any]:
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

    def generate_questions(
        self,
        grade: str,
        subject: str | None = None,
        topics: list[str] | None = None,
        difficulty: str | None = None,
        num_questions: int = 10,
        allow_similar: bool = False,
        content_chunks_limit: int = 20,
    ) -> dict[str, Any]:
        return self._generate_in_batches(
            grade=grade, subject=subject, topics=topics, difficulty=difficulty,
            num_questions=num_questions, content_chunks_limit=content_chunks_limit,
            batch_size=15, allow_similar=allow_similar,
            existing_questions=[], validate_naplan=True,
        )

    # ============================================================
    # CORE PIPELINE
    # ============================================================

    def _generate_in_batches(
        self, grade, subject, topics, difficulty,
        num_questions, content_chunks_limit, batch_size,
        allow_similar, existing_questions, validate_naplan=True,
    ):
        stored_questions = self._load_all_stored_questions(grade, subject)
        logger.info(
            f"Loaded {len(stored_questions)} previously stored questions (cross-run dedup)"
        )

        combined_existing = stored_questions + existing_questions
        all_new_questions: list[Question] = []
        duplicates_avoided = 0
        naplan_rejected = 0
        batch_num = 0
        max_batches = 15

        while len(all_new_questions) < num_questions and batch_num < max_batches:
            remaining = num_questions - len(all_new_questions)
            request_count = min(batch_size, remaining) + 5
            batch_num += 1

            logger.info(
                f"Batch {batch_num}: Requesting {request_count} questions "
                f"({len(all_new_questions)}/{num_questions} done)..."
            )

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
            logger.info(
                f"Batch {batch_num}: Kept {take}. "
                f"Total: {len(all_new_questions)}/{num_questions}"
            )

        offset = len(stored_questions)
        for i, q in enumerate(all_new_questions):
            q.question_number = offset + i + 1

        self._store_questions(all_new_questions, grade)
        logger.info(
            f"Done: {len(all_new_questions)} new | "
            f"{duplicates_avoided} dupes avoided | {naplan_rejected} NAPLAN rejected"
        )

        return {
            "questions": all_new_questions,
            "status": "complete" if len(all_new_questions) >= num_questions else "partial",
            "duplicates_avoided": duplicates_avoided,
            "naplan_rejected": naplan_rejected,
            "new_questions_count": len(all_new_questions),
        }

    def _generate_batch_raw(
        self, chunks, grade, subject, topics, difficulty, num_questions, existing_questions
    ) -> list[Question]:
        prompt = self._build_prompt(
            chunks=chunks, grade=grade, subject=subject, topics=topics,
            difficulty=difficulty, num_questions=num_questions,
            existing_questions=existing_questions,
        )
        response_text = self._call_gemini(prompt)
        if not response_text:
            return []
        return self._parse_questions(response_text, grade, subject)

    # ============================================================
    # PROMPT  ← FIX: sub_subject must be specific, not a copy of subject
    # ============================================================

    def _build_prompt(
        self, chunks, grade, subject, topics, difficulty, num_questions, existing_questions
    ) -> str:
        chunks_text = ""
        for i, chunk in enumerate(chunks):
            payload = chunk.payload if hasattr(chunk, "payload") else {}
            content = (
                payload.get("content", str(chunk)) if isinstance(payload, dict) else str(chunk)
            )
            page = payload.get("page_number", "?") if isinstance(payload, dict) else "?"
            pdf = payload.get("pdf_source", "unknown") if isinstance(payload, dict) else "unknown"
            chunks_text += f"\n[Source {i + 1} | {pdf} p.{page}]\n{content}\n"

        naplan_examples = self._get_naplan_examples(grade, subject)

        existing_text = ""
        if existing_questions:
            existing_text = "\nDO NOT GENERATE QUESTIONS SIMILAR TO THESE:\n"
            for q in existing_questions[-40:]:
                existing_text += f"- {q.question_text[:100]}\n"

        diff_str = f"Difficulty: {difficulty}" if difficulty else "Mix of easy, medium, hard"
        subj_str = f"Subject: {subject}" if subject else "Subject: as per content"
        topic_str = f"Focus topics: {', '.join(topics)}" if topics else "Topics: varied"

        # FIX: Explicit sub_subject rule — must be the specific maths/language topic name
        sub_subject_rule = (
            "- sub_subject: The SPECIFIC topic name tested — e.g. 'Fractions', "
            "'Place Value', 'Multiplication', 'Money', 'Geometry', 'Probability'. "
            "NEVER copy the subject name (e.g. NEVER write 'Numeracy' or 'Mathematics'). "
            "Each question must have a different, precise sub_subject."
        )

        return f"""You are an expert NAPLAN exam question writer for Australian {grade} students.
Generate exactly {num_questions} NAPLAN-style multiple-choice questions from the BOOK CONTENT below.
Use NAPLAN EXAMPLES as your quality and style guide.

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
- Exactly 4 options labeled option_1 to option_4, only ONE correct answer
- Wrong options must be plausible
- Clear simple Australian English
- Vary topics and question formats
{sub_subject_rule}
{existing_text}

CRITICAL: Return ONLY a valid, complete JSON array. No markdown, no backticks, no explanation.
Ensure the array ends with a closing ] character.
[
  {{
    "question_text": "Full question text?",
    "option_1": "First option",
    "option_2": "Second option",
    "option_3": "Third option",
    "option_4": "Fourth option",
    "correct_option": 1,
    "explanation": "Why this answer is correct",
    "subject": "numeracy",
    "sub_subject": "Place Value",
    "difficulty": "easy|medium|hard",
    "skill_tested": "Identifying the value of a digit in a 4-digit number",
    "naplan_strand": "Number and Algebra",
    "pdf_source": "filename.pdf",
    "page_number": 1
  }}
]"""

    def _get_naplan_examples(self, grade: str, subject: str | None) -> str:
        """Fetch NAPLAN examples from Qdrant reference collection, or fall back to defaults."""
        try:
            subject_key = (subject or "general").lower().replace(" ", "_")
            ref_col = f"{grade}_{subject_key}_naplan_reference"
            query_embedding = self.embedding_client.embed_query(
                f"NAPLAN {grade} {subject or ''} example question"
            )
            results = self.qdrant_manager.search_questions(
                collection_name=ref_col,
                query_embedding=query_embedding,
                limit=5,
            )
            if results:
                examples = ""
                for i, r in enumerate(results):
                    payload = r.payload if hasattr(r, "payload") else {}
                    content = payload.get("content", "") if isinstance(payload, dict) else ""
                    if content:
                        examples += f"\n[NAPLAN Example {i + 1}]\n{content}\n"
                if examples:
                    return examples
        except Exception:
            pass
        return self._default_naplan_style(subject)

    def _default_naplan_style(self, subject: str | None) -> str:
        if subject and "numer" in subject.lower():
            return (
                "[NAPLAN Numeracy Style]\n"
                "- Place Value: 'What is the value of the digit 7 in 4735?' → 700\n"
                "- Fractions: 'Which fraction is equivalent to 1/2?' → 3/6\n"
                "- Money: 'Change from $20 after spending $13.50?' → $6.50\n"
                "- Measurement: 'A rectangle is 8cm long and 5cm wide. What is the area?' → 40cm²\n"
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
                        # 65536 prevents JSON truncation on large batches
                        max_output_tokens=65536,
                    ),
                )
                text = response.text
                logger.info(f"Gemini response: {len(text)} chars")
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

    def _parse_questions(
        self, response_text: str, grade: str, subject: str | None = None
    ) -> list[Question]:
        try:
            text = _clean_json_response(response_text)

            data = _salvage_partial_json(text)
            if not data:
                logger.error(f"No parseable JSON found. Preview: {response_text[:300]}")
                return []

            questions = []
            year = datetime.now().year
            class_name = _grade_to_class_name(grade)  # "grade5" → "Grade 5"

            for i, d in enumerate(data):
                if not str(d.get("question_text", "")).strip():
                    continue

                options = [d.get(f"option_{j}", "") for j in range(1, 5)]
                correct_idx = max(0, min(3, int(d.get("correct_option", 1)) - 1))
                correct_letter = ["A", "B", "C", "D"][correct_idx]

                # FIX: convert difficulty string → int
                diff_raw = str(d.get("difficulty", "medium")).lower().strip()
                diff_int = DIFFICULTY_STR_TO_INT.get(diff_raw, 2)

                # FIX: ensure sub_subject is specific, not a copy of subject
                raw_sub = d.get("sub_subject", "") or d.get("skill_tested", "")
                if _is_generic_sub_subject(raw_sub, subject or ""):
                    # Fall back to skill_tested, then naplan_strand, then topic hint
                    raw_sub = (
                        d.get("skill_tested", "")
                        or d.get("naplan_strand", "")
                        or "General"
                    )

                fields = dict(
                    # Required non-default Question fields
                    serial_number=i + 1,
                    class_name=class_name,
                    question_number=i + 1,
                    year=year,
                    grade=grade,
                    subject=d.get("subject", subject or ""),
                    sub_subject=raw_sub,
                    question_text=d.get("question_text", ""),
                    options=options,
                    correct_answer=correct_letter,
                    correct_option_index=correct_idx,
                    explanation=d.get("explanation", ""),
                    difficulty=diff_int,
                    pdf_source=d.get("pdf_source", ""),
                    page_number=int(d.get("page_number", 0)),
                    file_path=d.get("file_path", ""),
                    categories=f"{grade},{d.get('subject', subject or '')}",
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

    def _validate_batch_naplan(
        self, questions: list[Question], grade: str
    ) -> tuple[list[Question], int]:
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

Questions:
{json.dumps(questions_json, indent=2)}

Return ONLY a raw JSON array (no markdown, no backticks):
[
  {{"index": 0, "valid": true, "score": 8, "reason": "Clear and age-appropriate"}},
  ...
]"""

            response_text = self._call_gemini(prompt)
            if not response_text:
                return questions, 0

            text = _clean_json_response(response_text)
            results = _salvage_partial_json(text)
            if not results:
                return questions, 0

            valid_indices = {r["index"] for r in results if r.get("valid", True)}
            validated = [q for i, q in enumerate(questions) if i in valid_indices]
            rejected = len(questions) - len(validated)

            logger.info(f"NAPLAN validation: {len(validated)} passed, {rejected} rejected")
            return validated, rejected

        except Exception as e:
            logger.warning(f"NAPLAN validation failed, keeping all: {e}")
            return questions, 0

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
                    raw_diff = payload.get("difficulty", 2)
                    if isinstance(raw_diff, str):
                        raw_diff = DIFFICULTY_STR_TO_INT.get(raw_diff.lower(), 2)

                    fields = dict(
                        serial_number=payload.get(
                            "serial_number", payload.get("question_number", 0)
                        ),
                        class_name=payload.get(
                            "class_name", _grade_to_class_name(grade)
                        ),
                        question_number=payload.get("question_number", 0),
                        year=payload.get("year", datetime.now().year),
                        grade=payload.get("grade", grade),
                        subject=payload.get("subject", ""),
                        sub_subject=payload.get("sub_subject", ""),
                        question_text=payload.get("question_text", ""),
                        options=payload.get("options", []),
                        correct_answer=payload.get("correct_answer", "A"),
                        correct_option_index=payload.get("correct_option_index", 0),
                        explanation=payload.get("explanation", ""),
                        difficulty=raw_diff,
                    )
                    q = _make_question(
                        fields, fields["correct_answer"], fields["correct_option_index"]
                    )
                    if q:
                        questions.append(q)
            return questions
        except Exception as e:
            logger.warning(f"Could not load stored questions (first run?): {e}")
            return []

    def _filter_duplicates(
        self, questions: list[Question], existing: list[Question], allow_similar: bool
    ) -> tuple[list[Question], int]:
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

    def _retrieve_content_chunks(
        self,
        grade: str,
        subject: str | None = None,
        topics: list[str] | None = None,
        difficulty: str | None = None,
        limit: int = 20,
        collection_name: str | None = None,
    ):
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

    def _store_questions(self, questions: list[Question], grade: str) -> None:
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

    def check_capacity(
        self,
        grade: str,
        subject: str | None = None,
        topics: list[str] | None = None,
        difficulty: str | None = None,
    ) -> dict:
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