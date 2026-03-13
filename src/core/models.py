"""Data models for the question ingestion system."""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Question:
    """
    Represents a single question with all metadata.

    This schema includes LMS bulk-upload fields and enriched metadata for Qdrant storage.

    LMS Required fields (for bulk upload):
        serial_number, year, class_name, grade, subject, sub_subject, question_text,
        type, difficulty, option1-6, answer, answer_index, content, explanation,
        hints, passage, citation, settings, question_tags, question_image,
        file_path, pdf_source, page_number, artifacts, last_generated

    Enriched metadata fields (stored in Qdrant):
        ocr_confidence, validation_status, reasoning_steps, token_count,
        question_context, question_complexity, references, related_questions,
        images_tagged_count, images_path, artifacts_path, tags, processing_status,
        issues_found, source_chunk
    """

    # === LMS BULK-UPLOAD FIELDS ===
    # Identifiers
    serial_number: int  # Unique numeric ID (question_number)
    year: int
    class_name: str  # e.g., "Grade 5", "Year 3"
    grade: str  # e.g., "grade5"
    subject: str
    sub_subject: str

    # Question content
    question_text: str  # Plain text version
    content: str = ""  # HTML-formatted question with <p>, <img>, <math>, etc.

    # Question type and difficulty
    type: str = "MCQ_SA"  # MCQ_SA, MCQ_MA, YN, TF, ESSAY, etc.
    difficulty: int = 0  # 0-5 difficulty level

    # Options (up to 6)
    option1: str = ""
    option2: str = ""
    option3: str = ""
    option4: str = ""
    option5: str = ""
    option6: str = ""

    # Answer
    answer: str = ""  # Correct answer text
    answer_index: int = 0  # Zero-based index for options

    # Rich content fields
    explanation: str = ""  # HTML explanation with <p>, <math>, <img>, etc.
    hints: str = ""  # Comma-separated hints with inline HTML
    passage: str = ""  # HTML passage/context with <p>, formatting, <img>, etc.
    citation: str = ""  # Source reference

    # Metadata
    settings: str = ""  # Platform-specific settings (for ESSAY/AUDIO/VIDEO)
    question_tags: str = ""  # Comma-separated tags
    question_image: str = ""  # Filename for rendered question image
    file_path: str = ""  # Path to source PDF file
    pdf_source: str = ""  # Original PDF filename
    page_number: int = 0  # Page in source PDF

    artifacts: list[str] = field(default_factory=list)  # Generated image paths
    last_generated: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )  # ISO timestamp

    # Backward compatibility - kept for internal use
    question_number: int = 0  # Maps to serial_number
    options: list[str] = field(default_factory=list)  # Internal list form

    # === ENRICHED METADATA FIELDS (Advanced tracking) ===
    # Quality metrics
    ocr_confidence: float = 0.95  # Confidence score for OCR extraction (0-1)
    validation_status: str = "validated"  # Status: pending, validated, flagged, rejected

    # Generation metadata
    reasoning_steps: list[str] = field(default_factory=list)  # AI reasoning process steps
    token_count: int = 0  # Tokens used in generation

    # Content analysis
    question_context: str = ""  # Brief context about the question
    question_complexity: str = "medium"  # Complexity: easy, medium, hard, advanced

    # Relationships
    references: list[str] = field(default_factory=list)  # Referenced concepts/pages
    related_questions: list[int] = field(default_factory=list)  # Related question numbers

    # Image tracking
    images_tagged_count: int = 0  # Number of images associated
    images_path: list[str] = field(default_factory=list)  # Paths to all associated images
    artifacts_path: list[str] = field(default_factory=list)  # Paths to artifact files

    # Classification & tracking
    tags: list[str] = field(default_factory=list)  # Custom tags for categorization
    processing_status: str = "success"  # Status: success, partial, failed
    issues_found: list[str] = field(default_factory=list)  # Any issues during generation

    # Source tracking (for anti-hallucination)
    source_chunk: int = 0  # Index of content chunk this question was generated from

    def __post_init__(self):
        """Post-initialization to sync fields and populate defaults."""
        # Sync serial_number with question_number
        if not self.serial_number and self.question_number:
            self.serial_number = self.question_number
        elif not self.question_number and self.serial_number:
            self.question_number = self.serial_number

        # Populate individual option fields from options list if not set
        if self.options and not self.option1:
            self.option1 = self.options[0] if len(self.options) > 0 else ""
            self.option2 = self.options[1] if len(self.options) > 1 else ""
            self.option3 = self.options[2] if len(self.options) > 2 else ""
            self.option4 = self.options[3] if len(self.options) > 3 else ""
            self.option5 = self.options[4] if len(self.options) > 4 else ""
            self.option6 = self.options[5] if len(self.options) > 5 else ""

        # Populate options list from individual fields if not set
        if not self.options and self.option1:
            self.options = [
                opt
                for opt in [
                    self.option1,
                    self.option2,
                    self.option3,
                    self.option4,
                    self.option5,
                    self.option6,
                ]
                if opt
            ]

        # Generate HTML content from question_text if not set
        if not self.content and self.question_text:
            self.content = f"<p>{self.question_text}</p>"

        # Generate question_tags from metadata if not set
        if not self.question_tags:
            tag_parts = []
            if self.class_name:
                tag_parts.append(self.class_name)
            if self.subject:
                tag_parts.append(self.subject)
            if self.sub_subject:
                tag_parts.append(self.sub_subject)
            self.question_tags = ",".join(tag_parts)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Question to a dictionary.

        Returns:
            Dictionary representation with 'class' field renamed from 'class_name'.
        """
        data = asdict(self)
        # Rename class_name to class for JSON output
        data["class"] = data.pop("class_name")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Question":
        """
        Create a Question instance from a dictionary.

        Args:
            data: Dictionary containing question data. Can use either 'class' or 'class_name'.

        Returns:
            Question instance.
        """
        # Handle both 'class' and 'class_name' keys
        if "class" in data and "class_name" not in data:
            data["class_name"] = data.pop("class")

        # Filter out any extra keys that aren't in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)


@dataclass
class PDFMetadata:
    """Metadata extracted from the PDF file path."""

    grade: str  # e.g., "grade5"
    subject: str  # e.g., "numeracy"
    pdf_filename: str  # e.g., "example.pdf"
    pdf_path: str  # Full path to the PDF

    @property
    def qdrant_collection_name(self) -> str:
        """
        Generate the Qdrant collection name for this grade.

        Returns:
            Collection name in format: grade{N}_questions
        """
        return f"{self.grade}_questions"
