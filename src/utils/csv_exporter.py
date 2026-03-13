"""CSV export functionality for questions."""

import csv
import json
import logging
from pathlib import Path
from typing import Any

from src.core.models import Question

logger = logging.getLogger(__name__)


class CSVExporter:
    """Export questions to CSV format for LMS bulk upload."""

    # CSV column headers (in LMS bulk-upload order)
    CSV_HEADERS = [
        "serial_number",
        "year",
        "class",
        "grade",
        "subject",
        "sub_subject",
        "question_text",
        "type",
        "difficulty",
        "option1",
        "option2",
        "option3",
        "option4",
        "option5",
        "option6",
        "answer",
        "answer_index",
        "content",
        "explanation",
        "hints",
        "passage",
        "citation",
        "settings",
        "question_tags",
        "question_image",
        "file_path",
        "pdf_source",
        "page_number",
        "artifacts",
        "last_generated",
    ]

    @staticmethod
    def export_to_csv(
        questions: list[Question], output_path: Path | str, include_headers: bool = True
    ) -> None:
        """
        Export questions to CSV file.

        Args:
            questions: List of Question objects to export.
            output_path: Path to the output CSV file.
            include_headers: Whether to include header row.

        Raises:
            Exception: If export fails.
        """
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSVExporter.CSV_HEADERS)

                if include_headers:
                    writer.writeheader()

                for question in questions:
                    row = CSVExporter._question_to_csv_row(question)
                    writer.writerow(row)

            logger.info(f"Exported {len(questions)} questions to {output_path}")

        except Exception as e:
            logger.exception(f"Error exporting to CSV: {e}")
            raise

    @staticmethod
    def _question_to_csv_row(question: Question) -> dict[str, Any]:
        """
        Convert a Question object to a CSV row dictionary for LMS bulk upload.

        Args:
            question: Question object.

        Returns:
            Dictionary with CSV column names as keys matching LMS format.
        """
        # Handle artifacts - convert list to JSON string
        artifacts_str = json.dumps(question.artifacts) if question.artifacts else "[]"

        # Create row with all LMS fields
        return {
            "serial_number": question.serial_number,
            "year": question.year,
            "class": question.class_name,
            "grade": question.grade,
            "subject": question.subject,
            "sub_subject": question.sub_subject,
            "question_text": question.question_text,
            "type": question.type,
            "difficulty": question.difficulty,
            "option1": question.option1,
            "option2": question.option2,
            "option3": question.option3,
            "option4": question.option4,
            "option5": question.option5,
            "option6": question.option6,
            "answer": question.answer,
            "answer_index": question.answer_index,
            "content": question.content,
            "explanation": question.explanation,
            "hints": question.hints,
            "passage": question.passage,
            "citation": question.citation,
            "settings": question.settings,
            "question_tags": question.question_tags,
            "question_image": question.question_image,
            "file_path": question.file_path,
            "pdf_source": question.pdf_source,
            "page_number": question.page_number,
            "artifacts": artifacts_str,
            "last_generated": question.last_generated,
        }

    @staticmethod
    def append_to_csv(questions: list[Question], output_path: Path | str) -> None:
        """
        Append questions to an existing CSV file.

        Args:
            questions: List of Question objects to append.
            output_path: Path to the CSV file.

        Raises:
            Exception: If append fails.
        """
        output_path = Path(output_path)

        # Check if file exists
        file_exists = output_path.exists()

        try:
            with open(output_path, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSVExporter.CSV_HEADERS)

                # Write headers only if file is new
                if not file_exists:
                    writer.writeheader()

                for question in questions:
                    row = CSVExporter._question_to_csv_row(question)
                    writer.writerow(row)

            logger.info(f"Appended {len(questions)} questions to {output_path}")

        except Exception as e:
            logger.exception(f"Error appending to CSV: {e}")
            raise

    @staticmethod
    def export_with_summary(
        questions: list[Question], output_dir: Path | str, base_filename: str = "questions"
    ) -> dict[str, Path]:
        """
        Export questions to CSV and create a summary file.

        Args:
            questions: List of Question objects.
            output_dir: Directory for output files.
            base_filename: Base name for output files.

        Returns:
            Dictionary with paths to created files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Export questions to CSV
        csv_path = output_dir / f"{base_filename}.csv"
        CSVExporter.export_to_csv(questions, csv_path)

        # Create summary
        summary_path = output_dir / f"{base_filename}_summary.txt"
        CSVExporter._create_summary(questions, summary_path)

        return {"csv": csv_path, "summary": summary_path}

    @staticmethod
    def _create_summary(questions: list[Question], summary_path: Path) -> None:
        """Create a summary file for the generated questions."""
        try:
            # Collect statistics
            total_questions = len(questions)
            subjects = {q.subject for q in questions}
            sub_subjects = {q.sub_subject for q in questions}
            grades = {q.grade for q in questions}
            pdf_sources = {q.pdf_source for q in questions}

            # Count questions with artifacts
            questions_with_artifacts = sum(1 for q in questions if q.artifacts)

            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("QUESTION GENERATION SUMMARY\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Total Questions Generated: {total_questions}\n\n")

                f.write(f"Grades: {', '.join(sorted(grades))}\n")
                f.write(f"Subjects: {', '.join(sorted(subjects))}\n")
                f.write(f"Sub-subjects ({len(sub_subjects)}): {', '.join(sorted(sub_subjects))}\n")
                f.write(f"Source PDFs: {', '.join(sorted(pdf_sources))}\n\n")

                f.write(f"Questions with Visual Artifacts: {questions_with_artifacts}\n")
                f.write(
                    f"Questions without Artifacts: {total_questions - questions_with_artifacts}\n\n"
                )

                # Question number range
                if questions:
                    f.write(
                        f"Question Numbers: {min(q.question_number for q in questions)} - {max(q.question_number for q in questions)}\n"
                    )
                    f.write(f"Generation Time: {questions[0].last_generated}\n")

                f.write("\n" + "=" * 80 + "\n")

            logger.info(f"Created summary at {summary_path}")

        except Exception as e:
            logger.exception(f"Error creating summary: {e}")
