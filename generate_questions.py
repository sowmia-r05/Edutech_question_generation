"""
Main script for NAPLAN / Primary Math question generation.

Four modes:
  --mode exam         → Full exam set (Numeracy: 34q, Language Convention: 50q,
                        Primary Math: 35q across all 3 major areas)
  --mode subtopic     → Sub-topic questions (count varies by difficulty)
  --mode major_topic  → 18 questions for one primary math major area
                        at a chosen difficulty level
  --mode standard     → Custom number of questions (default)

NEW FLAG:
  --use-practice-test  → Combines book content (grade3_content) with practice test
                         examples (grade3_practice_test) for richer, style-matched
                         question generation. Requires both collections to be ingested.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from core import EmbeddingClient, QdrantManager
from core.models import Question
from generators import ImageGenerator, QuestionGeneratorV2
from generators.question_generator_v2 import (
    EXAM_SET_CONFIG,
    PRIMARY_MATH_MAJOR_TOPICS,
    SUBTOPIC_QUESTION_COUNT,
)
from utils import CSVExporter, S3Uploader

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/generate_questions.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EduTech Question Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  exam         Full exam set
                 Numeracy          = 34 questions
                 Language Conv.    = 50 questions
                 Primary Math      = 35 questions (all 3 major areas)
  subtopic     Questions for one sub-topic (easy=5, medium=8, hard=12, mixed=15)
  major_topic  18 questions for one primary math major area + difficulty
  standard     Custom number of questions (default)

EXAMPLES:

  # Standard — books only
  python generate_questions.py --grade grade3 --subject numeracy --num 20

  # Combined — books + practice test style (RECOMMENDED)
  python generate_questions.py --grade grade3 --subject numeracy --num 20 --use-practice-test

  # Full NAPLAN exam set
  python generate_questions.py --grade grade3 --subject numeracy --mode exam --use-practice-test

  # Full Primary Math set (35 Qs, all 3 areas)
  python generate_questions.py --grade grade3 --subject primary_math --mode exam

  # 18 questions — Number & Algebra — Easy
  python generate_questions.py --grade grade3 --mode major_topic \\
      --major-topic number_and_algebra --difficulty easy

  # 18 questions — Measurement & Geometry — Medium
  python generate_questions.py --grade grade3 --mode major_topic \\
      --major-topic measurement_and_geometry --difficulty medium

  # 18 questions — Statistics & Probability — Hard
  python generate_questions.py --grade grade3 --mode major_topic \\
      --major-topic statistics_and_probability --difficulty hard

  # Sub-topic with practice test style
  python generate_questions.py --grade grade3 --subject numeracy --mode subtopic \\
      --subtopic "Statistics and Data" --difficulty medium --use-practice-test

  # With images
  python generate_questions.py --grade grade3 --subject numeracy --num 10 \\
      --use-practice-test --generate-images

  # Ingest order (run these first):
  #   python ingest_content.py --grade grade3 --subject numeracy
  #   python ingest_content.py --grade grade3 --subject practice_test
        """,
    )

    parser.add_argument("--grade", required=True, help="Grade level e.g. grade3")
    parser.add_argument("--subject", help="Subject e.g. numeracy, language_convention, primary_math")
    parser.add_argument(
        "--mode",
        choices=["exam", "subtopic", "major_topic", "standard"],
        default="standard",
        help="Generation mode (default: standard)",
    )
    parser.add_argument("--subtopic", help="Sub-topic for subtopic mode e.g. fractions")
    parser.add_argument(
        "--major-topic",
        choices=list(PRIMARY_MATH_MAJOR_TOPICS.keys()),
        help=(
            "Major topic for major_topic mode. "
            "Choices: number_and_algebra | measurement_and_geometry | statistics_and_probability"
        ),
    )
    parser.add_argument("--num", type=int, help="Number of questions (standard mode)")
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard", "mixed"],
        help="Difficulty filter",
    )
    parser.add_argument(
        "--topics", nargs="+",
        help='Specific topics e.g. --topics "Addition" "Place Value"',
    )
    parser.add_argument(
        "--use-practice-test", action="store_true",
        help=(
            "Combine book content (grade_content) with practice test examples "
            "(grade_practice_test) for style-matched question generation. "
            "Requires both collections to be ingested."
        ),
    )
    parser.add_argument(
        "--no-preview", action="store_true",
        help="Skip capacity preview and prompt (requires --num)",
    )
    parser.add_argument(
        "--generate-images", action="store_true",
        help="Generate contextual images for each question",
    )
    parser.add_argument(
        "--image-style", type=str, default="colorful educational diagram",
        help="Style hint for image generation",
    )
    parser.add_argument(
        "--output", type=str,
        help="Custom output CSV path (default: output/questions_<grade>_<timestamp>.csv)",
    )
    parser.add_argument(
        "--model", type=str, default="gemini-2.5-flash",
        help="Gemini model for question generation (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--dump", action="store_true",
        help="Dump all existing questions from Qdrant to CSV without generating new ones",
    )
    parser.add_argument(
        "--regenerate-images", action="store_true",
        help="Regenerate images for existing questions",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    try:
        logger.info("=" * 80)
        logger.info("EDUTECH QUESTION GENERATION SYSTEM")
        logger.info("=" * 80)
        logger.info(f"Grade:      {args.grade}")
        logger.info(f"Subject:    {args.subject or 'All'}")
        logger.info(f"Mode:       {args.mode}")
        logger.info(
            f"Sources:    "
            f"{'book + practice_test (combined)' if args.use_practice_test else 'book only'}"
        )
        if args.mode == "exam":
            config = EXAM_SET_CONFIG.get(
                (args.subject or "").lower().replace(" ", "_"), {}
            )
            logger.info(f"Target Qs:  {config.get('total_questions', '?')}")
            logger.info(f"Distribution: {config.get('distribution', {})}")
        if args.mode == "major_topic":
            logger.info(f"Major topic: {args.major_topic or '(not set)'}")
            logger.info(f"Difficulty:  {args.difficulty or 'medium'}")
            logger.info(f"Target Qs:  18")
        logger.info(f"Model:      {args.model}")
        logger.info("=" * 80 + "\n")

        # ── Initialise clients ────────────────────────────────────────────────
        logger.info("Initializing clients...")
        embedding_client = EmbeddingClient()
        qdrant_manager = QdrantManager(embedding_dimension=embedding_client.dimension)
        question_generator = QuestionGeneratorV2(
            embedding_client=embedding_client,
            qdrant_manager=qdrant_manager,
            model_name=args.model,
        )

        # ── Dump mode ─────────────────────────────────────────────────────────
        if args.dump:
            logger.info("DUMP MODE: exporting existing questions to CSV")
            questions = question_generator._load_all_stored_questions(
                args.grade, args.subject
            )
            if not questions:
                logger.error("No questions found in Qdrant.")
                sys.exit(1)
            logger.info(f"Found {len(questions)} questions")
            _export_and_finish(questions, args, generate_images=False)
            return

        # ── Regenerate images mode ────────────────────────────────────────────
        if args.regenerate_images:
            logger.info("REGENERATE IMAGES MODE")
            questions = question_generator._load_all_stored_questions(
                args.grade, args.subject
            )
            if not questions:
                logger.error("No questions found in Qdrant.")
                sys.exit(1)
            logger.info(f"Found {len(questions)} questions — regenerating images...")
            _export_and_finish(questions, args, generate_images=True)
            return

        # ── Capacity check (standard mode only) ──────────────────────────────
        if args.mode == "standard" and not args.no_preview:
            capacity = question_generator.check_capacity(
                args.grade, args.subject, args.topics, args.difficulty
            )
            logger.info(
                f"Capacity:          ~{capacity['estimated_remaining']} questions available"
            )
            logger.info(f"Already generated: {capacity['already_generated']}")

            if args.use_practice_test:
                pt_col = f"{args.grade}_practice_test"
                try:
                    info = qdrant_manager.get_collection_info(pt_col)
                    logger.info(
                        f"Practice test collection: {info.get('points_count', 0)} chunks"
                    )
                except Exception:
                    logger.warning(
                        f"Practice test collection '{pt_col}' not found. "
                        "Run: python ingest_content.py --grade grade3 --subject practice_test"
                    )

            if capacity["available_chunks"] == 0:
                logger.error("No book content found. Run ingest first:")
                logger.error(f"  python ingest_content.py --grade {args.grade}")
                sys.exit(1)

            if not args.num:
                remaining = capacity["estimated_remaining"]
                print(f"\nEstimated remaining unique questions: ~{remaining}")
                try:
                    args.num = int(input("How many questions to generate? "))
                    if args.num <= 0:
                        sys.exit(1)
                except (ValueError, KeyboardInterrupt):
                    sys.exit(1)

        elif args.mode == "standard" and not args.num:
            logger.error("--num required in standard mode with --no-preview")
            sys.exit(1)

        if args.mode == "subtopic" and not args.subtopic:
            logger.error("--subtopic required for subtopic mode")
            sys.exit(1)

        if args.mode == "major_topic" and not args.major_topic:
            logger.error(
                "--major-topic required for major_topic mode. "
                "Choices: number_and_algebra | measurement_and_geometry | "
                "statistics_and_probability"
            )
            sys.exit(1)

        # ── GENERATE ──────────────────────────────────────────────────────────
        if args.mode == "exam":
            if not args.subject:
                logger.error("--subject required for exam mode")
                sys.exit(1)
            result = question_generator.generate_exam_set(
                grade=args.grade,
                subject=args.subject,
            )

        elif args.mode == "major_topic":
            result = question_generator.generate_major_topic_questions(
                grade=args.grade,
                major_topic=args.major_topic,
                difficulty=args.difficulty or "medium",
            )

        elif args.mode == "subtopic":
            result = question_generator.generate_subtopic_questions(
                grade=args.grade,
                subject=args.subject or "",
                subtopic=args.subtopic,
                difficulty=args.difficulty or "mixed",
            )

        else:  # standard
            result = question_generator.generate_questions(
                grade=args.grade,
                subject=args.subject,
                topics=args.topics,
                difficulty=args.difficulty,
                num_questions=args.num,
                content_chunks_limit=20,
                use_practice_test=args.use_practice_test,
            )

        questions = result["questions"]

        if not questions:
            logger.error("No questions generated.")
            logger.error(
                "Ensure PDFs are ingested:\n"
                f"  python ingest_content.py --grade {args.grade} --subject numeracy\n"
                + (
                    f"  python ingest_content.py --grade {args.grade} --subject practice_test"
                    if args.use_practice_test else ""
                )
            )
            sys.exit(1)

        sources = result.get("sources_used", ["book_content"])
        logger.info(f"\nGenerated {len(questions)} questions  (sources: {', '.join(sources)})")
        logger.info(f"Duplicates avoided: {result.get('duplicates_avoided', 0)}")
        logger.info(f"NAPLAN rejected:    {result.get('naplan_rejected', 0)}")
        logger.info(f"Status:             {result.get('status', 'unknown')}")

        _export_and_finish(questions, args, generate_images=args.generate_images)

    except KeyboardInterrupt:
        logger.warning("Generation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


def _export_and_finish(questions: list, args, generate_images: bool) -> None:
    """Export questions to CSV and optionally generate images."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    grade_str = args.grade
    output_path = args.output or f"output/questions_{grade_str}_{timestamp}.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if generate_images:
        logger.info(f"\nGenerating images for {len(questions)} questions...")
        s3_uploader = S3Uploader()
        image_generator = ImageGenerator(s3_uploader=s3_uploader)
        questions = image_generator.generate_images_for_questions(
            questions=questions,
            grade=args.grade,
            subject=args.subject or "primary_math",
            style=getattr(args, "image_style", "colorful educational diagram"),
        )

    exporter = CSVExporter()
    exporter.export(questions=questions, output_path=output_path)

    summary_path = output_path.replace(".csv", "_summary.txt")
    mode = getattr(args, "mode", "standard")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Generation Summary\n{'=' * 40}\n")
        f.write(f"Grade:     {args.grade}\n")
        f.write(f"Subject:   {getattr(args, 'subject', 'all') or 'all'}\n")
        f.write(f"Mode:      {mode}\n")
        if mode == "major_topic":
            f.write(f"Major Topic: {getattr(args, 'major_topic', '')}\n")
            f.write(f"Difficulty:  {getattr(args, 'difficulty', 'medium')}\n")
        f.write(
            f"Sources:   "
            f"{'book + practice_test' if getattr(args, 'use_practice_test', False) else 'book only'}\n"
        )
        f.write(f"Questions: {len(questions)}\n")
        f.write(f"CSV:       {output_path}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")

    logger.info(f"\n{'=' * 80}")
    logger.info("GENERATION COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Questions: {len(questions)}")
    logger.info(f"CSV file:  {output_path}")
    logger.info(f"Summary:   {summary_path}")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()