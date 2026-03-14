"""
Main script for NAPLAN question generation.

Three modes:
  --mode exam      → Full exam set (Numeracy: 34q, Language Convention: 50q)
  --mode subtopic  → Sub-topic questions (count varies by difficulty)
  --mode standard  → Custom number of questions (default)
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Fix Windows Unicode encoding
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent / "src"))

from core import EmbeddingClient, QdrantManager
from core.models import Question
from generators import ImageGenerator, QuestionGeneratorV2
from generators.question_generator_v2 import EXAM_SET_CONFIG, SUBTOPIC_QUESTION_COUNT
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
        description="NAPLAN Question Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  exam      Full NAPLAN exam set (Numeracy=34q, Language Convention=50q)
  subtopic  Questions for one sub-topic (count by difficulty: easy=5, medium=8, hard=12, mixed=15)
  standard  Custom number of questions (default)

EXAMPLES:
  # Full numeracy exam set
  python generate_questions.py --grade grade5 --subject numeracy --mode exam

  # Full language convention exam set
  python generate_questions.py --grade grade5 --subject language_convention --mode exam

  # Sub-topic questions
  python generate_questions.py --grade grade5 --subject numeracy --mode subtopic --subtopic fractions --difficulty medium

  # Standard custom count
  python generate_questions.py --grade grade5 --subject numeracy --num 50 --mode standard

  # With images
  python generate_questions.py --grade grade5 --subject numeracy --mode exam --generate-images

  # Ingest NAPLAN reference papers first
  python ingest_content.py --grade grade5 --subject numeracy_naplan_reference --folder input/grade5/naplan_papers/
        """,
    )

    parser.add_argument("--grade", required=True, help="Grade level e.g. grade5")
    parser.add_argument("--subject", help="Subject e.g. numeracy, language_convention, reading")
    parser.add_argument(
        "--mode", choices=["exam", "subtopic", "standard"], default="standard",
        help="Generation mode (default: standard)",
    )
    parser.add_argument("--subtopic", help="Sub-topic for subtopic mode e.g. fractions")
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "mixed"],
        help="Difficulty (subtopic mode) or filter (standard mode)",
    )
    parser.add_argument("--num", "-n", type=int, help="Number of questions (standard mode)")
    parser.add_argument("--topics", nargs="+", help="Filter by topics (standard mode)")
    parser.add_argument("--no-preview", action="store_true", help="Skip capacity check")
    parser.add_argument("--generate-images", action="store_true", help="Generate images")
    parser.add_argument("--image-style", default="educational illustration")
    parser.add_argument("--output", "-o", type=Path, help="Output file path")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    try:
        logger.info("=" * 80)
        logger.info("NAPLAN QUESTION GENERATION SYSTEM")
        logger.info("=" * 80)
        logger.info(f"Grade:      {args.grade}")
        logger.info(f"Subject:    {args.subject or 'All'}")
        logger.info(f"Mode:       {args.mode.upper()}")
        if args.mode == "subtopic":
            logger.info(f"Subtopic:   {args.subtopic}")
            logger.info(f"Difficulty: {args.difficulty or 'mixed'}")
        elif args.mode == "exam":
            config = EXAM_SET_CONFIG.get((args.subject or "numeracy").lower(), {})
            logger.info(f"Target:     {config.get('total_questions', '?')} questions")
            logger.info(f"Distribution: {config.get('distribution', {})}")
        logger.info(f"Model:      {args.model}")
        logger.info("=" * 80 + "\n")

        # Initialize
        logger.info("Initializing clients...")
        embedding_client = EmbeddingClient()
        qdrant_manager = QdrantManager(embedding_dimension=embedding_client.dimension)
        question_generator = QuestionGeneratorV2(
            embedding_client=embedding_client,
            qdrant_manager=qdrant_manager,
            model_name=args.model,
        )

        # Capacity check (skip for exam/subtopic modes which have fixed counts)
        if args.mode == "standard" and not args.no_preview:
            capacity = question_generator.check_capacity(
                args.grade, args.subject, args.topics, args.difficulty
            )
            logger.info(f"Capacity: {capacity['estimated_remaining']} questions available")
            logger.info(f"Already generated: {capacity['already_generated']}")

            if capacity["available_chunks"] == 0:
                logger.error("No content found. Run ingest first:")
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

        # Validate subtopic mode args
        if args.mode == "subtopic" and not args.subtopic:
            logger.error("--subtopic required for subtopic mode")
            logger.error("Example: --subtopic fractions")
            sys.exit(1)

        # ── GENERATE ──────────────────────────────────────────
        if args.mode == "exam":
            if not args.subject:
                logger.error("--subject required for exam mode")
                logger.error("Available: numeracy, language_convention, reading")
                sys.exit(1)
            result = question_generator.generate_exam_set(
                grade=args.grade, subject=args.subject
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
            )

        questions = result["questions"]

        if not questions:
            logger.error("No questions generated. Check if PDFs are ingested.")
            logger.error(f"Run: python ingest_content.py --grade {args.grade}")
            sys.exit(1)

        # ── STATS ─────────────────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("GENERATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Mode:              {result.get('mode', args.mode).upper()}")
        logger.info(f"Status:            {result.get('status', '').upper()}")
        logger.info(f"Questions:         {len(questions)}")
        if result.get("target"):
            logger.info(f"Target:            {result['target']}")
        if result.get("duplicates_avoided"):
            logger.info(f"Duplicates avoided:{result['duplicates_avoided']}")
        if result.get("naplan_rejected"):
            logger.info(f"NAPLAN rejected:   {result['naplan_rejected']}")

        # Difficulty breakdown
        diff_counts = {}
        for q in questions:
            diff_counts[q.difficulty] = diff_counts.get(q.difficulty, 0) + 1
        logger.info(f"Difficulty breakdown: {diff_counts}")
        logger.info("=" * 80 + "\n")

        # ── IMAGES ────────────────────────────────────────────
        if args.generate_images:
            logger.info(f"Generating images for {len(questions)} questions...")
            try:
                s3_uploader = S3Uploader()
                if not s3_uploader.verify_bucket_access():
                    logger.error("Cannot access S3. Check AWS credentials.")
                    logger.warning("Skipping image generation.")
                else:
                    image_generator = ImageGenerator(s3_uploader=s3_uploader)
                    s3_urls = image_generator.generate_images_batch(
                        questions=questions, image_style=args.image_style
                    )
                    for q in questions:
                        if q.question_number in s3_urls:
                            url = s3_urls[q.question_number]
                            q.question_image = url
                            q.artifacts = q.artifacts or []
                            q.artifacts.append(url)
                            q.images_path.append(url)
                            q.images_tagged_count = len(q.images_path)
                    logger.info(f"Images generated: {len(s3_urls)}/{len(questions)}")
            except Exception as e:
                logger.warning(f"Image generation error: {e} - continuing without images")

        # ── EXPORT ────────────────────────────────────────────
        output_path = _build_output_path(args, result)
        csv_exporter = CSVExporter()
        out = csv_exporter.export_to_xlsx(
            questions=questions,
            output_path=str(output_path),
            grade=args.grade,
            template_path="ImportQuestionsTemplate.xlsx",
        )

        logger.info("=" * 80)
        logger.info("COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Questions: {len(questions)}")
        logger.info(f"Output:    {out}")
        logger.info("=" * 80 + "\n")

        # Sample preview
        logger.info("Sample Questions:")
        for i, q in enumerate(questions[:3], 1):
            logger.info(f"\n{i}. [{q.difficulty.upper()}] {q.question_text}")
            for j, opt in enumerate(q.options):
                marker = ">" if j == q.correct_option_index else " "
                logger.info(f"   [{marker}] {opt}")
        if len(questions) > 3:
            logger.info(f"\n... and {len(questions) - 3} more questions")

    except KeyboardInterrupt:
        logger.warning("Cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


def _build_output_path(args, result: dict) -> Path:
    if args.output:
        return args.output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = result.get("mode", args.mode)
    subject = (args.subject or "all").replace(" ", "_")
    if mode == "subtopic" and args.subtopic:
        name = f"questions_{args.grade}_{subject}_{args.subtopic}_{args.difficulty or 'mixed'}_{timestamp}"
    elif mode == "exam":
        name = f"exam_set_{args.grade}_{subject}_{timestamp}"
    else:
        name = f"questions_{args.grade}_{subject}_{timestamp}"
    return output_dir / f"{name}.xlsx"


if __name__ == "__main__":
    main()