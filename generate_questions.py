"""
Main script for EduTech / Primary Math question generation.

Modes:
  --mode exam         → Full exam set
  --mode subtopic     → Sub-topic questions
  --mode major_topic  → 18 questions for one primary math major area
  --mode standard     → Custom number of questions (default)

Image generation (--generate-images):
  1. Generates image for each question and uploads to S3
  2. Downloads image from S3
  3. Compresses image to UNDER 20KB using Pillow
  4. Encodes as base64 (data:image/jpeg;base64,...)
  5. Stores on q.image_base64 — exporter embeds it in question_text HTML:
       <p>Question text<br /><img src="data:image/jpeg;base64,..."/></p>
"""

import argparse
import base64
import io
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
from generators import QuestionGeneratorV2
from generators.hybrid_image_generator import HybridImageGenerator
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

# ── Image compression limit ────────────────────────────────────────────────────
MAX_IMAGE_BYTES = 20 * 1024   # 20 KB


def _compress_image(image_bytes: bytes) -> bytes:
    """
    Compress image bytes to stay under MAX_IMAGE_BYTES (20 KB).
    Tries progressively smaller sizes and lower JPEG quality.
    Returns compressed JPEG bytes.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("Pillow not installed — skipping compression. Run: pip install Pillow")
        return image_bytes

    if len(image_bytes) <= MAX_IMAGE_BYTES:
        logger.info(f"Image already under 20KB ({len(image_bytes)} bytes) — no compression needed")
        return image_bytes

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Try progressively smaller scale + lower JPEG quality
    for scale in [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]:
        w = max(int(img.width * scale), 60)
        h = max(int(img.height * scale), 60)
        resized = img.resize((w, h), Image.LANCZOS)

        for quality in [85, 70, 55, 40, 30, 20, 15]:
            buf = io.BytesIO()
            resized.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            if len(data) <= MAX_IMAGE_BYTES:
                logger.info(
                    f"Compressed: {len(image_bytes)//1024}KB → "
                    f"{len(data)//1024}KB "
                    f"(scale={scale}, quality={quality})"
                )
                return data

    # Absolute last resort
    buf = io.BytesIO()
    img.resize((80, 60)).save(buf, format="JPEG", quality=15, optimize=True)
    data = buf.getvalue()
    logger.warning(f"Aggressively compressed to {len(data)} bytes")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EduTech Question Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  exam         Full exam set (Primary Math = 35 Qs across all 3 areas)
  subtopic     Questions for one sub-topic
  major_topic  18 questions for one primary math major area + difficulty
  standard     Custom number of questions

EXAMPLES:

  # 18 questions — Number & Algebra — Easy
  python generate_questions.py --grade Year3 --mode major_topic \\
      --major-topic number_and_algebra --difficulty easy --no-preview

  # With images — compressed to <20KB, base64 embedded in question_text
  python generate_questions.py --grade Year3 --mode major_topic \\
      --major-topic number_and_algebra --difficulty easy --no-preview --generate-images

  # Full 35-question Primary Math set
  python generate_questions.py --grade Year3 --subject primary_math --mode exam --no-preview
        """,
    )

    parser.add_argument("--grade",    required=True, help="Grade level e.g. Year3")
    parser.add_argument("--subject",  help="Subject e.g. Numeracy, primary_math")
    parser.add_argument(
        "--mode",
        choices=["exam", "subtopic", "major_topic", "standard"],
        default="standard",
    )
    parser.add_argument("--subtopic",  help="Sub-topic for subtopic mode")
    parser.add_argument(
        "--major-topic",
        choices=list(PRIMARY_MATH_MAJOR_TOPICS.keys()),
        help="Major topic for major_topic mode",
    )
    parser.add_argument("--num",       type=int,  help="Number of questions (standard mode)")
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "mixed"],
    )
    parser.add_argument("--topics",    nargs="+", help="Specific topics")
    parser.add_argument("--use-practice-test", action="store_true")
    parser.add_argument("--no-preview",         action="store_true")
    parser.add_argument(
        "--generate-images", action="store_true",
        help="Generate images, compress to <20KB, embed as base64 in question_text",
    )
    parser.add_argument(
        "--image-style", type=str, default="colorful educational diagram",
    )
    parser.add_argument("--output",  type=str,  help="Custom output path (.xlsx)")
    parser.add_argument("--model",   type=str,  default="gemini-2.5-flash")
    parser.add_argument("--dump",               action="store_true")
    parser.add_argument("--regenerate-images",  action="store_true")
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
        logger.info(f"Sources:    {'book + practice_test' if args.use_practice_test else 'book only'}")
        if args.mode == "major_topic":
            logger.info(f"Major topic: {args.major_topic or '(not set)'}")
            logger.info(f"Difficulty:  {args.difficulty or 'medium'}")
            logger.info(f"Target Qs:  18")
        logger.info(f"Images:     {'yes — compressed <20KB, base64 in question_text' if args.generate_images else 'no'}")
        logger.info(f"Model:      {args.model}")
        logger.info("=" * 80 + "\n")

        # ── Init clients ──────────────────────────────────────────────────────
        logger.info("Initializing clients...")
        embedding_client   = EmbeddingClient()
        qdrant_manager     = QdrantManager(embedding_dimension=embedding_client.dimension)
        question_generator = QuestionGeneratorV2(
            embedding_client=embedding_client,
            qdrant_manager=qdrant_manager,
            model_name=args.model,
        )

        # ── Dump mode ─────────────────────────────────────────────────────────
        if args.dump:
            questions = question_generator._load_all_stored_questions(args.grade, args.subject)
            if not questions:
                logger.error("No questions found in Qdrant.")
                sys.exit(1)
            _export_and_finish(questions, args, generate_images=False)
            return

        # ── Regenerate images ─────────────────────────────────────────────────
        if args.regenerate_images:
            questions = question_generator._load_all_stored_questions(args.grade, args.subject)
            if not questions:
                logger.error("No questions found in Qdrant.")
                sys.exit(1)
            _export_and_finish(questions, args, generate_images=True)
            return

        # ── Capacity check ────────────────────────────────────────────────────
        if args.mode == "standard" and not args.no_preview:
            capacity = question_generator.check_capacity(
                args.grade, args.subject, args.topics, args.difficulty
            )
            logger.info(f"Capacity: ~{capacity['estimated_remaining']} questions available")
            if capacity["available_chunks"] == 0:
                logger.error("No content found. Run ingest first.")
                sys.exit(1)
            if not args.num:
                remaining = capacity["estimated_remaining"]
                print(f"\nEstimated remaining: ~{remaining}")
                try:
                    args.num = int(input("How many questions? "))
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
            logger.error("--major-topic required for major_topic mode")
            sys.exit(1)

        # ── Generate questions ────────────────────────────────────────────────
        if args.mode == "exam":
            if not args.subject:
                logger.error("--subject required for exam mode")
                sys.exit(1)
            result = question_generator.generate_exam_set(
                grade=args.grade, subject=args.subject,
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
        else:
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
            sys.exit(1)

        logger.info(f"\nGenerated {len(questions)} questions")
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
    """
    Export questions to xlsx.

    If generate_images=True:
      1. Generate image via Gemini → upload to S3
      2. Download image from S3
      3. Compress to <20KB using Pillow
      4. Base64 encode → store on q.image_base64
      5. CSVExporter embeds it in question_text:
           <p>text<br /><img src="data:image/jpeg;base64,..."/></p>
    """
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    grade_str   = args.grade
    output_path = args.output or f"output/questions_{grade_str}_{timestamp}.xlsx"

    # Ensure .xlsx extension
    if not output_path.endswith(".xlsx"):
        output_path = Path(output_path).with_suffix(".xlsx").as_posix()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if generate_images:
        logger.info(f"\nGenerating images for {len(questions)} questions...")
        s3_uploader     = S3Uploader()
        image_generator = HybridImageGenerator(s3_uploader=s3_uploader)
        image_style     = getattr(args, "image_style", "colorful educational diagram")

        # generate_images_batch → {question_number: s3_url}
        image_urls = image_generator.generate_images_batch(
            questions=questions,
            image_style=image_style,
        )

        for q in questions:
            url = image_urls.get(q.question_number)
            if not url:
                continue

            # Attach S3 URL to question
            q.artifacts      = [url]
            q.question_image = url

            # ── Download → compress → base64 ─────────────────────────────────
            try:
                import urllib.request
                with urllib.request.urlopen(url, timeout=20) as resp:
                    raw_bytes = resp.read()

                logger.info(
                    f"Q{q.question_number}: downloaded {len(raw_bytes)//1024}KB from S3"
                )

                # Compress to under 20KB
                compressed = _compress_image(raw_bytes)

                # Detect format for data URI
                mime = "image/jpeg"
                if compressed[:4] == b"\x89PNG":
                    mime = "image/png"

                b64_str = base64.b64encode(compressed).decode("utf-8")
                q.image_base64 = f"data:{mime};base64,{b64_str}"

                logger.info(
                    f"Q{q.question_number}: base64 ready "
                    f"({len(compressed)//1024}KB compressed, "
                    f"{len(b64_str)//1024}KB base64 string)"
                )

            except Exception as e:
                logger.warning(
                    f"Q{q.question_number}: image download/compress failed — {e}. "
                    "S3 URL still saved in image_url column."
                )

        logger.info(f"Images processed: {len(image_urls)}/{len(questions)}")

    # ── Export to xlsx using exact template ───────────────────────────────────
    exporter = CSVExporter()
    exporter.export_to_xlsx(
        questions=questions,
        output_path=output_path,
        grade=grade_str,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_path = output_path.replace(".xlsx", "_summary.txt")
    mode = getattr(args, "mode", "standard")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Generation Summary\n{'=' * 40}\n")
        f.write(f"Grade:     {args.grade}\n")
        f.write(f"Subject:   {getattr(args, 'subject', 'all') or 'all'}\n")
        f.write(f"Mode:      {mode}\n")
        if mode == "major_topic":
            f.write(f"Major Topic: {getattr(args, 'major_topic', '')}\n")
            f.write(f"Difficulty:  {getattr(args, 'difficulty', 'medium')}\n")
        f.write(f"Images:    {'yes (<20KB, base64 in question_text)' if generate_images else 'no'}\n")
        f.write(f"Questions: {len(questions)}\n")
        f.write(f"File:      {output_path}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")

    logger.info(f"\n{'=' * 80}")
    logger.info("GENERATION COMPLETE")
    logger.info(f"{'=' * 80}")
    logger.info(f"Questions: {len(questions)}")
    logger.info(f"File:      {output_path}")
    logger.info(f"Summary:   {summary_path}")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()