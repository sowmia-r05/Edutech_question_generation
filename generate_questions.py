"""Main script to generate questions from stored PDF content with capacity preview."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core import EmbeddingClient, QdrantManager
from core.models import Question
from generators import ImageGenerator, QuestionGeneratorV2
from utils import CSVExporter, S3Uploader

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/generate_questions.log"),
    ],
)

logger = logging.getLogger(__name__)


def check_capacity(
    question_generator: QuestionGeneratorV2,
    grade: str,
    subject: str | None,
    topics: list[str] | None,
    difficulty: str | None,
) -> dict:
    """
    Check generation capacity with smart sampling to estimate realistic availability.

    Returns:
        Dictionary with capacity information
    """
    logger.info("\n" + "=" * 80)
    logger.info("SMART CAPACITY CHECK")
    logger.info("=" * 80)

    # Get content chunks available
    content_collection = f"{grade}_content"
    chunks = question_generator._retrieve_content_chunks(
        collection_name=content_collection,
        subject=subject,
        topics=topics,
        difficulty=difficulty,
        limit=50,  # Get more for better estimation
    )

    if not chunks:
        return {
            "available": 0,
            "existing": 0,
            "capacity": 0,
            "chunks": 0,
            "realistic_available": 0,
            "status": "no_content",
        }

    # Get existing questions
    questions_collection = f"{grade}_questions"
    existing_questions = question_generator._get_existing_questions(
        questions_collection, subject, topics
    )

    logger.info(f"📊 Content chunks found: {len(chunks)}")
    logger.info(f"📝 Already generated: {len(existing_questions)} questions")

    # Smart estimation: Generate a small sample to test actual success rate
    if len(existing_questions) > 0:
        logger.info("\n🔍 Analyzing content uniqueness with test sample...")
        try:
            # Generate 5 test questions to check duplicate rate
            test_result = question_generator.generate_questions(
                grade=grade,
                subject=subject,
                topics=topics,
                difficulty=difficulty,
                num_questions=5,
                content_chunks_limit=10,
                allow_similar=False,
            )

            test_generated = len(test_result.get("questions", []))
            test_duplicates = test_result.get("duplicates_avoided", 0)

            # Calculate success rate (questions that weren't duplicates)
            if test_generated + test_duplicates > 0:
                success_rate = test_generated / (test_generated + test_duplicates)
                logger.info(
                    f"   Sample: {test_generated} unique / {test_generated + test_duplicates} attempted "
                    f"(success rate: {success_rate:.1%})"
                )
            else:
                success_rate = 0.5  # Default if test failed
                logger.info("   Could not determine success rate, using default estimate")

        except Exception as e:
            logger.warning(f"Test sample failed: {e}. Using default estimates.")
            success_rate = 0.7  # Conservative estimate

    else:
        # No existing questions, so success rate should be high
        success_rate = 0.9
        logger.info("   No existing questions - high success rate expected")

    # Calculate estimates
    theoretical_capacity = len(chunks) * 4
    realistic_capacity = int(theoretical_capacity * success_rate)
    realistic_available = max(0, realistic_capacity - len(existing_questions))

    logger.info("\n📈 CAPACITY ESTIMATES:")
    logger.info(f"   Theoretical maximum: ~{theoretical_capacity} questions")
    logger.info(f"   Realistic estimate: ~{realistic_available} NEW unique questions")
    logger.info(f"   (Based on {success_rate:.1%} uniqueness rate)")

    if realistic_available < 10:
        logger.warning("\n⚠️  LOW CAPACITY WARNING")
        logger.warning(f"   Only ~{realistic_available} unique questions can be generated")
        logger.warning("   💡 Consider: Ingesting more PDFs or removing filters")

    logger.info("=" * 80 + "\n")

    return {
        "available": theoretical_capacity - len(existing_questions),  # Optimistic
        "realistic_available": realistic_available,  # Realistic
        "existing": len(existing_questions),
        "capacity": theoretical_capacity,
        "chunks": len(chunks),
        "success_rate": success_rate,
        "status": "ok" if realistic_available > 0 else "exhausted",
    }


def main() -> None:
    """Main entry point for question generation."""
    parser = argparse.ArgumentParser(
        description="Generate questions from ingested PDF content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check capacity and generate for grade5
  python generate_questions.py --grade grade5

  # Generate specific number without preview
  python generate_questions.py --grade grade5 --num 10 --no-preview

  # Generate with images
  python generate_questions.py --grade grade5 --num 10 --generate-images

  # Filter by subject and difficulty
  python generate_questions.py --grade grade5 --subject numeracy --difficulty hard
        """,
    )

    parser.add_argument(
        "--grade", type=str, required=True, help="Grade level (e.g., grade5, grade4)"
    )
    parser.add_argument("--subject", type=str, help="Filter by subject (e.g., numeracy, english)")
    parser.add_argument("--topics", type=str, nargs="+", help="Filter by specific topics")
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        help="Filter by difficulty level",
    )
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        help="Number of questions to generate (if not provided, will prompt)",
    )
    parser.add_argument(
        "--no-preview", action="store_true", help="Skip capacity preview and generate directly"
    )
    parser.add_argument(
        "--generate-images", action="store_true", help="Generate contextual images for questions"
    )
    parser.add_argument(
        "--image-style", type=str, default="educational diagram", help="Style for generated images"
    )
    parser.add_argument("--output", "-o", type=Path, help="Output CSV file path")
    parser.add_argument(
        "--gemini-model", type=str, default="gemini-2.5-flash", help="Gemini model to use"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump all existing questions from Qdrant to CSV (skips generation)",
    )
    parser.add_argument(
        "--regenerate-images",
        action="store_true",
        help="Regenerate images for existing questions and update with S3 URLs",
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    try:
        logger.info("=" * 80)
        logger.info("QUESTION GENERATION SYSTEM")
        logger.info("=" * 80)
        logger.info(f"Grade: {args.grade}")
        logger.info(f"Subject: {args.subject or 'All'}")
        logger.info(f"Topics: {args.topics or 'All'}")
        logger.info(f"Difficulty: {args.difficulty or 'Mixed'}")
        logger.info(f"Model: {args.gemini_model}")
        logger.info("=" * 80 + "\n")

        # Initialize clients
        logger.info("Initializing clients...")
        embedding_client = EmbeddingClient()
        qdrant_manager = QdrantManager(embedding_dimension=embedding_client.dimension)

        # Handle --dump flag: export existing questions without generation
        if args.dump:
            logger.info("\n" + "=" * 80)
            logger.info("DUMPING EXISTING QUESTIONS FROM QDRANT")
            logger.info("=" * 80 + "\n")

            questions_collection = f"{args.grade}_questions"
            logger.info(f"Fetching questions from collection: {questions_collection}")

            # Search with empty query to get all questions
            query_embedding = embedding_client.embed_query("questions")
            results = qdrant_manager.search_questions(
                collection_name=questions_collection,
                query_embedding=query_embedding,
                limit=10000,  # Get all questions
            )

            if not results:
                logger.error(f"❌ No questions found in collection {questions_collection}")
                sys.exit(1)

            # Convert to Question objects
            questions = []
            for result in results:
                try:
                    q_data = result["payload"]
                    if q_data.get("type") == "question":
                        questions.append(Question.from_dict(q_data))
                except Exception as e:
                    logger.warning(f"Skipping invalid question: {e}")
                    continue

            logger.info(f"Found {len(questions)} questions")

            # Apply filters if provided
            filtered_questions = questions
            if args.subject:
                filtered_questions = [q for q in filtered_questions if q.subject == args.subject]
                logger.info(
                    f"Filtered by subject '{args.subject}': {len(filtered_questions)} questions"
                )

            if args.topics:
                filtered_questions = [
                    q
                    for q in filtered_questions
                    if any(topic in q.sub_subject for topic in args.topics)
                ]
                logger.info(
                    f"Filtered by topics {args.topics}: {len(filtered_questions)} questions"
                )

            if args.difficulty:
                # Map difficulty string to numeric range
                difficulty_map = {"easy": (0, 2), "medium": (2, 4), "hard": (4, 6)}
                min_diff, max_diff = difficulty_map.get(args.difficulty, (0, 6))
                filtered_questions = [
                    q for q in filtered_questions if min_diff <= q.difficulty < max_diff
                ]
                logger.info(
                    f"Filtered by difficulty '{args.difficulty}': {len(filtered_questions)} questions"
                )

            if not filtered_questions:
                logger.error("❌ No questions match the specified filters")
                sys.exit(1)

            # Export to CSV
            if args.output:
                output_path = args.output
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path("output")
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"questions_{args.grade}_dump_{timestamp}.csv"

            logger.info(f"\nExporting {len(filtered_questions)} questions to CSV...")
            files = CSVExporter.export_with_summary(
                questions=filtered_questions,
                output_dir=output_path.parent,
                base_filename=output_path.stem,
            )

            # Final summary
            logger.info(f"\n{'=' * 80}")
            logger.info("✅ DUMP COMPLETE")
            logger.info(f"{'=' * 80}")
            logger.info(f"Questions exported: {len(filtered_questions)}")
            logger.info(f"CSV file: {files['csv']}")
            logger.info(f"Summary: {files['summary']}")
            logger.info(f"{'=' * 80}\n")

            # Show sample questions
            logger.info("Sample Questions:\n")
            for i, q in enumerate(filtered_questions[:3], 1):
                logger.info(f"{i}. {q.question_text}")
                for j, opt in enumerate(q.options):
                    marker = "✓" if j == q.answer_index else " "
                    logger.info(f"   [{marker}] {opt}")
                logger.info("")

            if len(filtered_questions) > 3:
                logger.info(f"... and {len(filtered_questions) - 3} more questions\n")

            sys.exit(0)

        # Handle --regenerate-images flag: regenerate images for existing questions
        if args.regenerate_images:
            logger.info("\n" + "=" * 80)
            logger.info("REGENERATING IMAGES FOR EXISTING QUESTIONS")
            logger.info("=" * 80 + "\n")

            questions_collection = f"{args.grade}_questions"
            logger.info(f"Fetching questions from collection: {questions_collection}")

            # Search with empty query to get all questions
            query_embedding = embedding_client.embed_query("questions")
            results = qdrant_manager.search_questions(
                collection_name=questions_collection,
                query_embedding=query_embedding,
                limit=10000,  # Get all questions
            )

            if not results:
                logger.error(f"❌ No questions found in collection {questions_collection}")
                sys.exit(1)

            # Convert to Question objects
            questions = []
            for result in results:
                try:
                    q_data = result["payload"]
                    if q_data.get("type") == "question":
                        questions.append(Question.from_dict(q_data))
                except Exception as e:
                    logger.warning(f"Skipping invalid question: {e}")
                    continue

            logger.info(f"Found {len(questions)} questions")

            # Apply filters if provided
            filtered_questions = questions
            if args.subject:
                filtered_questions = [q for q in filtered_questions if q.subject == args.subject]
                logger.info(
                    f"Filtered by subject '{args.subject}': {len(filtered_questions)} questions"
                )

            if args.topics:
                filtered_questions = [
                    q
                    for q in filtered_questions
                    if any(topic in q.sub_subject for topic in args.topics)
                ]
                logger.info(
                    f"Filtered by topics {args.topics}: {len(filtered_questions)} questions"
                )

            if not filtered_questions:
                logger.error("❌ No questions match the specified filters")
                sys.exit(1)

            # Initialize S3 uploader
            logger.info("\nInitializing S3 uploader...")
            try:
                s3_uploader = S3Uploader()

                # Verify S3 bucket access
                if not s3_uploader.verify_bucket_access():
                    logger.error("❌ Cannot access S3 bucket. Please check your AWS credentials.")
                    sys.exit(1)

                # Initialize image generator with S3 uploader
                logger.info("\nGenerating images and uploading to S3...")
                image_generator = ImageGenerator(s3_uploader=s3_uploader)

                # Generate images for all questions
                s3_urls = image_generator.generate_images_batch(
                    questions=filtered_questions, image_style=args.image_style
                )

                # Update questions with S3 URLs
                logger.info(f"\nUpdating {len(s3_urls)} questions with S3 URLs...")
                for question in filtered_questions:
                    if question.question_number in s3_urls:
                        s3_url = s3_urls[question.question_number]

                        # Update question_image field with full S3 URL
                        question.question_image = s3_url

                        # Update artifacts with S3 URL
                        if not question.artifacts:
                            question.artifacts = []
                        if s3_url not in question.artifacts:
                            question.artifacts.append(s3_url)

                        # Update enriched image tracking fields
                        if s3_url not in question.images_path:
                            question.images_path.append(s3_url)
                        if s3_url not in question.artifacts_path:
                            question.artifacts_path.append(s3_url)
                        question.images_tagged_count = len(question.images_path)

                        # Add image tag to HTML content and explanation
                        img_tag = f'<img src="{s3_url}" alt="Question {question.question_number} illustration" style="max-width: 100%; height: auto;" />'

                        # Add to content if not already present
                        if s3_url not in question.content:
                            question.content = f"{question.content}\n{img_tag}"

                        # Add to explanation if not already present
                        if question.explanation and s3_url not in question.explanation:
                            question.explanation = f"{img_tag}\n{question.explanation}"

                logger.info(
                    f"✅ Updated {len(s3_urls)}/{len(filtered_questions)} questions with S3 URLs"
                )

                # Update Qdrant with S3 URLs in metadata
                if s3_urls:
                    logger.info("\nUpdating Qdrant database with S3 URLs...")
                    updated_questions = [
                        q for q in filtered_questions if q.question_number in s3_urls
                    ]
                    updated_embeddings = embedding_client.embed_questions_batch(updated_questions)
                    qdrant_manager.upsert_questions(
                        collection_name=questions_collection,
                        questions=updated_questions,
                        embeddings=updated_embeddings,
                    )
                    logger.info("✅ Database updated")

                # Export to CSV
                if args.output:
                    output_path = args.output
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = Path("output")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"questions_{args.grade}_regenerated_{timestamp}.csv"

                logger.info(f"\nExporting {len(filtered_questions)} questions to CSV...")
                files = CSVExporter.export_with_summary(
                    questions=filtered_questions,
                    output_dir=output_path.parent,
                    base_filename=output_path.stem,
                )

                # Final summary
                logger.info(f"\n{'=' * 80}")
                logger.info("✅ REGENERATION COMPLETE")
                logger.info(f"{'=' * 80}")
                logger.info(f"Total questions: {len(filtered_questions)}")
                logger.info(f"Images generated: {len(s3_urls)}")
                logger.info(f"CSV file: {files['csv']}")
                logger.info(f"Summary: {files['summary']}")
                logger.info(f"{'=' * 80}\n")

            except ValueError as ve:
                logger.exception(f"Configuration error: {ve}")
                logger.warning("Please configure S3 settings in .env file")
                sys.exit(1)
            except Exception as e:
                logger.exception(f"Error regenerating images: {e}")
                sys.exit(1)

            sys.exit(0)

        # Normal generation flow continues...
        question_generator = QuestionGeneratorV2(
            embedding_client=embedding_client,
            qdrant_manager=qdrant_manager,
            model_name=args.gemini_model,
        )

        # Check capacity unless --no-preview
        if not args.no_preview:
            capacity_info = check_capacity(
                question_generator, args.grade, args.subject, args.topics, args.difficulty
            )

            if capacity_info["status"] == "no_content":
                logger.error("❌ No content found. Please ingest PDFs first:")
                logger.error(f"   python ingest_content.py --grade {args.grade}")
                sys.exit(1)

            if capacity_info["status"] == "exhausted":
                logger.warning("⚠️  Content appears exhausted!")
                logger.warning("   Consider ingesting more PDFs or changing filters.")
                sys.exit(1)

            # Prompt user for number of questions if not provided
            if not args.num:
                realistic = capacity_info.get("realistic_available", capacity_info["available"])
                print(f"\n💡 Realistic estimate: ~{realistic} new unique questions available")
                print(
                    f"   (Success rate: {capacity_info.get('success_rate', 0.7):.0%} based on content analysis)"
                )
                try:
                    num_questions = int(input("\nHow many questions would you like to generate? "))
                    if num_questions <= 0:
                        logger.error("Number must be positive")
                        sys.exit(1)
                    if num_questions > realistic:
                        logger.warning(
                            f"⚠️  Requesting {num_questions} but only ~{realistic} realistically available."
                        )
                        logger.warning(
                            "   Will attempt to generate, but may produce fewer unique questions."
                        )
                    args.num = num_questions
                except (ValueError, KeyboardInterrupt):
                    logger.exception("\nInvalid input or cancelled")
                    sys.exit(1)
        elif not args.num:
            logger.error("--num required when using --no-preview")
            sys.exit(1)

        # Generate questions
        logger.info(f"\n{'=' * 80}")
        logger.info(f"GENERATING {args.num} QUESTIONS")
        logger.info(f"{'=' * 80}\n")

        result = question_generator.generate_questions(
            grade=args.grade,
            subject=args.subject,
            topics=args.topics,
            difficulty=args.difficulty,
            num_questions=args.num,
            content_chunks_limit=20,
        )

        questions = result["questions"]
        status = result["status"]

        if not questions:
            logger.error("\n❌ No questions generated. Check if content exists.")
            logger.info(f"Run: python ingest_content.py --grade {args.grade}")
            sys.exit(1)

        # Report statistics
        logger.info(f"\n{'=' * 80}")
        logger.info("GENERATION STATISTICS")
        logger.info(f"{'=' * 80}")
        logger.info(f"Status: {status.upper()}")
        logger.info(f"Questions generated: {len(questions)}")
        if result.get("new_questions_count"):
            logger.info(f"  ├─ New: {result['new_questions_count']}")
        if result.get("existing_questions_used"):
            logger.info(f"  └─ Reused: {result['existing_questions_used']}")
        if result.get("duplicates_avoided"):
            logger.info(f"Duplicates prevented: {result['duplicates_avoided']}")
        logger.info(f"{'=' * 80}\n")

        # Generate images if requested
        if args.generate_images:
            logger.info(f"{'=' * 80}")
            logger.info("GENERATING AND UPLOADING IMAGES TO S3")
            logger.info(f"{'=' * 80}")
            logger.info(f"Generating images for {len(questions)} questions...")

            try:
                # Initialize S3 uploader
                s3_uploader = S3Uploader()

                # Verify S3 bucket access
                if not s3_uploader.verify_bucket_access():
                    logger.error("❌ Cannot access S3 bucket. Please check your AWS credentials.")
                    logger.warning("Skipping image generation...")
                else:
                    # Initialize image generator with S3 uploader
                    image_generator = ImageGenerator(s3_uploader=s3_uploader)
                    s3_urls = image_generator.generate_images_batch(
                        questions=questions, image_style=args.image_style
                    )

                    # Update questions with S3 URLs
                    for question in questions:
                        if question.question_number in s3_urls:
                            s3_url = s3_urls[question.question_number]

                            # Update question_image field with full S3 URL
                            question.question_image = s3_url

                            # Update artifacts with S3 URL
                            if not question.artifacts:
                                question.artifacts = []
                            question.artifacts.append(s3_url)

                            # Update enriched image tracking fields
                            question.images_path.append(s3_url)
                            question.artifacts_path.append(s3_url)
                            question.images_tagged_count = len(question.images_path)

                            # Add image tag to HTML content and explanation
                            img_tag = f'<img src="{s3_url}" alt="Question {question.question_number} illustration" style="max-width: 100%; height: auto;" />'

                            # Add to content if not already present
                            if s3_url not in question.content:
                                question.content = f"{question.content}\n{img_tag}"

                            # Add to explanation if not already present
                            if question.explanation and s3_url not in question.explanation:
                                question.explanation = f"{img_tag}\n{question.explanation}"

                    logger.info(
                        f"✅ Generated and uploaded {len(s3_urls)}/{len(questions)} images to S3"
                    )

                    # Update Qdrant with S3 URLs in metadata
                    if s3_urls:
                        logger.info("Updating database with S3 image URLs...")
                        updated_questions = [q for q in questions if q.question_number in s3_urls]
                        updated_embeddings = embedding_client.embed_questions_batch(
                            updated_questions
                        )
                        qdrant_manager.upsert_questions(
                            collection_name=f"{args.grade}_questions",
                            questions=updated_questions,
                            embeddings=updated_embeddings,
                        )
                        logger.info("✅ Database updated with S3 URLs")

                    logger.info(f"{'=' * 80}\n")

            except ValueError as ve:
                logger.exception(f"Configuration error: {ve}")
                logger.warning("Skipping image generation. Please configure S3 settings.")
            except Exception as e:
                logger.exception(f"Error generating images: {e}")
                logger.warning("Continuing without images...")

        # Export to CSV
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"questions_{args.grade}_{timestamp}.csv"

        logger.info("Exporting to CSV...")
        files = CSVExporter.export_with_summary(
            questions=questions, output_dir=output_path.parent, base_filename=output_path.stem
        )

        # Final summary
        logger.info(f"\n{'=' * 80}")
        logger.info("✅ GENERATION COMPLETE")
        logger.info(f"{'=' * 80}")
        logger.info(f"Questions: {len(questions)}")
        logger.info(f"CSV file: {files['csv']}")
        logger.info(f"Summary: {files['summary']}")
        logger.info(f"{'=' * 80}\n")

        # Show sample questions
        logger.info("Sample Questions:\n")
        for i, q in enumerate(questions[:3], 1):
            logger.info(f"{i}. {q.question_text}")
            for j, opt in enumerate(q.options):
                marker = "✓" if j == q.answer_index else " "
                logger.info(f"   [{marker}] {opt}")
            logger.info("")

        if len(questions) > 3:
            logger.info(f"... and {len(questions) - 3} more questions\n")

    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Generation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
