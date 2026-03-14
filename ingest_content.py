"""Content ingestion script for analyzing PDFs and storing content chunks."""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core import EmbeddingClient, PDFMetadata, QdrantManager
from utils import ContentExtractor

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/ingest_content.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)


def get_collection_name(grade: str, subject: str) -> str:
    """
    Determine the correct Qdrant collection name based on grade and subject.

    Rules:
    - If subject ends with '_naplan_reference' → use as-is: grade5_numeracy_naplan_reference
    - Otherwise → grade5_content (standard content collection)

    Examples:
    - grade5, numeracy              → grade5_content
    - grade5, numeracy_naplan_reference → grade5_numeracy_naplan_reference
    - grade5, language_convention   → grade5_content
    - grade5, language_convention_naplan_reference → grade5_language_convention_naplan_reference
    """
    if subject.endswith("_naplan_reference"):
        return f"{grade}_{subject}"
    return f"{grade}_content"


class ContentIngestionPipeline:
    """Pipeline for ingesting PDF content and storing it for later question generation."""

    def __init__(
        self,
        content_extractor: ContentExtractor,
        embedding_client: EmbeddingClient,
        qdrant_manager: QdrantManager,
    ):
        self.content_extractor = content_extractor
        self.embedding_client = embedding_client
        self.qdrant_manager = qdrant_manager

    def ingest_pdf(self, pdf_metadata: PDFMetadata) -> dict:
        """Ingest a single PDF file into the correct collection."""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing: {pdf_metadata.pdf_filename}")
        logger.info(f"{'=' * 80}")

        try:
            # Extract content from PDF
            logger.info("Extracting content...")
            chunks = self.content_extractor.extract_content(
                pdf_path=pdf_metadata.pdf_path, pdf_metadata=pdf_metadata
            )

            if not chunks:
                logger.warning("No content chunks extracted")
                return {"success": False, "chunks": 0}

            logger.info(f"Extracted {len(chunks)} content chunks")

            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.embedding_client.embed_content_chunks(chunks)
            logger.info(f"Generated {len(embeddings)} embeddings")

            # ── FIX: use correct collection name based on subject ──
            collection_name = get_collection_name(pdf_metadata.grade, pdf_metadata.subject)
            logger.info(f"Storing in collection: {collection_name}")

            self.qdrant_manager.upsert_content_chunks(
                collection_name=collection_name, chunks=chunks, embeddings=embeddings
            )

            # Summary statistics
            topics = set()
            concepts = []
            for chunk in chunks:
                topics.update(chunk.topics)
                concepts.extend(chunk.concepts)

            logger.info(f"\n{'=' * 80}")
            logger.info("INGESTION COMPLETE")
            logger.info(f"{'=' * 80}")
            logger.info(f"Chunks stored: {len(chunks)}")
            logger.info(f"Collection:    {collection_name}")
            logger.info(f"Topics identified: {len(topics)}")
            logger.info(f"Concepts extracted: {len(set(concepts))}")
            logger.info(f"{'=' * 80}\n")

            return {
                "success": True,
                "chunks": len(chunks),
                "topics": len(topics),
                "concepts": len(set(concepts)),
                "collection": collection_name,
            }

        except Exception as e:
            logger.error(f"Error ingesting {pdf_metadata.pdf_filename}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}


def find_pdfs(
    root_dir: Path, grade: str | None = None, subject: str | None = None
) -> list[PDFMetadata]:
    """Find all PDF files matching the grade and subject filters."""
    pdf_files = []
    root_path = Path(root_dir)

    if not root_path.exists():
        logger.error(f"Directory not found: {root_dir}")
        return []

    pattern = grade if grade else "*"
    pattern += "/*" if not subject else f"/{subject}"
    pattern += "/*.pdf"

    for pdf_path in root_path.glob(pattern):
        parts = pdf_path.relative_to(root_path).parts
        if len(parts) >= 3:
            grade_name = parts[0]
            subject_name = parts[1]

            pdf_metadata = PDFMetadata(
                grade=grade_name,
                subject=subject_name,
                pdf_filename=pdf_path.name,
                pdf_path=str(pdf_path),
            )
            pdf_files.append(pdf_metadata)

    return pdf_files


def main() -> None:
    """Main entry point for content ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest PDF content for question generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest books (stored in grade5_content)
  python ingest_content.py --grade grade5 --subject numeracy

  # Ingest NAPLAN reference papers (stored in grade5_numeracy_naplan_reference)
  python ingest_content.py --grade grade5 --subject numeracy_naplan_reference

  # Ingest language convention books
  python ingest_content.py --grade grade5 --subject language_convention

  # Ingest language convention NAPLAN papers
  python ingest_content.py --grade grade5 --subject language_convention_naplan_reference
        """,
    )

    parser.add_argument(
        "--root-dir", type=Path, default=Path("input"),
        help="Root directory containing PDF files (default: input/)",
    )
    parser.add_argument("--grade", type=str, help="Filter by grade (e.g., grade5)")
    parser.add_argument("--subject", type=str, help="Filter by subject (e.g., numeracy)")
    parser.add_argument(
        "--gemini-model", type=str, default="gemini-2.5-pro",
        help="Gemini model for content analysis (default: gemini-2.5-pro)",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)

    try:
        logger.info("=" * 80)
        logger.info("CONTENT INGESTION SYSTEM")
        logger.info("=" * 80)
        logger.info(f"Root directory: {args.root_dir}")
        logger.info(f"Grade filter:   {args.grade or 'All'}")
        logger.info(f"Subject filter: {args.subject or 'All'}")
        logger.info(f"Model:          {args.gemini_model}")

        # Show which collection will be used
        if args.grade and args.subject:
            col = get_collection_name(args.grade, args.subject)
            logger.info(f"Target collection: {col}")

        logger.info("=" * 80 + "\n")

        # Find PDF files
        logger.info("Searching for PDF files...")
        pdf_files = find_pdfs(args.root_dir, args.grade, args.subject)

        if not pdf_files:
            logger.error("No PDF files found matching the criteria")
            logger.info(f"Expected structure: {args.root_dir}/gradeX/subject/*.pdf")
            sys.exit(1)

        logger.info(f"Found {len(pdf_files)} PDF file(s)\n")

        # Initialize pipeline
        logger.info("Initializing ingestion pipeline...")
        content_extractor = ContentExtractor(model_name=args.gemini_model)
        embedding_client = EmbeddingClient()
        qdrant_manager = QdrantManager(embedding_dimension=embedding_client.dimension)

        pipeline = ContentIngestionPipeline(
            content_extractor=content_extractor,
            embedding_client=embedding_client,
            qdrant_manager=qdrant_manager,
        )

        # Process each PDF
        results = []
        for i, pdf_metadata in enumerate(pdf_files, 1):
            logger.info(f"\nProcessing {i}/{len(pdf_files)}")
            result = pipeline.ingest_pdf(pdf_metadata)
            results.append(result)

        # Final summary
        successful = sum(1 for r in results if r.get("success"))
        total_chunks = sum(r.get("chunks", 0) for r in results)
        collections_used = set(r.get("collection", "") for r in results if r.get("success"))

        logger.info(f"\n{'=' * 80}")
        logger.info("INGESTION SUMMARY")
        logger.info(f"{'=' * 80}")
        logger.info(f"PDFs processed:     {successful}/{len(pdf_files)}")
        logger.info(f"Total chunks stored:{total_chunks}")
        logger.info(f"Collections used:   {', '.join(collections_used)}")
        logger.info(f"{'=' * 80}\n")

        if successful < len(pdf_files):
            logger.warning(f"{len(pdf_files) - successful} PDF(s) failed to process")

    except KeyboardInterrupt:
        logger.warning("Ingestion cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()