"""Generate contextual illustrative images for questions using Gemini."""

import logging
import os
import re
import time

from src.core.models import Question
from src.utils.s3_uploader import S3Uploader

logger = logging.getLogger(__name__)

# ── Question-type detection keywords ─────────────────────────────────────────
_DATA_KEYWORDS = re.compile(
    r"\b(tally|table|chart|graph|bar graph|pie chart|pictograph|column graph|"
    r"votes|voted|survey|results|frequency|data|shows|recorded|listed)\b",
    re.IGNORECASE,
)
_GEOMETRY_KEYWORDS = re.compile(
    r"\b(shape|triangle|square|rectangle|circle|pentagon|hexagon|polygon|angle|"
    r"symmetry|perimeter|area|volume|cube|prism|cylinder|cone|net|sides|vertices|"
    r"diagonal|parallel|perpendicular)\b",
    re.IGNORECASE,
)
_MEASUREMENT_KEYWORDS = re.compile(
    r"\b(ruler|measure|length|width|height|mass|weight|thermometer|scale|clock|"
    r"time|temperature|capacity|litre|millilitre|kg|cm|mm|km|metre)\b",
    re.IGNORECASE,
)
_MONEY_KEYWORDS = re.compile(
    r"\b(coin|coins|note|dollar|\$|cent|buy|pay|cost|price|change|shop|bought)\b",
    re.IGNORECASE,
)
_FRACTION_KEYWORDS = re.compile(
    r"\b(fraction|half|quarter|third|shaded|divided|equal parts|number line)\b",
    re.IGNORECASE,
)
_PATTERN_KEYWORDS = re.compile(
    r"\b(pattern|sequence|next|continue|rule|arrange|array|grid)\b",
    re.IGNORECASE,
)


def _detect_question_type(question_text: str, sub_subject: str) -> str:
    """
    Classify the question into one of several visual types.
    Returns: 'data_chart' | 'geometry' | 'measurement' | 'money' |
             'fraction' | 'pattern' | 'word_problem'
    """
    text = f"{question_text} {sub_subject}".lower()

    if _DATA_KEYWORDS.search(text):
        return "data_chart"
    if _GEOMETRY_KEYWORDS.search(text):
        return "geometry"
    if _MEASUREMENT_KEYWORDS.search(text):
        return "measurement"
    if _MONEY_KEYWORDS.search(text):
        return "money"
    if _FRACTION_KEYWORDS.search(text):
        return "fraction"
    if _PATTERN_KEYWORDS.search(text):
        return "pattern"
    return "word_problem"


class ImageGenerator:
    """Generate contextual images using Gemini native image generation."""

    # Correct model for Gemini native image output
    DEFAULT_MODEL = "gemini-2.0-flash-exp-image-generation"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = DEFAULT_MODEL,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        s3_uploader: S3Uploader | None = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not set in .env file.")

        try:
            from google import genai
            self.genai = genai
            self.client = genai.Client(api_key=self.api_key)
        except ImportError:
            raise ImportError("Run: pip install google-genai")

        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.s3_uploader = s3_uploader
        logger.info(f"Initialized ImageGenerator with model: {model_name}")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_question_image(
        self,
        question: Question,
        image_style: str = "educational illustration",
    ) -> str | None:
        """Generate a contextual image for a question and upload to S3."""
        try:
            prompt = self._create_image_prompt(question)
            q_type = _detect_question_type(
                question.question_text, question.sub_subject or ""
            )
            logger.info(
                f"Generating image for Q{question.question_number} "
                f"[{q_type}]: {question.sub_subject or question.subject}"
            )

            image_bytes = self._generate_with_retry(prompt)
            if not image_bytes:
                logger.warning(
                    f"No image bytes returned for Q{question.question_number}"
                )
                return None

            if self.s3_uploader:
                s3_url = self.s3_uploader.upload_image(image_bytes, question)
                logger.info(f"Uploaded image to S3: {s3_url}")
                return s3_url

            logger.warning("No S3 uploader configured — image bytes generated but not stored.")
            return None

        except Exception as e:
            logger.error(
                f"Error generating image for Q{question.question_number}: {e}"
            )
            return None

    def generate_images_batch(
        self,
        questions: list[Question],
        image_style: str = "educational illustration",
    ) -> dict[int, str]:
        """Generate images for a list of questions, returns {question_number: s3_url}."""
        image_urls: dict[int, str] = {}
        for question in questions:
            try:
                time.sleep(1.5)  # Rate-limit between requests
                s3_url = self.generate_question_image(question, image_style)
                if s3_url:
                    image_urls[question.question_number] = s3_url
            except Exception as e:
                logger.warning(
                    f"Skipping image Q{question.question_number}: {e}"
                )
                continue

        logger.info(f"Generated {len(image_urls)}/{len(questions)} images")
        return image_urls

    # ── Smart prompt builder ──────────────────────────────────────────────────

    def _create_image_prompt(self, question: Question) -> str:
        """
        Build a Gemini image prompt tailored to the question type.

        - Data/chart questions  → draw the actual table/tally/graph with labels
        - Geometry questions    → draw the shape, clean diagram, labels allowed
        - Measurement questions → draw the object being measured with a scale
        - Money questions       → draw the coins/notes described
        - Fraction questions    → draw the shape divided into parts, shaded if needed
        - Pattern questions     → draw the sequence of shapes/objects
        - Word problems         → illustrative scene, no text/numbers
        """
        q_text = question.question_text
        sub = question.sub_subject or question.subject
        q_type = _detect_question_type(q_text, sub)

        base_style = (
            "Clean, colourful, educational illustration for a primary school student. "
            "White background. Simple cartoon style. Bright primary colours."
        )

        if q_type == "data_chart":
            # For tally/table/graph questions the IMAGE IS the data — we must draw it
            return (
                f"{base_style}\n"
                f"Draw a clear, accurate educational chart or table that represents "
                f"the data described in this question:\n\"{q_text}\"\n"
                f"Instructions:\n"
                f"- If tally marks are mentioned, draw a tally table with the correct "
                f"tally marks and category labels.\n"
                f"- If a bar graph / column graph is described, draw the graph with "
                f"correct bars and axis labels.\n"
                f"- If a table of numbers is described, draw a neat table with those values.\n"
                f"- Labels, numbers, and category names ARE required so the data is readable.\n"
                f"- Do NOT show which answer is correct.\n"
                f"- Large readable text, suitable for a student worksheet."
            )

        if q_type == "geometry":
            return (
                f"{base_style}\n"
                f"Draw a clear geometric diagram for this question:\n\"{q_text}\"\n"
                f"Instructions:\n"
                f"- Draw the shape(s) described accurately with correct proportions.\n"
                f"- You may include dimension labels (e.g. '8 cm') if the question "
                f"mentions specific measurements.\n"
                f"- Show any angles, lines of symmetry, or shaded regions mentioned.\n"
                f"- No working out, no answer, just the diagram.\n"
                f"- Clean lines, large and clear."
            )

        if q_type == "measurement":
            return (
                f"{base_style}\n"
                f"Draw a clear measurement diagram for this question:\n\"{q_text}\"\n"
                f"Instructions:\n"
                f"- Draw the object being measured (ruler, thermometer, scale, clock, etc.).\n"
                f"- Show the scale markings and units clearly.\n"
                f"- Do NOT indicate the answer on the instrument.\n"
                f"- Large readable markings suitable for a student."
            )

        if q_type == "money":
            return (
                f"{base_style}\n"
                f"Draw an illustration of the money described in this question:\n\"{q_text}\"\n"
                f"Instructions:\n"
                f"- Draw the specific coins and/or notes mentioned.\n"
                f"- Show Australian currency (gold $1/$2 coins, silver coins, "
                f"green $100, red $50, blue $10, brown $5 notes, etc.).\n"
                f"- You may show the dollar/cent values on the coins/notes.\n"
                f"- Do NOT show the answer or total.\n"
                f"- Clear, realistic-looking cartoon coins/notes."
            )

        if q_type == "fraction":
            return (
                f"{base_style}\n"
                f"Draw a fraction diagram for this question:\n\"{q_text}\"\n"
                f"Instructions:\n"
                f"- Draw a shape (circle, rectangle, or number line as appropriate) "
                f"divided into the correct equal parts.\n"
                f"- Shade the fraction described if mentioned.\n"
                f"- You may label the fraction parts.\n"
                f"- Do NOT show the answer.\n"
                f"- Simple, large, easy to read."
            )

        if q_type == "pattern":
            return (
                f"{base_style}\n"
                f"Draw the pattern or sequence described in this question:\n\"{q_text}\"\n"
                f"Instructions:\n"
                f"- Draw the sequence of shapes, colours, or objects shown in the pattern.\n"
                f"- Leave the last position blank or with a question mark placeholder "
                f"if the question asks 'what comes next'.\n"
                f"- Do NOT show the answer.\n"
                f"- Clear, evenly spaced, colourful shapes."
            )

        # Default: word problem — illustrative scene, NO text or numbers
        return (
            f"{base_style}\n"
            f"Draw a simple illustrative scene for this maths word problem:\n\"{q_text}\"\n"
            f"Instructions:\n"
            f"- Show the physical objects or scenario described "
            f"(e.g. bags of apples, groups of students, stacks of boxes).\n"
            f"- The image should help a student visualise the problem.\n"
            f"- Do NOT include any text, numbers, labels, or equations.\n"
            f"- Do NOT show the answer.\n"
            f"- Simple cartoon objects, clear and colourful."
        )

    # ── Gemini API call ───────────────────────────────────────────────────────

    def _generate_with_retry(self, prompt: str) -> bytes | None:
        """Call Gemini image generation and return raw image bytes."""
        from google.genai import types

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )

                # Extract image bytes from response parts
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        logger.info(
                            f"Image generated: {len(part.inline_data.data)} bytes, "
                            f"mime={part.inline_data.mime_type}"
                        )
                        return part.inline_data.data

                logger.warning(
                    f"No image part in Gemini response on attempt {attempt + 1}. "
                    f"Text response: "
                    f"{response.candidates[0].content.parts[0].text[:100] if response.candidates else 'none'}"
                )
                return None

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Retrying image generation in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(
                        f"Image generation failed after {self.max_retries} attempts: {e}"
                    )
                    return None

        return None