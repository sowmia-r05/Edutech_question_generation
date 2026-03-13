"""Generate contextual images for questions using Gemini's image generation."""

import logging
import os
import time
from typing import TYPE_CHECKING

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from src.core.models import Question
from src.utils.s3_uploader import S3Uploader

if TYPE_CHECKING:
    from PIL.ImageFont import FreeTypeFont
    from PIL.ImageFont import ImageFont as ImageFontType

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate educational images for questions."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-flash-image",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        s3_uploader: S3Uploader | None = None,
    ):
        """
        Initialize image generator.

        Args:
            api_key: Google Gemini API key.
            model_name: Image generation model (default: gemini-2.5-flash-image).
            max_retries: Max retry attempts.
            retry_delay: Delay between retries.
            s3_uploader: S3Uploader instance for uploading images to S3.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            msg = "Gemini API key not provided."
            raise ValueError(msg)

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.s3_uploader = s3_uploader

        logger.info(f"Initialized ImageGenerator with model: {model_name}")
        if self.s3_uploader:
            logger.info("S3 uploader configured for direct cloud storage")
        else:
            logger.warning("No S3 uploader configured - images will not be stored")

    def generate_question_image(
        self, question: Question, image_style: str = "educational diagram"
    ) -> str | None:
        """
        Generate an image for a question and upload to S3.

        Args:
            question: Question object.
            image_style: Style description for image generation.

        Returns:
            S3 URL of generated image, or None if generation fails.
        """
        try:
            # Create image prompt based on question
            prompt = self._create_image_prompt(question, image_style)

            # Generate image
            logger.info(f"Generating image for Q{question.question_number}: {question.sub_subject}")
            image_data = self._generate_with_retry(prompt)

            if not image_data:
                return None

            # Upload to S3 if uploader is configured
            if self.s3_uploader:
                s3_url = self.s3_uploader.upload_image(image_data, question)
                logger.info(f"Uploaded image to S3: {s3_url}")
                return s3_url

            logger.warning("No S3 uploader configured - image not stored")
            return None

        except Exception as e:
            logger.exception(f"Error generating image for Q{question.question_number}: {e}")
            return None

    def generate_images_batch(
        self, questions: list[Question], image_style: str = "educational diagram"
    ) -> dict[int, str]:
        """
        Generate images for multiple questions and upload to S3.

        Args:
            questions: List of Question objects.
            image_style: Style for image generation.

        Returns:
            Dictionary mapping question_number to S3 URL.
        """
        image_urls = {}

        for question in questions:
            try:
                # Add delay to avoid rate limits
                time.sleep(1)

                s3_url = self.generate_question_image(question, image_style)
                if s3_url:
                    image_urls[question.question_number] = s3_url

            except Exception as e:
                logger.warning(f"Skipping image for Q{question.question_number}: {e}")
                continue

        logger.info(f"Generated and uploaded {len(image_urls)}/{len(questions)} images to S3")
        return image_urls

    def _create_image_prompt(self, question: Question, style: str) -> str:
        """Create prompt for image generation."""
        # Extract key concepts for visualization
        subject = question.sub_subject or question.subject
        question_text = question.question_text

        # Build educational image prompt
        return f"""Create a clear, educational {style} for a {question.grade} student learning about {subject}.

The image should help visualize this question:
"{question_text}"

Style requirements:
- Simple, clean educational illustration
- Appropriate for {question.grade} students
- Clear labels and annotations if needed
- Professional textbook-quality diagram
- Colors should be vibrant but not distracting
- Focus on clarity and educational value

Subject: {question.subject}
Topic: {subject}
Difficulty: Based on the question complexity

Make it visually engaging and educational, similar to diagrams found in mathematics or science textbooks.
"""

    def _generate_with_retry(self, prompt: str) -> bytes | None:
        """Generate image with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.4,  # Lower for consistent educational style
                    ),
                )

                # Extract image data
                if response.parts:
                    for part in response.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            # Get image bytes
                            return part.inline_data.data

                logger.warning("No image data in response")
                return None

            except google_exceptions.ResourceExhausted:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    logger.warning(f"Rate limit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.exception("Max retries reached")
                    return None

            except Exception as e:
                logger.exception(f"Generation error: {e}")
                return None

        return None

    def create_placeholder_image(self, question: Question, text: str = "Question Image") -> str:
        """
        Create a simple placeholder image and upload to S3 when generation fails.

        Args:
            question: Question object.
            text: Text to display on placeholder.

        Returns:
            S3 URL of placeholder image, or empty string if creation fails.
        """
        try:
            from io import BytesIO

            from PIL import Image, ImageDraw, ImageFont

            # Create simple placeholder
            width, height = 800, 600
            img = Image.new("RGB", (width, height), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            # Add text
            font: FreeTypeFont | ImageFontType
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
            except Exception:
                font = ImageFont.load_default()

            text_lines = [
                f"Question {question.question_number}",
                f"{question.subject} - {question.sub_subject}",
                "",
                text,
            ]

            y_offset = height // 3
            for line in text_lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
                x = (width - text_width) // 2
                draw.text((x, y_offset), line, fill=(100, 100, 100), font=font)
                y_offset += 60

            # Save to bytes buffer
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            image_data = buffer.getvalue()

            # Upload to S3 if uploader is configured
            if self.s3_uploader:
                s3_url = self.s3_uploader.upload_image(image_data, question)
                logger.info(f"Uploaded placeholder to S3: {s3_url}")
                return s3_url

            logger.warning("No S3 uploader configured - placeholder not stored")
            return ""

        except Exception as e:
            logger.exception(f"Failed to create placeholder: {e}")
            return ""
