"""Generate purely illustrative images for questions using Gemini."""

import logging
import os
import time

from src.core.models import Question
from src.utils.s3_uploader import S3Uploader

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Generate clean illustrative images using Gemini - no text, no explanation."""

    def __init__(self, api_key: str | None = None,
                 model_name: str = "gemini-2.5-flash-image",
                 max_retries: int = 3, retry_delay: float = 2.0,
                 s3_uploader: S3Uploader | None = None):

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

    def generate_question_image(self, question: Question,
                                 image_style: str = "educational illustration") -> str | None:
        """Generate a purely illustrative image and upload to S3."""
        try:
            prompt = self._create_image_prompt(question)
            logger.info(f"Generating image for Q{question.question_number}: {question.sub_subject or question.subject}")
            image_bytes = self._generate_with_retry(prompt)

            if not image_bytes:
                return None

            if self.s3_uploader:
                s3_url = self.s3_uploader.upload_image(image_bytes, question)
                logger.info(f"Uploaded image to S3: {s3_url}")
                return s3_url

            logger.warning("No S3 uploader configured - image not stored")
            return None

        except Exception as e:
            logger.error(f"Error generating image for Q{question.question_number}: {e}")
            return None

    def generate_images_batch(self, questions: list[Question],
                               image_style: str = "educational illustration") -> dict[int, str]:
        """Generate illustrative images for a list of questions."""
        image_urls = {}
        for question in questions:
            try:
                time.sleep(1.5)
                s3_url = self.generate_question_image(question, image_style)
                if s3_url:
                    image_urls[question.question_number] = s3_url
            except Exception as e:
                logger.warning(f"Skipping image Q{question.question_number}: {e}")
                continue

        logger.info(f"Generated {len(image_urls)}/{len(questions)} images")
        return image_urls

    def _create_image_prompt(self, question: Question) -> str:
        """
        Purely visual diagram — like a textbook illustration.
        No text, no labels, no numbers, no answer shown.
        """
        subject = question.sub_subject or question.subject
        q_text = question.question_text

        return (
            f"A clean simple colourful educational diagram for a primary school student. "
            f"Topic: {subject}. "
            f"Draw a visual illustration of the physical scenario in this question: {q_text}. "
            f"The image must contain NO text whatsoever. "
            f"NO numbers, NO letters, NO labels, NO equations, NO words. "
            f"Do NOT show the answer or any solution. "
            f"Just draw the physical objects or shapes described, like a textbook diagram. "
            f"Style: clean cartoon illustration, white background, bright simple colours, "
            f"suitable for primary school children."
        )

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

                # Extract image bytes from response
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        return part.inline_data.data

                logger.warning(f"No image in response on attempt {attempt + 1}")
                return None

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Retrying image in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Image generation failed: {e}")
                    return None

        return None