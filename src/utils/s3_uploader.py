"""S3 uploader utility for managing image uploads to AWS S3."""

import logging
import os
from io import BytesIO

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError

from src.core.models import Question

logger = logging.getLogger(__name__)


class S3Uploader:
    """Upload and manage images in AWS S3."""

    def __init__(
        self,
        bucket_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        base_path: str = "edutech/images",
    ):
        """
        Initialize S3 uploader.

        Args:
            bucket_name: S3 bucket name (or from env: S3_BUCKET_NAME).
            aws_access_key_id: AWS access key (or from env: AWS_ACCESS_KEY_ID).
            aws_secret_access_key: AWS secret key (or from env: AWS_SECRET_ACCESS_KEY).
            region_name: AWS region (or from env: AWS_REGION, default: us-east-1).
            endpoint_url: Custom S3 endpoint URL (or from env: S3_ENDPOINT_URL).
            base_path: Base path prefix for all uploads (default: edutech/images).
        """
        self.bucket_name = bucket_name or os.getenv("S3_BUCKET_NAME")
        if not self.bucket_name:
            msg = "S3 bucket name not provided. Set S3_BUCKET_NAME environment variable."
            raise ValueError(msg)

        self.base_path = base_path.strip("/")
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL")

        # Initialize S3 client
        try:
            # Base client credentials
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=self.region_name,
                endpoint_url=self.endpoint_url,
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                ) if self.endpoint_url else None,
            )

            if self.endpoint_url:
                logger.info(f"Using custom S3 endpoint: {self.endpoint_url}")
            logger.info(f"Initialized S3 client for bucket: {self.bucket_name}")
            logger.info(f"Region: {self.region_name}, Base path: {self.base_path}")
        except NoCredentialsError as e:
            msg = "AWS credentials not found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            raise ValueError(msg) from e

    def upload_image(
        self,
        image_data: bytes,
        question: Question,
        content_type: str = "image/png",
    ) -> str:
        """
        Upload image to S3 and return public URL.

        Args:
            image_data: Image bytes to upload.
            question: Question object for path generation.
            content_type: MIME type (default: image/png).

        Returns:
            S3 public URL of the uploaded image.

        Raises:
            Exception: If upload fails.
        """
        try:
            # Generate S3 key path: base_path/grade/subject/filename
            subject_slug = question.subject.lower().replace(" ", "_")
            filename = self._generate_filename(question)
            s3_key = f"{self.base_path}/{question.grade}/{subject_slug}/{filename}"

            # Upload to S3
            logger.info(f"Uploading image to S3: {s3_key}")
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=BytesIO(image_data),
                ContentType=content_type,
                ACL="public-read",  # Make publicly accessible
            )

            # Generate public URL
            s3_url = self._get_public_url(s3_key)
            logger.info(f"Image uploaded successfully: {s3_url}")

            return s3_url

        except ClientError as e:
            logger.exception(f"S3 upload failed: {e}")
            raise

    def upload_batch(
        self,
        images_data: dict[int, bytes],
        questions: list[Question],
    ) -> dict[int, str]:
        """
        Upload multiple images to S3.

        Args:
            images_data: Dictionary mapping question_number to image bytes.
            questions: List of Question objects.

        Returns:
            Dictionary mapping question_number to S3 URL.
        """
        s3_urls = {}
        question_map = {q.question_number: q for q in questions}

        for question_number, image_data in images_data.items():
            try:
                question = question_map.get(question_number)
                if not question:
                    logger.warning(f"Question {question_number} not found in batch")
                    continue

                s3_url = self.upload_image(image_data, question)
                s3_urls[question_number] = s3_url

            except Exception as e:
                logger.warning(f"Failed to upload image for Q{question_number}: {e}")
                continue

        logger.info(f"Uploaded {len(s3_urls)}/{len(images_data)} images to S3")
        return s3_urls

    def delete_image(self, s3_url: str) -> bool:
        """
        Delete image from S3.

        Args:
            s3_url: Full S3 URL of the image.

        Returns:
            True if deletion succeeded, False otherwise.
        """
        try:
            # Extract S3 key from URL
            s3_key = self._extract_key_from_url(s3_url)
            if not s3_key:
                logger.warning(f"Could not extract S3 key from URL: {s3_url}")
                return False

            logger.info(f"Deleting image from S3: {s3_key}")
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Image deleted successfully: {s3_key}")
            return True

        except ClientError as e:
            logger.exception(f"S3 deletion failed: {e}")
            return False

    def _generate_filename(self, question: Question) -> str:
        """
        Generate filename for the image.

        Args:
            question: Question object.

        Returns:
            Filename like: q123_topic_name.png
        """
        sub_subject_slug = question.sub_subject.lower().replace(" ", "_")
        return f"q{question.question_number}_{sub_subject_slug}.png"

    def _get_public_url(self, s3_key: str) -> str:
        """
        Generate public URL for S3 object.

        Args:
            s3_key: S3 object key.

        Returns:
            Public URL.
        """
        # Use custom endpoint URL if provided, otherwise use standard AWS S3 format
        if self.endpoint_url:
            # Remove trailing slash from endpoint if present
            endpoint = self.endpoint_url.rstrip("/")
            return f"{endpoint}/{self.bucket_name}/{s3_key}"

        # Standard AWS S3 public URL format
        return f"https://{self.bucket_name}.s3.{self.region_name}.amazonaws.com/{s3_key}"

    def _extract_key_from_url(self, s3_url: str) -> str | None:
        """
        Extract S3 key from full URL.

        Args:
            s3_url: Full S3 URL.

        Returns:
            S3 key or None if extraction fails.
        """
        try:
            if self.endpoint_url:
                # Custom endpoint: https://endpoint.url/bucket/path/to/file.png
                endpoint = self.endpoint_url.rstrip("/")
                prefix = f"{endpoint}/{self.bucket_name}/"
                if s3_url.startswith(prefix):
                    return s3_url[len(prefix) :]
            elif f"{self.bucket_name}.s3" in s3_url:
                # AWS S3: https://bucket.s3.region.amazonaws.com/path/to/file.png
                parts = s3_url.split(f"{self.bucket_name}.s3.{self.region_name}.amazonaws.com/")
                if len(parts) == 2:
                    return parts[1]
            return None
        except Exception as e:
            logger.exception(f"Error extracting S3 key: {e}")
            return None

    def verify_bucket_access(self) -> bool:
        """
        Verify that we have access to the S3 bucket.

        Returns:
            True if bucket is accessible, False otherwise.
        """
        try:
            # For custom endpoints, try listing objects instead of head_bucket
            # as head_bucket may not work correctly with all S3-compatible services
            if self.endpoint_url:
                # Try to list objects (limit to 1) to verify access
                self.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=1)
            else:
                # For AWS S3, use head_bucket
                self.s3_client.head_bucket(Bucket=self.bucket_name)

            logger.info(f"✅ S3 bucket '{self.bucket_name}' is accessible")
            return True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_msg = e.response.get("Error", {}).get("Message", str(e))

            if error_code in {"404", "NoSuchBucket"}:
                logger.error(
                    f"❌ S3 bucket '{self.bucket_name}' does not exist. Error: {error_msg}",
                    exc_info=True,
                )
            elif error_code in {"403", "AccessDenied"}:
                logger.error(
                    f"❌ Access denied to S3 bucket '{self.bucket_name}'. Error: {error_msg}",
                    exc_info=True,
                )
            else:
                logger.error(
                    f"❌ Error accessing bucket '{self.bucket_name}' at {self.endpoint_url or 'AWS S3'}. "
                    f"Code: {error_code}, Message: {error_msg}",
                    exc_info=True,
                )
            return False
