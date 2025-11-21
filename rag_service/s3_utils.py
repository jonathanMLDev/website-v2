"""
S3 Utilities for RAG Vector Data.

Provides functions to upload and download ChromaDB vector data to/from S3.
"""

import os
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

import structlog

logger = structlog.get_logger(__name__)

if not BOTO3_AVAILABLE:
    logger.warning("boto3 not available. S3 operations will not work.")


class RAGS3Manager:
    """Manager for uploading/downloading RAG vector data to/from S3."""

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        """
        Initialize S3 manager.

        Args:
            bucket_name: S3 bucket name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            endpoint_url: S3 endpoint URL (for DigitalOcean Spaces, etc.)
            region_name: AWS region name
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for S3 operations")

        self.bucket_name = bucket_name or os.getenv("RAG_S3_BUCKET_NAME")
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv(
            "AWS_SECRET_ACCESS_KEY"
        )
        self.endpoint_url = endpoint_url or os.getenv("AWS_S3_ENDPOINT_URL")
        self.region_name = region_name or os.getenv("AWS_S3_REGION_NAME", "us-east-1")

        if not self.bucket_name:
            raise ValueError(
                "bucket_name must be provided or set in RAG_S3_BUCKET_NAME env var"
            )

        # Initialize S3 client
        s3_config = {
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "region_name": self.region_name,
        }

        if self.endpoint_url:
            s3_config["endpoint_url"] = self.endpoint_url

        self.s3_client = boto3.client("s3", **s3_config)
        self.logger = logger.bind(component="RAGS3Manager")

    def upload_vector_data(
        self,
        chroma_db_path: str,
        s3_key_prefix: str = "rag/vector_data",
        include_cache: bool = False,
    ) -> dict:
        """
        Upload ChromaDB vector data to S3.

        Args:
            chroma_db_path: Path to ChromaDB directory
            s3_key_prefix: S3 key prefix for the upload
            include_cache: Whether to include query cache in upload

        Returns:
            Dictionary with upload statistics
        """
        self.logger.info("Starting upload of vector data", path=chroma_db_path)

        chroma_path = Path(chroma_db_path)
        if not chroma_path.exists():
            raise ValueError(f"ChromaDB path does not exist: {chroma_db_path}")

        # Create temporary tar file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        s3_key = f"{s3_key_prefix}/chromadb_backup_{timestamp}.tar.gz"

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Create tar.gz archive
            self.logger.info("Creating archive of ChromaDB data")
            with tarfile.open(tmp_path, "w:gz") as tar:
                # Add ChromaDB directories
                for retriever_type_dir in chroma_path.iterdir():
                    if retriever_type_dir.is_dir():
                        self.logger.info(
                            "Adding directory to archive", name=retriever_type_dir.name
                        )
                        tar.add(
                            retriever_type_dir,
                            arcname=f"chromadb/{retriever_type_dir.name}",
                            recursive=True,
                        )

                # Optionally include cache
                if include_cache:
                    cache_path = chroma_path.parent / "query_cache"
                    if cache_path.exists():
                        self.logger.info("Adding query cache to archive")
                        tar.add(
                            cache_path,
                            arcname="query_cache",
                            recursive=True,
                        )

            # Upload to S3
            file_size = os.path.getsize(tmp_path)
            self.logger.info(
                "Uploading to S3", size_mb=file_size / (1024 * 1024), s3_key=s3_key
            )

            self.s3_client.upload_file(
                tmp_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    "Metadata": {
                        "timestamp": timestamp,
                        "uploaded_at": datetime.now().isoformat(),
                    }
                },
            )

            self.logger.info("Successfully uploaded to S3", s3_key=s3_key)

            return {
                "status": "success",
                "s3_key": s3_key,
                "bucket": self.bucket_name,
                "file_size_mb": file_size / (1024 * 1024),
                "timestamp": timestamp,
            }

        except ClientError as e:
            self.logger.exception("Error uploading to S3", error=str(e))
            raise
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def download_vector_data(
        self,
        s3_key: Optional[str] = None,
        chroma_db_path: str = None,
        extract_to: Optional[str] = None,
        latest: bool = True,
    ) -> dict:
        """
        Download ChromaDB vector data from S3.

        Args:
            s3_key: Specific S3 key to download (if None, downloads latest)
            chroma_db_path: Path where ChromaDB should be extracted
            extract_to: Alternative path to extract to (overrides chroma_db_path)
            latest: If True and s3_key is None, download latest backup

        Returns:
            Dictionary with download statistics
        """
        self.logger.info("Starting download of vector data from S3")

        # Determine S3 key
        if s3_key is None and latest:
            s3_key = self._get_latest_backup_key()
            if not s3_key:
                raise ValueError("No backup found in S3")

        if not s3_key:
            raise ValueError("s3_key must be provided or latest=True")

        # Determine extraction path
        if extract_to:
            extract_path = Path(extract_to)
        elif chroma_db_path:
            extract_path = Path(chroma_db_path).parent
        else:
            extract_path = Path("data")

        extract_path.mkdir(parents=True, exist_ok=True)

        # Download from S3
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.logger.info("Downloading from S3", s3_key=s3_key)
            self.s3_client.download_file(self.bucket_name, s3_key, tmp_path)

            file_size = os.path.getsize(tmp_path)
            self.logger.info("Downloaded from S3", size_mb=file_size / (1024 * 1024))

            # Extract archive
            self.logger.info("Extracting archive", path=extract_path)
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(path=extract_path)

            self.logger.info("Successfully extracted vector data")

            return {
                "status": "success",
                "s3_key": s3_key,
                "extract_path": str(extract_path),
                "file_size_mb": file_size / (1024 * 1024),
            }

        except ClientError as e:
            self.logger.exception("Error downloading from S3", error=str(e))
            raise
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _get_latest_backup_key(self, prefix: str = "rag/vector_data") -> Optional[str]:
        """
        Get the latest backup key from S3.

        Args:
            prefix: S3 key prefix to search

        Returns:
            Latest backup key or None
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
            )

            if "Contents" not in response:
                return None

            # Sort by last modified date (most recent first)
            objects = sorted(
                response["Contents"],
                key=lambda x: x["LastModified"],
                reverse=True,
            )

            if objects:
                return objects[0]["Key"]

            return None

        except ClientError as e:
            self.logger.exception("Error listing S3 objects", error=str(e))
            return None

    def list_backups(self, prefix: str = "rag/vector_data", limit: int = 10) -> list:
        """
        List available backups in S3.

        Args:
            prefix: S3 key prefix to search
            limit: Maximum number of backups to return

        Returns:
            List of backup dictionaries with key, size, and last_modified
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
            )

            if "Contents" not in response:
                return []

            backups = []
            for obj in response["Contents"][:limit]:
                backups.append(
                    {
                        "key": obj["Key"],
                        "size_mb": obj["Size"] / (1024 * 1024),
                        "last_modified": obj["LastModified"].isoformat(),
                    }
                )

            return sorted(backups, key=lambda x: x["last_modified"], reverse=True)

        except ClientError as e:
            self.logger.exception("Error listing backups", error=str(e))
            return []


def upload_vector_data_to_s3(
    chroma_db_path: str,
    bucket_name: Optional[str] = None,
    s3_key_prefix: str = "rag/vector_data",
) -> dict:
    """
    Convenience function to upload vector data to S3.

    Args:
        chroma_db_path: Path to ChromaDB directory
        bucket_name: S3 bucket name (optional, uses env var if not provided)
        s3_key_prefix: S3 key prefix

    Returns:
        Dictionary with upload statistics
    """
    manager = RAGS3Manager(bucket_name=bucket_name)
    return manager.upload_vector_data(chroma_db_path, s3_key_prefix)


def download_vector_data_from_s3(
    chroma_db_path: str,
    bucket_name: Optional[str] = None,
    latest: bool = True,
    s3_key: Optional[str] = None,
) -> dict:
    """
    Convenience function to download vector data from S3.

    Args:
        chroma_db_path: Path where ChromaDB should be extracted
        bucket_name: S3 bucket name (optional, uses env var if not provided)
        latest: If True, download latest backup
        s3_key: Specific S3 key to download (overrides latest)

    Returns:
        Dictionary with download statistics
    """
    manager = RAGS3Manager(bucket_name=bucket_name)
    return manager.download_vector_data(
        s3_key=s3_key,
        chroma_db_path=chroma_db_path,
        latest=latest,
    )
