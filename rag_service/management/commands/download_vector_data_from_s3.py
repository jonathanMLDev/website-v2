"""
Django management command to download vector data from S3.

This command should be run during environment setup to restore ChromaDB data.

Usage:
    python manage.py download_vector_data_from_s3
    python manage.py download_vector_data_from_s3 --s3-key rag/vector_data/chromadb_backup_20250101_120000.tar.gz
    python manage.py download_vector_data_from_s3 --list-backups
"""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from rag_service.s3_utils import RAGS3Manager, download_vector_data_from_s3


class Command(BaseCommand):
    help = "Download vector data (ChromaDB) from S3"

    def add_arguments(self, parser):
        parser.add_argument(
            "--s3-key",
            type=str,
            help="Specific S3 key to download (if not provided, downloads latest)",
        )
        parser.add_argument(
            "--chroma-db-path",
            type=str,
            help="Path where ChromaDB should be extracted (default: data/chromadb)",
        )
        parser.add_argument(
            "--bucket-name",
            type=str,
            help="S3 bucket name (default: from RAG_S3_BUCKET_NAME env var or settings)",
        )
        parser.add_argument(
            "--list-backups",
            action="store_true",
            help="List available backups in S3",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite existing ChromaDB data if it exists",
        )

    def handle(self, *args, **options):
        if options["list_backups"]:
            self._list_backups(options)
            return

        self.stdout.write(
            self.style.SUCCESS("Starting download of vector data from S3...")
        )

        # Determine ChromaDB path
        chroma_db_path = options.get("chroma_db_path")
        if not chroma_db_path:
            # Try to get from settings or use default
            chroma_db_path = getattr(
                settings,
                "RAG_CHROMA_DIR",
                Path(settings.BASE_DIR) / "data" / "chromadb",
            )

        chroma_path = Path(chroma_db_path)

        # Check if ChromaDB already exists
        if chroma_path.exists() and not options["force"]:
            self.stdout.write(
                self.style.WARNING(
                    f"ChromaDB already exists at {chroma_path}. "
                    "Use --force to overwrite."
                )
            )
            return

        try:
            # Download from S3
            result = download_vector_data_from_s3(
                chroma_db_path=str(chroma_path),
                bucket_name=options.get("bucket_name"),
                latest=options.get("s3_key") is None,
                s3_key=options.get("s3_key"),
            )

            if result["status"] == "success":
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Successfully downloaded vector data from S3!\n"
                        f"  S3 Key: {result['s3_key']}\n"
                        f"  Extracted to: {result['extract_path']}\n"
                        f"  Size: {result['file_size_mb']:.2f} MB"
                    )
                )
            else:
                raise CommandError(
                    f"Download failed: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            raise CommandError(f"Error downloading vector data: {e}")

    def _list_backups(self, options):
        """List available backups in S3."""
        self.stdout.write("Listing available backups in S3...")

        try:
            manager = RAGS3Manager(bucket_name=options.get("bucket_name"))
            backups = manager.list_backups(limit=20)

            if not backups:
                self.stdout.write(self.style.WARNING("No backups found in S3"))
                return

            self.stdout.write(
                self.style.SUCCESS(f"\nFound {len(backups)} backup(s):\n")
            )
            for i, backup in enumerate(backups, 1):
                self.stdout.write(
                    f"  {i}. {backup['key']}\n"
                    f"     Size: {backup['size_mb']:.2f} MB\n"
                    f"     Last Modified: {backup['last_modified']}\n"
                )

        except Exception as e:
            raise CommandError(f"Error listing backups: {e}")
