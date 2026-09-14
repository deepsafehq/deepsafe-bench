"""
Shared infrastructure clients and FastAPI dependency helpers.

Provides Redis and MinIO clients plus the MINIO_JOBS_BUCKET constant used
across the gateway.  Import from this module rather than constructing clients
ad-hoc to ensure a single shared connection is reused everywhere.
"""

import logging
import os

import redis as redis_lib
from minio import Minio

logger = logging.getLogger(__name__)

# Redis client for job state. decode_responses=True so values are str, not bytes.
redis_client = redis_lib.from_url(
    os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)

# MinIO client for temporary file storage during job processing.
_minio_url = os.getenv("MINIO_URL", "localhost:9000")
_minio_access_key = os.getenv("MINIO_ACCESS_KEY", "")
_minio_secret_key = os.getenv("MINIO_SECRET_KEY", "")
_minio_secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

if not _minio_access_key or not _minio_secret_key:
    logger.warning(
        "MINIO_ACCESS_KEY or MINIO_SECRET_KEY not set. "
        "Async detection (MinIO) will be unavailable."
    )
    minio_client = None
else:
    minio_client = Minio(
        _minio_url,
        access_key=_minio_access_key,
        secret_key=_minio_secret_key,
        secure=_minio_secure,
    )

MINIO_JOBS_BUCKET = "jobs"
