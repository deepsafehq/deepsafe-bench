"""API key model, generation, and verification utilities."""

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timezone

from database import Base
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID

KEY_PREFIX = "ds_live_"


class ApiKey(Base):
    """Stores hashed API keys for the public /v1/ API."""

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False, index=True)
    key_hash = Column(String, unique=True, nullable=False, index=True)
    key_prefix = Column(String, nullable=False)
    name = Column(String, nullable=False)
    tier = Column(String, nullable=False, default="free")
    scans_used = Column(Integer, nullable=False, default=0)
    period_start = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    last_used_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)


def generate_api_key() -> str:
    """Generate a new API key: ds_live_ + 32 random hex chars."""
    return KEY_PREFIX + secrets.token_hex(16)


def hash_key(key: str) -> str:
    """SHA-256 hash of the full API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_key(key: str, stored_hash: str) -> bool:
    """Check if a plaintext key matches a stored hash (constant-time)."""
    return hmac.compare_digest(hash_key(key), stored_hash)
