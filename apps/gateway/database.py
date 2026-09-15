"""
DeepSafe Database Module
SQLAlchemy ORM models and database session management for storing analysis history.
"""

import logging
import os
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

# Database configuration.
#
# Production must set DATABASE_URL to PostgreSQL. Local installs should not
# have to stand up Postgres just to try the thing, so without DATABASE_URL we
# fall back to a SQLite file and say so. Setting DEEPSAFE_ENV=production
# restores the hard failure, so a real deployment cannot silently run on
# SQLite because someone forgot to configure it.
DATABASE_URL = os.getenv("DATABASE_URL", "")
if not DATABASE_URL:
    _is_testing = (
        "pytest" in os.getenv("_", "")
        or "PYTEST_CURRENT_TEST" in os.environ
        or "pytest" in sys.modules
    )
    from config import IS_PRODUCTION as _is_production

    if _is_testing:
        DATABASE_URL = "sqlite:///./deepsafe_test.db"
        logger.info("Using SQLite test database (DATABASE_URL not set).")
    elif _is_production:
        raise RuntimeError(
            "DATABASE_URL is not set and DEEPSAFE_ENV=production. "
            "Set DATABASE_URL to a PostgreSQL connection string."
        )
    else:
        DATABASE_URL = "sqlite:///./deepsafe_local.db"
        logger.warning(
            "DATABASE_URL not set; using SQLite at ./deepsafe_local.db. "
            "Fine for a local install. Set DATABASE_URL to PostgreSQL for "
            "anything multi-user."
        )

# Ensure SSL for all PostgreSQL connections (required by Supabase).
if DATABASE_URL.startswith("postgresql") and "sslmode" not in DATABASE_URL:
    separator = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL = f"{DATABASE_URL}{separator}sslmode=require"

_engine_kwargs = {}
if DATABASE_URL.startswith("postgresql"):
    _engine_kwargs.update(
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300,
    )

engine = create_engine(DATABASE_URL, **_engine_kwargs)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


@contextmanager
def get_db_session():
    """Context manager for safe database session handling.

    Usage::

        with get_db_session() as db:
            db.add(record)
            db.commit()

    Rolls back on exception, always closes the session.
    """
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class AnalysisHistory(Base):
    """Store analysis results for auditing and reporting."""

    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(String, index=True, nullable=True)  # Supabase user UUID
    media_type = Column(String, nullable=False)  # image, video, audio
    media_name = Column(String, nullable=True)

    # Analysis Results
    verdict = Column(String, nullable=False)  # "real" or "fake"
    confidence = Column(Float, nullable=False)
    ensemble_method = Column(String, nullable=False)  # "stacking", "voting", "average"
    ensemble_score = Column(Float, nullable=False)

    # Metadata
    inference_time = Column(Float, nullable=True)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Optional: Store full JSON response
    full_response = Column(Text, nullable=True)

    def __repr__(self):
        return f"<AnalysisHistory(id={self.id}, request_id={self.request_id}, verdict={self.verdict})>"


class ModelPerformance(Base):
    """Per-model metrics for each analysis run."""

    __tablename__ = "model_performance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(
        Integer, ForeignKey("analysis_history.id"), nullable=False, index=True
    )
    user_id = Column(String, index=True, nullable=True)
    model_name = Column(String, nullable=False)
    model_score = Column(Float, nullable=True)
    raw_output = Column(Text, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    model_version = Column(String, nullable=True)
    status = Column(String, nullable=False)  # success, error, timeout
    error_message = Column(Text, nullable=True)
    media_type = Column(String, nullable=False)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )


class CostTracking(Base):
    """Per-analysis cost attribution for unit economics."""

    __tablename__ = "cost_tracking"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_id = Column(
        Integer, ForeignKey("analysis_history.id"), nullable=False, index=True
    )
    user_id = Column(String, index=True, nullable=True)
    credits_consumed = Column(Integer, nullable=False, default=1)
    model_count = Column(Integer, nullable=False)
    total_inference_ms = Column(Float, nullable=True)
    media_type = Column(String, nullable=False)
    file_size_bytes = Column(BigInteger, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )


class AnalyticsEvent(Base):
    """Product analytics events — replaces PostHog."""

    __tablename__ = "analytics_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=True)
    event_name = Column(String, index=True, nullable=False)
    properties = Column(Text, nullable=True)  # JSON string
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for FastAPI routes to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
