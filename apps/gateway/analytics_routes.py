"""Analytics API endpoints for customer-facing dashboard."""

import csv
import io
import logging
from datetime import date, timedelta
from typing import Literal, Optional

from auth import get_current_user
from database import (
    AnalysisHistory,
    CostTracking,
    ModelPerformance,
    get_db,
)
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import case, func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["Analytics"])

_EXPORT_ROW_LIMIT = 10_000


class AnalyticsTimeRange(BaseModel):
    """Query parameters for time-ranged analytics with period granularity."""

    period: Literal["daily", "weekly", "monthly"] = "daily"
    from_date: Optional[date] = None
    to_date: Optional[date] = None


class AnalyticsDateRange(BaseModel):
    """Query parameters for date-ranged analytics."""

    from_date: Optional[date] = None
    to_date: Optional[date] = None


class ExportParams(AnalyticsDateRange):
    """Query parameters for data export."""

    format: Literal["csv"] = "csv"


def _default_range(
    from_date: Optional[date],
    to_date: Optional[date],
) -> tuple[date, date]:
    """Return a (from_date, to_date) pair, defaulting to last 30 days.

    Args:
        from_date: Optional start of the range.
        to_date: Optional end of the range.

    Returns:
        Tuple of (from_date, to_date) with defaults applied.
    """
    if to_date is None:
        to_date = date.today()
    if from_date is None:
        from_date = to_date - timedelta(days=30)
    return from_date, to_date


def _period_trunc(period: str, column):
    """Return a SQLAlchemy expression that truncates a datetime column to the given period.

    Args:
        period: One of 'daily', 'weekly', 'monthly'.
        column: The SQLAlchemy column to truncate.

    Returns:
        A SQLAlchemy expression for date truncation.
    """
    if period == "monthly":
        return func.date_trunc("month", column)
    if period == "weekly":
        return func.date_trunc("week", column)
    return func.date(column)


@router.get("/summary")
async def get_summary(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all-time aggregate statistics for the authenticated user.

    Args:
        current_user: JWT payload from the Authorization header.
        db: SQLAlchemy database session.

    Returns:
        Dict with total_analyses, credit_balance, total_credits_consumed,
        avg_confidence, avg_inference_ms, verdicts, media_types, fake_rate.
    """
    user_id = current_user["sub"]

    total = (
        db.query(func.count(AnalysisHistory.id))
        .filter(AnalysisHistory.user_id == user_id)
        .scalar()
        or 0
    )

    avg_conf = (
        db.query(func.avg(AnalysisHistory.confidence))
        .filter(AnalysisHistory.user_id == user_id)
        .scalar()
    )

    verdict_counts = dict(
        db.query(AnalysisHistory.verdict, func.count())
        .filter(AnalysisHistory.user_id == user_id)
        .group_by(AnalysisHistory.verdict)
        .all()
    )

    media_counts = dict(
        db.query(AnalysisHistory.media_type, func.count())
        .filter(AnalysisHistory.user_id == user_id)
        .group_by(AnalysisHistory.media_type)
        .all()
    )

    total_credits = (
        db.query(func.sum(CostTracking.credits_consumed))
        .filter(CostTracking.user_id == user_id)
        .scalar()
        or 0
    )

    avg_speed = (
        db.query(func.avg(CostTracking.total_inference_ms))
        .filter(CostTracking.user_id == user_id)
        .scalar()
    )

    return {
        "total_analyses": total,
        "total_credits_consumed": total_credits,
        "avg_confidence": round(avg_conf, 4) if avg_conf else None,
        "avg_inference_ms": round(avg_speed, 1) if avg_speed else None,
        "verdicts": verdict_counts,
        "media_types": media_counts,
        "fake_rate": (
            round(verdict_counts.get("fake", 0) / total, 4) if total > 0 else 0
        ),
    }


@router.get("/usage")
async def get_usage(
    period: Literal["daily", "weekly", "monthly"] = Query("daily"),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return usage trends grouped by date period and media type.

    Args:
        period: Grouping granularity — daily, weekly, or monthly.
        from_date: Start of the date range (defaults to 30 days ago).
        to_date: End of the date range (defaults to today).
        current_user: JWT payload from the Authorization header.
        db: SQLAlchemy database session.

    Returns:
        List of dicts, each with period_start, media_type, and count.
    """
    user_id = current_user["sub"]
    from_date, to_date = _default_range(from_date, to_date)

    period_col = _period_trunc(period, AnalysisHistory.timestamp)

    rows = (
        db.query(
            period_col.label("period_start"),
            AnalysisHistory.media_type,
            func.count().label("count"),
        )
        .filter(
            AnalysisHistory.user_id == user_id,
            func.date(AnalysisHistory.timestamp) >= from_date,
            func.date(AnalysisHistory.timestamp) <= to_date,
        )
        .group_by("period_start", AnalysisHistory.media_type)
        .order_by("period_start")
        .all()
    )

    return [
        {
            "period_start": str(row.period_start),
            "media_type": row.media_type,
            "count": row.count,
        }
        for row in rows
    ]


@router.get("/models")
async def get_model_performance(
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return aggregate detection performance statistics.

    Returns a single summary across all detection modules — no per-model
    breakdown is exposed to prevent fingerprinting of internal architecture.

    Args:
        from_date: Start of the date range (defaults to 30 days ago).
        to_date: End of the date range (defaults to today).
        current_user: JWT payload from the Authorization header.
        db: SQLAlchemy database session.

    Returns:
        Dict with aggregate avg_time_ms, total_runs, and error_rate.
    """
    user_id = current_user["sub"]
    from_date, to_date = _default_range(from_date, to_date)

    row = (
        db.query(
            func.avg(ModelPerformance.inference_time_ms).label("avg_time_ms"),
            func.count().label("total_runs"),
            func.sum(case((ModelPerformance.status == "error", 1), else_=0)).label(
                "error_count"
            ),
        )
        .filter(
            ModelPerformance.user_id == user_id,
            func.date(ModelPerformance.created_at) >= from_date,
            func.date(ModelPerformance.created_at) <= to_date,
        )
        .one()
    )

    total = row.total_runs or 0
    return {
        "avg_time_ms": (
            round(row.avg_time_ms, 1) if row.avg_time_ms is not None else None
        ),
        "total_runs": total,
        "error_rate": (round(row.error_count / total, 4) if total > 0 else 0),
    }


@router.get("/costs")
async def get_costs(
    period: Literal["daily", "weekly", "monthly"] = Query("daily"),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return credit consumption trends over time.

    Args:
        period: Grouping granularity — daily, weekly, or monthly.
        from_date: Start of the date range (defaults to 30 days ago).
        to_date: End of the date range (defaults to today).
        current_user: JWT payload from the Authorization header.
        db: SQLAlchemy database session.

    Returns:
        List of dicts with period_start, credits_consumed, analyses, cost_per_analysis.
    """
    user_id = current_user["sub"]
    from_date, to_date = _default_range(from_date, to_date)

    period_col = _period_trunc(period, CostTracking.created_at)

    rows = (
        db.query(
            period_col.label("period_start"),
            func.sum(CostTracking.credits_consumed).label("credits_consumed"),
            func.count().label("analyses"),
        )
        .filter(
            CostTracking.user_id == user_id,
            func.date(CostTracking.created_at) >= from_date,
            func.date(CostTracking.created_at) <= to_date,
        )
        .group_by("period_start")
        .order_by("period_start")
        .all()
    )

    return [
        {
            "period_start": str(row.period_start),
            "credits_consumed": row.credits_consumed,
            "analyses": row.analyses,
            "cost_per_analysis": (
                round(row.credits_consumed / row.analyses, 4) if row.analyses > 0 else 0
            ),
        }
        for row in rows
    ]


@router.get("/export")
async def export_csv(
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export analysis history as a CSV file (max 10,000 rows).

    Args:
        from_date: Start of the date range (defaults to 30 days ago).
        to_date: End of the date range (defaults to today).
        current_user: JWT payload from the Authorization header.
        db: SQLAlchemy database session.

    Returns:
        StreamingResponse with Content-Disposition: attachment; filename=export.csv
    """
    user_id = current_user["sub"]
    from_date, to_date = _default_range(from_date, to_date)

    rows = (
        db.query(AnalysisHistory)
        .filter(
            AnalysisHistory.user_id == user_id,
            func.date(AnalysisHistory.timestamp) >= from_date,
            func.date(AnalysisHistory.timestamp) <= to_date,
        )
        .order_by(AnalysisHistory.timestamp.desc())
        .limit(_EXPORT_ROW_LIMIT)
        .all()
    )

    def _generate():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "id",
                "request_id",
                "media_type",
                "media_name",
                "verdict",
                "confidence",
                "inference_time",
                "timestamp",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.id,
                    row.request_id,
                    row.media_type,
                    row.media_name,
                    row.verdict,
                    row.confidence,
                    row.inference_time,
                    row.timestamp,
                ]
            )
        buf.seek(0)
        yield buf.read()

    headers = {
        "Content-Disposition": "attachment; filename=deepsafe_export.csv",
    }
    return StreamingResponse(
        _generate(),
        media_type="text/csv",
        headers=headers,
    )
