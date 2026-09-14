"""
Celery application factory for DeepSafe.

This module is the entry point for the Celery worker:
    celery -A celery_app worker --beat --loglevel=info --concurrency=4

It is also imported by main.py to enqueue tasks via run_detection.delay().
include=["main", "tasks"] tells the worker to import both modules at startup
so all @celery_app.task-decorated functions are registered, including the
periodic beat tasks defined in tasks.py.
"""

import os

from celery import Celery
from celery.schedules import crontab

celery_app = Celery(
    "deepsafe",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
    include=["main", "tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
)

celery_app.conf.beat_schedule = {
    "refresh-usage-summary": {
        "task": "tasks.refresh_usage_summary",
        "schedule": crontab(minute=0),  # every hour, on the hour
    },
}
