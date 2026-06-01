"""Scheduler for the daily stock picks pipeline.

Uses APScheduler to run the daily picks pipeline at a user-configured
time.  The scheduler runs in-process alongside the Streamlit app and
persists its schedule to a JSON file so it survives restarts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "scheduler_config.json"
_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "scheduler_log.json"

_scheduler: Optional[BackgroundScheduler] = None
_lock = Lock()

JOB_ID = "daily_picks_pipeline"


# ------------------------------------------------------------------
# Config persistence
# ------------------------------------------------------------------

def _load_config() -> dict:
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _save_config(cfg: dict) -> None:
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _append_log(entry: dict) -> None:
    """Append a run-log entry (max 100 entries kept)."""
    logs: list[dict] = []
    if _LOG_PATH.exists():
        try:
            with open(_LOG_PATH) as f:
                logs = json.load(f)
        except Exception:
            logs = []
    logs.append(entry)
    logs = logs[-100:]  # keep last 100
    with open(_LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)


def get_run_log() -> list[dict]:
    """Return the scheduler run log."""
    if _LOG_PATH.exists():
        try:
            with open(_LOG_PATH) as f:
                return json.load(f)
        except Exception:
            return []
    return []


# ------------------------------------------------------------------
# Pipeline runner (called by scheduler)
# ------------------------------------------------------------------

def _run_pipeline_job() -> None:
    """Execute the daily picks pipeline, log the result, and send email."""
    start = datetime.now()
    logger.info("Scheduler: starting daily picks pipeline run...")

    try:
        from stock_predictor.pipeline.daily_picks import run_daily_picks

        result = run_daily_picks()
        n_picks = len(result) if result is not None else 0
        elapsed = (datetime.now() - start).total_seconds()

        email_sent = False
        try:
            from stock_predictor.pipeline.email_notifier import (
                is_email_configured,
                send_picks_email,
            )
            if is_email_configured():
                email_sent = send_picks_email(
                    result if result is not None else __import__("pandas").DataFrame()
                )
        except Exception as email_err:
            logger.warning("Scheduler: email notification failed — %s", email_err)

        _append_log({
            "timestamp": start.isoformat(),
            "status": "success",
            "picks": n_picks,
            "elapsed_seconds": round(elapsed, 1),
            "email_sent": email_sent,
        })
        logger.info("Scheduler: pipeline completed — %d picks in %.1fs (email=%s)",
                     n_picks, elapsed, email_sent)

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        _append_log({
            "timestamp": start.isoformat(),
            "status": "error",
            "error": str(e),
            "elapsed_seconds": round(elapsed, 1),
            "email_sent": False,
        })
        logger.exception("Scheduler: pipeline failed — %s", e)


# ------------------------------------------------------------------
# Scheduler management
# ------------------------------------------------------------------

def get_scheduler() -> BackgroundScheduler:
    """Return the singleton scheduler, creating it if needed."""
    global _scheduler
    with _lock:
        if _scheduler is None:
            _scheduler = BackgroundScheduler(daemon=True)
            _scheduler.start()
    return _scheduler


def get_schedule_config() -> dict:
    """Return the current schedule configuration."""
    return _load_config()


def schedule_pipeline(
    *,
    hour: int = 6,
    minute: int = 0,
    frequency: str = "daily",
    day_of_week: str = "mon-fri",
) -> dict:
    """Schedule (or reschedule) the daily picks pipeline.

    Args:
        hour: Hour to run (0-23, UTC).
        minute: Minute to run (0-59).
        frequency: 'daily' or 'weekly'.
        day_of_week: Cron day-of-week (e.g. 'mon-fri', 'mon', '0-6').
            Ignored when frequency is 'daily'.

    Returns:
        Dict with the schedule details.
    """
    scheduler = get_scheduler()

    # Remove existing job if any
    if scheduler.get_job(JOB_ID):
        scheduler.remove_job(JOB_ID)

    if frequency == "weekly":
        trigger = CronTrigger(
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
        )
    else:
        # daily — default to weekdays only (market days)
        trigger = CronTrigger(
            day_of_week="mon-fri",
            hour=hour,
            minute=minute,
        )

    scheduler.add_job(
        _run_pipeline_job,
        trigger=trigger,
        id=JOB_ID,
        replace_existing=True,
        name="Daily Stock Picks Pipeline",
    )

    cfg = {
        "enabled": True,
        "hour": hour,
        "minute": minute,
        "frequency": frequency,
        "day_of_week": day_of_week,
        "updated_at": datetime.now().isoformat(),
    }
    _save_config(cfg)

    next_run = scheduler.get_job(JOB_ID).next_run_time
    logger.info("Pipeline scheduled: %s at %02d:%02d UTC, next run: %s",
                frequency, hour, minute, next_run)

    return {
        **cfg,
        "next_run": str(next_run) if next_run else None,
    }


def stop_schedule() -> None:
    """Remove the scheduled pipeline job."""
    scheduler = get_scheduler()
    if scheduler.get_job(JOB_ID):
        scheduler.remove_job(JOB_ID)

    cfg = _load_config()
    cfg["enabled"] = False
    cfg["updated_at"] = datetime.now().isoformat()
    _save_config(cfg)
    logger.info("Pipeline schedule stopped")


def is_scheduled() -> bool:
    """Return True if a pipeline job is currently scheduled."""
    scheduler = get_scheduler()
    return scheduler.get_job(JOB_ID) is not None


def get_next_run() -> Optional[str]:
    """Return the next scheduled run time as an ISO string."""
    scheduler = get_scheduler()
    job = scheduler.get_job(JOB_ID)
    if job and job.next_run_time:
        return job.next_run_time.isoformat()
    return None


def restore_schedule() -> None:
    """Restore the schedule from saved config (call on app startup)."""
    cfg = _load_config()
    if cfg.get("enabled"):
        schedule_pipeline(
            hour=cfg.get("hour", 6),
            minute=cfg.get("minute", 0),
            frequency=cfg.get("frequency", "daily"),
            day_of_week=cfg.get("day_of_week", "mon-fri"),
        )
        logger.info("Restored pipeline schedule from config")
