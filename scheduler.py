"""
Job Scheduler.
Runs all bot jobs on schedule.
Start this process and leave it running.
"""

import os
import sys
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger.add(
    "logs/scheduler.log",
    rotation="1 week",
    retention="1 month",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

scheduler = BlockingScheduler(timezone="America/New_York")


def job_morning_scan():
    """7:00 AM ET — data refresh + market scan + trade execution."""
    logger.info("🕖 JOB: Morning scan starting...")
    try:
        from engine import run_morning_scan
        run_morning_scan()
    except Exception as e:
        logger.error(f"Morning scan failed: {e}")


def job_midday_update():
    """12:00 PM ET — refresh forecasts with latest data."""
    logger.info("🕛 JOB: Midday update starting...")
    try:
        from engine import run_morning_scan
        run_morning_scan()
    except Exception as e:
        logger.error(f"Midday update failed: {e}")


def job_settlement_check():
    """8:45 AM ET — check for settled contracts."""
    logger.info("🕗 JOB: Settlement check starting...")
    try:
        from engine import run_settlement_check
        run_settlement_check()
    except Exception as e:
        logger.error(f"Settlement check failed: {e}")


def job_data_ingestion():
    """6:00 AM ET — pull fresh economic data."""
    logger.info("📥 JOB: Data ingestion starting...")
    try:
        from data.ingestion import run_data_ingestion
        run_data_ingestion()
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")


def job_circuit_breaker_check():
    """6:00 PM ET — daily risk check."""
    logger.info("🔒 JOB: Circuit breaker check starting...")
    try:
        from execution.risk_manager import RiskManager
        rm = RiskManager()
        rm.check_circuit_breakers()
    except Exception as e:
        logger.error(f"Circuit breaker check failed: {e}")


def job_monthly_retrain():
    """
    1st of every month at 2:00 AM — retrain model with latest data.
    """
    logger.info("🧠 JOB: Monthly model retrain starting...")
    try:
        from engine import train_and_save_model
        train_and_save_model()
    except Exception as e:
        logger.error(f"Monthly retrain failed: {e}")


def setup_jobs():
    """Register all scheduled jobs."""
    
    # Data ingestion — every trading day at 6:00 AM ET
    scheduler.add_job(
        job_data_ingestion,
        CronTrigger(day_of_week="mon-fri", hour=6, minute=0),
        id="data_ingestion",
        name="Daily Data Ingestion",
        misfire_grace_time=600
    )
    
    # Morning scan — every trading day at 7:00 AM ET
    scheduler.add_job(
        job_morning_scan,
        CronTrigger(day_of_week="mon-fri", hour=7, minute=0),
        id="morning_scan",
        name="Morning Market Scan",
        misfire_grace_time=600
    )
    
    # Settlement check — every trading day at 8:45 AM ET
    scheduler.add_job(
        job_settlement_check,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=45),
        id="settlement_check",
        name="Settlement Check",
        misfire_grace_time=300
    )
    
    # Midday refresh — every trading day at 12:00 PM ET
    scheduler.add_job(
        job_midday_update,
        CronTrigger(day_of_week="mon-fri", hour=12, minute=0),
        id="midday_update",
        name="Midday Market Update",
        misfire_grace_time=600
    )
    
    # Circuit breaker check — every day at 6:00 PM ET
    scheduler.add_job(
        job_circuit_breaker_check,
        CronTrigger(day_of_week="mon-fri", hour=18, minute=0),
        id="circuit_breaker",
        name="Daily Circuit Breaker Check",
        misfire_grace_time=3600
    )
    
    # Monthly retrain — 1st of every month at 2:00 AM
    scheduler.add_job(
        job_monthly_retrain,
        CronTrigger(day=1, hour=2, minute=0),
        id="monthly_retrain",
        name="Monthly Model Retrain",
        misfire_grace_time=7200
    )
    
    logger.info("✅ All jobs registered.")

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("  KALSHI TRADING BOT — SCHEDULER STARTING")
    logger.info(f"  Mode: {os.getenv('TRADING_MODE', 'paper').upper()}")
    logger.info("=" * 50)
    
    setup_jobs()
    
    try:
        logger.info("Scheduler running. Press Ctrl+C to stop.")
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user.")
        sys.exit(0)
