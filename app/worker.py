import json
import logging
import os
from datetime import datetime

from celery import Celery

from app.database import SessionLocal
from app.models import AnalysisJob

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("atletico", broker=_REDIS_URL, backend=_REDIS_URL)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@celery_app.task(name="run_inference")
def run_inference(task_id: str):
    from app.analysis import run_analysis

    logger.info("[%s] Task received", task_id)
    db = SessionLocal()
    try:
        job = db.get(AnalysisJob, task_id)
        job.status = "processing"
        db.commit()
        logger.info("[%s] Status → processing  video=%s", task_id, job.video_path)

        output_path = f"uploads/{task_id}_annotated.mp4"
        stats = run_analysis(job.video_path, output_path)
        job.verdict = json.dumps(stats)
        job.status = "completed"
        job.annotated_video_url = output_path
        job.completed_at = datetime.utcnow()
        db.commit()
        logger.info("[%s] Status → completed  stats=%s", task_id, stats)
    except Exception as exc:
        logger.error("[%s] Task failed: %s", task_id, exc, exc_info=True)
        job.status = "failed"
        db.commit()
        raise
    finally:
        db.close()
