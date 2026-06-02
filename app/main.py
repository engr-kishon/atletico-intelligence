import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Generator

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.database import Base, SessionLocal, engine
from app.models import AnalysisJob
from app.schemas import AnalyzeVideoResponse, AnalysisResultResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Atlético Intelligence", lifespan=lifespan)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
    }


@app.post("/analyze-video", response_model=AnalyzeVideoResponse)
async def analyze_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    from app.worker import run_inference

    task_id = str(uuid.uuid4())
    video_path = f"uploads/{task_id}.mp4"

    with open(video_path, "wb") as f:
        f.write(await file.read())

    job = AnalysisJob(id=task_id, status="pending", video_path=video_path)
    db.add(job)
    db.commit()

    run_inference.delay(task_id)

    return AnalyzeVideoResponse(task_id=task_id, message="Video queued for analysis")


@app.get("/get-analysis-result", response_model=AnalysisResultResponse)
def get_analysis_result(task_id: str, db: Session = Depends(get_db)):
    job = db.get(AnalysisJob, task_id)
    if not job:
        raise HTTPException(status_code=404, detail="Task not found")

    return AnalysisResultResponse(
        task_id=job.id,
        status=job.status,
        verdict=job.verdict,
        video_url=job.annotated_video_url,
        created_at=job.created_at,
        completed_at=job.completed_at,
    )
