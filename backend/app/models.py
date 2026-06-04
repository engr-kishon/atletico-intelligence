import uuid
from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String, default="pending")
    video_path: Mapped[str] = mapped_column(String)
    verdict: Mapped[str | None] = mapped_column(String, nullable=True)
    annotated_video_url: Mapped[str | None] = mapped_column(
        String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True)
