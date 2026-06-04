from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class AnalyzeVideoResponse(BaseModel):
    task_id: str
    message: str


class AnalysisResultResponse(BaseModel):
    task_id: str
    status: str
    verdict: Optional[str] = None
    video_url: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
