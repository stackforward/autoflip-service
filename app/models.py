"""Pydantic models for AutoFlip API request/response schemas."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator


class TaskStatus(str, Enum):
    queued = "queued"
    downloading = "downloading"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class TaskCreateRequest(BaseModel):
    url: Optional[str] = None
    path: Optional[str] = None

    @model_validator(mode="after")
    def require_url_or_path(self):
        if not self.url and not self.path:
            raise ValueError("Either 'url' or 'path' must be provided")
        if self.url and self.path:
            raise ValueError("Provide only one of 'url' or 'path', not both")
        return self


class TaskCreateResponse(BaseModel):
    task_id: str
    status: TaskStatus


class TaskDetail(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float = 0.0
    input: Optional[str] = None
    output_path: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class TaskSummary(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float = 0.0
    created_at: Optional[datetime] = None


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = None
    active_tasks: int
    max_parallel_tasks: int
