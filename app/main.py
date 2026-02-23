"""FastAPI application for AutoFlip video processing service."""

import asyncio
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import httpx
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .models import (
    HealthResponse,
    TaskCreateRequest,
    TaskCreateResponse,
    TaskDetail,
    TaskStatus,
    TaskSummary,
)
from .worker import load_model, run_autoflip

# ── Config ───────────────────────────────────────────────────────────────────
MAX_PARALLEL_TASKS = int(os.environ.get("MAX_PARALLEL_TASKS", "2"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DOWNLOADS_DIR = DATA_DIR / "downloads"
OUTPUTS_DIR = DATA_DIR / "outputs"

DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Task state ───────────────────────────────────────────────────────────────
tasks: Dict[str, TaskDetail] = {}
semaphore = asyncio.Semaphore(MAX_PARALLEL_TASKS)
executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL_TASKS)

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoFlip Service", version="1.0.0")


@app.on_event("startup")
async def startup():
    """Pre-load the YOLO model at startup."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(executor, load_model)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sanitize_filename(name: str) -> str:
    name = os.path.basename(name)
    name = re.sub(r"[^\w.\-]", "_", name)
    return name or "video.mp4"


async def _download_file(url: str, dest: Path) -> None:
    async with httpx.AsyncClient(follow_redirects=True, timeout=600) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                    f.write(chunk)


def _run_task_sync(task_id: str, input_path: str, output_path: str) -> None:
    """Run autoflip synchronously (called in thread pool)."""
    task = tasks[task_id]
    task.status = TaskStatus.processing
    task.started_at = datetime.now(timezone.utc)

    def progress_cb(p: float):
        task.progress = round(p * 100, 1)

    try:
        run_autoflip(
            input_path,
            output_path,
            progress_callback=progress_cb,
            video_format=(task.video_format.value if task.video_format else "PORTRAIT"),
            target_aspect_ratio=task.target_aspect_ratio,
        )
        task.status = TaskStatus.completed
        task.output_path = output_path
        task.progress = 100.0
    except Exception as e:
        task.status = TaskStatus.failed
        task.error = str(e)
    finally:
        task.completed_at = datetime.now(timezone.utc)


async def _process_task(task_id: str, url: str | None, path: str | None) -> None:
    """Acquire semaphore, optionally download, then run autoflip in thread."""
    task = tasks[task_id]
    try:
        async with semaphore:
            # Resolve input path
            if url:
                task.status = TaskStatus.downloading
                filename = _sanitize_filename(url.split("/")[-1].split("?")[0])
                if not filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
                    filename += ".mp4"
                download_path = DOWNLOADS_DIR / f"{task_id}_{filename}"
                await _download_file(url, download_path)
                input_path = str(download_path)
            else:
                input_path = path

            task.input = input_path
            output_path = str(OUTPUTS_DIR / f"{task_id}.mp4")

            # Run in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, _run_task_sync, task_id, input_path, output_path)
    except Exception as e:
        task.status = TaskStatus.failed
        task.error = str(e)
        task.completed_at = datetime.now(timezone.utc)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.post("/tasks", response_model=TaskCreateResponse, status_code=201)
async def create_task(req: TaskCreateRequest):
    # Check concurrency — reject if all slots are taken
    active = sum(
        1 for t in tasks.values()
        if t.status in (TaskStatus.queued, TaskStatus.downloading, TaskStatus.processing)
    )
    if active >= MAX_PARALLEL_TASKS:
        raise HTTPException(status_code=429, detail="Max parallel tasks reached. Try again later.")

    # Validate local path
    if req.path:
        if not os.path.isfile(req.path):
            raise HTTPException(status_code=400, detail=f"File not found: {req.path}")

    task_id = str(uuid.uuid4())
    task = TaskDetail(
        task_id=task_id,
        status=TaskStatus.queued,
        input=req.url or req.path,
        video_format=req.video_format,
        target_aspect_ratio=req.target_aspect_ratio,
        created_at=datetime.now(timezone.utc),
    )
    tasks[task_id] = task

    # Fire and forget
    asyncio.create_task(_process_task(task_id, req.url, req.path))

    return TaskCreateResponse(task_id=task_id, status=TaskStatus.queued)


@app.get("/tasks/{task_id}", response_model=TaskDetail)
async def get_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.get("/tasks", response_model=list[TaskSummary])
async def list_tasks():
    return [
        TaskSummary(
            task_id=t.task_id,
            status=t.status,
            progress=t.progress,
            video_format=t.video_format,
            created_at=t.created_at,
        )
        for t in tasks.values()
    ]


@app.get("/tasks/{task_id}/download")
async def download_task(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    task = tasks[task_id]
    if task.status != TaskStatus.completed:
        raise HTTPException(status_code=400, detail=f"Task is {task.status.value}, not completed")
    if not task.output_path or not os.path.isfile(task.output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(
        task.output_path,
        media_type="video/mp4",
        filename=f"autoflip_{task_id}.mp4",
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    gpu_available = torch.cuda.is_available()
    active = sum(
        1 for t in tasks.values()
        if t.status in (TaskStatus.queued, TaskStatus.downloading, TaskStatus.processing)
    )
    return HealthResponse(
        status="ok",
        gpu_available=gpu_available,
        gpu_name=torch.cuda.get_device_name(0) if gpu_available else None,
        cuda_version=torch.version.cuda if gpu_available else None,
        active_tasks=active,
        max_parallel_tasks=MAX_PARALLEL_TASKS,
    )
