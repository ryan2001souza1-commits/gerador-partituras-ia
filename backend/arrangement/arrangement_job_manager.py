"""
Job manager em memória para arranjos (Etapa 7). Reiniciar perde jobs.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class ArrangementJob:
    job_id: str
    file_id: str
    status: str  # queued | running | completed | failed
    message: str
    created_at: str
    updated_at: str
    config: Optional[Dict] = None
    results: Optional[Dict] = None
    error: Optional[str] = None
    already_completed: bool = False


_JOBS: Dict[str, ArrangementJob] = {}
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def create_arrangement_job(file_id: str, status: str = "queued",
                           message: str = "Analisando melodia...",
                           config: Optional[Dict] = None) -> ArrangementJob:
    job_id = str(uuid.uuid4())
    now = _now_iso()
    job = ArrangementJob(job_id=job_id, file_id=file_id, status=status,
                         message=message, created_at=now, updated_at=now, config=config)
    with _LOCK:
        _JOBS[job_id] = job
    return job


def get_arrangement_job(job_id: str) -> Optional[ArrangementJob]:
    with _LOCK:
        return _JOBS.get(job_id)


def update_arrangement_job(job_id: str, **kwargs) -> Optional[ArrangementJob]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)
        job.updated_at = _now_iso()
        return job


def has_active_arrangement_job() -> bool:
    with _LOCK:
        return any(j.status in ("queued", "running") for j in _JOBS.values())


def get_active_arrangement_job() -> Optional[ArrangementJob]:
    with _LOCK:
        for j in _JOBS.values():
            if j.status in ("queued", "running"):
                return j
        return None


def arrangement_job_to_dict(job: ArrangementJob) -> Dict:
    return {
        "job_id": job.job_id, "file_id": job.file_id, "status": job.status,
        "message": job.message, "created_at": job.created_at,
        "updated_at": job.updated_at, "config": job.config,
        "results": job.results, "error": job.error,
        "already_completed": job.already_completed,
    }


def clear_arrangement_jobs() -> None:
    with _LOCK:
        _JOBS.clear()
