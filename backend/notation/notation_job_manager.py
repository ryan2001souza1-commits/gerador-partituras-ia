"""
Job manager em memória para geração de partitura (Etapa 6).

Separado dos jobs de Demucs/Basic Pitch. Reiniciar o servidor perde jobs.
Somente um job de MusicXML ativo por vez (queued/running).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


@dataclass
class NotationJob:
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


_JOBS: Dict[str, NotationJob] = {}
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def create_notation_job(
    file_id: str,
    status: str = "queued",
    message: str = "Preparando partitura...",
    config: Optional[Dict] = None,
) -> NotationJob:
    job_id = str(uuid.uuid4())
    now = _now_iso()
    job = NotationJob(
        job_id=job_id,
        file_id=file_id,
        status=status,
        message=message,
        created_at=now,
        updated_at=now,
        config=config,
    )
    with _LOCK:
        _JOBS[job_id] = job
    return job


def get_notation_job(job_id: str) -> Optional[NotationJob]:
    with _LOCK:
        return _JOBS.get(job_id)


def update_notation_job(job_id: str, **kwargs) -> Optional[NotationJob]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)
        job.updated_at = _now_iso()
        return job


def has_active_notation_job() -> bool:
    with _LOCK:
        return any(j.status in ("queued", "running") for j in _JOBS.values())


def get_active_notation_job() -> Optional[NotationJob]:
    with _LOCK:
        for j in _JOBS.values():
            if j.status in ("queued", "running"):
                return j
        return None


def notation_job_to_dict(job: NotationJob) -> Dict:
    return {
        "job_id": job.job_id,
        "file_id": job.file_id,
        "status": job.status,
        "message": job.message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "config": job.config,
        "results": job.results,
        "error": job.error,
        "already_completed": job.already_completed,
    }


def clear_notation_jobs() -> None:
    with _LOCK:
        _JOBS.clear()
