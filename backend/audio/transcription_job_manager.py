"""
Job manager simples em memória para transcrição Basic Pitch.

Separado do job_manager de Demucs para não misturar jobs e não quebrar
a proteção de 1 job por vez de cada tipo.

Limitação: reiniciar servidor perde jobs.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, List

@dataclass
class TranscriptionJob:
    job_id: str
    file_id: str
    status: str  # queued, running, completed, failed
    message: str
    created_at: str
    updated_at: str
    stems: Optional[List[str]] = None
    results: Optional[Dict] = None
    error: Optional[str] = None
    already_completed: bool = False

_JOBS: Dict[str, TranscriptionJob] = {}
_LOCK = threading.Lock()

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def create_transcription_job(file_id: str, status: str = "queued", message: str = "Preparando transcrição...") -> TranscriptionJob:
    job_id = str(uuid.uuid4())
    now = _now_iso()
    job = TranscriptionJob(
        job_id=job_id,
        file_id=file_id,
        status=status,
        message=message,
        created_at=now,
        updated_at=now,
    )
    with _LOCK:
        _JOBS[job_id] = job
    return job

def get_transcription_job(job_id: str) -> Optional[TranscriptionJob]:
    with _LOCK:
        return _JOBS.get(job_id)

def update_transcription_job(job_id: str, **kwargs):
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)
        job.updated_at = _now_iso()
        return job

def has_active_transcription_job() -> bool:
    with _LOCK:
        for j in _JOBS.values():
            if j.status in ("queued", "running"):
                return True
        return False

def get_active_transcription_job() -> Optional[TranscriptionJob]:
    with _LOCK:
        for j in _JOBS.values():
            if j.status in ("queued", "running"):
                return j
        return None

def transcription_job_to_dict(job: TranscriptionJob) -> Dict:
    return {
        "job_id": job.job_id,
        "file_id": job.file_id,
        "status": job.status,
        "message": job.message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "stems": job.stems,
        "results": job.results,
        "error": job.error,
        "already_completed": job.already_completed,
    }

def clear_transcription_jobs():
    with _LOCK:
        _JOBS.clear()
