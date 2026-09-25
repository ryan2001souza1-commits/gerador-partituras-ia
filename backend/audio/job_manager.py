"""
Job manager simples em memória para separação Demucs.

Limitação documentada: reiniciar servidor perde status de jobs ativos.
Protege contra acesso concorrente com threading.Lock.
Permite apenas um job ativo por vez (queued/running) para proteger CPU/RAM.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List

@dataclass
class Job:
    job_id: str
    file_id: str
    status: str  # queued, running, completed, failed
    message: str
    created_at: str
    updated_at: str
    stems: Optional[List[str]] = None
    error: Optional[str] = None
    already_completed: bool = False

_JOBS: Dict[str, Job] = {}
_LOCK = threading.Lock()

def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def create_job(file_id: str, status: str = "queued", message: str = "Preparando separação...") -> Job:
    job_id = str(uuid.uuid4())
    now = _now_iso()
    job = Job(
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

def get_job(job_id: str) -> Optional[Job]:
    with _LOCK:
        return _JOBS.get(job_id)

def update_job(job_id: str, **kwargs):
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)
        job.updated_at = _now_iso()
        return job

def list_jobs() -> List[Job]:
    with _LOCK:
        return list(_JOBS.values())

def has_active_job() -> bool:
    """
    Verifica se há job ativo (queued ou running).
    Para proteger CPU/RAM, permite somente um por vez.
    """
    with _LOCK:
        for j in _JOBS.values():
            if j.status in ("queued", "running"):
                return True
        return False

def get_active_job() -> Optional[Job]:
    with _LOCK:
        for j in _JOBS.values():
            if j.status in ("queued", "running"):
                return j
        return None

def job_to_dict(job: Job) -> Dict:
    return {
        "job_id": job.job_id,
        "file_id": job.file_id,
        "status": job.status,
        "message": job.message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "stems": job.stems,
        "error": job.error,
        "already_completed": job.already_completed,
    }

def clear_jobs():
    # Apenas para testes
    with _LOCK:
        _JOBS.clear()
