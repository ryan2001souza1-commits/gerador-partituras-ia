"""
Etapa 8.3 — Pipeline: job manager para processamento completo.

Gerencia jobs de pipeline completo: analysis → demucs → transcription →
drums → score → arrangement (opcional).

Cada stage tem status independente + flag cached para resume.
Thread-safe com threading.Lock. Atomic double-start protection.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class StageInfo:
    """Status de uma etapa do pipeline."""
    name: str
    label: str
    status: str = "pending"  # pending | running | completed | failed | skipped
    cached: bool = False
    progress: int = 0  # 0-100
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class PipelineJob:
    """Job completo do pipeline."""
    job_id: str
    file_id: str
    status: str  # queued | running | completed | failed
    current_stage: Optional[str] = None
    current_stage_label: str = ""
    progress_percent: int = 0
    created_at: str = ""
    updated_at: str = ""
    completed_at: Optional[str] = None
    error: Optional[str] = None
    stages: Dict[str, StageInfo] = field(default_factory=dict)
    config: Optional[Dict] = None

    def init_stages(self, with_arrangement: bool = True):
        """Inicializa stages do pipeline."""
        stage_defs = [
            ("analysis", "Análise musical"),
            ("demucs", "Separação de instrumentos"),
            ("transcription", "Transcrição de notas"),
            ("drums", "Transcrição de bateria"),
            ("score", "Geração da partitura"),
        ]
        if with_arrangement:
            stage_defs.append(("arrangement", "Criação do arranjo"))
        self.stages = {
            name: StageInfo(name=name, label=label)
            for name, label in stage_defs
        }


_JOBS: Dict[str, PipelineJob] = {}
_LOCK = threading.RLock()  # RLock: permite reentrada (exclusive cria dentro do lock)


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def create_pipeline_job(file_id: str, config: Optional[Dict] = None,
                        with_arrangement: bool = True) -> PipelineJob:
    """Cria novo job de pipeline."""
    job_id = str(uuid.uuid4())
    now = _now_iso()
    job = PipelineJob(
        job_id=job_id,
        file_id=file_id,
        status="queued",
        created_at=now,
        updated_at=now,
        config=config,
    )
    job.init_stages(with_arrangement=with_arrangement)
    with _LOCK:
        _JOBS[job_id] = job
    return job


def create_pipeline_job_exclusive(file_id: str, config: Optional[Dict] = None,
                                   with_arrangement: bool = True) -> Optional[PipelineJob]:
    """Cria job APENAS se não houver outro pipeline ativo. Atomic."""
    with _LOCK:
        for j in _JOBS.values():
            if j.status in ("queued", "running"):
                return None
        return create_pipeline_job(file_id, config, with_arrangement)


def get_pipeline_job(job_id: str) -> Optional[PipelineJob]:
    with _LOCK:
        return _JOBS.get(job_id)


def has_active_pipeline_job() -> bool:
    with _LOCK:
        return any(j.status in ("queued", "running") for j in _JOBS.values())


def update_pipeline_job(job_id: str, **kwargs) -> Optional[PipelineJob]:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        for k, v in kwargs.items():
            if hasattr(job, k):
                setattr(job, k, v)
        job.updated_at = _now_iso()
        return job


def update_stage(job_id: str, stage_name: str, **kwargs) -> Optional[StageInfo]:
    """Atualiza uma etapa do pipeline."""
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return None
        stage = job.stages.get(stage_name)
        if not stage:
            return None
        for k, v in kwargs.items():
            if hasattr(stage, k):
                setattr(stage, k, v)
        if kwargs.get("status") == "running":
            stage.started_at = _now_iso()
            job.current_stage = stage_name
            job.current_stage_label = stage.label
        if kwargs.get("status") in ("completed", "failed", "skipped"):
            stage.completed_at = _now_iso()
        job.updated_at = _now_iso()
        return stage


def compute_progress(job: PipelineJob) -> int:
    """Computa progresso global baseado em stages.

    Pesos documentados:
    analysis: 10, demucs: 40, transcription: 30, drums: 5, score: 7, arrangement: 8
    Stage cacheada = imediatamente concluída.
    Nunca retrocede.
    """
    weights = {
        "analysis": 10, "demucs": 40, "transcription": 30,
        "drums": 5, "score": 7, "arrangement": 8,
    }
    total_weight = sum(weights.get(s, 0) for s in job.stages)
    earned = 0
    for name, stage in job.stages.items():
        w = weights.get(name, 0)
        if stage.status in ("completed", "skipped"):
            earned += w
        elif stage.status == "running":
            # Stage em progresso: proporcional ao sub-progresso
            earned += int(w * stage.progress / 100)
    pct = int(100 * earned / total_weight) if total_weight > 0 else 0
    # Nunca retrocede
    return max(pct, job.progress_percent)


def pipeline_job_to_dict(job: PipelineJob) -> Dict[str, Any]:
    """Serializa job para API response."""
    return {
        "job_id": job.job_id,
        "file_id": job.file_id,
        "status": job.status,
        "current_stage": job.current_stage,
        "current_stage_label": job.current_stage_label,
        "progress_percent": compute_progress(job),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "completed_at": job.completed_at,
        "error": job.error,
        "config": job.config,
        "stages": {
            name: {
                "name": s.name,
                "label": s.label,
                "status": s.status,
                "cached": s.cached,
                "progress": s.progress,
                "error": s.error,
            }
            for name, s in job.stages.items()
        },
    }


def clear_pipeline_jobs() -> None:
    with _LOCK:
        _JOBS.clear()
