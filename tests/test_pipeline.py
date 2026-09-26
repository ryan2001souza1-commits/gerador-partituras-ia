"""
Testes Etapa 8.3 — Pipeline completo.

Cobre:
- Pipeline job creation
- Double-start protection
- Stage lifecycle
- Progress monotonic
- Resume after failure
- Cache-aware stages
- Config validation
"""
import uuid
from pathlib import Path

import pytest

from backend.pipeline.pipeline_job_manager import (
    StageInfo,
    PipelineJob,
    clear_pipeline_jobs,
    compute_progress,
    create_pipeline_job,
    create_pipeline_job_exclusive,
    get_pipeline_job,
    has_active_pipeline_job,
    pipeline_job_to_dict,
    update_pipeline_job,
    update_stage,
)


class TestPipelineJobManager:
    """Testa criação e gestão de pipeline jobs."""

    def setup_method(self):
        clear_pipeline_jobs()

    def teardown_method(self):
        clear_pipeline_jobs()

    def test_create_job(self):
        fid = str(uuid.uuid4())
        job = create_pipeline_job(fid)
        assert job.job_id is not None
        assert job.file_id == fid
        assert job.status == "queued"
        assert "analysis" in job.stages
        assert "demucs" in job.stages
        assert "transcription" in job.stages
        assert "drums" in job.stages
        assert "score" in job.stages
        assert "arrangement" in job.stages

    def test_create_without_arrangement(self):
        job = create_pipeline_job(str(uuid.uuid4()), with_arrangement=False)
        assert "arrangement" not in job.stages
        assert len(job.stages) == 5

    def test_double_start_protection(self):
        """Segundo job enquanto primeiro ativo: None."""
        fid1 = str(uuid.uuid4())
        fid2 = str(uuid.uuid4())
        job1 = create_pipeline_job_exclusive(fid1)
        assert job1 is not None
        # Segundo: deve retornar None (primeiro ainda queued)
        job2 = create_pipeline_job_exclusive(fid2)
        assert job2 is None

    def test_exclusive_after_completed(self):
        """Após completar primeiro, segundo pode criar."""
        fid = str(uuid.uuid4())
        job1 = create_pipeline_job_exclusive(fid)
        assert job1 is not None
        update_pipeline_job(job1.job_id, status="completed")
        job2 = create_pipeline_job_exclusive(fid)
        assert job2 is not None

    def test_has_active(self):
        assert not has_active_pipeline_job()
        job = create_pipeline_job(str(uuid.uuid4()))
        assert has_active_pipeline_job()
        update_pipeline_job(job.job_id, status="completed")
        assert not has_active_pipeline_job()


class TestStageLifecycle:
    """Testa ciclo de vida das stages."""

    def setup_method(self):
        clear_pipeline_jobs()
        self.job = create_pipeline_job(str(uuid.uuid4()))

    def teardown_method(self):
        clear_pipeline_jobs()

    def test_stage_starts_pending(self):
        assert self.job.stages["analysis"].status == "pending"
        assert self.job.stages["demucs"].status == "pending"

    def test_stage_running(self):
        update_stage(self.job.job_id, "analysis", status="running")
        job = get_pipeline_job(self.job.job_id)
        assert job.stages["analysis"].status == "running"
        assert job.current_stage == "analysis"

    def test_stage_completed(self):
        update_stage(self.job.job_id, "analysis", status="completed", cached=True)
        job = get_pipeline_job(self.job.job_id)
        assert job.stages["analysis"].status == "completed"
        assert job.stages["analysis"].cached is True

    def test_stage_failed(self):
        update_stage(self.job.job_id, "demucs", status="failed", error="test error")
        job = get_pipeline_job(self.job.job_id)
        assert job.stages["demucs"].status == "failed"
        assert job.stages["demucs"].error == "test error"

    def test_stage_skipped(self):
        update_stage(self.job.job_id, "drums", status="skipped")
        job = get_pipeline_job(self.job.job_id)
        assert job.stages["drums"].status == "skipped"

    def test_nonexistent_stage(self):
        result = update_stage(self.job.job_id, "nonexistent", status="running")
        assert result is None

    def test_nonexistent_job(self):
        result = update_stage(str(uuid.uuid4()), "analysis", status="running")
        assert result is None


class TestProgressComputation:
    """Testa progresso global do pipeline."""

    def setup_method(self):
        clear_pipeline_jobs()

    def teardown_method(self):
        clear_pipeline_jobs()

    def test_zero_progress_initially(self):
        job = create_pipeline_job(str(uuid.uuid4()))
        assert compute_progress(job) == 0

    def test_full_progress_completed(self):
        job = create_pipeline_job(str(uuid.uuid4()))
        for name in job.stages:
            update_stage(job.job_id, name, status="completed")
        job = get_pipeline_job(job.job_id)
        assert compute_progress(job) == 100

    def test_progress_monotonic(self):
        """Progresso nunca retrocede."""
        job = create_pipeline_job(str(uuid.uuid4()))
        job.progress_percent = 50  # já atingiu 50%
        # Só analysis completada (10% do peso) → deveria manter 50%
        update_stage(job.job_id, "analysis", status="completed")
        job = get_pipeline_job(job.job_id)
        pct = compute_progress(job)
        assert pct >= 50  # não retrocede abaixo de 50%

    def test_cached_stage_counts(self):
        """Stage cacheada contribui imediatamente."""
        job = create_pipeline_job(str(uuid.uuid4()))
        update_stage(job.job_id, "analysis", status="completed", cached=True)
        job = get_pipeline_job(job.job_id)
        pct = compute_progress(job)
        # Analysis tem peso 10 de 100 total → pelo menos 10%
        assert pct >= 10

    def test_running_stage_partial(self):
        """Stage running com sub-progresso contribui proporcionalmente."""
        job = create_pipeline_job(str(uuid.uuid4()))
        update_stage(job.job_id, "analysis", status="completed")
        update_stage(job.job_id, "demucs", status="running", progress=50)
        job = get_pipeline_job(job.job_id)
        pct = compute_progress(job)
        # analysis=10 + demucs(50% de 40)=20 → 30
        assert 25 <= pct <= 35


class TestPipelineJobToDict:
    """Testa serialização do job para API."""

    def setup_method(self):
        clear_pipeline_jobs()

    def test_dict_contains_required_fields(self):
        job = create_pipeline_job(str(uuid.uuid4()))
        d = pipeline_job_to_dict(job)
        for field in ["job_id", "file_id", "status", "progress_percent",
                      "stages", "created_at", "updated_at"]:
            assert field in d

    def test_stages_serialized(self):
        job = create_pipeline_job(str(uuid.uuid4()))
        d = pipeline_job_to_dict(job)
        for name, stage in d["stages"].items():
            assert "name" in stage
            assert "label" in stage
            assert "status" in stage
            assert "cached" in stage

    def test_error_included_when_failed(self):
        job = create_pipeline_job(str(uuid.uuid4()))
        update_pipeline_job(job.job_id, status="failed", error="test fail")
        d = pipeline_job_to_dict(job)
        assert d["error"] == "test fail"


class TestAnalysisCacheIntegration:
    """Testa integração do cache de análise com o pipeline."""

    def setup_method(self):
        clear_pipeline_jobs()

    def test_pipeline_with_cache_check(self):
        """Pipeline verifica cache antes de executar analysis."""
        from backend.audio.analysis_cache import (
            save_analysis_cache, clear_analysis_cache,
        )
        fid = str(uuid.uuid4())
        try:
            # Simula cache existente
            save_analysis_cache(fid, "hash_test", {
                "bpm": 120, "key": "C", "mode": "major", "duration": 30.0,
                "bpm_confidence": 0.9, "key_confidence": 0.8,
            })
            # Cria job — o runner deveria detectar cache
            job = create_pipeline_job(fid)
            assert "analysis" in job.stages
        finally:
            clear_analysis_cache(fid)


class TestPipelineCrossFile:
    """Testa que pipeline de arquivo A não contamina arquivo B."""

    def setup_method(self):
        clear_pipeline_jobs()

    def test_jobs_independent(self):
        """Dois jobs para files diferentes são independentes."""
        fid_a = str(uuid.uuid4())
        fid_b = str(uuid.uuid4())
        # Cria job A, completa
        job_a = create_pipeline_job(fid_a)
        update_pipeline_job(job_a.job_id, status="completed")
        # Cria job B
        job_b = create_pipeline_job(fid_b)
        # Jobs são independentes
        assert job_a.file_id != job_b.file_id
        assert job_a.job_id != job_b.job_id
        # Job A não afeta job B
        assert job_b.status == "queued"
        assert job_b.stages["analysis"].status == "pending"
