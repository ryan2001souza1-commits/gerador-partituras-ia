"""
Etapa 8.3.1 — Pipeline runner com cache baseado em HASH/IDENTITY.

Cada stage:
1. Valida cache com metadata (não apenas file_exists)
2. Se cache válido → SKIP (cached=true)
3. Se cache inválido → EXECUTA
4. Após executar → SALVA metadata com hashes das entradas

Cadeia: audio_hash → analysis → demucs(stem_hashes) → transcription →
drums → score → arrangement
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("uvicorn.error")

BASE_DIR = Path(__file__).resolve().parents[2]


async def run_pipeline(job_id: str, file_id: str, config: Dict[str, Any]):
    """Executa pipeline completo com cache identity-based."""
    from backend.pipeline.pipeline_job_manager import (
        update_pipeline_job, update_stage, compute_progress, get_pipeline_job,
    )
    from backend.pipeline import cache_identity as ci

    try:
        update_pipeline_job(job_id, status="running",
                           current_stage_label="Iniciando pipeline...")
        logger.info(f"Pipeline {job_id} started file_id={file_id}")

        # Config do usuário
        cleanup_profile = config.get("cleanup_profile", "natural")
        arrangement_style = config.get("arrangement_style", "automatic")
        include_drums = config.get("include_drums", True)
        include_originals = config.get("include_original_parts", True)
        dynamics = config.get("dynamics", "automatic")
        instruments = config.get("instruments", [])
        with_arrangement = bool(instruments)

        # Localiza arquivo
        from app import _find_upload_path
        upload_path = _find_upload_path(file_id)
        if upload_path is None:
            raise FileNotFoundError("Arquivo não encontrado.")

        # COMPUTA AUDIO HASH UMA VEZ — base de toda a cadeia
        audio_hash = ci.compute_file_hash(Path(upload_path))
        if audio_hash is None:
            raise RuntimeError("Não foi possível calcular hash do áudio.")

        # ==============================================================
        # STAGE 1: ANÁLISE MUSICAL
        # ==============================================================
        update_stage(job_id, "analysis", status="running")
        try:
            if ci.is_analysis_cache_valid(file_id, audio_hash):
                update_stage(job_id, "analysis", status="completed",
                           cached=True, progress=100)
                logger.info(f"Pipeline {job_id}: analysis CACHE VALID (hash match)")
            else:
                from backend.audio.music_analysis import analyze_music
                result = await asyncio.to_thread(
                    analyze_music, Path(upload_path), None, file_id)
                if result.error:
                    raise RuntimeError(f"Análise falhou: {result.error}")
                # Salva metadata da analysis
                ci.save_analysis_metadata(file_id, audio_hash)
                update_stage(job_id, "analysis", status="completed",
                           cached=False, progress=100)
                logger.info(f"Pipeline {job_id}: analysis computed + metadata saved")
        except Exception as e:
            update_stage(job_id, "analysis", status="failed", error=str(e))
            update_pipeline_job(job_id, status="failed",
                               error=f"Falha na análise musical: {e}")
            return
        _update_progress(job_id)

        # ==============================================================
        # STAGE 2: DEMUCS
        # ==============================================================
        update_stage(job_id, "demucs", status="running")
        demucs_model = "htdemucs"
        try:
            if ci.is_demucs_cache_valid(file_id, audio_hash, demucs_model):
                update_stage(job_id, "demucs", status="completed",
                           cached=True, progress=100)
                logger.info(f"Pipeline {job_id}: demucs CACHE VALID (hash+stems match)")
            else:
                from backend.audio.stem_separator import (
                    separate_stems_async, DEMUCS_TIMEOUT, DEMUCS_MODEL,
                )
                await separate_stems_async(upload_path, file_id,
                                          timeout=DEMUCS_TIMEOUT)
                # Verifica stems produzidos
                stem_hashes = ci.get_stem_hashes(file_id)
                if stem_hashes is None:
                    raise RuntimeError("Demucs não produziu stems válidos.")
                # Salva metadata com hashes
                if not ci.save_demucs_metadata(file_id, audio_hash, demucs_model):
                    logger.warning(f"Pipeline {job_id}: demucs metadata save falhou")
                update_stage(job_id, "demucs", status="completed",
                           cached=False, progress=100)
                logger.info(f"Pipeline {job_id}: demucs computed + metadata saved")
        except Exception as e:
            update_stage(job_id, "demucs", status="failed", error=str(e))
            update_pipeline_job(job_id, status="failed",
                               error=f"Falha na separação: {e}")
            return
        _update_progress(job_id)

        # Recupera stem hashes (já validados acima)
        stem_hashes = ci.get_stem_hashes(file_id)
        if stem_hashes is None:
            update_pipeline_job(job_id, status="failed",
                               error="Stems não encontrados após separação.")
            return

        # ==============================================================
        # STAGE 3: TRANSCRIPTION (Basic Pitch — vocals/bass/other)
        # ==============================================================
        update_stage(job_id, "transcription", status="running")
        try:
            # Config hash para transcrição
            from backend.audio.transcriber import BASIC_PITCH_DEFAULTS, STEM_FREQ_RANGES
            trans_config = {
                "defaults": {k: v for k, v in BASIC_PITCH_DEFAULTS.items()},
                "stem_freq_ranges": {k: v for k, v in STEM_FREQ_RANGES.items()},
            }
            trans_config_hash = ci.compute_config_hash(trans_config)

            all_valid = ci.are_all_transcriptions_valid(file_id, trans_config_hash)
            if all_valid:
                update_stage(job_id, "transcription", status="completed",
                           cached=True, progress=100)
                logger.info(f"Pipeline {job_id}: transcription CACHE VALID")
            else:
                from backend.audio.transcriber import (
                    transcribe_stem_async, BASIC_PITCH_TIMEOUT,
                )
                from backend.audio.stem_separator import STEMS_DIR

                for idx, stem in enumerate(ci.TRANSCRIBED_STEMS):
                    stem_path = STEMS_DIR / file_id / f"{stem}.wav"
                    if not stem_path.is_file():
                        raise FileNotFoundError(f"Stem não encontrado: {stem}")
                    update_stage(job_id, "transcription",
                               progress=int(100 * idx / 3))
                    await transcribe_stem_async(
                        stem_path, file_id, stem, timeout=BASIC_PITCH_TIMEOUT)
                    # Salva metadata deste stem com o hash do stem
                    ci.save_transcription_metadata(
                        file_id, stem, stem_hashes[stem], trans_config_hash)

                update_stage(job_id, "transcription", status="completed",
                           cached=False, progress=100)
                logger.info(f"Pipeline {job_id}: transcription computed + metadata saved")
        except Exception as e:
            update_stage(job_id, "transcription", status="failed", error=str(e))
            update_pipeline_job(job_id, status="failed",
                               error=f"Falha na transcrição: {e}")
            return
        _update_progress(job_id)

        # ==============================================================
        # STAGE 4: DRUMS (opcional)
        # ==============================================================
        if include_drums:
            update_stage(job_id, "drums", status="running")
            try:
                # Config hash para drums
                from backend.drums.drum_utils import DRUM_VERSION
                from backend.musical.cleanup import CLEANUP_PROFILES
                drums_config = {
                    "drum_version": DRUM_VERSION,
                    "cleanup_profile": cleanup_profile,
                    "time_signature": config.get("time_signature", "4/4"),
                }
                drums_config_hash = ci.compute_config_hash(drums_config)
                drums_stem_hash = stem_hashes.get("drums", "")

                if ci.is_drums_cache_valid(file_id, drums_stem_hash, drums_config_hash):
                    update_stage(job_id, "drums", status="completed",
                               cached=True, progress=100)
                    logger.info(f"Pipeline {job_id}: drums CACHE VALID")
                else:
                    from backend.drums.drum_transcriber import (
                        transcribe_drums_async, get_drums_wav, DRUM_TIMEOUT,
                    )
                    from backend.notation.score_generator import get_music_context

                    drums_wav = get_drums_wav(file_id)
                    if drums_wav is None:
                        raise FileNotFoundError("Stem drums não encontrado.")

                    ctx = get_music_context(file_id)
                    tempo = ctx.get("tempo") or 120
                    beat_offset = float(ctx.get("beat_offset") or 0.0)

                    await transcribe_drums_async(
                        drums_wav, file_id, tempo, beat_offset,
                        time_signature=config.get("time_signature", "4/4"),
                        cleanup_profile=cleanup_profile,
                        timeout=DRUM_TIMEOUT,
                    )
                    ci.save_drums_metadata(file_id, drums_stem_hash, drums_config_hash)
                    update_stage(job_id, "drums", status="completed",
                               cached=False, progress=100)
                    logger.info(f"Pipeline {job_id}: drums computed + metadata saved")
            except Exception as e:
                update_stage(job_id, "drums", status="failed", error=str(e))
                update_pipeline_job(job_id, status="failed",
                                   error=f"Falha na bateria: {e}")
                return
        else:
            update_stage(job_id, "drums", status="skipped")
        _update_progress(job_id)

        # ==============================================================
        # STAGE 5: SCORE (MusicXML)
        # ==============================================================
        update_stage(job_id, "score", status="running")
        try:
            from backend.notation.score_generator import (
                are_transcriptions_ready, generate_score_async,
                get_score_paths, NOTATION_TIMEOUT, get_music_context,
            )
            from backend.notation.score_utils import config_key
            from backend.musical.cleanup import validate_cleanup_profile

            ready, missing = are_transcriptions_ready(file_id)
            if not ready:
                raise RuntimeError(f"Transcrições incompletas: {missing}")

            ctx = get_music_context(file_id)
            tempo_val = config.get("tempo") or ctx.get("tempo") or 120
            time_sig = config.get("time_signature", "4/4")
            quant = config.get("quantization", "1/16")
            key_mode = config.get("key_mode", "auto")
            profile = validate_cleanup_profile(cleanup_profile)

            expected_config_key = config_key(int(tempo_val), time_sig, quant,
                                            key_mode, cleanup_profile=profile)

            # Identity: config_key + stem hashes das transcrições
            score_identity = {
                "config_key": expected_config_key,
                "vocals_stem_hash": stem_hashes.get("vocals", ""),
                "bass_stem_hash": stem_hashes.get("bass", ""),
                "other_stem_hash": stem_hashes.get("other", ""),
                "audio_hash": audio_hash,
            }

            if ci.is_score_cache_valid(file_id, score_identity):
                update_stage(job_id, "score", status="completed",
                           cached=True, progress=100)
                logger.info(f"Pipeline {job_id}: score CACHE VALID")
            else:
                model = await generate_score_async(
                    file_id,
                    tempo=int(tempo_val),
                    time_signature=time_sig,
                    quantization=quant,
                    key_mode=key_mode,
                    cleanup_profile=profile,
                    timeout=NOTATION_TIMEOUT,
                )
                ci.save_score_metadata(file_id, score_identity)
                update_stage(job_id, "score", status="completed",
                           cached=False, progress=100)
                logger.info(f"Pipeline {job_id}: score computed + metadata saved")
        except Exception as e:
            update_stage(job_id, "score", status="failed", error=str(e))
            update_pipeline_job(job_id, status="failed",
                               error=f"Falha na partitura: {e}")
            return
        _update_progress(job_id)

        # ==============================================================
        # STAGE 6: ARRANGEMENT (opcional)
        # ==============================================================
        if with_arrangement:
            await _run_arrangement_stage(
                job_id, file_id, config, instruments, arrangement_style,
                dynamics, include_originals, include_drums, cleanup_profile,
                audio_hash, stem_hashes)
        else:
            _finish_pipeline(job_id,
                           "Partitura pronta. Selecione instrumentos para criar um arranjo.")

    except Exception as e:
        logger.error(f"Pipeline {job_id} erro inesperado: {e}", exc_info=True)
        update_pipeline_job(job_id, status="failed",
                           error=f"Erro inesperado: {e}")


async def _run_arrangement_stage(job_id: str, file_id: str, config: Dict,
                                instruments: list, arrangement_style: str,
                                dynamics: str, include_originals: bool,
                                include_drums: bool, cleanup_profile: str,
                                audio_hash: str, stem_hashes: Dict):
    """Stage de arranjo com cache identity-based."""
    from backend.pipeline.pipeline_job_manager import (
        update_pipeline_job, update_stage, compute_progress, get_pipeline_job,
    )
    from backend.pipeline import cache_identity as ci

    update_stage(job_id, "arrangement", status="running")
    try:
        from backend.arrangement.arrangement_generator import (
            generate_arrangement_async, read_base_score,
            arrangement_config_key, get_arrangement_paths, ARRANGE_TIMEOUT,
        )

        base = read_base_score(file_id)
        if not base:
            raise FileNotFoundError("Partitura base necessária.")

        base_ck = str(base.get("config_key", ""))
        arr_config_key = arrangement_config_key(
            sorted(instruments), "automatic", include_originals,
            cleanup_profile=cleanup_profile,
            arrangement_style=arrangement_style,
            include_drums=include_drums,
            dynamics=dynamics,
        )

        # Identity: config_key + base_config_key + score stems
        arr_identity = {
            "config_key": arr_config_key,
            "base_config_key": base_ck,
            "audio_hash": audio_hash,
        }

        if ci.is_arrangement_cache_valid(file_id, arr_identity):
            update_stage(job_id, "arrangement", status="completed",
                       cached=True, progress=100)
            logger.info(f"Pipeline {job_id}: arrangement CACHE VALID")
            _finish_pipeline(job_id, "Pipeline concluído.")
            return

        await generate_arrangement_async(
            file_id,
            instruments=sorted(instruments),
            mode="automatic",
            include_original_parts=include_originals,
            cleanup_profile=cleanup_profile,
            arrangement_style=arrangement_style,
            include_drums=include_drums,
            dynamics=dynamics,
            timeout=ARRANGE_TIMEOUT,
        )
        ci.save_arrangement_metadata(file_id, arr_identity)
        update_stage(job_id, "arrangement", status="completed",
                   cached=False, progress=100)
        logger.info(f"Pipeline {job_id}: arrangement computed + metadata saved")
        _finish_pipeline(job_id, "Pipeline concluído.")

    except Exception as e:
        update_stage(job_id, "arrangement", status="failed", error=str(e))
        update_pipeline_job(job_id, status="failed",
                           error=f"Falha no arranjo: {e}")


def _update_progress(job_id: str):
    from backend.pipeline.pipeline_job_manager import (
        update_pipeline_job, compute_progress, get_pipeline_job,
    )
    job = get_pipeline_job(job_id)
    if job:
        update_pipeline_job(job_id, progress_percent=compute_progress(job))


def _finish_pipeline(job_id: str, message: str):
    from backend.pipeline.pipeline_job_manager import update_pipeline_job
    from datetime import datetime
    completed_at = datetime.utcnow().isoformat() + "Z"
    update_pipeline_job(job_id, status="completed", error=None,
                       completed_at=completed_at,
                       current_stage=None,
                       current_stage_label=message)
    logger.info(f"Pipeline {job_id} COMPLETED")
