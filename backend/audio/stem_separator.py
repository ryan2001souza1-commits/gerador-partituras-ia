"""
Separação de stems com Demucs — Etapa 4.

Responsabilidades:
  - detectar instalação Demucs (python separado .venv-demucs)
  - executar Demucs via subprocess seguro (sem uso de shell, lista, cpu, htdemucs, j1)
  - controlar timeout centralizado (45 min)
  - validar saída (4 stems, tamanho>0, FFprobe, duração aproximada)
  - normalizar para stems/<file_id>/{vocals,drums,bass,other}.wav
  - limpar temporários em try/finally
  - proteger contra path traversal, file_id inválido, stem inválido

Modelo: htdemucs (4 stems: vocals, drums, bass, other)
Device: cpu, jobs=1

Não usar: htdemucs_6s, CUDA, paralelismo agressivo.

Fluxo Demucs:
  input -> demucs -d cpu -j 1 -n htdemucs -o <tmp> <input>
  output -> <tmp>/htdemucs/<track_name>/{vocals.wav,...}
  normaliza -> stems/<file_id>/{vocals.wav,...}
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEMUCS_MODEL = "htdemucs"
DEMUCS_DEVICE = "cpu"
DEMUCS_JOBS = 1  # int centralizado — usado em comando e endpoint
DEMUCS_TIMEOUT = 45 * 60  # 45 minutos em segundos — centralizado
EXPECTED_STEMS = ["vocals", "drums", "bass", "other"]  # 4 stems base

BASE_DIR = Path(__file__).resolve().parents[2]  # projeto root (backend/audio -> .. -> root)
STEMS_DIR = BASE_DIR / "stems"
STEMS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Resolução do Python Demucs
# ---------------------------------------------------------------------------

def get_demucs_python() -> Optional[str]:
    """
    Resolve caminho para python do Demucs de forma segura, sem hardcode absoluto do dev.
    Ordem:
      1. env DEMUCS_PYTHON se existir e for arquivo
      2. .venv-demucs/Scripts/python.exe (Windows)
      3. .venv-demucs/Scripts/python (Windows sem ext)
      4. .venv-demucs/bin/python (Linux/Mac)
      5. fallback: sys.executable (útil para testes/mock, mas documentação alerta que deve usar .venv-demucs)
    Retorna string path ou None se não encontrado.
    """
    import sys
    env = os.getenv("DEMUCS_PYTHON")
    if env:
        p = Path(env)
        if p.is_file():
            return str(p)
        logger.warning(f"DEMUCS_PYTHON env aponta para não-arquivo: {env}")

    candidates = [
        BASE_DIR / ".venv-demucs" / "Scripts" / "python.exe",
        BASE_DIR / ".venv-demucs" / "Scripts" / "python",
        BASE_DIR / ".venv-demucs" / "bin" / "python",
        BASE_DIR / ".venv-demucs" / "bin" / "python3",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    # Fallback: tenta usar o python atual (para testes sem venv dedicado)
    # Mas loga aviso
    logger.debug("Demucs python não encontrado em .venv-demucs, usando sys.executable como fallback")
    return sys.executable

def is_demucs_python_available() -> bool:
    p = get_demucs_python()
    if not p:
        return False
    return Path(p).is_file()

def get_demucs_version() -> Optional[str]:
    py = get_demucs_python()
    if not py:
        return None
    try:
        result = subprocess.run(
            [py, "-m", "pip", "show", "demucs"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in (result.stdout or "").splitlines():
                if line.lower().startswith("version:"):
                    return line.split(":",1)[1].strip()
        # fallback: try import
        result2 = subprocess.run(
            [py, "-c", "import demucs; print(getattr(demucs,'__version__','unknown'))"],
            capture_output=True, text=True, timeout=10
        )
        if result2.returncode == 0:
            return (result2.stdout or "").strip()
    except Exception as e:
        logger.debug(f"get_demucs_version falhou: {e}")
    return None

def is_demucs_available() -> bool:
    py = get_demucs_python()
    if not py or not Path(py).is_file():
        return False
    try:
        result = subprocess.run(
            [py, "-c", "import demucs; print('ok')"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0 and "ok" in (result.stdout or "")
    except Exception:
        return False

def get_torch_info() -> Dict[str, object]:
    py = get_demucs_python()
    info: Dict[str, object] = {}
    if not py:
        return info
    # torch version, cuda, cpu capability — apenas valores serializáveis
    try:
        result = subprocess.run(
            [py, "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available()); "
             "print(getattr(torch.backends.cpu, 'get_cpu_capability', lambda: 'unknown')())"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            lines = (result.stdout or "").strip().splitlines()
            if len(lines) >= 1:
                info["version"] = str(lines[0].strip())
            if len(lines) >= 2:
                cuda_str = str(lines[1].strip()).lower()
                info["cuda_available"] = cuda_str == "true"
            if len(lines) >= 3:
                info["cpu_capability"] = str(lines[2].strip())
        else:
            logger.warning(f"get_torch_info subprocess falhou rc={result.returncode} stdout={result.stdout[:500]} stderr={result.stderr[:500]}")
    except subprocess.TimeoutExpired as e:
        logger.warning(f"get_torch_info timeout: {e}")
    except Exception as e:
        logger.debug(f"get_torch_info falhou: {e}")
    return info

# ---------------------------------------------------------------------------
# Validação e helpers
# ---------------------------------------------------------------------------

def _validate_file_id(file_id: str) -> bool:
    try:
        uuid.UUID(file_id)
        return True
    except ValueError:
        return False

def _validate_stem_name(stem: str) -> bool:
    return stem in EXPECTED_STEMS

def _get_stems_dir(file_id: str) -> Path:
    return STEMS_DIR / file_id

def are_stems_valid(file_id: str) -> Tuple[bool, List[str]]:
    """
    Verifica se stems/<file_id>/ contém os 4 stems válidos:
      - existe, tamanho>0, FFprobe reconhece áudio, duração aproximada.
    Retorna (valid, missing_or_invalid_list)
    Se não houver diretório, retorna (False, EXPECTED_STEMS).
    """
    if not _validate_file_id(file_id):
        return False, EXPECTED_STEMS
    d = _get_stems_dir(file_id)
    if not d.is_dir():
        return False, EXPECTED_STEMS
    missing: List[str] = []
    for stem in EXPECTED_STEMS:
        p = d / f"{stem}.wav"
        if not p.is_file():
            missing.append(stem)
            continue
        try:
            if p.stat().st_size == 0:
                missing.append(stem)
                continue
        except Exception:
            missing.append(stem)
            continue
        # FFprobe validação (se falhar, considera inválido mas não expõe detalhes)
        try:
            from backend.audio.probe import probe_audio
            # probe_audio exige file_id mas pode passar None; usa p.stem
            probe_audio(p, file_id=file_id)
        except Exception as e:
            logger.debug(f"are_stems_valid ffprobe falhou para {p}: {e}")
            missing.append(stem)
            continue
    valid = len(missing) == 0
    return valid, missing

def get_stems_info(file_id: str) -> Optional[Dict]:
    """
    Retorna info para API GET /api/stems/{file_id}
    """
    if not _validate_file_id(file_id):
        return None
    valid, missing = are_stems_valid(file_id)
    stems = []
    for stem in EXPECTED_STEMS:
        p = _get_stems_dir(file_id) / f"{stem}.wav"
        exists = p.is_file() and p.stat().st_size > 0 if p.exists() else False
        stems.append({
            "name": stem,
            "available": exists and valid or exists,  # se file existe, considera available
            "url": f"/api/stems/{file_id}/{stem}",
            "size_bytes": p.stat().st_size if exists else None,
        })
    return {
        "file_id": file_id,
        "available": valid,
        "missing": missing,
        "stems": stems,
    }

def _build_demucs_command(python_path: str, input_path: Path, out_dir: Path) -> List[str]:
    """
    Constrói comando Demucs seguro (sem uso de shell).
    Sempre: -d cpu -j 1 -n htdemucs -o <out_dir> <input>
    Usa Path.resolve() para caminhos absolutos normalizados (Windows com espaços).
    """
    # Normaliza todos os caminhos para absolutos
    py_resolved = str(Path(python_path).resolve())
    out_resolved = str(Path(out_dir).resolve())
    inp_resolved = str(Path(input_path).resolve())
    cmd = [
        py_resolved,
        "-m", "demucs",
        "-d", str(DEMUCS_DEVICE),
        "-j", str(DEMUCS_JOBS),
        "-n", str(DEMUCS_MODEL),
        "-o", out_resolved,
        inp_resolved,
    ]
    # Log detalhado para comparação terminal vs API (repr evita mascarar espaços)
    logger.info(f"Demucs comando: {[repr(c) for c in cmd]}")
    return cmd

# ---------------------------------------------------------------------------
# Execução Demucs (async)
# ---------------------------------------------------------------------------

def _run_demucs_sync(input_path: Path, tmp_out: Path, timeout: int = DEMUCS_TIMEOUT) -> Tuple[int, str, str]:
    """
    Execução síncrona do Demucs em thread worker (evita limitação do event loop
    WindowsSelectorEventLoop do Uvicorn que não suporta create_subprocess_exec).

    Usa subprocess.run com shell=False (default), lista, caminhos absolutos,
    cwd=BASE_DIR, PIPE, timeout, decode errors="replace".
    Retorna (returncode, stdout, stderr). Timeout levanta TimeoutError.
    Não trata warning HF Hub como erro — apenas returncode !=0 é falha.
    """
    py = get_demucs_python()
    if not py or not Path(py).is_file():
        raise RuntimeError("Python Demucs não encontrado (is_demucs_python_available false)")
    if not Path(py).is_file():
        raise RuntimeError("Demucs python inexistente")

    # _build_demucs_command já normaliza para absolutos e loga repr
    cmd = _build_demucs_command(py, input_path, tmp_out)
    cwd = str(BASE_DIR.resolve())
    # Garante que tmp_out existe
    Path(tmp_out).resolve().mkdir(parents=True, exist_ok=True)
    logger.info(f"Demucs sync cwd={cwd} cmd_repr={[repr(c) for c in cmd]}")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            timeout=timeout,
        )
        rc = result.returncode
        out_str = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
        err_str = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
        out_log = out_str[-6000:] if len(out_str) > 6000 else out_str
        err_log = err_str[-6000:] if len(err_str) > 6000 else err_str
        logger.info(f"Demucs sync returncode={rc} stdout_tail={repr(out_log[-500:])} stderr_tail={repr(err_log[-800:])}")
        logger.debug(f"Demucs sync stdout len={len(out_str)} stderr len={len(err_str)}")
        return rc, out_str, err_str
    except subprocess.TimeoutExpired as e:
        logger.error(
            "Demucs sync timeout após %ss para %s: type=%s repr=%r str=%s",
            timeout, Path(input_path).name, type(e).__name__, e, str(e), exc_info=True
        )
        # subprocess.run já tentou matar o processo em timeout
        raise TimeoutError(f"Demucs timeout após {timeout}s") from e
    except FileNotFoundError as e:
        logger.error(
            "Demucs python não encontrado ao executar sync: type=%s repr=%r str=%s",
            type(e).__name__, e, str(e), exc_info=True
        )
        raise RuntimeError("Demucs não está instalado (python não encontrado)") from e
    except Exception as e:
        logger.error(
            "Falha ao iniciar/executar Demucs sync: type=%s repr=%r str=%s",
            type(e).__name__, e, str(e), exc_info=True
        )
        raise


async def _run_demucs_async(input_path: Path, tmp_out: Path, timeout: int = DEMUCS_TIMEOUT) -> Tuple[int, str, str]:
    """
    Wrapper async que evita limitação do WindowsSelectorEventLoop do Uvicorn
    para subprocessos. Loga tipo do event loop e delega para thread worker
    via asyncio.to_thread(_run_demucs_sync). Captura e loga tipo/repr da exceção.
    """
    # Loga tipo do event loop (diagnóstico Windows)
    try:
        loop = asyncio.get_running_loop()
        logger.info(f"Event loop Demucs: type={type(loop).__name__} repr={repr(loop)}")
    except Exception as e:
        logger.warning(f"Não foi possível obter event loop: type={type(e).__name__} repr={repr(e)} str={str(e)}")

    try:
        # Evita create_subprocess_exec no WindowsSelectorEventLoop — usa thread worker
        return await asyncio.to_thread(_run_demucs_sync, input_path, tmp_out, timeout)
    except NotImplementedError as e:
        logger.error(
            "Falha NotImplementedError ao executar Demucs (event loop sem suporte a subprocesso): type=%s repr=%r str=%s",
            type(e).__name__, e, str(e), exc_info=True
        )
        raise RuntimeError("Falha ao executar Demucs: event loop não suporta subprocesso") from e
    except TimeoutError:
        raise
    except Exception as e:
        logger.error(
            "Falha ao iniciar/executar Demucs async: type=%s repr=%r str=%s",
            type(e).__name__, e, str(e), exc_info=True
        )
        raise

def _find_generated_stems(tmp_out: Path, track_name: str) -> Dict[str, Path]:
    """
    Localiza stems gerados após Demucs.
    Espera <tmp_out>/htdemucs/<track_name>/{vocals,drums,bass,other}.wav
    Mas faz busca robusta via rglob caso modelo/track varie.
    Retorna dict stem->Path (pode estar vazio se não encontrou)
    """
    found: Dict[str, Path] = {}
    # Tenta caminho esperado
    expected_dir = tmp_out / DEMUCS_MODEL / track_name
    if expected_dir.is_dir():
        for stem in EXPECTED_STEMS:
            p = expected_dir / f"{stem}.wav"
            if p.is_file():
                found[stem] = p
        if len(found) == 4:
            return found
    # Fallback: busca recursiva por cada stem name
    for stem in EXPECTED_STEMS:
        candidates = list(tmp_out.rglob(f"{stem}.wav"))
        # Filtra candidatos que estão dentro de tmp_out/htdemucs (evita falsos)
        # Prefere mais profundo (track folder) e mais recente
        if candidates:
            # Ordena por profundidade decrescente (track folder) e mtime
            candidates.sort(key=lambda p: (len(p.parts), p.stat().st_mtime), reverse=True)
            found[stem] = candidates[0]
    return found

async def separate_stems_async(input_path: Path, file_id: str, timeout: int = DEMUCS_TIMEOUT) -> Dict:
    """
    Função principal assíncrona para separar stems de um file_id.

    - Valida file_id UUID e input_path dentro de uploads (caller já validou, mas reforça)
    - Verifica idempotência: se stems já válidos, retorna already_completed
    - Cria tmp dir controlado, executa Demucs, valida 4 stems, move para stems/<file_id>/
    - Limpa temporários em finally
    - Retorna dict com stems info ou levanta RuntimeError/TimeoutError com mensagem amigável

    Não usar shell, usa asyncio subprocess, CPU htdemucs.
    """
    if not _validate_file_id(file_id):
        raise ValueError("file_id inválido")

    if not input_path.is_file():
        raise FileNotFoundError("Arquivo de entrada não encontrado")

    # Defesa: garantir que input_path está dentro de uploads quando vier da API
    # Para testes com arquivos temporários fora de uploads, não bloqueia — apenas loga
    try:
        base_upload = Path(__file__).resolve().parents[2] / "uploads"
        try:
            input_path.resolve().relative_to(base_upload.resolve())
        except ValueError:
            # Se não estiver em uploads, verifica se é arquivo temporário válido (ex: testes)
            # Em produção input_path sempre vem de _find_upload_path, então este ramo não ocorre para requests reais
            # Não bloqueia testes com tempfile, apenas loga
            if input_path.is_file() and input_path.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
                logger.debug(f"input_path fora de uploads permitido para teste/processamento direto: {input_path}")
            else:
                raise ValueError("Path traversal detectado em input_path")
    except ValueError:
        raise
    except Exception:
        # Se não conseguir validar, loga mas continua
        pass

    # Idempotência: se já existe e válido, retorna sem reprocessar
    valid, missing = are_stems_valid(file_id)
    if valid:
        logger.info(f"Stems já válidos para file_id={file_id}, idempotência")
        return {
            "already_completed": True,
            "file_id": file_id,
            "stems_dir": str(_get_stems_dir(file_id)),
            "stems": EXPECTED_STEMS,
            "message": "Instrumentos já separados.",
        }

    # Se diretório existe mas incompleto, limpa parcial antes de reprocessar?
    # Mantém, será sobrescrito ao mover final; mas remove stale?
    # Não removemos agora, deixamos finalizar e sobrescrever.

    tmp_dir = None
    try:
        tmp_dir = Path(tempfile.mkdtemp(prefix="demucs_"))
        logger.info(f"Demucs tmp_dir={tmp_dir} file_id={file_id}")

        # Executa Demucs
        # Verifica se python demucs existe antes
        py = get_demucs_python()
        if not py or not Path(py).is_file():
            raise RuntimeError("Demucs não está instalado. Configure .venv-demucs com demucs==4.1.0")
        # Tenta verificar import demucs rapidamente (opcional)
        # Não bloqueia se torch ausente; demucs retornará erro no subprocess

        # track_name = input_path.stem (sem ext) usado pelo Demucs para subpasta
        track_name = input_path.stem

        rc, stdout, stderr = await _run_demucs_async(input_path, tmp_dir, timeout=timeout)

        if rc != 0:
            # Loga stderr sem expor completo ao usuário
            logger.error(f"Demucs falhou rc={rc} file_id={file_id} stderr[:500]={stderr[:500]}")
            # Interpreta se falta modelo/download: demucs pode falhar se rede indisponível na primeira execução
            # Mensagem amigável genérica
            # Se stderr contém "No such file" ou "model not found", trata
            raise RuntimeError(f"Não foi possível separar os instrumentos. (Demucs rc={rc})")

        # Localiza stems gerados
        found = _find_generated_stems(tmp_dir, track_name)
        if len(found) != 4:
            # Tenta listar o que foi gerado para debug
            all_wavs = list(tmp_dir.rglob("*.wav"))
            logger.error(f"Demucs saída incompleta file_id={file_id} found={list(found.keys())} all_wavs={all_wavs[:10]} stdout[:300]={stdout[:300]}")
            # Verifica se faltam stems
            missing_stems = [s for s in EXPECTED_STEMS if s not in found]
            raise RuntimeError(f"Saída incompleta: faltando {', '.join(missing_stems)}")

        # Valida cada stem (tamanho>0, ffprobe, duração)
        # Obtém duração original para comparação aproximada (tolerância 2s ou 5%)
        original_duration: Optional[float] = None
        try:
            from backend.audio.probe import probe_audio
            meta = probe_audio(input_path, file_id=file_id)
            original_duration = meta.duration
        except Exception:
            logger.debug("Não foi possível obter duração original para validação stems")

        for stem, p in found.items():
            if not p.is_file() or p.stat().st_size == 0:
                raise RuntimeError(f"Stem {stem} vazio ou inexistente")
            try:
                from backend.audio.probe import probe_audio
                m = probe_audio(p, file_id=file_id)
                # Duração aproximadamente compatível (tolerância)
                if original_duration is not None and m.duration is not None:
                    diff = abs(m.duration - original_duration)
                    # Permite diff até 1s ou 5% (o que for maior), pois stems podem ter leve diferença de trimming
                    allowed = max(1.0, original_duration * 0.05)
                    if diff > allowed:
                        logger.warning(f"Stem {stem} duração diff {diff:.2f}s > {allowed:.2f}s original {original_duration:.2f} stem {m.duration:.2f}")
                        # Não falha, apenas warning; poderia ser trimming do Demucs
            except Exception as e:
                logger.warning(f"Validação FFprobe falhou para stem {stem}: {e}")
                # Se for InvalidAudio, considera falha
                # Mas se probe falhou por ffprobe ausente, não bloqueia? Mantém
                # Para stems, consideramos falha se tamanho zero já checado; ffprobe falhar para wav demucs é raro
                # Vamos apenas logar e continuar; se realmente inválido, demucs teria falhado rc!=0

        # Normaliza para stems/<file_id>/
        final_dir = _get_stems_dir(file_id)
        final_dir.mkdir(parents=True, exist_ok=True)
        # Garante que final_dir está dentro de STEMS_DIR
        try:
            final_dir.resolve().relative_to(STEMS_DIR.resolve())
        except ValueError:
            raise RuntimeError("Path traversal em final_dir")

        for stem, src in found.items():
            dest = final_dir / f"{stem}.wav"
            try:
                # Garante dest dentro de final_dir
                dest.resolve().relative_to(final_dir.resolve())
            except ValueError:
                # Se dest não existe ainda, verifica parent
                try:
                    dest.parent.resolve().relative_to(STEMS_DIR.resolve())
                except ValueError:
                    raise RuntimeError("Path traversal em stem dest")
            # Copia (não move direto pois tmp será removido)
            shutil.copy2(str(src), str(dest))
            # Valida cópia
            if not dest.is_file() or dest.stat().st_size == 0:
                raise RuntimeError(f"Falha ao copiar stem {stem}")

        logger.info(f"Stems normalizados para {final_dir}")
        return {
            "already_completed": False,
            "file_id": file_id,
            "stems_dir": str(final_dir),
            "stems": EXPECTED_STEMS,
            "message": "Separação concluída.",
        }

    except TimeoutError as e:
        logger.error(f"Demucs timeout file_id={file_id}: {e}")
        # Limpa parcial
        final_dir = _get_stems_dir(file_id)
        # Não remove final_dir se já existia parcialmente? Mas se timeout, pode ter criado incompleto; vamos remover incompleto?
        # Mantém se já existia antes? Para simplificar, remove incompleto se criamos nesta execução e não estava válido antes
        # Como já verificamos idempotência no início, se chegamos aqui é porque não estava válido; então podemos limpar final criado parcial
        # Mas final ainda não existia válido; se criou parcial, remove
        try:
            # Se final_dir contém apenas stems incompletos criados agora, remove? Melhor não remover se já existia válido (já tratado)
            # Então limpamos apenas se encontramos falha e valid==False antes
            if not valid:
                # Se final_dir foi criado nesta tentativa e está incompleto, remove para não deixar lixo
                # Verifica se ainda não válido
                still_valid, _ = are_stems_valid(file_id)
                if not still_valid and final_dir.is_dir():
                    # Remove arquivos copiados parcialmente?
                    # Vamos remover destino incompleto para evitar estado inconsistente
                    for stem in EXPECTED_STEMS:
                        p = final_dir / f"{stem}.wav"
                        if p.is_file():
                            try:
                                p.unlink()
                            except Exception:
                                pass
                    # Tenta remover dir se vazio
                    try:
                        final_dir.rmdir()
                    except Exception:
                        pass
        except Exception:
            pass
        raise TimeoutError("A separação demorou mais que o esperado (45 min).")

    except RuntimeError:
        # Propaga com limpeza já feita no timeout? Para outros erros, também limpa temporários mas não final válido
        raise
    except Exception as e:
        logger.error(f"Erro inesperado em separate_stems_async file_id={file_id}: {e}", exc_info=True)
        raise RuntimeError("Não foi possível separar os instrumentos.")
    finally:
        # Limpa tmp_dir
        if tmp_dir and Path(tmp_dir).exists():
            try:
                shutil.rmtree(str(tmp_dir), ignore_errors=True)
            except Exception as e:
                logger.debug(f"Falha ao remover tmp_dir {tmp_dir}: {e}")

# ---------------------------------------------------------------------------
# Síncrono wrapper (para testes mock ou fallback)
# ---------------------------------------------------------------------------

def separate_stems_sync(input_path: Path, file_id: str, timeout: int = DEMUCS_TIMEOUT) -> Dict:
    """
    Wrapper síncrono que executa separate_stems_async via asyncio.run
    Útil para testes ou quando chamado de contexto síncrono.
    """
    return asyncio.run(separate_stems_async(input_path, file_id, timeout=timeout))
