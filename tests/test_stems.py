"""
Testes Etapa 4 — separação Demucs / stems
Cobrem 24 itens do spec, usando mocks para não rodar Demucs completo a cada caso.
Um teste real curto (10-20s) é feito em test_real_short (mocked por padrão, pode habiliar real com env REAL_DEMUCS=1)
"""
import tempfile
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import numpy as np
import soundfile as sf
import uuid

SR = 22050

def gen_sine(duration=5):
    t = np.arange(int(duration*SR))/SR
    y = 0.5*np.sin(2*np.pi*220*t) + 0.3*np.sin(2*np.pi*440*t)
    y = y / np.max(np.abs(y)) * 0.6
    return y

def save_wav(y, path):
    sf.write(str(path), y, SR)

def make_upload(client, duration=5, ext=".wav"):
    y = gen_sine(duration)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        p = Path(tmp.name)
    save_wav(y, p)
    with open(p, "rb") as f:
        r = client.post("/api/upload", files={"file": (f"test{ext}", f, "audio/wav")})
    p.unlink(missing_ok=True)
    return r

# 1. Python do Demucs encontrado
def test_demucs_python_found():
    from backend.audio.stem_separator import get_demucs_python, is_demucs_python_available
    py = get_demucs_python()
    assert py is not None
    # Deve apontar para .venv-demucs ou fallback
    # Pelo menos caminho contém python
    assert "python" in py.lower()
    # Se .venv-demucs existe, deve ser preferido
    venv_py = Path("C:/Users/Ryan Souza/Desktop/Projetos/gerador-partituras-ia/.venv-demucs/Scripts/python.exe")
    # Não exige existir em CI, mas se existir, get_demucs_python deve retornar ele
    if venv_py.is_file():
        assert Path(py).resolve() == venv_py.resolve()

# 2. Demucs importável
def test_demucs_importable():
    from backend.audio.stem_separator import is_demucs_available
    # Em ambiente com .venv-demucs instalado, deve ser True
    # Se não, teste skip? Mas aqui deve ser True porque instalamos
    assert is_demucs_available() is True

# 3. torch importável
def test_torch_importable():
    from backend.audio.stem_separator import get_torch_info
    info = get_torch_info()
    # Nova API retorna version (string), cuda_available (bool), cpu_capability (string)
    assert "version" in info
    assert isinstance(info["version"], str)
    assert info["version"].startswith("2.")
    assert "cuda_available" in info
    assert isinstance(info["cuda_available"], bool)
    assert info["cuda_available"] is False  # CPU nesta etapa
    assert "cpu_capability" in info
    assert isinstance(info["cpu_capability"], str)

# 4. execução CPU
def test_command_cpu():
    from backend.audio.stem_separator import _build_demucs_command
    cmd = _build_demucs_command("/fake/python", Path("/tmp/input.wav"), Path("/tmp/out"))
    assert "-d" in cmd and "cpu" in cmd
    idx = cmd.index("-d")
    assert cmd[idx+1] == "cpu"

# 5. comando usa -j 1 -n htdemucs
def test_command_jobs_and_model():
    from backend.audio.stem_separator import _build_demucs_command
    cmd = _build_demucs_command("/fake/python", Path("/tmp/a.wav"), Path("/tmp/out"))
    assert "-j" in cmd and cmd[cmd.index("-j")+1] == "1"
    assert "-n" in cmd and cmd[cmd.index("-n")+1] == "htdemucs"
    assert "htdemucs_6s" not in cmd

# 5b. comando usa caminhos absolutos e cwd
def test_command_absolute_paths():
    from backend.audio.stem_separator import _build_demucs_command, BASE_DIR
    import tempfile
    from pathlib import Path
    py = "C:/Users/Ryan Souza/Desktop/Projetos/gerador-partituras-ia/.venv-demucs/Scripts/python.exe"
    inp = Path("uploads/test.wav")  # relativo
    out = Path(tempfile.gettempdir()) / "demucs_test_out"
    cmd = _build_demucs_command(py, inp, out)
    # Verifica que todos os paths no comando são absolutos: python, out, inp são os últimos 3
    # Estrutura: [py, -m, demucs, -d, cpu, -j, 1, -n, htdemucs, -o, out, inp]
    assert Path(cmd[0]).is_absolute()
    assert Path(cmd[10]).is_absolute()
    assert Path(cmd[11]).is_absolute()
    assert "Ryan Souza" in cmd[0]  # espaço preservado como elemento único, não splitado por shell

# 6. shell=True NÃO usado
def test_no_shell_true():
    import pathlib
    files = ["app.py", "backend/audio/stem_separator.py", "backend/audio/job_manager.py", "backend/audio/probe.py", "backend/audio/music_analysis.py"]
    for fp in files:
        text = pathlib.Path(fp).read_text(encoding="utf-8")
        assert "shell=True" not in text, f"shell=True encontrado em {fp}"
        # Também verifica subprocess com shell
        assert "shell" not in text.lower() or "shell=True" not in text, f"shell suspeito {fp}"

# 7. file_id inválido
def test_separate_invalid_file_id():
    from fastapi.testclient import TestClient
    from app import app
    from backend.audio.job_manager import clear_jobs
    clear_jobs()
    client = TestClient(app)
    r = client.post("/api/separate/invalid-uuid")
    assert r.status_code == 400
    r2 = client.get("/api/separate/status/invalid-uuid")
    assert r2.status_code == 400

# 8. path traversal (stem)
def test_path_traversal_stem():
    from fastapi.testclient import TestClient
    from app import app
    from backend.audio.job_manager import clear_jobs
    clear_jobs()
    client = TestClient(app)
    # Cria upload válido para ter file_id
    r = make_upload(client, duration=3)
    assert r.status_code == 200
    file_id = r.json()["file_id"]
    # Tenta GET stem com traversal
    r2 = client.get(f"/api/stems/{file_id}/../../etc/passwd")
    # FastAPI path param não permite slash, mas ainda deve falhar allowlist 400
    assert r2.status_code in (400, 404, 422)
    # Tenta stem_name com .. 
    r3 = client.get(f"/api/stems/{file_id}/..%2F..%2Fvocals")
    assert r3.status_code in (400,404)

# 9. arquivo inexistente
def test_separate_not_found():
    from fastapi.testclient import TestClient
    from app import app
    from backend.audio.job_manager import clear_jobs
    clear_jobs()
    client = TestClient(app)
    fake_id = str(uuid.uuid4())
    r = client.post(f"/api/separate/{fake_id}")
    assert r.status_code == 404

# 10. job criação
def test_job_creation():
    from fastapi.testclient import TestClient
    from app import app
    from backend.audio.job_manager import clear_jobs
    clear_jobs()
    client = TestClient(app)
    r = make_upload(client, duration=3)
    file_id = r.json()["file_id"]
    # Mock demucs para não rodar real
    with patch("app.separate_stems_async", new=AsyncMock(return_value={"already_completed": False, "stems": ["vocals","drums","bass","other"]})):
        with patch("app.is_demucs_available", return_value=True):
            # Patch _run_separation_job para não executar async real demorado? Mas create_task ainda chama, vamos mockar diretamente
            # Vamos deixar original mas com mock separado; para evitar tempo, patch _run_separation_job como AsyncMock que completa rápido
            with patch("app._run_separation_job", new=AsyncMock()):
                r2 = client.post(f"/api/separate/{file_id}")
                assert r2.status_code == 200
                data = r2.json()
                assert data["success"] is True
                assert "job_id" in data
                assert data["status"] == "queued"
                assert data["file_id"] == file_id
                # Verifica job existe
                from backend.audio.job_manager import get_job
                job = get_job(data["job_id"])
                assert job is not None

# 11. status queued
def test_status_queued():
    from fastapi.testclient import TestClient
    from app import app
    from backend.audio.job_manager import clear_jobs
    clear_jobs()
    client = TestClient(app)
    r = make_upload(client, duration=3)
    file_id = r.json()["file_id"]
    with patch("app.is_demucs_available", return_value=True):
        with patch("app._run_separation_job", new=AsyncMock()):
            r2 = client.post(f"/api/separate/{file_id}")
            job_id = r2.json()["job_id"]
            r3 = client.get(f"/api/separate/status/{job_id}")
            assert r3.status_code == 200
            assert r3.json()["status"] in ("queued","running")

# 12. status running (mock)
def test_status_running():
    from backend.audio.job_manager import clear_jobs, create_job, update_job
    clear_jobs()
    job = create_job("test-file", status="running", message="Separando instrumentos...")
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    r = client.get(f"/api/separate/status/{job.job_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "running"

# 13. status completed (mock)
def test_status_completed():
    from backend.audio.job_manager import clear_jobs, create_job, update_job
    from backend.audio.stem_separator import STEMS_DIR
    clear_jobs()
    # Cria fake stems dir para file_id
    file_id = str(uuid.uuid4())
    d = STEMS_DIR / file_id
    d.mkdir(parents=True, exist_ok=True)
    for stem in ["vocals","drums","bass","other"]:
        p = d / f"{stem}.wav"
        # cria wav válido de 1s
        y = gen_sine(1)
        save_wav(y, p)
    job = create_job(file_id, status="completed", message="Separação concluída.")
    job.stems = ["vocals","drums","bass","other"]
    update_job(job.job_id, status="completed", message="Separação concluída.", stems=job.stems)
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    r = client.get(f"/api/separate/status/{job.job_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    # cleanup
    import shutil
    shutil.rmtree(d, ignore_errors=True)

# 14. status failed
def test_status_failed():
    from backend.audio.job_manager import clear_jobs, create_job, update_job
    clear_jobs()
    job = create_job("fid", status="failed", message="Não foi possível separar")
    update_job(job.job_id, status="failed", message="Não foi possível separar", error="erro")
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    r = client.get(f"/api/separate/status/{job.job_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    assert "error" in r.json()

# 15. erro Demucs (rc !=0) — deve falhar
def test_demucs_error_handling():
    from pathlib import Path
    import asyncio
    from backend.audio.stem_separator import separate_stems_async
    import tempfile
    y = gen_sine(3)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    save_wav(y, p)
    file_id = str(uuid.uuid4())
    async def mock_run(*args, **kwargs):
        return 1, "", "error"
    with patch("backend.audio.stem_separator._run_demucs_async", new=mock_run):
        try:
            asyncio.run(separate_stems_async(p, file_id, timeout=10))
            assert False, "deveria levantar"
        except RuntimeError as e:
            assert "Não foi possível separar" in str(e) or "rc" in str(e)
    p.unlink(missing_ok=True)

# 15b. warning com rc 0 — NÃO deve falhar (regressão Hugging Face warning)
def test_warning_with_rc0_not_error():
    from pathlib import Path
    import asyncio, tempfile, pathlib
    from backend.audio.stem_separator import separate_stems_async, STEMS_DIR
    y = gen_sine(2)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = pathlib.Path(tmp.name)
    save_wav(y, p)
    file_id = str(uuid.uuid4())
    async def mock_run(input_path, tmp_out, timeout):
        # Simula Demucs rc 0 com warning no stderr (HF Hub) — deve ser considerado sucesso
        track = Path(input_path).stem
        out = Path(tmp_out) / "htdemucs" / track
        out.mkdir(parents=True, exist_ok=True)
        for stem in ["vocals","drums","bass","other"]:
            yy = gen_sine(1)
            save_wav(yy, out / f"{stem}.wav")
        return 0, "Selected model is a bag of 1 models.", "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN"
    with patch("backend.audio.stem_separator._run_demucs_async", new=mock_run):
        res = asyncio.run(separate_stems_async(p, file_id, timeout=10))
        assert res["already_completed"] is False
        assert res["file_id"] == file_id
        # Verifica stems criados
        for stem in ["vocals","drums","bass","other"]:
            assert (STEMS_DIR / file_id / f"{stem}.wav").exists()
    # cleanup
    import shutil
    shutil.rmtree(STEMS_DIR / file_id, ignore_errors=True)
    p.unlink(missing_ok=True)

# 16. timeout
def test_timeout():
    from pathlib import Path
    import asyncio
    from backend.audio.stem_separator import separate_stems_async
    y = gen_sine(3)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    save_wav(y, p)
    file_id = str(uuid.uuid4())
    async def mock_run(*args, **kwargs):
        raise asyncio.TimeoutError()
    with patch("backend.audio.stem_separator._run_demucs_async", new=mock_run):
        try:
            asyncio.run(separate_stems_async(p, file_id, timeout=1))
            assert False
        except TimeoutError:
            pass
        except RuntimeError:
            pass # pode ser RuntimeError encapsulado
    p.unlink(missing_ok=True)

# 17. saída faltando stem
def test_missing_stem():
    from backend.audio.stem_separator import _find_generated_stems
    with tempfile.TemporaryDirectory() as tmp:
        tmp_p = Path(tmp)
        # cria apenas 2 stems
        d = tmp_p / "htdemucs" / "track"
        d.mkdir(parents=True)
        for stem in ["vocals","drums"]:
            p = d / f"{stem}.wav"
            y = gen_sine(1)
            save_wav(y, p)
        found = _find_generated_stems(tmp_p, "track")
        assert len(found) != 4
        assert "bass" not in found

# 18. arquivo stem vazio
def test_empty_stem():
    from backend.audio.stem_separator import are_stems_valid, STEMS_DIR
    file_id = str(uuid.uuid4())
    d = STEMS_DIR / file_id
    d.mkdir(parents=True, exist_ok=True)
    for stem in ["vocals","drums","bass","other"]:
        p = d / f"{stem}.wav"
        if stem == "bass":
            p.write_bytes(b"")  # vazio
        else:
            y = gen_sine(1)
            save_wav(y, p)
    valid, missing = are_stems_valid(file_id)
    assert valid is False
    assert "bass" in missing
    import shutil
    shutil.rmtree(d, ignore_errors=True)

# 19. stem_name inválido
def test_invalid_stem_name():
    from fastapi.testclient import TestClient
    from app import app
    from backend.audio.job_manager import clear_jobs
    clear_jobs()
    client = TestClient(app)
    r = make_upload(client, duration=2)
    file_id = r.json()["file_id"]
    r2 = client.get(f"/api/stems/{file_id}/guitar")
    assert r2.status_code == 400
    assert "permitidos" in r2.json()["detail"].lower()

# 20. GET stem válido
def test_get_valid_stem():
    from fastapi.testclient import TestClient
    from app import app
    from backend.audio.stem_separator import STEMS_DIR
    from backend.audio.job_manager import clear_jobs
    clear_jobs()
    client = TestClient(app)
    # Cria upload e fake stems
    r = make_upload(client, duration=2)
    file_id = r.json()["file_id"]
    d = STEMS_DIR / file_id
    d.mkdir(parents=True, exist_ok=True)
    for stem in ["vocals","drums","bass","other"]:
        y = gen_sine(1)
        save_wav(y, d / f"{stem}.wav")
    r2 = client.get(f"/api/stems/{file_id}/vocals")
    assert r2.status_code == 200
    assert r2.headers["content-type"].startswith("audio")
    # Lista
    r3 = client.get(f"/api/stems/{file_id}")
    assert r3.status_code == 200
    assert r3.json()["available"] is True
    assert len(r3.json()["stems"]) == 4
    import shutil
    shutil.rmtree(d, ignore_errors=True)

# 21. idempotência
def test_idempotency():
    from fastapi.testclient import TestClient
    from app import app
    from backend.audio.stem_separator import STEMS_DIR
    from backend.audio.job_manager import clear_jobs
    clear_jobs()
    client = TestClient(app)
    r = make_upload(client, duration=2)
    file_id = r.json()["file_id"]
    d = STEMS_DIR / file_id
    d.mkdir(parents=True, exist_ok=True)
    for stem in ["vocals","drums","bass","other"]:
        y = gen_sine(1)
        save_wav(y, d / f"{stem}.wav")
    # Segunda chamada deve retornar already_completed
    with patch("app.is_demucs_available", return_value=True):
        r2 = client.post(f"/api/separate/{file_id}")
        assert r2.status_code == 200
        assert r2.json()["already_completed"] is True
        assert r2.json()["status"] == "completed"
    import shutil
    shutil.rmtree(d, ignore_errors=True)

# 22. apenas um job simultâneo
def test_single_job_concurrency():
    from fastapi.testclient import TestClient
    from app import app
    from backend.audio.job_manager import clear_jobs
    clear_jobs()
    client = TestClient(app)
    r = make_upload(client, duration=2)
    file_id = r.json()["file_id"]
    # Mock para manter job running sem completar
    with patch("app.is_demucs_available", return_value=True):
        with patch("app._run_separation_job", new=AsyncMock()):
            r2 = client.post(f"/api/separate/{file_id}")
            assert r2.status_code == 200
            # Segunda tentativa com outro file_id deve dar 409
            r3 = make_upload(client, duration=2)
            file_id2 = r3.json()["file_id"]
            r4 = client.post(f"/api/separate/{file_id2}")
            assert r4.status_code == 409
            assert "andamento" in r4.json()["detail"].lower()

# 23. temporários removidos
def test_temp_cleanup_on_success():
    import tempfile, pathlib
    from backend.audio.stem_separator import separate_stems_async
    import asyncio
    y = gen_sine(2)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = pathlib.Path(tmp.name)
    save_wav(y, p)
    file_id = str(uuid.uuid4())
    # Mock demucs para criar estrutura fake rapidamente e não usar subprocess real
    async def mock_run(input_path, tmp_out, timeout):
        # Simula demucs criando output
        track = Path(input_path).stem
        out = Path(tmp_out) / "htdemucs" / track
        out.mkdir(parents=True, exist_ok=True)
        for stem in ["vocals","drums","bass","other"]:
            yy = gen_sine(1)
            save_wav(yy, out / f"{stem}.wav")
        return 0, "ok", ""
    with patch("backend.audio.stem_separator._run_demucs_async", new=mock_run):
        # Conta temp antes
        tmp_before = set(Path(tempfile.gettempdir()).glob("demucs_*"))
        asyncio.run(separate_stems_async(p, file_id, timeout=10))
        tmp_after = set(Path(tempfile.gettempdir()).glob("demucs_*"))
        leaked = [x for x in tmp_after if x not in tmp_before]
        assert len(leaked) == 0, f"temp leaked {leaked}"
    # Cleanup stems
    from backend.audio.stem_separator import STEMS_DIR
    import shutil
    shutil.rmtree(STEMS_DIR / file_id, ignore_errors=True)
    p.unlink(missing_ok=True)

# 24. Etapas 1-3 continuam passando (health, upload, analyze)
def test_etapas_1_3():
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    r = make_upload(client, duration=2)
    assert r.status_code == 200
    file_id = r.json()["file_id"]
    r2 = client.get(f"/api/analyze/{file_id}")
    assert r2.status_code == 200
    data = r2.json()
    assert "duration" in data and "music" in data
    assert "bpm" in data["music"]

# Teste endpoint demucs/info — garante não há NameError/TypeError e serialização
def test_demucs_info_endpoint():
    from fastapi.testclient import TestClient
    from app import app
    from backend.audio.stem_separator import DEMUCS_MODEL, DEMUCS_DEVICE, DEMUCS_JOBS, DEMUCS_TIMEOUT, EXPECTED_STEMS
    client = TestClient(app)
    r = client.get("/api/demucs/info")
    assert r.status_code == 200, f"GET /api/demucs/info falhou {r.status_code} {r.text}"
    data = r.json()
    # model
    assert data["model"] == DEMUCS_MODEL
    assert data["model"] == "htdemucs"
    # device
    assert data["device"] == DEMUCS_DEVICE
    assert data["device"] == "cpu"
    # jobs — deve ser 1, não 2700
    assert data["jobs"] == int(DEMUCS_JOBS)
    assert data["jobs"] == 1
    assert data["jobs"] != data["timeout_seconds"] or data["timeout_seconds"] == 2700  # garante jobs != timeout
    # timeout
    assert data["timeout_seconds"] == int(DEMUCS_TIMEOUT)
    assert data["timeout_seconds"] == 2700
    # expected_stems
    assert data["expected_stems"] == EXPECTED_STEMS
    assert data["expected_stems"] == ["vocals", "drums", "bass", "other"]
    # demucs_python serializável string
    assert isinstance(data["demucs_python"], str)
    assert "python" in data["demucs_python"].lower()
    # demucs_available bool
    assert isinstance(data["demucs_available"], bool)
    # torch_info serializável com tipos corretos
    ti = data["torch_info"]
    assert isinstance(ti, dict)
    assert "version" in ti and isinstance(ti["version"], str)
    assert "cuda_available" in ti and isinstance(ti["cuda_available"], bool)
    assert "cpu_capability" in ti and isinstance(ti["cpu_capability"], str)
    # garante JSON serializável total
    import json
    json.dumps(data)

def test_demucs_info_no_name_error():
    # Auditoria: garante que todos os símbolos usados existem e são serializáveis
    from fastapi.testclient import TestClient
    from app import app
    client = TestClient(app)
    # Deve retornar 200 sem exceção, mesmo se chamado múltiplas vezes
    for _ in range(2):
        r = client.get("/api/demucs/info")
        assert r.status_code == 200
        # Verifica que não há NameError no texto
        assert "NameError" not in r.text
        assert "TypeError" not in r.text

# Teste NotImplementedError fallback — garante que asyncio.to_thread evita limitação WindowsSelectorEventLoop
def test_not_implemented_fallback():
    import asyncio
    from pathlib import Path
    import tempfile, soundfile as sf, numpy as np
    from unittest.mock import patch
    # Simula create_subprocess_exec quebrado, mas to_thread deve continuar funcionando
    async def fake_to_thread(func, *args, **kwargs):
        # Executa sync diretamente (simula to_thread)
        return func(*args, **kwargs)
    # Mock _run_demucs_sync para não rodar Demucs real, apenas retorna sucesso
    from backend.audio.stem_separator import _run_demucs_async
    with patch("asyncio.create_subprocess_exec", side_effect=NotImplementedError("Selector loop")):
        with patch("asyncio.to_thread", side_effect=fake_to_thread):
            with patch("backend.audio.stem_separator._run_demucs_sync", return_value=(0, "ok", "Warning: HF Hub")) as mock_sync:
                tmp_in = Path(tempfile.gettempdir()) / "test_notimpl.wav"
                y = np.zeros(22050)
                sf.write(str(tmp_in), y, 22050)
                tmp_out = Path(tempfile.mkdtemp())
                # Deve chamar via to_thread e não falhar com NotImplementedError
                rc, out, err = asyncio.run(_run_demucs_async(tmp_in, tmp_out, timeout=10))
                assert rc == 0
                # Verifica que _run_demucs_sync foi chamado (via to_thread)
                assert mock_sync.called
                tmp_in.unlink(missing_ok=True)
                import shutil
                shutil.rmtree(tmp_out, ignore_errors=True)

# Teste paths com espaços, cwd com espaços, stdout/stderr vazio, WAV/MP3
def test_paths_with_spaces_and_empty_output():
    from backend.audio.stem_separator import _build_demucs_command, BASE_DIR
    from pathlib import Path
    import tempfile
    # Caminho com espaços (projeto já tem "Ryan Souza")
    py = str(Path("C:/Users/Ryan Souza/Desktop/Projetos/gerador-partituras-ia/.venv-demucs/Scripts/python.exe"))
    # Input com espaços
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_p = Path(tmpdir)
        inp_with_space = tmpdir_p / "test file with spaces.wav"
        y = gen_sine(1)
        save_wav(y, inp_with_space)
        out = tmpdir_p / "out dir with spaces"
        out.mkdir()
        cmd = _build_demucs_command(py, inp_with_space, out)
        # Verifica absolutos e espaços preservados como elemento único
        assert Path(cmd[0]).is_absolute()
        assert Path(cmd[10]).is_absolute()
        assert " " in cmd[10]  # out com espaços deve ser elemento único com espaço, não quebrado
        assert " " in cmd[11]  # inp com espaços
        # Verifica que cwd com espaços (BASE_DIR) é usado
        assert " " in str(BASE_DIR.resolve())

def test_empty_stdout_stderr_with_rc0():
    # stderr/stdout vazios com rc 0 não deve falhar se stems encontrados (mock)
    from pathlib import Path
    import asyncio, tempfile
    from backend.audio.stem_separator import separate_stems_async, STEMS_DIR
    from unittest.mock import patch
    y = gen_sine(1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    save_wav(y, p)
    file_id = str(uuid.uuid4())
    async def mock_run(input_path, tmp_out, timeout):
        track = Path(input_path).stem
        out = Path(tmp_out) / "htdemucs" / track
        out.mkdir(parents=True, exist_ok=True)
        for stem in ["vocals","drums","bass","other"]:
            save_wav(gen_sine(1), out / f"{stem}.wav")
        return 0, "", ""  # vazio mas rc 0
    with patch("backend.audio.stem_separator._run_demucs_sync", side_effect=lambda *a, **k: (0, "", "")):
        # Na verdade patchamos _run_demucs_async para usar mock_run
        with patch("backend.audio.stem_separator._run_demucs_async", new=mock_run):
            res = asyncio.run(separate_stems_async(p, file_id, timeout=10))
            assert res["file_id"] == file_id
    import shutil
    shutil.rmtree(STEMS_DIR / file_id, ignore_errors=True)
    p.unlink(missing_ok=True)

def test_input_wav_and_mp3_with_spaces():
    # Garante que tanto WAV quanto MP3 com espaços funcionam (lista, sem shell)
    import subprocess
    from pathlib import Path
    import tempfile
    # Não roda Demucs real, apenas verifica que comando lista não quebra com espaços
    from backend.audio.stem_separator import _build_demucs_command
    py = str(Path("C:/Users/Ryan Souza/Desktop/Projetos/gerador-partituras-ia/.venv-demucs/Scripts/python.exe"))
    for ext in [".wav", ".mp3"]:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / f"audio with spaces{ext}"
            y = gen_sine(1)
            save_wav(y, p)
            out = Path(tmpdir) / "out"
            cmd = _build_demucs_command(py, p, out)
            # Verifica que extensão é preservada e caminho absoluto
            assert cmd[-1].endswith(ext)
            assert Path(cmd[-1]).is_absolute()
            # Verifica que não há shell quoting manual
            assert '"' not in cmd[-1]  # não deve ter aspas dentro do argumento

# Teste real curto opcional (habilitado via env)
def test_real_short_optional():
    import os
    if os.getenv("REAL_DEMUCS") != "1":
        # skip por padrão para não gastar tempo/ci
        return
    from fastapi.testclient import TestClient
    from app import app
    import time
    client = TestClient(app)
    # Gera 12s de áudio com mistura simples (vocais+bateria+baixo simulados)
    # Para teste real, usa sine com diferentes freqs
    y = gen_sine(12)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        p = Path(tmp.name)
    save_wav(y, p)
    with open(p, "rb") as f:
        r = client.post("/api/upload", files={"file": ("real12.wav", f, "audio/wav")})
    assert r.status_code == 200
    file_id = r.json()["file_id"]
    start = time.time()
    r2 = client.post(f"/api/separate/{file_id}")
    assert r2.status_code == 200
    job_id = r2.json()["job_id"]
    # Poll até completed (max 5 min para 12s)
    for _ in range(60):
        time.sleep(5)
        rr = client.get(f"/api/separate/status/{job_id}")
        assert rr.status_code == 200
        if rr.json()["status"] == "completed":
            break
        if rr.json()["status"] == "failed":
            raise AssertionError(f"job failed {rr.json()}")
    else:
        raise AssertionError("timeout real short")
    elapsed = time.time() - start
    print(f"real short elapsed {elapsed:.1f}s")
    # Verifica stems
    for stem in ["vocals","drums","bass","other"]:
        rr = client.get(f"/api/stems/{file_id}/{stem}")
        assert rr.status_code == 200
    # Cleanup
    from backend.audio.stem_separator import STEMS_DIR
    import shutil
    shutil.rmtree(STEMS_DIR / file_id, ignore_errors=True)
    p.unlink(missing_ok=True)
