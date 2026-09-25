#!/usr/bin/env python
"""
Worker isolado para Basic Pitch — executado via .venv-basicpitch.

Responsabilidades:
- carregar Basic Pitch (ONNX/TF)
- executar prediction no stem.wav
- salvar MIDI
- serializar note events para JSON
- imprimir resumo JSON em stdout
- exit 0 em sucesso, !=0 em falha

Não depende de FastAPI. Executado como:
  .venv-basicpitch/Scripts/python.exe backend/workers/basic_pitch_worker.py
    --input <stem.wav> --output-midi <out.mid> --output-json <out.json>
    [--minimum-frequency 30] [--maximum-frequency 500] [--stem bass]

Caminhos devem ser absolutos. O worker é chamado via subprocess com shell=False.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Nota: este worker roda em .venv-basicpitch, onde basic_pitch e onnxruntime estão disponíveis.
# Não importar no .venv principal.

def midi_pitch_to_note_name(pitch: int) -> str:
    """Converte MIDI pitch (0-127) para nome internacional, ex: 60 -> C4."""
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (pitch // 12) - 1
    name = names[pitch % 12]
    return f"{name}{octave}"

def parse_args():
    p = argparse.ArgumentParser(description="Basic Pitch worker")
    p.add_argument("--input", required=True, help="Caminho absoluto para stem.wav")
    p.add_argument("--output-midi", required=True, help="Caminho absoluto para saida .mid")
    p.add_argument("--output-json", required=True, help="Caminho absoluto para saida .json com eventos")
    p.add_argument("--stem", default="other", help="Nome do stem (vocals/bass/other) para metadados")
    p.add_argument("--file-id", default="", help="file_id para metadados JSON")
    p.add_argument("--minimum-frequency", type=float, default=None, help="Frequencia minima Hz")
    p.add_argument("--maximum-frequency", type=float, default=None, help="Frequencia maxima Hz")
    p.add_argument("--onset-threshold", type=float, default=0.5, help="onset_threshold (default 0.5)")
    p.add_argument("--frame-threshold", type=float, default=0.3, help="frame_threshold (default 0.3)")
    p.add_argument("--minimum-note-length", type=float, default=127.7, help="minimum_note_length ms (default 127.7)")
    p.add_argument("--midi-tempo", type=float, default=120.0, help="midi_tempo (default 120)")
    p.add_argument("--multiple-pitch-bends", action="store_true", help="multiple_pitch_bends")
    p.add_argument("--no-melodia-trick", action="store_true", help="desativa melodia_trick")
    return p.parse_args()

def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_midi = Path(args.output_midi).resolve()
    output_json = Path(args.output_json).resolve()

    # Validações básicas
    if not input_path.is_file():
        print(json.dumps({"error": f"Input não encontrado: {input_path}"}), file=sys.stderr)
        sys.exit(2)
    if input_path.stat().st_size == 0:
        print(json.dumps({"error": f"Input vazio: {input_path}"}), file=sys.stderr)
        sys.exit(3)

    # Garante diretórios de saída existem
    output_midi.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Parâmetros de frequência por stem (se não passados explicitamente, usa defaults do transcriber)
    # O worker apenas repassa o que recebeu; validação de ranges é feita no transcriber.

    try:
        from basic_pitch.inference import predict
    except Exception as e:
        print(json.dumps({"error": f"Falha ao importar basic_pitch: {e}"}), file=sys.stderr)
        sys.exit(10)

    # Prepara kwargs para predict
    predict_kwargs = {
        "onset_threshold": args.onset_threshold,
        "frame_threshold": args.frame_threshold,
        "minimum_note_length": args.minimum_note_length,
        "midi_tempo": args.midi_tempo,
        "multiple_pitch_bends": args.multiple_pitch_bends,
        "melodia_trick": not args.no_melodia_trick,
    }
    if args.minimum_frequency is not None:
        predict_kwargs["minimum_frequency"] = args.minimum_frequency
    if args.maximum_frequency is not None:
        predict_kwargs["maximum_frequency"] = args.maximum_frequency

    try:
        # predict retorna (model_output, midi_data, note_events)
        # note_events: List[Tuple[float, float, int, float, Optional[List[int]]]]
        #   (start, end, pitch, amplitude, pitch_bends)
        model_output, midi_data, note_events = predict(str(input_path), **predict_kwargs)
    except Exception as e:
        import traceback
        print(json.dumps({"error": f"Falha na predição Basic Pitch: {e}", "traceback": traceback.format_exc()}), file=sys.stderr)
        sys.exit(11)

    # Salva MIDI
    try:
        midi_data.write(str(output_midi))
    except Exception as e:
        import traceback
        print(json.dumps({"error": f"Falha ao salvar MIDI: {e}", "traceback": traceback.format_exc()}), file=sys.stderr)
        sys.exit(12)

    # Valida MIDI salvo
    try:
        if not output_midi.is_file() or output_midi.stat().st_size == 0:
            print(json.dumps({"error": "MIDI não gerado ou vazio"}), file=sys.stderr)
            sys.exit(13)
        # Tenta abrir com pretty_midi para validar
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(output_midi))
        # Verifica se tem instrumentos/tracks (pode ser 0 notas, mas deve ter pelo menos 1 instrumento)
        # Se não houver notas, não é erro técnico, mas warning
        notes_in_midi = sum(len(instr.notes) for instr in pm.instruments) if pm.instruments else 0
    except Exception as e:
        import traceback
        print(json.dumps({"error": f"MIDI inválido: {e}", "traceback": traceback.format_exc()}), file=sys.stderr)
        sys.exit(14)

    # Converte note_events para JSON estruturado
    # Cada evento: start, end, duration, pitch, note, velocity, amplitude, confidence, pitch_bends
    # amplitude do Basic Pitch é float 0-1; convertemos para velocity 0-127 e mantemos amplitude
    events = []
    try:
        for ev in note_events or []:
            # ev pode ser tuple de 4 ou 5 elementos
            if len(ev) == 5:
                start, end, pitch, amplitude, pitch_bends = ev
            elif len(ev) == 4:
                start, end, pitch, amplitude = ev
                pitch_bends = None
            else:
                # Caso inesperado, tenta desempacotar
                start, end, pitch = ev[0], ev[1], ev[2]
                amplitude = ev[3] if len(ev) > 3 else 0.5
                pitch_bends = ev[4] if len(ev) > 4 else None

            start_f = float(start)
            end_f = float(end)
            duration = float(end_f - start_f) if end_f > start_f else 0.0
            pitch_i = int(pitch)
            # Amplitude 0-1 -> velocity 0-127, clamp
            try:
                amp_f = float(amplitude)
            except:
                amp_f = 0.5
            amp_f = max(0.0, min(1.0, amp_f))
            velocity = int(round(amp_f * 127))
            velocity = max(0, min(127, velocity))
            note_name = midi_pitch_to_note_name(pitch_i)

            # Pitch bends: Basic Pitch retorna lista de ints ou None
            # Mantemos como está, mas garantimos serializável
            if pitch_bends is not None:
                try:
                    pb = [int(x) for x in pitch_bends] if isinstance(pitch_bends, (list, tuple)) else None
                except:
                    pb = None
            else:
                pb = None

            event = {
                "start": round(start_f, 6),
                "end": round(end_f, 6),
                "duration": round(duration, 6),
                "pitch": pitch_i,
                "note": note_name,
                "velocity": velocity,
                "amplitude": round(amp_f, 4),
                # Mantemos confidence como alias de amplitude para compatibilidade, documentado como amplitude
                "confidence": round(amp_f, 4),
                "strength": round(amp_f, 4),
            }
            if pb is not None:
                event["pitch_bends"] = pb
            # Opcional: incluir pitch_bends se multiple_pitch_bends True

            events.append(event)
    except Exception as e:
        import traceback
        print(json.dumps({"error": f"Falha ao serializar eventos: {e}", "traceback": traceback.format_exc()}), file=sys.stderr)
        sys.exit(15)

    # Ordena por start time
    events.sort(key=lambda x: x["start"])

    # Calcula duração do áudio original (para metadados)
    duration = 0.0
    try:
        import soundfile as sf
        info = sf.info(str(input_path))
        duration = float(info.duration) if hasattr(info, 'duration') else 0.0
    except:
        # fallback: usa último evento end ou 0
        if events:
            duration = max(ev["end"] for ev in events)
        else:
            duration = 0.0

    notes_count = len(events)
    warning = None
    if notes_count == 0:
        warning = "Nenhuma nota detectada."

    # Monta JSON final
    output_data = {
        "file_id": args.file_id,
        "stem": args.stem,
        "notes_count": notes_count,
        "duration": round(duration, 3),
        "events": events,
    }
    if warning:
        output_data["warning"] = warning
    # Inclui parâmetros usados para auditoria
    output_data["parameters"] = {
        "minimum_frequency": args.minimum_frequency,
        "maximum_frequency": args.maximum_frequency,
        "onset_threshold": args.onset_threshold,
        "frame_threshold": args.frame_threshold,
        "minimum_note_length": args.minimum_note_length,
        "midi_tempo": args.midi_tempo,
        "multiple_pitch_bends": args.multiple_pitch_bends,
        "melodia_trick": not args.no_melodia_trick,
    }

    try:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        import traceback
        print(json.dumps({"error": f"Falha ao salvar JSON: {e}", "traceback": traceback.format_exc()}), file=sys.stderr)
        sys.exit(16)

    # Valida JSON salvo
    try:
        if not output_json.is_file() or output_json.stat().st_size == 0:
            print(json.dumps({"error": "JSON não gerado ou vazio"}), file=sys.stderr)
            sys.exit(17)
        with open(output_json, "r", encoding="utf-8") as f:
            json.load(f)
    except Exception as e:
        print(json.dumps({"error": f"JSON inválido: {e}"}), file=sys.stderr)
        sys.exit(18)

    # Sucesso: imprime resumo JSON em stdout para o chamador capturar
    summary = {
        "file_id": args.file_id,
        "stem": args.stem,
        "notes_count": notes_count,
        "midi_path": str(output_midi),
        "json_path": str(output_json),
        "duration": round(duration, 3),
        "warning": warning,
    }
    print(json.dumps(summary))
    sys.exit(0)

if __name__ == "__main__":
    main()
