"""Cálculo de BPM/GAP, cuantización a beats y escritura del txt de UltraStar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .notes import Note

LANGUAGE_NAMES = {
    "es": "Español",
    "ca": "Español",
    "gl": "Español",
    "eu": "Español",
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
}
MIDI_C4 = 60  # tono 0 de UltraStar
MAX_LINE_CHARS = 45


@dataclass
class BeatNote:
    kind: str
    start: int
    length: int
    pitch: int
    text: str
    line: int


def detect_bpm(audio_path: str | Path) -> float:
    """Tempo del acompañamiento, duplicado hasta caer en el rango 200-400 que pide UltraStar."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    if bpm <= 0:
        raise RuntimeError(f"No se ha podido estimar el tempo de {audio_path}")
    while bpm < 200:
        bpm *= 2
    while bpm > 400:
        bpm /= 2
    return round(bpm, 2)


def beat_seconds(bpm: float) -> float:
    """Duración de un beat del txt: UltraStar usa cuartos del pulso indicado en #BPM."""
    return 15.0 / bpm


def quantize(notes: list[Note], bpm: float) -> tuple[float, list[BeatNote]]:
    """Devuelve (GAP en ms, notas en beats). El beat 0 es el inicio de la primera nota."""
    unit = beat_seconds(bpm)
    gap = notes[0].start
    out: list[BeatNote] = []
    for i, note in enumerate(notes):
        start = round((note.start - gap) / unit)
        end = round((note.end - gap) / unit)
        if out:
            prev = out[-1]
            overlap = prev.start + prev.length - start
            if overlap > 0:
                shrink = min(overlap, prev.length - 1)
                prev.length -= shrink
                start += overlap - shrink
        length = max(1, end - start)
        last_in_line = i + 1 == len(notes) or notes[i + 1].line != note.line
        text = note.text + (" " if note.word_end and not last_in_line else "")
        out.append(BeatNote(note.kind, start, length, note.pitch - MIDI_C4, text, note.line))
    return gap * 1000.0, out


def _decimal(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def render(headers: dict[str, str], bpm: float, gap_ms: float, notes: list[BeatNote]) -> str:
    lines = [f"#{key}:{value}" for key, value in headers.items() if value]
    lines.append(f"#BPM:{_decimal(bpm)}")
    lines.append(f"#GAP:{_decimal(gap_ms)}")
    for i, note in enumerate(notes):
        if i and note.line != notes[i - 1].line:
            prev_end = notes[i - 1].start + notes[i - 1].length
            lines.append(f"- {prev_end + min((note.start - prev_end) // 2, 8)}")
        lines.append(f"{note.kind} {note.start} {note.length} {note.pitch} {note.text}")
    lines.append("E")
    return "\n".join(lines) + "\n"
