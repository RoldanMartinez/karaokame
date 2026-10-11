"""Conversión de sílabas temporizadas en notas con tono."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import medfilt

from .align import Word
from .pitch import PitchTrack
from .syllables import ends_with_vowel_sound, starts_with_vowel_sound


@dataclass
class Note:
    start: float
    end: float
    text: str
    line: int
    word_end: bool  # la nota cierra una palabra (lleva espacio detrás)
    pitch: int | None = None  # nota MIDI
    kind: str = ":"

    @property
    def duration(self) -> float:
        return self.end - self.start


def build_notes(words: list[Word], language: str, sinalefa: bool = True) -> list[Note]:
    """Una nota por sílaba; opcionalmente funde las sinalefas ("te ha", "to en")."""
    notes: list[Note] = []
    for word in words:
        last = len(word.syllables) - 1
        for i, (text, (start, end)) in enumerate(zip(word.syllables, word.syllable_spans)):
            note = Note(start, end, text, word.line, word_end=i == last)
            prev = notes[-1] if notes else None
            if (
                sinalefa
                and language == "es"
                and i == 0
                and prev is not None
                and prev.word_end
                and prev.line == note.line
                and ends_with_vowel_sound(prev.text)
                and starts_with_vowel_sound(text)
                and note.start - prev.end < 0.08
                and min(prev.duration, note.duration) < 0.16
            ):
                prev.text = f"{prev.text} {text}"
                prev.end = end
                prev.word_end = note.word_end
            else:
                notes.append(note)
    return notes


def _pitch_segments(track: PitchTrack, t0: float, t1: float, min_seg: float) -> list[tuple[float, float, int]]:
    """Divide [t0, t1] en tramos de tono estable. Lista vacía si no hay voz."""
    sl = track.span(t0, t1)
    raw = track.midi[sl]
    voiced = track.voiced[sl]
    if not voiced.any():
        return []
    idx = np.arange(len(raw))
    filled = np.interp(idx, idx[voiced], raw[voiced])
    kernel = min(9, len(filled) - (1 - len(filled) % 2))
    smooth = medfilt(filled, kernel) if kernel >= 3 else filled
    rounded = np.round(smooth).astype(int)

    # Tramos [inicio, fin, valor] de valor constante.
    cuts = np.flatnonzero(np.diff(rounded)) + 1
    bounds = np.concatenate(([0], cuts, [len(rounded)]))
    runs = [[int(a), int(b), int(rounded[a])] for a, b in zip(bounds[:-1], bounds[1:])]

    def merge_equal(rs):
        out = [rs[0]]
        for r in rs[1:]:
            if r[2] == out[-1][2]:
                out[-1][1] = r[1]
            else:
                out.append(r)
        return out

    min_frames = max(2, int(round(min_seg / track.hop_s)))
    while len(runs) > 1:
        shortest = min(range(len(runs)), key=lambda i: runs[i][1] - runs[i][0])
        if runs[shortest][1] - runs[shortest][0] >= min_frames:
            break
        # El tramo corto (adorno, vibrato, ataque) se absorbe en el vecino de tono más próximo.
        neighbours = [i for i in (shortest - 1, shortest + 1) if 0 <= i < len(runs)]
        target = min(neighbours, key=lambda i: (abs(runs[i][2] - runs[shortest][2]), runs[i][0] - runs[i][1]))
        runs[target][0] = min(runs[target][0], runs[shortest][0])
        runs[target][1] = max(runs[target][1], runs[shortest][1])
        del runs[shortest]
        runs = merge_equal(runs)

    segments = []
    for a, b, value in runs:
        values = raw[a:b][voiced[a:b]]
        pitch = int(round(float(np.median(values)))) if len(values) else value
        segments.append([a, b, pitch])
    segments = merge_equal(segments)

    span = t1 - t0
    n = len(raw)
    return [(t0 + span * a / n, t0 + span * b / n, p) for a, b, p in segments]


def assign_pitch(notes: list[Note], track: PitchTrack, melisma: bool = True, min_seg: float = 0.15) -> list[Note]:
    """Asigna tono a cada nota; las sílabas con varios tonos se parten en notas "~"."""
    out: list[Note] = []
    for note in notes:
        segments = _pitch_segments(track, note.start, note.end, min_seg)
        if not segments:
            out.append(note)  # sin voz detectada: se rellena después
            continue
        if not melisma or len(segments) == 1 or len(segments) > 4:
            median = track.median_midi(note.start, note.end)
            note.pitch = int(round(median)) if median is not None else segments[0][2]
            out.append(note)
            continue
        for i, (start, end, pitch) in enumerate(segments):
            last = i == len(segments) - 1
            out.append(
                Note(
                    start,
                    end,
                    note.text if i == 0 else "~",
                    note.line,
                    word_end=note.word_end and last,
                    pitch=pitch,
                )
            )

    known = [n.pitch for n in out if n.pitch is not None]
    if not known:
        raise RuntimeError("CREPE no ha detectado tono en ninguna nota")
    last = known[0]
    for note in out:
        if note.pitch is None:
            note.pitch = last
        last = note.pitch

    _fix_octaves(out)
    return out


def _fix_octaves(notes: list[Note], window: int = 4, jump: int = 9) -> None:
    """Corrige saltos de octava espurios comparando con las notas vecinas."""
    pitches = np.array([n.pitch for n in notes])
    for i, note in enumerate(notes):
        neighbours = np.concatenate((pitches[max(0, i - window) : i], pitches[i + 1 : i + 1 + window]))
        if not len(neighbours):
            continue
        reference = float(np.median(neighbours))
        while note.pitch - reference >= jump:
            note.pitch -= 12
        while reference - note.pitch >= jump:
            note.pitch += 12
