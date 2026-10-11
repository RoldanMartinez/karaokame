"""Alineación de la letra conocida con las palabras oídas por Whisper.

La letra del cfg es la referencia: Whisper solo aporta los tiempos. Se alinean
ambas secuencias de palabras (Needleman-Wunsch con similitud difusa), las
palabras emparejadas heredan el tiempo de Whisper y las que quedan sin pareja
se reparten sobre los tramos con voz que haya entre sus vecinas.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np

from .pitch import PitchTrack
from .syllables import syllabify
from .transcribe import HeardWord

MATCH_THRESHOLD = 0.5
GAP_PENALTY = -0.4
NO_ROOM_PENALTY = -0.3
ROOM_SECONDS = 0.4
SECONDS_PER_SYLLABLE = 0.4  # estimación para palabras sin vecinas ancladas


@dataclass
class Word:
    text: str
    line: int
    syllables: list[str]
    start: float | None = None
    end: float | None = None
    conf: float = 0.0  # similitud con la palabra oída, rebajada si Whisper dudaba; 0 si es interpolada
    heard: str = ""
    syllable_spans: list[tuple[float, float]] = field(default_factory=list)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in text if c.isalnum() and not unicodedata.combining(c))


@lru_cache(maxsize=None)
def similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def build_words(lines: list[str], language: str) -> list[Word]:
    words = []
    for n, line in enumerate(lines):
        for token in line.split():
            words.append(Word(text=token, line=n, syllables=syllabify(token, language)))
    return words


def match_sequences(
    ref: list[str], hyp: list[str], room: list[bool] | None = None
) -> list[tuple[int, int, float]]:
    """Alineación global; devuelve los pares (i_ref, j_hyp, similitud) emparejados.

    `room[j]` indica si antes de la palabra oída j (o tras la última, j = m) hay
    tiempo libre. Dejar sin pareja una palabra de la letra donde no lo hay
    penaliza más: así, ante frases repetidas, los huecos caen donde caben.
    """
    n, m = len(ref), len(hyp)
    skip_ref = np.full(m + 1, GAP_PENALTY)
    if room is not None:
        skip_ref[~np.asarray(room)] += NO_ROOM_PENALTY
    score = np.zeros((n + 1, m + 1))
    score[:, 0] = np.arange(n + 1) * skip_ref[0]
    score[0, :] = np.arange(m + 1) * GAP_PENALTY
    move = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 diagonal, 1 arriba, 2 izquierda
    move[1:, 0] = 1
    move[0, 1:] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = similarity(ref[i - 1], hyp[j - 1])
            options = (
                score[i - 1, j - 1] + 2 * sim - 1,
                score[i - 1, j] + skip_ref[j],
                score[i, j - 1] + GAP_PENALTY,
            )
            best = int(np.argmax(options))
            score[i, j] = options[best]
            move[i, j] = best

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        step = move[i, j]
        if step == 0:
            sim = similarity(ref[i - 1], hyp[j - 1])
            if sim >= MATCH_THRESHOLD:
                pairs.append((i - 1, j - 1, sim))
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def _distribute(words: list[Word], t0: float, t1: float, track: PitchTrack) -> None:
    """Reparte palabras sin ancla en [t0, t1], preferentemente sobre los tramos con voz."""
    weights = np.array([len(w.syllables) for w in words], dtype=float)
    cuts = np.concatenate(([0.0], np.cumsum(weights))) / weights.sum()
    intervals = track.voiced_intervals(t0, t1)
    voiced_total = sum(e - s for s, e in intervals)

    if voiced_total < 0.12 * weights.sum():
        intervals, voiced_total = [(t0, max(t1, t0))], max(t1 - t0, 0.0)

    def at(position: float, side: str) -> float:
        """Tiempo real en el que se lleva acumulado `position` segundos de voz."""
        acc = 0.0
        for s, e in intervals:
            length = e - s
            if position < acc + length or (side == "end" and position <= acc + length):
                return s + (position - acc)
            acc += length
        return intervals[-1][1]

    for word, a, b in zip(words, cuts, cuts[1:]):
        word.start = at(a * voiced_total, "start")
        word.end = max(word.start, at(b * voiced_total, "end"))


def assign_times(words: list[Word], heard: list[HeardWord], track: PitchTrack) -> list[HeardWord]:
    """Asigna tiempos a las palabras de la letra; devuelve las palabras oídas que no están en ella."""
    if not heard:
        raise RuntimeError("Whisper no ha detectado ninguna palabra en la pista de voz")

    edges = [0.0] + [t for h in heard for t in (h.start, h.end)] + [track.duration]
    room = [b - a >= ROOM_SECONDS for a, b in zip(edges[::2], edges[1::2])]
    pairs = match_sequences([normalize(w.text) for w in words], [normalize(h.text) for h in heard], room)
    matched = {j for _, j, _ in pairs}
    unmatched = [h for j, h in enumerate(heard) if j not in matched]
    if not pairs:
        raise RuntimeError("Lo transcrito por Whisper no se parece en nada a la letra indicada")

    for i, j, sim in pairs:
        h = heard[j]
        words[i].start, words[i].end = h.start, max(h.end, h.start)
        words[i].conf, words[i].heard = sim * min(1.0, h.prob / 0.5), h.text

    # Tramos de palabras sin ancla entre dos anclas (o los extremos de la canción).
    i = 0
    while i < len(words):
        if words[i].start is not None:
            i += 1
            continue
        j = i
        while j < len(words) and words[j].start is None:
            j += 1
        run = words[i:j]
        span = SECONDS_PER_SYLLABLE * sum(len(w.syllables) for w in run)
        t1 = words[j].start if j < len(words) else None
        t0 = words[i - 1].end if i > 0 else None
        if t0 is None:
            t0 = max(0.0, t1 - span)
        if t1 is None:
            t1 = min(track.duration, t0 + span)
        _distribute(run, t0, max(t0, t1), track)
        i = j
    return unmatched


def refine_boundaries(words: list[Word], track: PitchTrack) -> None:
    """Ajusta inicios y finales de palabra a lo que realmente suena en la pista de voz."""
    hop = track.hop_s
    voiced = track.voiced

    for k, word in enumerate(words):
        prev_start = words[k - 1].start if k else 0.0
        # Inicio: si hay una transición silencio->voz cerca, se pega a ella.
        lo, hi = track.frame(max(prev_start + hop, word.start - 0.15)), track.frame(word.start + 0.25)
        rises = [f for f in range(max(lo, 1), hi + 1) if voiced[f] and not voiced[f - 1]]
        if rises:
            target = min(rises, key=lambda f: abs(f * hop - word.start)) * hop
            if target < word.end - 0.03 or word.conf == 0:
                word.start = target
                word.end = max(word.end, word.start + 0.03)
        if k and words[k - 1].end > word.start:
            words[k - 1].end = word.start

    for k, word in enumerate(words):
        limit = words[k + 1].start if k + 1 < len(words) else track.duration
        limit = min(limit, word.end + 4.0)
        sl = track.span(word.start, word.end)
        idx = np.flatnonzero(voiced[sl])
        if len(idx):
            # Recorta la cola en silencio...
            trimmed = (sl.start + idx[-1] + 1) * hop
            if trimmed - word.start >= 0.06:
                word.end = min(word.end, trimmed)
        # ...y alarga las notas sostenidas mientras siga sonando la voz.
        f = track.frame(word.end)
        if voiced[max(f - 1, 0)]:
            while f < len(voiced) and voiced[f] and f * hop < limit:
                f += 1
            word.end = max(word.end, min(f * hop, limit))
        word.end = max(word.end, word.start + 0.02)


def _pitch_steps(track: PitchTrack, t0: float, t1: float) -> list[float]:
    """Instantes en los que el tono (redondeado al semitono) cambia de forma estable."""
    sl = track.span(t0, t1)
    midi = np.where(track.voiced[sl], np.round(track.midi[sl]), np.nan)
    steps = []
    for f in range(3, len(midi) - 3):
        before, after = midi[f - 3 : f], midi[f : f + 3]
        if np.isnan(before).any() or np.isnan(after).any():
            continue
        if (before == before[0]).all() and (after == after[0]).all() and before[0] != after[0]:
            steps.append((sl.start + f) * track.hop_s)
    return steps


def split_syllables(words: list[Word], track: PitchTrack, min_len: float = 0.07) -> None:
    """Reparte el tiempo de cada palabra entre sus sílabas.

    Parte de un reparto uniforme y mueve cada frontera al ataque o cambio de
    tono más cercano, si lo hay dentro de un margen razonable.
    """
    for word in words:
        k = len(word.syllables)
        s, e = word.start, word.end
        if k == 1 or e - s < k * min_len:
            edges = np.linspace(s, e, k + 1)
            word.syllable_spans = list(zip(edges[:-1], edges[1:]))
            continue
        candidates = [t for t in track.onsets if s < t < e] + _pitch_steps(track, s, e)
        step = (e - s) / k
        bounds, prev = [s], s
        for i in range(1, k):
            expected = s + i * step
            lo = max(prev + min_len, expected - 0.45 * step)
            hi = min(e - (k - i) * min_len, expected + 0.45 * step)
            near = [t for t in candidates if lo <= t <= hi]
            if near:
                prev = min(near, key=lambda t: abs(t - expected))
            else:
                prev = min(max(expected, lo), hi) if lo <= hi else expected
            bounds.append(prev)
        bounds.append(e)
        word.syllable_spans = list(zip(bounds[:-1], bounds[1:]))


def heard_between(heard: list[HeardWord], t0: float, t1: float) -> str:
    return " ".join(h.text for h in heard if t0 <= (h.start + h.end) / 2 <= t1)
