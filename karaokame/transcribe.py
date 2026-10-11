"""Transcripción de la voz con Whisper (faster-whisper) con tiempos por palabra."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .pitch import SAMPLE_RATE, PitchTrack

LONG_WORD = 2.0  # s; por encima, Whisper probablemente ha estirado la palabra sobre algo que no oyó
MIN_GAP = 2.5  # s sin palabras...
MIN_GAP_VOICED = 1.0  # ...pero con al menos esta voz dentro: se vuelve a transcribir
PAD = 0.2
MIN_WORD = 0.03  # s; las palabras de duración nula son bucles de alucinación
MIN_PROB_RETRY = 0.2


@dataclass
class HeardWord:
    text: str
    start: float
    end: float
    prob: float


def _run(model, audio: np.ndarray, language: str, prompt: str | None, offset: float = 0.0) -> list[HeardWord]:
    segments, _ = model.transcribe(
        audio,  # mono a 16 kHz; evita depender del decodificador PyAV de faster-whisper
        language=language,
        word_timestamps=True,
        initial_prompt=prompt,
        # Sin contexto previo: evita que Whisper entre en bucle en los estribillos.
        condition_on_previous_text=False,
        vad_filter=False,
        beam_size=5,
    )
    words: list[HeardWord] = []
    for segment in segments:
        for w in segment.words or []:
            text = w.word.strip()
            if text and w.end - w.start >= MIN_WORD:
                words.append(HeardWord(text, offset + float(w.start), offset + float(w.end), float(w.probability)))
    return words


def _trim_long_words(words: list[HeardWord], track: PitchTrack) -> None:
    """Acorta las palabras estiradas hasta el final del tramo de voz en el que empiezan."""
    for w in words:
        if w.end - w.start <= LONG_WORD:
            continue
        intervals = [(s, e) for s, e in track.voiced_intervals(w.start, w.end) if e > w.start + 0.05]
        w.end = max(w.start + 0.2, intervals[0][1]) if intervals else w.start + 0.5


def _unheard_gaps(words: list[HeardWord], track: PitchTrack) -> list[tuple[float, float]]:
    """Huecos largos entre palabras en los que, sin embargo, hay voz."""
    edges = [0.0] + [t for w in words for t in (w.start, w.end)] + [track.duration]
    gaps = []
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= MIN_GAP and sum(e - s for s, e in track.voiced_intervals(a, b)) >= MIN_GAP_VOICED:
            gaps.append((a, b))
    return gaps


def transcribe(
    audio: np.ndarray,
    language: str,
    model_name: str = "medium",
    device: str = "cpu",
    lyrics: list[str] | None = None,
    track: PitchTrack | None = None,
) -> list[HeardWord]:
    """Transcribe la voz. Con `track`, repasa los tramos con voz en los que no se oyó nada."""
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    lyrics = lyrics or []
    words = _run(model, audio, language, " ".join(lyrics[:6]) or None)
    if track is None:
        return words

    _trim_long_words(words, track)
    duration = len(audio) / SAMPLE_RATE
    for a, b in _unheard_gaps(words, track):
        lo, hi = max(0.0, a - PAD), min(duration, b + PAD)
        # Como pista, las líneas de la letra que caen aproximadamente en esa zona.
        centre = int(len(lyrics) * a / duration)
        prompt = " ".join(lyrics[max(0, centre - 3) : centre + 5]) or None
        clip = audio[int(lo * SAMPLE_RATE) : int(hi * SAMPLE_RATE)]
        extra = _run(model, clip, language, prompt, offset=lo)
        _trim_long_words(extra, track)
        words += [
            w
            for w in extra
            if w.prob >= MIN_PROB_RETRY and a <= (w.start + w.end) / 2 <= b and track.voiced[track.span(w.start, w.end)].any()
        ]
    words.sort(key=lambda w: w.start)
    return words


def detect_language(audio: np.ndarray, model_name: str = "medium", device: str = "cpu") -> tuple[str, float]:
    """Idioma del primer minuto cantado según Whisper: (código, probabilidad)."""
    import librosa
    from faster_whisper import WhisperModel

    _, (start, _) = librosa.effects.trim(audio, top_db=30)  # salta el silencio inicial
    clip = audio[start : start + 60 * SAMPLE_RATE]
    model = WhisperModel(model_name, device=device, compute_type="float16" if device == "cuda" else "int8")
    _, info = model.transcribe(clip, language=None, language_detection_segments=2)
    return info.language, float(info.language_probability)


def to_lines(words: list[HeardWord], max_chars: int = 40, pause: float = 0.7) -> list[str]:
    """Agrupa las palabras oídas en líneas de karaoke: corta en las pausas, tras puntuación o por longitud."""
    lines: list[list[HeardWord]] = []
    for word in words:
        current = lines[-1] if lines else None
        if current:
            text = " ".join(w.text for w in current)
            last = current[-1].text[-1]
            ended = last in ".!?" or (last in ",;:" and len(text) >= 15)  # fin de frase, o coma con la línea ya larga
            if word.start - current[-1].end > pause or ended or len(text) + 1 + len(word.text) > max_chars:
                current = None
        if current is None:
            lines.append([])
        lines[-1].append(word)
    out = []
    for line in lines:
        text = " ".join(w.text for w in line).rstrip(".,;:")
        if text:
            out.append(text[0].upper() + text[1:])
    return out


def to_json(words: list[HeardWord]) -> list[dict]:
    return [asdict(w) for w in words]


def from_json(data: list[dict]) -> list[HeardWord]:
    return [HeardWord(**d) for d in data]
