"""Análisis de la pista de voz: tono (CREPE), sonoridad y candidatos a frontera de sílaba."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
HOP = 160  # 10 ms


@dataclass
class PitchTrack:
    times: np.ndarray  # segundos, un valor por frame
    midi: np.ndarray  # nota MIDI (float), válida solo donde `voiced`
    periodicity: np.ndarray  # confianza de CREPE, 0..1
    rms_db: np.ndarray
    voiced: np.ndarray  # bool
    onsets: np.ndarray  # segundos; ataques detectados en la voz

    @property
    def hop_s(self) -> float:
        return HOP / SAMPLE_RATE

    @property
    def duration(self) -> float:
        return float(self.times[-1]) if len(self.times) else 0.0

    def frame(self, t: float) -> int:
        return int(np.clip(round(t / self.hop_s), 0, len(self.times) - 1))

    def span(self, t0: float, t1: float) -> slice:
        a = self.frame(t0)
        return slice(a, max(a + 1, self.frame(t1)))

    def voiced_intervals(self, t0: float, t1: float) -> list[tuple[float, float]]:
        """Tramos con voz dentro de [t0, t1]."""
        sl = self.span(t0, t1)
        v = self.voiced[sl].astype(np.int8)
        if not v.any():
            return []
        edges = np.diff(np.concatenate(([0], v, [0])))
        starts = np.flatnonzero(edges == 1) + sl.start
        ends = np.flatnonzero(edges == -1) + sl.start
        return [(max(t0, s * self.hop_s), min(t1, e * self.hop_s)) for s, e in zip(starts, ends)]

    def median_midi(self, t0: float, t1: float) -> float | None:
        sl = self.span(t0, t1)
        values = self.midi[sl][self.voiced[sl]]
        return float(np.median(values)) if len(values) else None


def _clean_mask(mask: np.ndarray, fill: int, drop: int) -> np.ndarray:
    """Rellena huecos de menos de `fill` frames y elimina islas de menos de `drop`."""

    def remove_short_runs(m: np.ndarray, max_len: int) -> np.ndarray:
        m = m.copy()
        edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
        for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if e - s < max_len:
                m[s:e] = False
        return m

    filled = ~remove_short_runs(~mask, fill)
    # No se rellena el silencio inicial/final aunque sea corto.
    first, last = np.argmax(mask), len(mask) - np.argmax(mask[::-1])
    if mask.any():
        filled[:first] = False
        filled[last:] = False
    return remove_short_runs(filled, drop)


def analyze(
    audio: np.ndarray,
    model: str = "full",
    device: str = "cpu",
    periodicity_threshold: float = 0.4,
    silence_db: float = 45.0,
) -> PitchTrack:
    """Extrae el tono con CREPE (decodificación Viterbi) sobre audio mono a 16 kHz."""
    import librosa
    import torch
    import torchcrepe

    wav = torch.from_numpy(audio).float().unsqueeze(0)
    freq, periodicity = torchcrepe.predict(
        wav,
        SAMPLE_RATE,
        hop_length=HOP,
        fmin=65.0,
        fmax=1050.0,
        model=model,
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
        batch_size=1024,
        device=device,
    )
    periodicity = torchcrepe.filter.median(periodicity, 3)
    freq = freq.squeeze(0).cpu().numpy()
    periodicity = periodicity.squeeze(0).cpu().numpy()

    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=HOP)[0]
    n = min(len(freq), len(rms))
    freq, periodicity, rms = freq[:n], periodicity[:n], rms[:n]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-7))

    voiced = (periodicity > periodicity_threshold) & (rms_db > rms_db.max() - silence_db)
    voiced = _clean_mask(voiced, fill=6, drop=5)

    onsets = librosa.onset.onset_detect(
        y=audio, sr=SAMPLE_RATE, hop_length=HOP, backtrack=True, units="time"
    )

    return PitchTrack(
        times=np.arange(n) * HOP / SAMPLE_RATE,
        midi=librosa.hz_to_midi(np.maximum(freq, 1e-3)),
        periodicity=periodicity,
        rms_db=rms_db,
        voiced=voiced,
        onsets=np.asarray(onsets, dtype=float),
    )


def save(track: PitchTrack, path) -> None:
    np.savez_compressed(
        path,
        times=track.times,
        midi=track.midi,
        periodicity=track.periodicity,
        rms_db=track.rms_db,
        voiced=track.voiced,
        onsets=track.onsets,
    )


def load(path) -> PitchTrack:
    with np.load(path) as data:
        return PitchTrack(**{k: data[k] for k in data.files})
