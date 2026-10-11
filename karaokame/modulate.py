"""Análisis tonal de la voz y transposición de una canción ya generada."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ConfigError, SongConfig
from .pitch import PitchTrack
from .ultrastar import MIDI_C4

NOTE_NAMES = ["Do", "Do#", "Re", "Re#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]
# Nombre habitual de cada tonalidad (con bemoles donde es lo corriente).
MAJOR_KEYS = ["Do", "Re♭", "Re", "Mi♭", "Mi", "Fa", "Fa#", "Sol", "La♭", "La", "Si♭", "Si"]
MINOR_KEYS = ["Do", "Do#", "Re", "Mi♭", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "Si♭", "Si"]

# Perfiles de Krumhansl-Kessler: peso perceptivo de cada grado en una tonalidad mayor y menor.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

NOTE_LINE = re.compile(r"^([:*FRG]) (-?\d+) (\d+) (-?\d+) (.*)$")
MAX_SEMITONES = 12
MIN_GAIN = 0.02  # mejora mínima (fracción del tiempo cantado) para que compense transportar
RANGE_NOTE = re.compile(r"^(do|re|mi|fa|sol|la|si)(#|b|♭)?(-?\d)$")
NATURALS = {"do": 0, "re": 2, "mi": 4, "fa": 5, "sol": 7, "la": 9, "si": 11}


@dataclass
class Key:
    tonic: int  # clase de nota, 0 = Do
    minor: bool
    score: float  # correlación con el perfil, -1..1

    def name(self, shift: int = 0) -> str:
        names, mode = (MINOR_KEYS, "menor") if self.minor else (MAJOR_KEYS, "mayor")
        return f"{names[(self.tonic + shift) % 12]} {mode}"


@dataclass
class Analysis:
    seconds_by_midi: dict[int, float]  # tiempo cantado en cada nota
    pitch_classes: np.ndarray  # fracción del tiempo cantado en cada clase de nota (12)
    keys: list[Key]  # tonalidades candidatas, de más a menos probable
    lowest: int
    highest: int
    usual_low: int  # percentiles 5 y 95 del tiempo cantado
    usual_high: int

    @property
    def key(self) -> Key:
        return self.keys[0]


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def estimate_keys(pitch_classes: np.ndarray) -> list[Key]:
    """Algoritmo de Krumhansl-Schmuckler: correlación del histograma con los 24 perfiles."""
    keys = []
    for tonic in range(12):
        rotated = np.roll(pitch_classes, -tonic)
        for minor, profile in ((False, MAJOR_PROFILE), (True, MINOR_PROFILE)):
            keys.append(Key(tonic, minor, float(np.corrcoef(rotated, profile)[0, 1])))
    return sorted(keys, key=lambda k: -k.score)


def analyze(track: PitchTrack) -> Analysis:
    midi = np.round(track.midi[track.voiced]).astype(int)
    if not len(midi):
        raise RuntimeError("No se ha detectado voz en la pista principal")
    values, counts = np.unique(midi, return_counts=True)
    seconds = {int(v): float(c) * track.hop_s for v, c in zip(values, counts)}
    pitch_classes = np.zeros(12)
    for v, s in seconds.items():
        pitch_classes[v % 12] += s
    pitch_classes /= pitch_classes.sum()
    low, high = np.percentile(midi, [5, 95])
    return Analysis(
        seconds_by_midi=seconds,
        pitch_classes=pitch_classes,
        keys=estimate_keys(pitch_classes),
        lowest=int(values[0]),
        highest=int(values[-1]),
        usual_low=int(round(low)),
        usual_high=int(round(high)),
    )


def parse_note(text: str) -> int:
    """'do4' -> 60 (do central), 'fa#3' -> 54, 'sib2' -> 46."""
    match = RANGE_NOTE.match(text.strip().lower())
    if not match:
        raise ConfigError(f"nota no válida: {text!r} (ejemplos: do3, fa#3, sib2)")
    name, accidental, octave = match.groups()
    return NATURALS[name] + {"#": 1, "b": -1, "♭": -1, None: 0}[accidental] + 12 * (int(octave) + 1)


def parse_range(text: str) -> tuple[int, int]:
    """'do3_mi4' -> (48, 64): notas MIDI más grave y más aguda, ambas incluidas."""
    parts = re.split(r"[_\s-]+", text.strip())
    if len(parts) != 2:
        raise ConfigError(f"rango no válido: {text!r} (se espera algo como do3_mi4)")
    low, high = parse_note(parts[0]), parse_note(parts[1])
    if low >= high:
        raise ConfigError(f"rango no válido: {text!r} (la primera nota debe ser la más grave)")
    return low, high


@dataclass
class Fit:
    semitones: int  # transposición del audio, entre -6 y +6
    octaves: int  # octavas por encima (+) o por debajo (-) a las que se canta
    inside: float  # fracción del tiempo cantado que queda dentro del rango
    inside_before: float  # la misma fracción sin transportar ni cambiar de octava

    @property
    def note_shift(self) -> int:
        return self.semitones + 12 * self.octaves


def best_fit(analysis: Analysis, low: int, high: int) -> Fit:
    """Transposición que deja el mayor tiempo cantado posible dentro de [low, high].

    El audio se mueve como mucho seis semitonos: cantar una octava arriba o
    abajo no necesita otro acompañamiento, así que la octava se elige aparte.
    A igualdad de tiempo dentro, gana la que menos se sale y la que menos mueve.
    Si transportar mejora menos de MIN_GAIN respecto a no hacerlo, no se transporta.
    """
    total = sum(analysis.seconds_by_midi.values())

    def measure(shift: int) -> tuple[float, float]:
        inside = excess = 0.0
        for midi, seconds in analysis.seconds_by_midi.items():
            note = midi + shift
            if low <= note <= high:
                inside += seconds
            else:
                excess += seconds * (low - note if note < low else note - high)
        return inside / total, excess / total

    candidates = []
    for semitones in range(-6, 7):
        for octaves in range(-4, 5):
            inside, excess = measure(semitones + 12 * octaves)
            candidates.append((-round(inside, 4), round(excess, 4), abs(semitones), abs(octaves), semitones, octaves, inside))
    best = min(candidates)
    untouched = min(c for c in candidates if c[4] == 0)
    if best[6] - untouched[6] < MIN_GAIN:
        best = untouched
    return Fit(semitones=best[4], octaves=best[5], inside=best[6], inside_before=measure(0)[0])


def _bar(fraction: float, width: int = 40) -> str:
    return "█" * int(round(fraction * width)) or ("▏" if fraction > 0 else "")


def render(analysis: Analysis, title: str, shift: int = 0) -> str:
    """Pantalla de texto con la tonalidad, el rango y la distribución de notas."""
    a = analysis
    out = [f"═══ {title}" + (f"  (transportada {shift:+d} semitonos)" if shift else "") + " ═══", ""]
    best = a.key
    others = ", ".join(f"{k.name(shift)} ({k.score:.2f})" for k in a.keys[1:3])
    out.append(f"Tonalidad estimada: {best.name(shift)} ({best.score:.2f})   alternativas: {others}")
    out.append(
        f"Rango de la voz:    {note_name(a.lowest + shift)} – {note_name(a.highest + shift)}"
        f"   habitual (5-95 %): {note_name(a.usual_low + shift)} – {note_name(a.usual_high + shift)}"
    )
    out += ["", "Distribución por nota (todas las octavas juntas):"]
    top = a.pitch_classes.max()
    for pc in range(12):
        share = a.pitch_classes[(pc - shift) % 12]
        mark = " ← tónica" if pc == (best.tonic + shift) % 12 else ""
        out.append(f"  {NOTE_NAMES[pc]:<4} {_bar(share / top):<40} {100 * share:5.1f} %{mark}")

    out += ["", "Distribución por altura:"]
    total = sum(a.seconds_by_midi.values())
    peak = max(a.seconds_by_midi.values())
    for midi in range(a.highest, a.lowest - 1, -1):
        s = a.seconds_by_midi.get(midi, 0.0)
        if s / total < 0.002 and not a.usual_low <= midi <= a.usual_high:
            continue  # notas sueltas en los extremos: casi siempre ruido del detector
        out.append(f"  {note_name(midi + shift):<5} {_bar(s / peak):<40} {100 * s / total:5.1f} %")
    return "\n".join(out)


def suffix(semitones: int) -> str:
    return f"_{semitones:+d}"


def transpose_txt(text: str, semitones: int, mp3_name: str, note_shift: int | None = None) -> str:
    """Sube o baja el tono de todas las notas y apunta #MP3 al audio transportado.

    `note_shift` permite mover las notas del txt un número distinto de semitonos
    que el audio (la transposición más un cambio de octava).
    """
    note_shift = semitones if note_shift is None else note_shift
    out = []
    for line in text.splitlines():
        match = NOTE_LINE.match(line)
        if match:
            kind, start, length, pitch, lyric = match.groups()
            line = f"{kind} {start} {length} {int(pitch) + note_shift} {lyric}"
        elif line.startswith("#MP3:"):
            line = f"#MP3:{mp3_name}"
        elif line.startswith("#TITLE:"):
            line = f"{line} ({semitones:+d})"  # para distinguirla de la original en el juego
        out.append(line)
    return "\n".join(out) + "\n"


def _bitrate(path: Path) -> str:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=bit_rate", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    digits = result.stdout.strip()
    return f"{min(320, max(96, round(int(digits) / 1000)))}k" if digits.isdigit() else "256k"


def transpose_audio(source: Path, target: Path, semitones: int, voice: bool = False) -> None:
    """Cambia el tono sin alterar la duración (Rubber Band a través de ffmpeg)."""
    ratio = 2 ** (semitones / 12)
    # En la voz se conservan los formantes para que no suene a "ardilla" ni a "gigante".
    formant = "preserved" if voice else "shifted"
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
            "-af", f"rubberband=pitch={ratio:.8f}:pitchq=quality:formant={formant}",
            "-codec:a", "libmp3lame", "-b:a", _bitrate(source), str(target),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg no ha podido transportar {source.name}: {result.stderr.strip()}")


def generate(cfg: SongConfig, root: Path, semitones: int, note_shift: int | None = None, log=print) -> Path:
    """Escribe en songs/ la canción transportada (<name>_X.txt y .mp3) y la voz de referencia en work/."""
    songs_dir = root / "songs"
    work_dir = root / "work" / cfg.name
    source_txt = songs_dir / f"{cfg.name}.txt"
    if not source_txt.is_file():
        raise RuntimeError(f"No existe {source_txt}: genera antes la canción con bin/karaokame.py")

    base = cfg.name + suffix(semitones)
    mp3 = songs_dir / f"{base}.mp3"
    txt = songs_dir / f"{base}.txt"
    lead = work_dir / f"{cfg.lead.stem}{suffix(semitones)}.mp3"

    log(f"Transportando el acompañamiento {semitones:+d} semitonos...")
    transpose_audio(cfg.accompaniment, mp3, semitones)
    log(f"Transportando la voz {semitones:+d} semitonos...")
    transpose_audio(cfg.lead, lead, semitones, voice=True)
    txt.write_text(transpose_txt(source_txt.read_text(encoding="utf-8-sig"), semitones, mp3.name, note_shift), encoding="utf-8")
    for path in (txt, mp3, lead):
        log(f"Generado: {path}")
    return txt
