"""Localiza la canción que hay en data/ y crea su cfg."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import ConfigError, parse_cfg

# Sufijos de ficheros que generan los propios scripts a partir de la canción.
DERIVED_SUFFIXES = ("_V", "_A", "_I", "_video")
NOT_MEDIA = {".txt", ".cfg", ".jpg", ".jpeg", ".png", ".webp", ".gif", ".srt", ".lrc", ".json", ".md", ".pdf"}
DEFAULT_FIELDS = ["name", "audio", "lead", "accompaniment", "lyrics", "language", "intrument", "score", "range"]
TEMPLATE = "template.cfg"


def has_audio(path: Path) -> bool:
    """True si ffprobe encuentra una pista de audio, sea cual sea el contenedor (mp3, mp4, mkv, webm, flac...)."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def find_songs(data_dir: Path) -> list[Path]:
    """Ficheros de data/ con audio que no son pistas derivadas de otra canción."""
    songs = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.suffix.lower() in NOT_MEDIA:
            continue
        if path.stem.endswith(DERIVED_SUFFIXES) or not has_audio(path):
            continue
        songs.append(path)
    return songs


def find_lyrics(song: Path) -> Path | None:
    """El txt con el nombre de la canción o, si no lo hay, el único txt de la carpeta."""
    same_name = song.with_suffix(".txt")
    if same_name.is_file():
        return same_name
    others = sorted(p for p in song.parent.glob("*.txt") if p.is_file())
    return others[0] if len(others) == 1 else None


def pick_song(data_dir: Path, songs_dir: Path) -> Path:
    """La canción a procesar. Si hay varias, la única que aún no tiene txt generado en songs/."""
    if not data_dir.is_dir():
        raise ConfigError(f"No existe la carpeta {data_dir}")
    songs = find_songs(data_dir)
    if not songs:
        raise ConfigError(f"No hay ningún fichero de audio o vídeo en {data_dir}")
    if len(songs) > 1:
        pending = [s for s in songs if not (songs_dir / f"{s.stem}.txt").is_file()]
        if len(pending) != 1:
            names = ", ".join(s.name for s in (pending or songs))
            raise ConfigError(
                f"En {data_dir} solo debe haber una canción y hay varias "
                f"{'sin generar' if pending else 'ya generadas'} ({names}). "
                "Deja solo una o indica su cfg como argumento."
            )
        songs = pending
    return songs[0]


def ensure_cfg(song: Path, config_dir: Path, log=print) -> Path:
    """Crea config/<canción>.cfg si no existe, partiendo de template.cfg si lo hay."""
    path = config_dir / f"{song.stem}.cfg"
    if path.is_file():
        log(f"Canción encontrada: {song.name} (se usa el cfg existente {path.name})")
        return path

    template = config_dir / TEMPLATE
    values = dict.fromkeys(DEFAULT_FIELDS, "")
    if template.is_file():
        values = {**parse_cfg(template)}
        for key in ("name", "audio", "lyrics"):
            values.setdefault(key, "")
    # Las pistas y la letra son de cada canción: nunca se heredan de la plantilla.
    values.update(name=song.stem, audio=str(song), lead="", accompaniment="", lyrics="")
    for key in ("cover", "video"):
        values.pop(key, None)
    lyrics = find_lyrics(song)
    if lyrics:
        values["lyrics"] = str(lyrics)

    config_dir.mkdir(exist_ok=True)
    path.write_text("".join(f'{k}="{v}"\n' if v else f"{k}=\n" for k, v in values.items()), encoding="utf-8")
    log(f"Canción encontrada: {song.name}")
    log(f"Creado {path}" + (f" con la letra de {lyrics.name}" if lyrics else " sin letra (se transcribirá de la voz)"))
    return path
