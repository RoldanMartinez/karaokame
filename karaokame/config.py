"""Lectura y validación de los ficheros .cfg que describen una canción."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".aac"}

# Campos opcionales que, si están en el cfg, tienen prioridad sobre lo que proponga el LLM.
METADATA_KEYS = ("artist", "title", "genre", "year", "creator")


class ConfigError(ValueError):
    pass


@dataclass
class SongConfig:
    path: Path
    name: str
    lead: Path
    accompaniment: Path
    lyrics: Path
    language: str
    metadata: dict[str, str] = field(default_factory=dict)
    range: str = ""  # rango deseado para cantar, p. ej. "do3_mi4"; vacío = no transportar
    cover: Path | None = None
    video: Path | None = None

    def read_lyrics(self) -> list[str]:
        """Devuelve las líneas no vacías de la letra."""
        try:
            text = self.lyrics.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ConfigError(f"{self.lyrics}: la letra debe ser texto UTF-8 ({exc})") from exc
        lines = [" ".join(line.split()) for line in text.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            raise ConfigError(f"{self.lyrics}: la letra está vacía")
        return lines


def parse_cfg(path: Path) -> dict[str, str]:
    """Parsea líneas `clave=valor` o `clave="valor"`. Ignora vacías y comentarios (#)."""
    values: dict[str, str] = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"{path}:{n}: se esperaba clave=valor: {raw!r}")
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip().lower()] = value
    return values


def load_config(path: str | Path) -> SongConfig:
    path = Path(path)
    values = parse_cfg(path)

    missing = [k for k in ("name", "lead", "accompaniment", "lyrics", "language") if not values.get(k)]
    if missing:
        raise ConfigError(f"{path}: faltan campos obligatorios: {', '.join(missing)}")

    def resolve(key: str) -> Path:
        p = Path(values[key]).expanduser()
        if not p.is_absolute():
            p = (path.parent / p).resolve()
        if not p.is_file():
            raise ConfigError(f"{path}: '{key}' apunta a un fichero que no existe: {p}")
        return p

    name = values["name"]
    if name.lower().endswith(".txt"):
        name = name[:-4]

    lyrics = resolve("lyrics")
    if lyrics.suffix.lower() in AUDIO_EXTS:
        raise ConfigError(
            f"{path}: 'lyrics' apunta a un fichero de audio ({lyrics.name}); "
            "debe ser un fichero de texto con la letra"
        )

    return SongConfig(
        path=path,
        name=name,
        lead=resolve("lead"),
        accompaniment=resolve("accompaniment"),
        lyrics=lyrics,
        language=values["language"].strip().lower(),
        metadata={k: values[k] for k in METADATA_KEYS if values.get(k)},
        range=values.get("range", "").strip(),
        cover=resolve("cover") if values.get("cover") else None,
        video=resolve("video") if values.get("video") else None,
    )


def update_cfg(path: str | Path, updates: dict[str, str]) -> None:
    """Cambia o añade campos de un cfg sin tocar el resto de líneas."""
    path = Path(path)
    pending = dict(updates)
    lines = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        key = raw.split("=", 1)[0].strip().lower() if "=" in raw and not raw.lstrip().startswith("#") else None
        if key in pending:
            raw = f'{key}="{pending.pop(key)}"'
        lines.append(raw)
    while lines and not lines[-1].strip():
        lines.pop()
    lines += [f'{key}="{value}"' for key, value in pending.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
