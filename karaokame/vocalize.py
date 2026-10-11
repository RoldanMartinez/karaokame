"""Separación de voz y acompañamiento a partir de la canción completa (campo audio del cfg)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import ConfigError, parse_cfg, update_cfg

ROOT = Path(__file__).resolve().parent.parent
WORKER = Path(__file__).resolve().parent / "uvr_mdxc.py"
DEFAULT_UVR = ROOT / "ultimatevocalremovergui"
DEFAULT_MODEL = "MDX23C-8KFFT-InstVoc_HQ.ckpt"
DIRECT_FORMATS = {".mp3", ".wav", ".flac", ".ogg"}  # los lee UVR sin ayuda; el resto pasa antes por ffmpeg
OOM_EXIT = 3  # código con el que uvr_mdxc.py avisa de que la GPU se ha quedado sin memoria
MP4_CODECS = {"h264", "hevc", "mpeg4", "av1"}  # se copian tal cual; el resto se recodifica


@dataclass
class Options:
    uvr_dir: Path = DEFAULT_UVR
    model: Path = Path(DEFAULT_MODEL)
    segment_size: int = 256
    overlap: int = 8
    device: str = "auto"
    bitrate: str = "320k"
    python: str | None = None
    force: bool = False


def log(message: str) -> None:
    print(f"[vocalizame] {message}", flush=True)


def ffmpeg(*args: str) -> None:
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg ha fallado: {result.stderr.strip()}")


def video_streams(path: Path) -> list[dict]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
         "stream=index,codec_name:stream_disposition=attached_pic", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe no puede leer {path}: {result.stderr.strip()}")
    return json.loads(result.stdout).get("streams", [])


def extract_picture(audio: Path, force: bool) -> dict[str, str]:
    """Extrae el vídeo a mp4, o la carátula incrustada a jpg. Devuelve los campos del cfg a añadir."""
    streams = video_streams(audio)
    videos = [s for s in streams if not s.get("disposition", {}).get("attached_pic")]
    covers = [s for s in streams if s.get("disposition", {}).get("attached_pic")]

    if videos:
        target = audio.with_suffix(".mp4")
        if target.resolve() == audio.resolve():
            target = audio.with_name(f"{audio.stem}_video.mp4")
        if force or not target.exists():
            log(f"El fichero contiene vídeo ({videos[0]['codec_name']}): extrayendo a {target.name}...")
            codec = ["-c:v", "copy"] if videos[0]["codec_name"] in MP4_CODECS else ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p"]
            ffmpeg("-i", str(audio), "-map", f"0:{videos[0]['index']}", "-an", *codec, "-movflags", "+faststart", str(target))
        return {"video": str(target)}

    if covers:
        target = audio.with_suffix(".jpg")
        if force or not target.exists():
            log(f"El fichero no contiene vídeo, solo una carátula incrustada: extrayendo a {target.name}...")
            ffmpeg("-i", str(audio), "-map", f"0:{covers[0]['index']}", "-frames:v", "1", "-q:v", "2", str(target))
        return {"cover": str(target)}

    log("El fichero no contiene vídeo ni carátula.")
    return {}


def separate(audio: Path, vocals: Path, instrumental: Path, args: Options) -> None:
    uvr = args.uvr_dir.resolve()
    model = args.model if args.model.is_absolute() else uvr / "models" / "MDX_Net_Models" / args.model
    if not (uvr / "lib_v5" / "tfc_tdf_v3.py").is_file():
        raise RuntimeError(f"No encuentro Ultimate Vocal Remover en {uvr} (opción --uvr-dir)")
    if not model.is_file():
        raise RuntimeError(f"No encuentro el modelo {model}")

    # Por defecto, el intérprete del entorno de UVR: es el que tiene PyTorch con GPU.
    python = args.python or (uvr / ".venv" / "bin" / "python")
    if not Path(python).exists():
        python = sys.executable

    work = ROOT / "work"
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vocalizame_", dir=work) as tmp:
        wav_vocals, wav_instrumental = Path(tmp) / "vocals.wav", Path(tmp) / "instrumental.wav"
        if audio.suffix.lower() not in DIRECT_FORMATS:
            log(f"Extrayendo el audio de {audio.name}...")
            decoded = Path(tmp) / "mix.wav"
            ffmpeg("-i", str(audio), "-map", "0:a:0", "-vn", "-ac", "2", "-ar", "44100", str(decoded))
            audio = decoded
        def run_worker(device: str) -> int:
            command = [
                str(python), "-W", "ignore", str(WORKER), "--repo", str(uvr), "--model", str(model), "--input", str(audio),
                "--vocals", str(wav_vocals), "--instrumental", str(wav_instrumental),
                "--segment-size", str(args.segment_size), "--overlap", str(args.overlap), "--device", device,
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            shown = -1
            for line in process.stdout:
                line = line.strip()
                if line.startswith("DEVICE"):
                    log(f"Dispositivo: {line.split()[1]}")
                elif line.startswith("PROGRESS"):
                    done, total = map(int, line.split()[1].split("/"))
                    percent = 100 * done // total
                    if percent // 10 > shown // 10:
                        log(f"  {percent} %")
                    shown = percent
            errors = process.stderr.read()
            code = process.wait()
            if code not in (0, OOM_EXIT):
                sys.stderr.write(errors)
            return code

        log(f"Separando con {model.stem} (segmento {args.segment_size}, solape {args.overlap})...")
        code = run_worker(args.device)
        if code == OOM_EXIT and args.device == "auto":
            log("La GPU no tiene memoria libre suficiente (¿otro programa la está usando?): se repite en CPU, que es mucho más lento.")
            code = run_worker("cpu")
        if code != 0:
            raise RuntimeError("La separación ha fallado" + (": la GPU no tiene memoria suficiente" if code == OOM_EXIT else " (ver el error anterior)"))

        log(f"Codificando los mp3 a {args.bitrate}...")
        for wav, mp3 in ((wav_vocals, vocals), (wav_instrumental, instrumental)):
            ffmpeg("-i", str(wav), "-codec:a", "libmp3lame", "-b:a", args.bitrate, str(mp3))


def run(config: Path, opts: Options) -> dict[str, str]:
    """Separa la canción del cfg, extrae vídeo o carátula y actualiza el cfg. Devuelve los campos escritos."""
    values = parse_cfg(config)
    if not values.get("audio"):
        raise ConfigError(f"{config}: falta el campo 'audio' con la canción completa")
    audio = Path(values["audio"]).expanduser()
    if not audio.is_absolute():
        audio = (config.parent / audio).resolve()
    if not audio.is_file():
        raise ConfigError(f"{config}: 'audio' apunta a un fichero que no existe: {audio}")

    vocals = audio.with_name(f"{audio.stem}_V.mp3")
    instrumental = audio.with_name(f"{audio.stem}_A.mp3")
    if not opts.force and vocals.exists() and instrumental.exists():
        log(f"Ya existen {vocals.name} y {instrumental.name}; no se repite la separación (--force para rehacerla).")
    else:
        separate(audio, vocals, instrumental, opts)

    updates = {"lead": str(vocals), "accompaniment": str(instrumental), **extract_picture(audio, opts.force)}
    update_cfg(config, updates)
    for key, value in updates.items():
        log(f"{config.name}: {key}=\"{value}\"")
    return updates
