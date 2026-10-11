#!/usr/bin/env python3
"""Genera canciones de UltraStar WorldParty a partir de los cfg de config/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from karaokame import llm  # noqa: E402
from karaokame import vocalize  # noqa: E402
from karaokame import discover  # noqa: E402
from karaokame.config import ConfigError  # noqa: E402
from karaokame.pipeline import Options, log, prepare, run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="*", type=Path, help="ficheros .cfg (por defecto, se crea el de la canción que haya en data/)")
    parser.add_argument("--whisper-model", default="medium", help="modelo de faster-whisper (small, medium, large-v3...)")
    parser.add_argument("--crepe-model", default="full", choices=["full", "tiny"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-llm", action="store_true", help="no usar Ollama (ni metadatos ni revisión)")
    parser.add_argument("--llm-model", default=llm.DEFAULT_MODEL)
    parser.add_argument("--ollama-url", default=llm.DEFAULT_URL)
    parser.add_argument("--no-sinalefa", action="store_true", help="no fundir vocales de palabras contiguas")
    parser.add_argument("--no-melisma", action="store_true", help="no partir sílabas con cambios de tono (~)")
    parser.add_argument("--creator", default="karaokame", help="valor de #CREATOR si el cfg no lo indica")
    parser.add_argument("--uvr-dir", type=Path, default=vocalize.DEFAULT_UVR, help="carpeta de ultimatevocalremovergui (para separar la voz)")
    parser.add_argument("--force", action="store_true", help="ignorar la caché de Whisper y CREPE y repetir la separación de voz")
    args = parser.parse_args()

    configs = args.configs
    if not configs:
        # Sin argumentos: la canción es el fichero de audio o vídeo que haya en data/.
        try:
            song = discover.pick_song(ROOT / "data", ROOT / "songs")
            configs = [discover.ensure_cfg(song, ROOT / "config", log=log)]
        except ConfigError as exc:
            print(f"[karaokame] ERROR: {exc}", file=sys.stderr)
            return 1

    opts = Options(
        whisper_model=args.whisper_model,
        crepe_model=args.crepe_model,
        device=args.device,
        use_llm=not args.no_llm,
        llm_model=args.llm_model,
        ollama_url=args.ollama_url,
        sinalefa=not args.no_sinalefa,
        melisma=not args.no_melisma,
        force=args.force,
        creator=args.creator,
        uvr_dir=args.uvr_dir,
    )

    failed = 0
    for path in configs:
        try:
            run(prepare(path, ROOT, opts), ROOT, opts)
        except (ConfigError, RuntimeError) as exc:
            print(f"[karaokame] ERROR en {path}: {exc}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
