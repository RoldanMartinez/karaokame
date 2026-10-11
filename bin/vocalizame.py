#!/usr/bin/env python3
"""Separa la voz y el acompañamiento de la canción indicada en un cfg y lo actualiza."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from karaokame import vocalize  # noqa: E402
from karaokame.config import ConfigError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="fichero .cfg con el campo audio")
    parser.add_argument("--uvr-dir", type=Path, default=vocalize.DEFAULT_UVR, help="carpeta de ultimatevocalremovergui")
    parser.add_argument("--model", type=Path, default=Path(vocalize.DEFAULT_MODEL), help="modelo MDX23C (.ckpt)")
    parser.add_argument("--segment-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--bitrate", default="320k", help="bitrate de los mp3 generados")
    parser.add_argument("--python", help="intérprete con el que ejecutar la separación (por defecto, el de UVR)")
    parser.add_argument("--force", action="store_true", help="rehacer aunque los ficheros ya existan")
    args = parser.parse_args()

    opts = vocalize.Options(
        uvr_dir=args.uvr_dir, model=args.model, segment_size=args.segment_size, overlap=args.overlap,
        device=args.device, bitrate=args.bitrate, python=args.python, force=args.force,
    )
    try:
        vocalize.run(args.config, opts)
    except (ConfigError, RuntimeError) as exc:
        print(f"[vocalizame] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
