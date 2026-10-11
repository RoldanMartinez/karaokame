#!/usr/bin/env python3
"""Analiza la tonalidad de la voz de una canción y la transporta X semitonos."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from karaokame import modulate  # noqa: E402
from karaokame.config import ConfigError, load_config  # noqa: E402
from karaokame.pipeline import lead_pitch, log, resolve_device  # noqa: E402


def ask_semitones(analysis: modulate.Analysis, title: str) -> int:
    """Diálogo: el usuario prueba transposiciones y confirma una. Devuelve 0 si no quiere ninguna."""
    limit = modulate.MAX_SEMITONES
    while True:
        answer = input(f"\nSemitonos a transportar (+ sube, - baja, entre -{limit} y +{limit}; Intro para salir): ").strip()
        if not answer or answer == "0":
            return 0
        try:
            semitones = int(answer)
        except ValueError:
            print("Escribe un número entero, por ejemplo -2 o +3.")
            continue
        if abs(semitones) > limit:
            print(f"El máximo es {limit} semitonos (una octava).")
            continue
        print()
        print(modulate.render(analysis, title, semitones))
        if input(f"\n¿Generar los ficheros transportados {semitones:+d}? [s/N] ").strip().lower() in ("s", "si", "sí", "y"):
            return semitones


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="fichero .cfg de la canción")
    parser.add_argument("-s", "--semitones", type=int, help="transportar directamente, sin preguntar (p. ej. -2)")
    parser.add_argument("--crepe-model", default="full", choices=["full", "tiny"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()
    if args.semitones is not None and abs(args.semitones) > modulate.MAX_SEMITONES:
        parser.error(f"--semitones debe estar entre -{modulate.MAX_SEMITONES} y +{modulate.MAX_SEMITONES}")

    try:
        cfg = load_config(args.config)
        work_dir = ROOT / "work" / cfg.name
        work_dir.mkdir(parents=True, exist_ok=True)
        track = lead_pitch(cfg.lead, work_dir, args.crepe_model, resolve_device(args.device))
        analysis = modulate.analyze(track)
        print()
        print(modulate.render(analysis, cfg.name))

        semitones = args.semitones
        if semitones is None:
            if not sys.stdin.isatty():
                return 0  # sin terminal interactiva solo se muestra el análisis
            semitones = ask_semitones(analysis, cfg.name)
        elif semitones:
            print()
            print(modulate.render(analysis, cfg.name, semitones))
        if semitones:
            print()
            modulate.generate(cfg, ROOT, semitones, log=log)
    except (ConfigError, RuntimeError) as exc:
        print(f"[modulame] ERROR: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
