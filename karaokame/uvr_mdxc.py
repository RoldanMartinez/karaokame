#!/usr/bin/env python3
"""Separación voz/acompañamiento con un modelo MDX23C de Ultimate Vocal Remover.

Reproduce, sin interfaz gráfica, lo que hace UVR con "MDX-Net" y un modelo
MDX23C: misma red (lib_v5/tfc_tdf_v3.py del repositorio), mismos pesos y el
mismo troceado con solape. Es un script independiente, sin dependencias del
paquete karaokame, para poder lanzarlo con el intérprete de UVR, que es el que
tiene PyTorch con GPU. Escribe dos WAV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

SAMPLE_RATE = 44100
VOCALS, INSTRUMENTAL = "Vocals", "Instrumental"
OOM_EXIT = 3  # la GPU se ha quedado sin memoria


def model_hash(path: Path) -> str:
    """Identificador que usa UVR: MD5 de los últimos 10 MB del fichero."""
    with open(path, "rb") as f:
        try:
            f.seek(-10000 * 1024, 2)
        except OSError:
            f.seek(0)
        return hashlib.md5(f.read()).hexdigest()


def namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
    return value


def load_model_config(repo: Path, model_path: Path):
    data_dir = repo / "models" / "MDX_Net_Models" / "model_data"
    table = json.loads((data_dir / "model_data.json").read_text())
    entry = table.get(model_hash(model_path))
    if not entry or "config_yaml" not in entry:
        raise SystemExit(f"{model_path.name} no es un modelo MDX23C conocido por UVR")
    with open(data_dir / "mdx_c_configs" / entry["config_yaml"]) as f:
        return namespace(yaml.load(f, Loader=yaml.FullLoader))


def demix(model, config, mix: np.ndarray, segment_size: int, overlap: int, device: str, batch_size: int = 1):
    """Igual que SeperateMDXC.demix de UVR: ventanas solapadas que se suman y promedian."""
    mix = torch.tensor(mix, dtype=torch.float32)
    chunk_size = config.audio.hop_length * (segment_size - 1)
    hop_size = chunk_size // overlap
    pad_size = hop_size - (mix.shape[1] - chunk_size) % hop_size
    mix = torch.cat([torch.zeros(2, chunk_size - hop_size), mix, torch.zeros(2, pad_size + chunk_size - hop_size)], 1)
    chunks = mix.unfold(1, chunk_size, hop_size).transpose(0, 1)

    stems = model.num_target_instruments
    out = torch.zeros(stems, *mix.shape) if stems > 1 else torch.zeros_like(mix)
    out = out.to(device)
    with torch.no_grad():
        done = 0
        for i in range(0, len(chunks), batch_size):
            for estimate in model(chunks[i : i + batch_size].to(device)):
                out[..., done * hop_size : done * hop_size + chunk_size] += estimate
                done += 1
            print(f"PROGRESS {done}/{len(chunks)}", flush=True)
    out = out[..., chunk_size - hop_size : -(pad_size + chunk_size - hop_size)] / overlap
    return out.cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True, help="carpeta de ultimatevocalremovergui")
    parser.add_argument("--model", type=Path, required=True, help="fichero .ckpt del modelo MDX23C")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocals", type=Path, required=True, help="WAV de salida con la voz")
    parser.add_argument("--instrumental", type=Path, required=True, help="WAV de salida con el acompañamiento")
    parser.add_argument("--segment-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    from lib_v5.tfc_tdf_v3 import TFC_TDF_net

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE {device}", flush=True)

    config = load_model_config(args.repo, args.model)
    instruments = list(config.training.instruments)
    if config.training.target_instrument or VOCALS not in instruments or INSTRUMENTAL not in instruments:
        raise SystemExit(f"{args.model.name} no separa en {VOCALS} e {INSTRUMENTAL}")

    model = TFC_TDF_net(config, device=device)
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.to(device).eval()

    mix, _ = librosa.load(str(args.input), mono=False, sr=SAMPLE_RATE)
    if mix.ndim == 1:
        mix = np.stack([mix, mix])

    try:
        sources = dict(zip(instruments, demix(model, config, mix, args.segment_size, args.overlap, device)))
    except torch.OutOfMemoryError:
        print("OOM", flush=True)
        return OOM_EXIT
    sf.write(str(args.vocals), sources[VOCALS].T, SAMPLE_RATE, subtype="PCM_16")
    sf.write(str(args.instrumental), sources[INSTRUMENTAL].T, SAMPLE_RATE, subtype="PCM_16")
    return 0


if __name__ == "__main__":
    sys.exit(main())
