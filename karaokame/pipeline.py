"""Orquestación: del cfg a songs/<name>.txt y songs/<name>.mp3."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import align, llm as llm_mod, modulate, notes as notes_mod, pitch as pitch_mod, transcribe, ultrastar, vocalize
from .config import SongConfig, load_config, parse_cfg, update_cfg

LOW_CONFIDENCE = 0.75
MAX_REVIEW_LINES = 25
SLOW_PACE = 1.0  # s por sílaba; fuera de [FAST_PACE, SLOW_PACE] la línea es sospechosa
FAST_PACE = 0.08
TRANSCRIBED_LYRICS = "lyrics_whisper.txt"
EXTRA_RUN = 3  # palabras oídas seguidas que no están en la letra para avisar


@dataclass
class Options:
    whisper_model: str = "medium"
    crepe_model: str = "full"
    device: str = "auto"
    use_llm: bool = True
    llm_model: str = llm_mod.DEFAULT_MODEL
    ollama_url: str = llm_mod.DEFAULT_URL
    sinalefa: bool = True
    melisma: bool = True
    force: bool = False
    creator: str = "karaokame"
    uvr_dir: Path = vocalize.DEFAULT_UVR


def log(message: str) -> None:
    print(f"[karaokame] {message}", flush=True)


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{path}:{stat.st_size}:{stat.st_mtime_ns}"


def _cached(path: Path, key: str, force: bool) -> bool:
    key_file = path.with_suffix(path.suffix + ".key")
    return not force and path.exists() and key_file.exists() and key_file.read_text() == key


def _mark(path: Path, key: str) -> None:
    path.with_suffix(path.suffix + ".key").write_text(key)


def _copy_accompaniment(source: Path, target: Path) -> None:
    if source.suffix.lower() == ".mp3":
        shutil.copyfile(source, target)
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-codec:a", "libmp3lame", "-q:a", "2", str(target)],
        check=True,
    )


def _metadata(cfg: SongConfig, lyrics: list[str], opts: Options, llm, warnings: list[str]) -> dict[str, str]:
    meta = {"artist": "", "title": "", "genre": "", "year": "", "creator": opts.creator}
    wanted = [k for k in ("artist", "title", "genre", "year") if k not in cfg.metadata]
    if llm and wanted:
        try:
            guess = llm_mod.guess_metadata(llm, cfg.name, lyrics, cfg.language)
            meta.update({k: guess[k] for k in wanted if guess.get(k)})
            warnings.append(
                f"Metadatos propuestos por {opts.llm_model} sin verificar ({', '.join(wanted)}): "
                "revísalos o fíjalos en el cfg."
            )
        except llm_mod.LLMError as exc:
            warnings.append(f"No se han podido obtener metadatos del LLM: {exc}")
    meta.update(cfg.metadata)
    if not meta["title"]:
        meta["title"] = cfg.name.replace("_", " ")
    if not meta["artist"]:
        meta["artist"] = "Desconocido"
        warnings.append("Artista desconocido: añade artist=\"...\" al cfg.")
    return meta


def load_lead(lead: Path) -> np.ndarray:
    import librosa

    return librosa.load(str(lead), sr=pitch_mod.SAMPLE_RATE, mono=True)[0]


def lead_pitch(lead: Path, work_dir: Path, crepe_model: str, device: str, force: bool = False):
    """Análisis de tono de la voz, reutilizando la caché de work/<name>/ si sigue vigente."""
    pitch_file = work_dir / f"pitch_{crepe_model}.npz"
    lead_key = _fingerprint(lead)
    if _cached(pitch_file, lead_key, force):
        log(f"CREPE: usando caché {pitch_file.name}")
        return pitch_mod.load(pitch_file)
    log(f"CREPE ({crepe_model}, {device}): analizando el tono de la voz...")
    track = pitch_mod.analyze(load_lead(lead), model=crepe_model, device=device)
    pitch_mod.save(track, pitch_file)
    _mark(pitch_file, lead_key)
    return track


def heard_words(lead: Path, language: str, work_dir: Path, opts: Options, device: str, lyrics: list[str], track):
    """Palabras oídas por Whisper en la voz, reutilizando la caché de work/<name>/ si sigue vigente."""
    whisper_file = work_dir / f"whisper_{opts.whisper_model}.json"
    whisper_key = f"{_fingerprint(lead)}:{language}:v3"
    if _cached(whisper_file, whisper_key, opts.force):
        log(f"Whisper: usando caché {whisper_file.name}")
        return transcribe.from_json(json.loads(whisper_file.read_text(encoding="utf-8")))
    log(f"Whisper ({opts.whisper_model}, {device}): transcribiendo la voz...")
    heard = transcribe.transcribe(load_lead(lead), language, opts.whisper_model, device, lyrics, track)
    whisper_file.write_text(json.dumps(transcribe.to_json(heard), ensure_ascii=False, indent=1), encoding="utf-8")
    _mark(whisper_file, whisper_key)
    return heard


def prepare(path: Path, root: Path, opts: Options) -> SongConfig:
    """Completa el cfg con lo que falte (voz y acompañamiento, letra, idioma) y lo carga."""
    values = parse_cfg(path)

    def present(key: str) -> bool:
        return bool(values.get(key)) and Path(values[key]).expanduser().is_file()

    if values.get("audio") and not (present("lead") and present("accompaniment")):
        log("El cfg no tiene voz y acompañamiento: se separan de 'audio'")
        if opts.use_llm:
            llm_mod.Ollama(opts.llm_model, opts.ollama_url).unload()  # la separación necesita la GPU entera
        values.update(vocalize.run(path, vocalize.Options(uvr_dir=opts.uvr_dir, force=opts.force)))

    if not values.get("lyrics") and present("lead"):
        # Sin letra: se usa como letra lo que Whisper entienda en la voz.
        if not values.get("language"):
            values["language"] = _detect_language(values, opts)
            update_cfg(path, {"language": values["language"]})
            log(f'{path.name}: language="{values["language"]}"')
        name = values.get("name", "")
        work_dir = root / "work" / (name[:-4] if name.lower().endswith(".txt") else name)
        work_dir.mkdir(parents=True, exist_ok=True)
        lead, device = Path(values["lead"]).expanduser(), resolve_device(opts.device)
        log("El cfg no tiene letra: se transcribe de la voz")
        track = lead_pitch(lead, work_dir, opts.crepe_model, device, opts.force)
        heard = heard_words(lead, values["language"], work_dir, opts, device, [], track)
        lyrics_file = work_dir / TRANSCRIBED_LYRICS
        lyrics_file.write_text("\n".join(transcribe.to_lines(heard)) + "\n", encoding="utf-8")
        values["lyrics"] = str(lyrics_file)
        update_cfg(path, {"lyrics": values["lyrics"]})
        log(f'{path.name}: lyrics="{lyrics_file}"')

    if not values.get("language"):
        language = _detect_language(values, opts) if present("lyrics") or present("lead") else None
        if language:
            update_cfg(path, {"language": language})
            log(f'{path.name}: language="{language}"')

    return load_config(path)


def _detect_language(values: dict[str, str], opts: Options) -> str | None:
    """Idioma a partir de la letra (LLM) o, si no se puede, de la voz (Whisper)."""
    lyrics = Path(values.get("lyrics", "")).expanduser()
    if opts.use_llm and lyrics.is_file():
        try:
            lines = [line.strip() for line in lyrics.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            language = llm_mod.detect_language(llm_mod.Ollama(opts.llm_model, opts.ollama_url), lines)
            log(f"El cfg no indica idioma: según la letra ({opts.llm_model}) es '{language}'")
            return language
        except (llm_mod.LLMError, UnicodeDecodeError, KeyError) as exc:
            log(f"No se ha podido deducir el idioma de la letra: {exc}")
    lead = Path(values.get("lead", "")).expanduser()
    if not lead.is_file():
        return None
    import librosa

    log(f"El cfg no indica idioma: detectando en la voz con Whisper ({opts.whisper_model})...")
    audio = librosa.load(str(lead), sr=pitch_mod.SAMPLE_RATE, mono=True)[0]
    language, probability = transcribe.detect_language(audio, opts.whisper_model, resolve_device(opts.device))
    log(f"Idioma detectado: {language} ({100 * probability:.0f} %)")
    return language


def fit_range(cfg: SongConfig, root: Path, track, warnings: list[str]) -> list[str]:
    """Transporta la canción para que la voz caiga en el rango pedido. Devuelve líneas para el informe."""
    low, high = modulate.parse_range(cfg.range)
    analysis = modulate.analyze(track)
    fit = modulate.best_fit(analysis, low, high)
    wanted = f"{modulate.note_name(low)}–{modulate.note_name(high)}"
    sung = f"{modulate.note_name(analysis.usual_low)}–{modulate.note_name(analysis.usual_high)}"
    octave = ""
    if fit.octaves:
        n = abs(fit.octaves)
        octave = f" cantando {'una octava' if n == 1 else f'{n} octavas'} {'por encima' if fit.octaves > 0 else 'por debajo'} de la voz original"
    lines = [
        f"- Rango pedido: {wanted} · rango habitual de la voz original: {sung} · tonalidad: {analysis.key.name()}",
        f"- Tiempo cantado dentro del rango sin transportar: {100 * fit.inside_before:.1f} %",
    ]
    if fit.semitones == 0:
        log(f"Rango {wanted}: no hace falta transportar{octave} ({100 * fit.inside:.0f} % dentro)")
        lines.append(f"- No se transporta: la canción ya encaja{octave} ({100 * fit.inside:.1f} % dentro del rango)")
    else:
        log(f"Rango {wanted}: se transporta {fit.semitones:+d} semitonos{octave} ({100 * fit.inside:.0f} % dentro)")
        txt = modulate.generate(cfg, root, fit.semitones, fit.note_shift, log=log)
        lines.append(
            f"- Transposición: {fit.semitones:+d} semitonos{octave}, a {analysis.key.name(fit.semitones)} "
            f"({100 * fit.inside:.1f} % dentro del rango) → `{txt.name}`"
        )
    if fit.inside < 0.9:
        warnings.append(
            f"Solo el {100 * fit.inside:.0f} % del tiempo cantado cabe en el rango {wanted}: "
            "la melodía es más amplia que el rango pedido."
        )
    return lines


def run(cfg: SongConfig, root: Path, opts: Options) -> Path:

    songs_dir = root / "songs"
    work_dir = root / "work" / cfg.name
    songs_dir.mkdir(exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(opts.device)
    lyrics = cfg.read_lyrics()
    warnings: list[str] = []
    if cfg.lyrics.name == TRANSCRIBED_LYRICS:
        warnings.append(
            f"La letra no viene de un fichero: es la transcripción de Whisper ({cfg.lyrics}). "
            "Corrígela y vuelve a ejecutar; los tiempos se recalculan en segundos."
        )

    # 1. Tono y sonoridad de la voz (CREPE)
    track = lead_pitch(cfg.lead, work_dir, opts.crepe_model, device, opts.force)

    # 2. Palabras con tiempos (Whisper)
    heard = heard_words(cfg.lead, cfg.language, work_dir, opts, device, lyrics, track)

    # 3. Alineación letra <-> voz
    log("Alineando la letra con la voz...")
    words = align.build_words(lyrics, cfg.language)
    unmatched = align.assign_times(words, heard, track)
    align.refine_boundaries(words, track)
    align.split_syllables(words, track)

    # 4. Notas
    notes = notes_mod.build_notes(words, cfg.language, sinalefa=opts.sinalefa)
    notes = notes_mod.assign_pitch(notes, track, melisma=opts.melisma)

    # 5. BPM, GAP y cuantización
    log("Calculando el BPM del acompañamiento...")
    bpm = ultrastar.detect_bpm(cfg.accompaniment)
    gap_ms, beat_notes = ultrastar.quantize(notes, bpm)

    # 6. LLM: metadatos y revisión
    llm = llm_mod.Ollama(opts.llm_model, opts.ollama_url) if opts.use_llm else None
    if llm:
        log(f"LLM ({opts.llm_model}): metadatos...")
    meta = _metadata(cfg, lyrics, opts, llm, warnings)

    line_stats = _line_stats(lyrics, words, heard)
    review = None
    suspects = [s for s in line_stats if s["reason"]]
    if llm and suspects:
        log(f"LLM: revisando {min(len(suspects), MAX_REVIEW_LINES)} líneas dudosas...")
        worst = sorted(suspects, key=lambda s: s["conf"])[:MAX_REVIEW_LINES]
        payload = [
            {"n": s["n"], "lyric": s["lyric"], "heard": s["heard"], "start": round(s["start"], 1), "end": round(s["end"], 1)}
            for s in sorted(worst, key=lambda s: s["n"])
        ]
        try:
            review = llm_mod.review_lines(llm, payload)
        except llm_mod.LLMError as exc:
            warnings.append(f"Revisión del LLM no disponible: {exc}")

    # 7. Salida
    mp3_name = f"{cfg.name}.mp3"
    headers = {
        "ARTIST": meta["artist"],
        "TITLE": meta["title"],
        "LANGUAGE": ultrastar.LANGUAGE_NAMES.get(cfg.language, cfg.language),
        "GENRE": meta["genre"],
        "YEAR": meta["year"],
        "CREATOR": meta["creator"],
        "MP3": mp3_name,
    }
    for tag, source in (("COVER", cfg.cover), ("VIDEO", cfg.video)):
        if source:
            headers[tag] = f"{cfg.name}{source.suffix.lower()}"
            shutil.copyfile(source, songs_dir / headers[tag])
    txt_path = songs_dir / f"{cfg.name}.txt"
    txt_path.write_text(ultrastar.render(headers, bpm, gap_ms, beat_notes), encoding="utf-8")
    _copy_accompaniment(cfg.accompaniment, songs_dir / mp3_name)

    # 8. Transposición al rango pedido
    range_lines = fit_range(cfg, root, track, warnings) if cfg.range else []

    for stat in line_stats:
        if len(stat["lyric"]) > ultrastar.MAX_LINE_CHARS:
            warnings.append(f"Línea {stat['n']} supera los {ultrastar.MAX_LINE_CHARS} caracteres: {stat['lyric']!r}")

    for run in _extra_runs(unmatched):
        warnings.append(
            f"Oído en {_clock(run[0].start)} pero no está en la letra: «{' '.join(h.text for h in run)}»"
        )
    report = _report(cfg, opts, meta, bpm, gap_ms, words, notes, track, line_stats, review, warnings, range_lines)
    (work_dir / "report.md").write_text(report, encoding="utf-8")
    log(f"Hecho: {txt_path}")
    log(f"Informe: {work_dir / 'report.md'}")
    return txt_path


def _line_stats(lyrics: list[str], words: list[align.Word], heard) -> list[dict]:
    stats = []
    for n, lyric in enumerate(lyrics):
        line_words = [w for w in words if w.line == n]
        start, end = line_words[0].start, line_words[-1].end
        stats.append(
            {
                "n": n + 1,
                "lyric": lyric,
                "start": start,
                "end": end,
                "conf": float(np.mean([w.conf for w in line_words])),
                "heard": align.heard_between(heard, start, end),
            }
        )
        pace = (end - start) / sum(len(w.syllables) for w in line_words)
        reasons = []
        if stats[-1]["conf"] < LOW_CONFIDENCE:
            reasons.append("mal oída")
        if pace > SLOW_PACE:
            reasons.append("muy lenta")
        if pace < FAST_PACE:
            reasons.append("muy rápida")
        stats[-1]["reason"] = ", ".join(reasons)
    return stats


def _extra_runs(unmatched) -> list[list]:
    """Grupos de palabras oídas con claridad, contiguas en el tiempo, que no aparecen en la letra."""
    runs: list[list] = [[]]
    for h in unmatched:
        if h.prob < 0.5:
            continue
        if runs[-1] and h.start - runs[-1][-1].end > 1.0:
            runs.append([])
        runs[-1].append(h)
    return [r for r in runs if len(r) >= EXTRA_RUN]


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


def _report(cfg, opts, meta, bpm, gap_ms, words, notes, track, line_stats, review, warnings, range_lines) -> str:
    anchored = sum(1 for w in words if w.conf > 0)
    exact = sum(1 for w in words if w.conf == 1.0)
    voiced_notes = sum(1 for n in notes if track.voiced[track.span(n.start, n.end)].any())
    pitches = [n.pitch for n in notes]
    suspects = [s for s in line_stats if s["reason"]]
    overlaps = sum(1 for a, b in zip(line_stats, line_stats[1:]) if b["start"] < a["end"] - 0.05)

    out = [
        f"# Informe de generación: {cfg.name}",
        "",
        f"- Artista / título: {meta['artist']} / {meta['title']}",
        f"- Género / año: {meta['genre'] or '-'} / {meta['year'] or '-'}",
        f"- Modelos: Whisper `{opts.whisper_model}`, CREPE `{opts.crepe_model}`"
        + (f", LLM `{opts.llm_model}`" if opts.use_llm else ", sin LLM"),
        f"- BPM: {bpm:.2f} (beat de {ultrastar.beat_seconds(bpm) * 1000:.1f} ms) · GAP: {gap_ms:.0f} ms",
        "",
        "## Calidad de la alineación",
        "",
        f"- Palabras de la letra: {len(words)}",
        f"- Emparejadas con Whisper: {anchored} ({100 * anchored / len(words):.1f} %), "
        f"de ellas idénticas {exact}",
        f"- Con tiempo interpolado (no oídas): {len(words) - anchored}",
        f"- Notas: {len(notes)} ({sum(1 for n in notes if n.text == '~')} de cambio de tono `~`)",
        f"- Notas con voz detectada por CREPE: {voiced_notes} ({100 * voiced_notes / len(notes):.1f} %)",
        f"- Rango de tonos (MIDI): {min(pitches)}-{max(pitches)}",
        f"- Líneas a revisar: {len(suspects)} de {len(line_stats)}",
        f"- Líneas solapadas en el tiempo: {overlaps}",
        "",
    ]
    if range_lines:
        out += ["## Rango y transposición", ""] + range_lines + [""]
    if warnings:
        out += ["## Avisos", ""] + [f"- {w}" for w in warnings] + [""]

    if suspects:
        verdicts = {entry["n"]: entry for entry in (review or {}).get("lines", [])}
        out += [
            "## Líneas a revisar a mano",
            "",
            "| # | Inicio | Fin | Conf. | Motivo | Letra | Oído por Whisper | LLM |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in suspects:
            v = verdicts.get(s["n"])
            llm_cell = f"{v['verdict']}: {v['comment']}" if v else "-"
            row = [str(s["n"]), _clock(s["start"]), _clock(s["end"]), f"{s['conf']:.2f}", s["reason"], s["lyric"], s["heard"] or "(nada)", llm_cell]
            out.append("| " + " | ".join(cell.replace("|", "/") for cell in row) + " |")
        out.append("")
        if review and review.get("summary"):
            out += [f"**Resumen del LLM:** {review['summary']}", ""]

    out += ["## Todas las líneas", "", "| # | Inicio | Fin | Conf. | Letra |", "|---|---|---|---|---|"]
    for s in line_stats:
        out.append(f"| {s['n']} | {_clock(s['start'])} | {_clock(s['end'])} | {s['conf']:.2f} | {s['lyric']} |")
    return "\n".join(out) + "\n"
