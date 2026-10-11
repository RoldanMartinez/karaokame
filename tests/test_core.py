import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from karaokame.align import match_sequences, normalize
from karaokame.notes import Note
from karaokame.syllables import syllabify
from karaokame.ultrastar import quantize, render


@pytest.mark.parametrize(
    "word, expected",
    [
        ("transforma", "trans-for-ma"),
        ("movimiento", "mo-vi-mien-to"),
        ("galaxia", "ga-la-xia"),
        ("había", "ha-bí-a"),
        ("Bahía", "Ba-hí-a"),
        ("caía", "ca-í-a"),
        ("darías", "da-rí-as"),
        ("aquel", "a-quel"),
        ("pagué", "pa-gué"),
        ("euro", "eu-ro"),
        ("luego", "lue-go"),
        ("aurora", "au-ro-ra"),
        ("sandalias", "san-da-lias"),
        ("hicieron", "hi-cie-ron"),
        ("vapor,", "va-por,"),
        ("rincón", "rin-cón"),
        ("simple", "sim-ple"),
        ("pierde", "pier-de"),
        ("compraste", "com-pras-te"),
        ("devolvería", "de-vol-ve-rí-a"),
        ("calle", "ca-lle"),
        ("perro", "pe-rro"),
        ("muy", "muy"),
        ("y", "y"),
        ("reyes", "re-yes"),
        ("instante", "ins-tan-te"),
    ],
)
def test_syllabify_es(word, expected):
    assert "-".join(syllabify(word, "es")) == expected


def test_match_sequences_tolerates_errors():
    ref = [normalize(w) for w in "Tu beso se hizo calor luego el calor movimiento".split()]
    hyp = [normalize(w) for w in "tu peso se hizo calor eh luego calor movimientos".split()]
    pairs = {(i, j) for i, j, _ in match_sequences(ref, hyp)}
    assert {(0, 0), (1, 1), (4, 4), (5, 6), (7, 7), (8, 8)} <= pairs
    assert all(i != 6 for i, _ in pairs)  # "el" no se oyó


def test_quantize_and_render():
    notes = [
        Note(10.00, 10.20, "Ho", 0, False, pitch=60),
        Note(10.20, 10.21, "la", 0, True, pitch=62),
        Note(10.21, 10.60, "tú", 0, True, pitch=64),
        Note(12.00, 12.50, "sol", 1, True, pitch=57),
    ]
    gap, beats = quantize(notes, 300.0)  # beat de 50 ms
    assert gap == pytest.approx(10000.0)
    assert [(b.start, b.length) for b in beats] == [(0, 4), (4, 1), (5, 7), (40, 10)]
    text = render({"TITLE": "Prueba"}, 300.0, gap, beats)
    assert text.splitlines() == [
        "#TITLE:Prueba",
        "#BPM:300,00",
        "#GAP:10000,00",
        ": 0 4 0 Ho",
        ": 4 1 2 la ",
        ": 5 7 4 tú",
        "- 20",
        ": 40 10 -3 sol",
        "E",
    ]


def test_estimate_key_and_transpose_txt():
    import numpy as np

    from karaokame import modulate

    # Escala de la menor con peso en la tónica y la quinta.
    histogram = np.zeros(12)
    for pc, weight in {9: 5, 11: 1, 0: 3, 2: 2, 4: 4, 5: 1.5, 7: 2}.items():
        histogram[pc] = weight
    key = modulate.estimate_keys(histogram / histogram.sum())[0]
    assert key.name() == "La menor"
    assert key.name(-2) == "Sol menor"
    assert modulate.note_name(60) == "Do4"

    source = "#TITLE:Prueba\n#MP3:a.mp3\n#BPM:300,00\n: 0 4 -1 Ho\n* 4 1 2 la \n- 20\nF 40 10 -3 sol\nE\n"
    assert modulate.transpose_txt(source, -2, "a_-2.mp3") == (
        "#TITLE:Prueba (-2)\n#MP3:a_-2.mp3\n#BPM:300,00\n: 0 4 -3 Ho\n* 4 1 0 la \n- 20\nF 40 10 -5 sol\nE\n"
    )


def test_range_and_best_fit():
    import numpy as np

    from karaokame import modulate
    from karaokame.config import ConfigError

    assert modulate.parse_range("do3_mi4") == (48, 64)
    assert modulate.parse_range("Fa#2_sib3") == (42, 58)
    with pytest.raises(ConfigError):
        modulate.parse_range("mi4_do3")

    def analysis(notes):
        seconds = {n: 1.0 for n in notes}
        return modulate.Analysis(seconds, np.ones(12) / 12, [], min(notes), max(notes), min(notes), max(notes))

    # Ya cabe: no se mueve.
    fit = modulate.best_fit(analysis(range(50, 60)), 48, 64)
    assert (fit.semitones, fit.octaves, fit.inside) == (0, 0, 1.0)
    # Tres semitonos por encima del tope: baja tres.
    fit = modulate.best_fit(analysis(range(55, 68)), 48, 64)
    assert (fit.semitones, fit.octaves, fit.inside) == (-3, 0, 1.0)
    # Voz de soprano algo más de una octava por encima: se canta una octava abajo y el audio apenas se mueve.
    fit = modulate.best_fit(analysis(range(65, 78)), 48, 64)
    assert (fit.semitones, fit.octaves, fit.note_shift) == (-1, -1, -13)
    assert fit.inside_before < 0.2

    source = "#MP3:a.mp3\n: 0 4 3 Ho\nE\n"
    assert modulate.transpose_txt(source, 1, "a_+1.mp3", note_shift=-11) == "#MP3:a_+1.mp3\n: 0 4 -8 Ho\nE\n"


def test_discover_and_lines(tmp_path, monkeypatch):
    from karaokame import discover
    from karaokame.config import ConfigError, parse_cfg
    from karaokame.transcribe import HeardWord, to_lines

    data, config, songs = tmp_path / "data", tmp_path / "config", tmp_path / "songs"
    for d in (data, config, songs):
        d.mkdir()
    for name in ("Una.mp4", "Una_V.mp3", "Una_A.mp3", "Una_video.mp4", "Una.jpg", "Una.txt", "Otra.mp3", "Otra.txt", "x.crdownload"):
        (data / name).write_text("x")
    monkeypatch.setattr(discover, "has_audio", lambda p: p.suffix in (".mp3", ".mp4"))

    assert [p.name for p in discover.find_songs(data)] == ["Otra.mp3", "Una.mp4"]
    with pytest.raises(ConfigError):
        discover.pick_song(data, songs)  # dos canciones sin generar
    (songs / "Otra.txt").write_text("x")
    song = discover.pick_song(data, songs)
    assert song.name == "Una.mp4"

    (config / "template.cfg").write_text('name=\naudio=\nlead="viejo.mp3"\nlyrics=\nrange="do3_mi4"\ncreator="yo"\n')
    values = parse_cfg(discover.ensure_cfg(song, config, log=lambda m: None))
    assert values == {"name": "Una", "audio": str(song), "lead": "", "accompaniment": "", "lyrics": str(data / "Una.txt"), "range": "do3_mi4", "creator": "yo"}

    words = [HeardWord(t, s, s + 0.3, 1.0) for t, s in [("hola,", 0), ("qué", 0.4), ("tal", 0.8), ("estás.", 1.2), ("Muy", 3.0), ("bien", 3.4)]]
    assert to_lines(words) == ["Hola, qué tal estás", "Muy bien"]
