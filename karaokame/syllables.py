"""División de palabras en sílabas.

Para español se usa un silabeador por reglas (el español es regular). Para el
resto de idiomas se recurre a los diccionarios de guionado de pyphen, que son
una aproximación razonable.
"""

from __future__ import annotations

from functools import lru_cache

VOWELS = set("aeiouáéíóúü")
STRONG = set("aeoáéó")
ACCENTED_WEAK = set("íú")
# Grupos consonánticos que no se separan (van juntos a la sílaba siguiente).
CLUSTERS = {"pr", "br", "tr", "dr", "cr", "gr", "fr", "kr", "pl", "bl", "cl", "gl", "fl", "kl"}

PYPHEN_LANGS = {"en": "en_US", "fr": "fr", "de": "de_DE", "it": "it_IT", "pt": "pt_PT", "ca": "ca", "gl": "gl"}


def _split_affixes(word: str) -> tuple[str, str, str]:
    """Separa la puntuación inicial y final del núcleo alfabético de la palabra."""
    start, end = 0, len(word)
    while start < end and not word[start].isalpha():
        start += 1
    while end > start and not word[end - 1].isalpha():
        end -= 1
    return word[:start], word[start:end], word[end:]


def _tokenize_es(core: str) -> list[tuple[int, int, bool]]:
    """Trocea en unidades (inicio, fin, es_vocal); los dígrafos cuentan como una consonante."""
    low = core.lower()
    n = len(low)
    tokens: list[tuple[int, int, bool]] = []
    i = 0
    while i < n:
        c = low[i]
        nxt = low[i + 1] if i + 1 < n else ""
        if c + nxt in ("ch", "ll", "rr", "qu"):
            tokens.append((i, i + 2, False))
            i += 2
            continue
        if c == "g" and nxt == "u" and i + 2 < n and low[i + 2] in "eiéí":
            tokens.append((i, i + 2, False))  # la u de gue/gui es muda
            i += 2
            continue
        if c == "y":
            is_vowel = i == n - 1 or nxt not in VOWELS  # "y", "muy", "hoy" frente a "yo", "reyes"
        else:
            is_vowel = c in VOWELS
        tokens.append((i, i + 1, is_vowel))
        i += 1
    return tokens


def _is_hiatus(a: str, b: str) -> bool:
    return (a in STRONG and b in STRONG) or a in ACCENTED_WEAK or b in ACCENTED_WEAK


def syllabify_es(word: str) -> list[str]:
    prefix, core, suffix = _split_affixes(word)
    if not core:
        return [word]
    low = core.lower()
    tokens = _tokenize_es(core)

    # Agrupa en núcleos vocálicos (diptongos/triptongos) y bloques de consonantes.
    groups: list[tuple[bool, list[tuple[int, int, bool]]]] = []
    for tok in tokens:
        start, _, is_vowel = tok
        if groups and groups[-1][0] == is_vowel:
            prev = groups[-1][1][-1]
            if is_vowel and _is_hiatus(low[prev[0]], low[start]):
                groups.append((True, [tok]))
            else:
                groups[-1][1].append(tok)
        else:
            groups.append((is_vowel, [tok]))

    nuclei = [i for i, (is_vowel, _) in enumerate(groups) if is_vowel]
    if len(nuclei) <= 1:
        return [word]

    # Puntos de corte (índices sobre `core`) entre núcleos consecutivos.
    cuts: list[int] = []
    for a, b in zip(nuclei, nuclei[1:]):
        if b == a + 1:  # hiato: dos núcleos seguidos
            cuts.append(groups[b][1][0][0])
            continue
        cons = groups[a + 1][1]
        m = len(cons)
        if m == 1:
            keep = 0
        else:
            last_two = low[cons[-2][0] : cons[-1][1]]
            keep = m - 2 if last_two in CLUSTERS else m - 1
        cuts.append(cons[keep][0])

    bounds = [0, *cuts, len(core)]
    parts = [core[i:j] for i, j in zip(bounds, bounds[1:])]
    parts[0] = prefix + parts[0]
    parts[-1] = parts[-1] + suffix
    return parts


@lru_cache(maxsize=None)
def _pyphen(lang: str):
    import pyphen

    code = PYPHEN_LANGS.get(lang, lang)
    if code not in pyphen.LANGUAGES:
        return None
    return pyphen.Pyphen(lang=code, left=1, right=1)


def syllabify(word: str, language: str = "es") -> list[str]:
    """Divide una palabra en sílabas; la concatenación de las partes es la palabra original."""
    if language == "es":
        return syllabify_es(word)
    dic = _pyphen(language)
    if dic is None:
        return [word]
    prefix, core, suffix = _split_affixes(word)
    if not core:
        return [word]
    parts = dic.inserted(core, hyphen="\x00").split("\x00")
    parts[0] = prefix + parts[0]
    parts[-1] = parts[-1] + suffix
    return parts


def starts_with_vowel_sound(text: str) -> bool:
    low = text.lower().lstrip("¡¿\"'(«")
    if low.startswith("h"):
        low = low[1:]
    if not low:
        return False
    if low[0] == "y":
        return len(low) == 1
    return low[0] in VOWELS


def ends_with_vowel_sound(text: str) -> bool:
    """True si acaba en vocal sin puntuación detrás (una coma rompe la sinalefa)."""
    low = text.lower()
    if not low:
        return False
    return low[-1] in VOWELS or (low[-1] == "y" and len(low) > 1 and low[-2] in VOWELS) or low == "y"
