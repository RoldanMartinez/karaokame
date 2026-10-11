"""Tareas de lenguaje con un modelo local servido por Ollama (por defecto Qwen3 4B).

El LLM no toca tiempos ni tonos: propone los metadatos de la cabecera y da una
segunda opinión sobre las líneas en las que la alineación ha quedado dudosa.
Si Ollama no responde, el pipeline sigue sin él.
"""

from __future__ import annotations

import json

import requests

DEFAULT_MODEL = "qwen3:4b-instruct-2507-q4_K_M"
DEFAULT_URL = "http://localhost:11434"


class LLMError(RuntimeError):
    pass


class Ollama:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL, timeout: float = 300.0):
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout

    def unload(self) -> None:
        """Pide a Ollama que suelte el modelo ya: libera la memoria de la GPU para el resto del proceso."""
        try:
            requests.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30)
        except requests.RequestException:
            pass

    def ask_json(self, system: str, user: str, schema: dict) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema,
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            response = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            if response.status_code == 400 and "think" in response.text:
                payload.pop("think")  # el modelo no admite el parámetro
                response = requests.post(f"{self.url}/api/chat", json=payload, timeout=self.timeout)
            response.raise_for_status()
            return json.loads(response.json()["message"]["content"])
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise LLMError(f"Ollama ({self.model}) no ha dado una respuesta válida: {exc}") from exc


METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "artist": {"type": "string"},
        "title": {"type": "string"},
        "genre": {"type": "string"},
        "year": {"type": "string"},
    },
    "required": ["artist", "title", "genre", "year"],
}


def guess_metadata(llm: Ollama, name: str, lyrics: list[str], language: str) -> dict[str, str]:
    system = (
        "Eres un documentalista musical. A partir del nombre de fichero y del comienzo de la letra "
        "de una canción, identifica artista, título, género y año de publicación. "
        "Solo la primera palabra del título y los nombres propios llevan mayúscula inicial. "
        "El género es una sola etiqueta en inglés (Pop, Rock, Folk, Latin, Singer-Songwriter...). "
        "Si no conoces el año con seguridad, deja 'year' como cadena vacía; no lo inventes."
    )
    user = f"Fichero: {name}\nIdioma: {language}\nLetra:\n" + "\n".join(lyrics[:8])
    data = llm.ask_json(system, user, METADATA_SCHEMA)
    year = "".join(c for c in str(data.get("year", "")) if c.isdigit())
    return {
        "artist": str(data.get("artist", "")).strip(),
        "title": str(data.get("title", "")).strip(),
        "genre": str(data.get("genre", "")).strip(),
        "year": year if len(year) == 4 else "",
    }


# Idiomas entre los que elige el LLM; todos los entiende Whisper.
LANGUAGES = {
    "es": "español", "en": "inglés", "it": "italiano", "fr": "francés", "de": "alemán", "pt": "portugués",
    "ca": "catalán", "gl": "gallego", "eu": "euskera", "la": "latín", "nl": "neerlandés", "ru": "ruso",
    "ja": "japonés", "ko": "coreano", "zh": "chino",
}
LANGUAGE_SCHEMA = {
    "type": "object",
    "properties": {"language": {"type": "string", "enum": list(LANGUAGES)}},
    "required": ["language"],
}


def detect_language(llm: Ollama, lyrics: list[str]) -> str:
    """Idioma de la letra. Un dialecto se asigna a la lengua estándar más próxima."""
    options = ", ".join(f"{code} ({name})" for code, name in LANGUAGES.items())
    system = (
        "Identifica el idioma de la letra de una canción y responde con su código. "
        f"Opciones: {options}. Si la letra está en un dialecto o lengua regional que no aparece "
        "(napolitano, siciliano, andaluz, occitano...), elige la lengua estándar más próxima de la lista."
    )
    return llm.ask_json(system, "\n".join(lyrics[:12]), LANGUAGE_SCHEMA)["language"]


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer"},
                    "verdict": {"type": "string", "enum": ["ok", "dudosa", "mal"]},
                    "comment": {"type": "string"},
                },
                "required": ["n", "verdict", "comment"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["lines", "summary"],
}


def review_lines(llm: Ollama, suspects: list[dict]) -> dict:
    """Segunda opinión sobre líneas dudosas: [{n, lyric, heard, start, end}]."""
    system = (
        "Revisas la sincronización de un karaoke. Para cada línea recibes la letra correcta ('lyric') "
        "y lo que un reconocedor de voz ha oído en el intervalo de tiempo asignado a esa línea ('heard'). "
        "El reconocedor se equivoca con frecuencia al transcribir voz cantada, así que no esperes texto idéntico. "
        "Veredicto por línea: 'ok' si lo oído es la misma frase (aunque con errores de transcripción o "
        "palabras sueltas de más o de menos); 'dudosa' si solo coincide en parte; 'mal' si es otra frase "
        "distinta o no se ha oído nada. Responde en español, con un comentario breve por línea y un "
        "resumen final de una o dos frases."
    )
    user = json.dumps(suspects, ensure_ascii=False, indent=1)
    data = llm.ask_json(system, user, REVIEW_SCHEMA)
    valid = {s["n"] for s in suspects}
    data["lines"] = [entry for entry in data.get("lines", []) if entry.get("n") in valid]
    return data
