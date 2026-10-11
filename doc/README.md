# karaokame

Genera canciones para UltraStar WorldParty (`.txt` + `.mp3`) a partir de una
pista de voz, una de acompañamiento y la letra, usando modelos locales:
Whisper para los tiempos, CREPE para los tonos y Qwen (Ollama) como revisor.

- [Uso](uso.md): instalación, fichero cfg, opciones y salidas.
- [Arquitectura](arquitectura.md): cómo funciona cada etapa y sus limitaciones.
- [vocalizame](vocalizame.md): separación de voz y acompañamiento a partir de la canción completa.
- [modulame](modulame.md): análisis de tonalidad y transposición de una canción.
- [Formato del txt](formato_ultrastar.md): qué se escribe en cada etiqueta y nota.
- [PDF](karaokame.pdf): formato del cfg, del txt y opciones del script (se regenera con `doc/build_pdf.py`).
