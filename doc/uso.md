# Uso

## Flujo completo

`bin/karaokame.py` hace todo el proceso. Lo más sencillo es dejar la canción en
`data/` y lanzarlo sin argumentos:

```bash
.venv/bin/python bin/karaokame.py
```

0. Busca la canción en `data/` y crea su cfg en `config/` (ver más abajo).
1. Si faltan `lead` y `accompaniment` y hay `audio`, separa la voz del
   acompañamiento (lo mismo que [vocalizame](vocalizame.md)) y los escribe en el cfg.
2. Si falta la letra, la transcribe de la voz con Whisper.
3. Si falta `language`, lo deduce de la letra con el LLM (o, sin LLM, de la voz
   con Whisper) y lo escribe en el cfg.
4. Genera `songs/<name>.txt` y `songs/<name>.mp3`.
5. Si hay `range`, analiza la voz (lo mismo que [modulame](modulame.md)) y
   genera además la versión transportada que mejor encaja en ese rango.

También se le puede pasar un cfg ya escrito; con `name` y `audio` basta.

## La carpeta data/

Sin argumentos, el script espera encontrar **una sola canción** en `data/`:

- **La canción** es cualquier fichero con una pista de audio, sea cual sea su
  formato: mp3, wav, flac, m4a, o un vídeo mp4, mkv o webm. Se comprueba con
  `ffprobe`, no por la extensión. Si es un vídeo, la imagen se extrae a un mp4
  sin audio y se usa como `#VIDEO`; si es un audio con carátula incrustada, la
  carátula se usa como `#COVER`.
- **La letra** es el `.txt` con el mismo nombre que la canción o, si no existe,
  el único `.txt` que haya en la carpeta. Si no hay ninguno, se usa como letra lo
  que Whisper transcriba de la voz; queda en `work/<name>/lyrics_whisper.txt`
  para corregirla y volver a ejecutar.
- Se ignoran los ficheros que generan los propios scripts (`*_V`, `*_A`,
  `*_video`), las imágenes y cualquier fichero sin audio.

El cfg se crea como `config/<nombre del fichero>.cfg`. Si existe
`config/template.cfg`, se parte de él: sirve para dejar fijados valores comunes
como `range` o `creator`. Si el cfg de la canción ya existe no se sobrescribe,
así que se puede editar y volver a lanzar.

Si en `data/` hay varias canciones, se procesa la única que todavía no tenga su
txt en `songs/`. Si hay varias pendientes, o varias ya generadas, el script se
detiene y pide dejar solo una o indicar el cfg.

## Requisitos

- Python 3.12 y `ffmpeg`.
- [Ollama](https://ollama.com) en marcha con un modelo Qwen3 4B (opcional; ver `--no-llm`).
- Para separar la voz: una copia de Ultimate Vocal Remover con su entorno virtual
  y el modelo MDX23C-InstVoc HQ (por defecto en `ultimatevocalremovergui/`).
- Unos 2 GB de disco para el entorno virtual y 1,5 GB para el modelo `medium` de Whisper.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
```

Se instala la variante CPU de PyTorch. Con GPU operativa se puede instalar la
variante CUDA y pasar `--device cuda`.

## Fichero de configuración

Un `.cfg` por canción en `config/`, con líneas `clave="valor"`:

| Campo | Obligatorio | Descripción |
|---|---|---|
| `name` | sí | Nombre base de los ficheros de salida (se ignora un `.txt` final) |
| `audio` | si faltan `lead` y `accompaniment` | Canción completa, de la que se separan voz y acompañamiento |
| `lead` | se genera | Audio con la pista principal aislada (la voz) |
| `accompaniment` | se genera | Audio del acompañamiento; es lo que sonará en el karaoke |
| `lyrics` | se genera | Fichero de texto UTF-8 con la letra, una línea de karaoke por línea; si está vacío se transcribe de la voz |
| `language` | se detecta | Código de idioma (`es`, `en`, ...); si está vacío se deduce de la letra |
| `range` | no | Rango en el que se quiere cantar, p. ej. `do3_mi4`; vacío = no transportar |
| `cover`, `video` | no | Carátula e imagen de fondo; se copian a `songs/` y dan `#COVER` y `#VIDEO`. Los rellena la separación si el audio los contiene |
| `artist`, `title`, `genre`, `year`, `creator` | no | Fijan la etiqueta correspondiente; si faltan, las propone el LLM |
| `intrument`, `score` | no | Reservados, todavía sin uso |

El rango se escribe como dos notas separadas por `_`: nombre en español (`do`,
`re`, `mi`, `fa`, `sol`, `la`, `si`), alteración opcional (`#` o `b`) y octava,
con `do4` como do central. `do3_mi4` va del do por debajo del do central al mi
por encima.

El script escribe en el cfg los campos que genera (`lead`, `accompaniment`,
`language`, `cover`, `video`), sin tocar el resto de líneas.

Las rutas relativas se resuelven respecto a la carpeta del cfg. Las líneas de
la letra se respetan tal cual como líneas del karaoke (conviene que no pasen de
45 caracteres; el informe avisa si alguna lo hace).

## Ejecución

```bash
.venv/bin/python bin/karaokame.py                      # la canción que haya en data/
.venv/bin/python bin/karaokame.py config/cancion.cfg   # uno concreto
```

Opciones principales:

| Opción | Por defecto | Efecto |
|---|---|---|
| `--whisper-model` | `medium` | Modelo de faster-whisper (`small`, `medium`, `large-v3`...) |
| `--crepe-model` | `full` | `tiny` es mucho más rápido y algo menos preciso |
| `--device` | `auto` | `cpu` o `cuda` |
| `--no-llm` | | No usa Ollama: sin metadatos propuestos ni revisión |
| `--llm-model` | `qwen3:4b-instruct-2507-q4_K_M` | Modelo de Ollama |
| `--no-sinalefa` | | Una nota por sílaba aunque dos vocales se canten juntas |
| `--no-melisma` | | No genera notas `~` para cambios de tono dentro de una sílaba |
| `--creator` | `karaokame` | Valor de `#CREATOR` |
| `--uvr-dir` | `ultimatevocalremovergui/` | Copia de Ultimate Vocal Remover para separar la voz |
| `--force` | | Recalcula Whisper y CREPE aunque haya caché y repite la separación de voz |

## Tiempos orientativos

En CPU (16 hilos), para una canción de 3:40: CREPE `full` unos 7 minutos,
Whisper `medium` unos 2-3 minutos y el resto segundos. `--crepe-model tiny`
es bastante más rápido (no medido aquí).

## Salidas

- `songs/<name>.txt`: canción de UltraStar.
- `songs/<name>.mp3`: copia del acompañamiento (referenciada por `#MP3`).
- `songs/<name>.jpg`, `songs/<name>.mp4`: carátula y vídeo, si el cfg los indica.
- `songs/<name>_X.txt`, `songs/<name>_X.mp3`: versión transportada X semitonos
  (por ejemplo `_-4`), si el cfg tiene `range` y hace falta transportar.
- `<audio>_V.mp3`, `<audio>_A.mp3`: voz y acompañamiento, junto al fichero `audio`.
- `config/<name>.cfg`: el cfg creado para la canción de `data/`.
- `work/<name>/lyrics_whisper.txt`: letra transcrita, si no había fichero de letra.
- `work/<name>/report.md`: informe de calidad con las líneas que conviene revisar a mano.
- `work/<name>/whisper_*.json`, `pitch_*.npz`: caché de los modelos. Mientras no
  cambie la pista de voz, las ejecuciones siguientes tardan segundos.

## Revisión posterior

El resultado es un buen punto de partida, no una canción terminada. Lo normal es
abrir el txt en un editor (Yass, UltraStar Creator o el editor del propio juego)
y repasar las líneas que el informe marca como dudosas. Las etiquetas `#BACKGROUND`
y `#MEDLEYSTARTBEAT`/`#MEDLEYENDBEAT` no se generan.

## Pruebas

```bash
.venv/bin/python -m pytest tests
```
