# vocalizame

Parte de la canción completa (campo `audio` del cfg), separa la voz del
acompañamiento con un modelo de Ultimate Vocal Remover y deja el cfg listo para
`karaokame.py`.

`bin/karaokame.py` hace esta separación de forma automática cuando al cfg le
faltan `lead` y `accompaniment`. Este script sirve para hacer solo ese paso o
para cambiar el modelo y sus parámetros.

## Uso

```bash
.venv/bin/python bin/vocalizame.py config/cancion.cfg
```

| Opción | Por defecto | Descripción |
|---|---|---|
| `config` | (obligatorio) | Fichero cfg; debe tener el campo `audio` |
| `--uvr-dir` | `ultimatevocalremovergui/` | Carpeta con la copia del repositorio de UVR |
| `--model` | `MDX23C-8KFFT-InstVoc_HQ.ckpt` | Modelo MDX23C, por nombre (en `models/MDX_Net_Models/` de UVR) o ruta |
| `--segment-size` | `256` | Tamaño de segmento, como en la interfaz de UVR |
| `--overlap` | `8` | Solape, como en la interfaz de UVR |
| `--device` | `auto` | `auto`, `cpu` o `cuda` |
| `--bitrate` | `320k` | Bitrate de los mp3 generados |
| `--python` | el de UVR | Intérprete con el que se ejecuta la separación |
| `--force` | | Rehace la separación y la extracción aunque los ficheros existan |

Los valores por defecto equivalen a la configuración de UVR "MDX-Net /
MDX23C-InstVoc HQ / Segment Size 256 / Overlap 8 / GPU Conversion / MP3".

## Qué genera

Junto al fichero de `audio`, con su mismo nombre base:

| Fichero | Campo del cfg | Contenido |
|---|---|---|
| `<audio>_V.mp3` | `lead` | Voz |
| `<audio>_A.mp3` | `accompaniment` | Acompañamiento |
| `<audio>.mp4` | `video` | Vídeo sin audio, si el fichero contiene una pista de vídeo real |
| `<audio>.jpg` | `cover` | Carátula, si lo único que hay es una imagen incrustada |

El cfg se reescribe cambiando solo esos campos; el resto de líneas queda igual.
Muchos mp3 llevan la carátula incrustada como una "pista de vídeo" de un solo
fotograma: eso no es vídeo, y se extrae como imagen en el campo `cover`. El vídeo
se copia sin recodificar si es H.264, HEVC, MPEG-4 o AV1; si no, se convierte a
H.264.

El campo `audio` puede ser cualquier fichero con una pista de audio. Los mp3,
wav, flac y ogg se leen directamente, como hace UVR; del resto (mp4, mkv, webm,
m4a...) se extrae antes el audio con `ffmpeg`. Si el fichero ya es un mp4, el
vídeo sin audio se guarda como `<audio>_video.mp4`.

Si los dos mp3 ya existen no se repite la separación (salvo con `--force`).

## Cómo funciona

`karaokame/uvr_mdxc.py` reproduce sin interfaz gráfica la separación MDX23C de
UVR: carga la misma red (`lib_v5/tfc_tdf_v3.py` del repositorio) y los mismos
pesos, trocea el audio en ventanas solapadas, las pasa por el modelo y promedia
los solapes. El modelo devuelve directamente las dos pistas, voz e instrumental.

Ese script se ejecuta como proceso aparte con el intérprete del entorno virtual
de UVR (`ultimatevocalremovergui/.venv`), que es el que tiene PyTorch con GPU;
el `.venv` de karaokame solo tiene la variante de CPU. Si ese entorno no existe
se usa el intérprete actual. Si la GPU no tiene memoria libre (por ejemplo,
porque otro programa la ocupa), la separación se repite en CPU, mucho más
lenta; `karaokame.py` descarga antes el modelo de Ollama para evitarlo. En una RTX 3050 una canción de 4 minutos tarda algo
más de 3 minutos.

Con la canción de prueba, el resultado es idéntico muestra a muestra al que
produce la interfaz de UVR con la misma configuración.
