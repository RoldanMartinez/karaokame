# modulame

Analiza la voz de una canción, muestra su tonalidad y su distribución de notas,
y la transporta los semitonos que se le indiquen para adaptarla a quien canta.

`bin/karaokame.py` hace esto mismo de forma automática cuando el cfg tiene el
campo `range`. Este script sirve para ver el análisis y elegir la transposición
a mano.

## Uso

```bash
.venv/bin/python bin/modulame.py config/cancion.cfg          # interactivo
.venv/bin/python bin/modulame.py config/cancion.cfg -s -3    # directo: baja 3 semitonos
```

| Opción | Por defecto | Descripción |
|---|---|---|
| `config` | (obligatorio) | Fichero cfg de la canción, el mismo que usa karaokame |
| `-s`, `--semitones` | | Transporta sin preguntar; entre -12 y +12 |
| `--crepe-model` | `full` | `full` o `tiny`; solo importa si no hay caché del análisis |
| `--device` | `auto` | `auto`, `cpu` o `cuda` |

La canción tiene que estar ya generada con `bin/karaokame.py`: modulame parte
de `songs/<name>.txt` y reutiliza el análisis de tono de `work/<name>/`.

## Qué muestra

- **Tonalidad estimada** y dos alternativas, con su puntuación (de -1 a 1).
- **Rango de la voz**: nota más grave y más aguda, y el rango habitual (entre
  los percentiles 5 y 95 del tiempo cantado), que es el que conviene mirar para
  decidir la transposición.
- **Distribución por nota**, juntando todas las octavas, con la tónica marcada.
- **Distribución por altura**, nota a nota de aguda a grave.

Las notas se nombran en español, con Do4 como do central. En modo interactivo se
escribe un número de semitonos (`-2`, `+3`), se ve la misma pantalla ya
transportada y se confirma con `s`; se pueden probar varios valores antes de
confirmar. Intro sin número sale sin generar nada.

## Qué genera

Para una transposición de X semitonos (por ejemplo `-3`):

| Fichero | Contenido |
|---|---|
| `songs/<name>_-3.mp3` | Acompañamiento transportado |
| `songs/<name>_-3.txt` | Txt de UltraStar con todas las notas desplazadas, `#MP3` apuntando al mp3 nuevo y `(-3)` añadido al título |
| `work/<name>/<voz>_-3.mp3` | Pista de voz transportada, como referencia |

El sufijo lleva siempre signo: `_+2`, `_-3`. La duración y el tempo no cambian,
así que `#BPM`, `#GAP` y los beats son los del txt original.

## Cómo funciona

- **Distribución de notas.** Sale del análisis de CREPE de la pista de voz: cada
  frame con voz se redondea al semitono y se acumula su duración.
- **Tonalidad.** Algoritmo de Krumhansl-Schmuckler: se compara el histograma de
  las doce clases de nota con los perfiles mayor y menor de Krumhansl-Kessler en
  las doce tónicas y se ordenan las 24 tonalidades por correlación. Se basa solo
  en la melodía, así que puede confundir una tonalidad con su relativa o su
  dominante; por eso se muestran alternativas.
- **Transposición del audio.** Rubber Band a través de `ffmpeg`, que cambia el
  tono sin tocar la duración. En la voz se conservan los formantes para que el
  timbre siga sonando natural. Cuantos más semitonos, más se nota el procesado.
