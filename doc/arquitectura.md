# Arquitectura

## Idea general

La letra ya se conoce (viene en el cfg), así que el problema no es transcribir
sino **alinear**: averiguar cuándo se canta cada sílaba y con qué tono. Cada
modelo hace solo aquello en lo que es fiable:

| Pieza | Modelo | Qué aporta |
|---|---|---|
| Voz y acompañamiento | MDX23C (Ultimate Vocal Remover) | Separa la canción completa en dos pistas |
| Tiempos de palabra | Whisper (`faster-whisper`) | Cuándo suena cada palabra en la pista de voz |
| Tono y sonoridad | CREPE (`torchcrepe`) | Nota cantada cada 10 ms y dónde hay voz |
| Tempo | `librosa` | BPM del acompañamiento |
| Metadatos y revisión | Qwen3 4B (Ollama) | Artista, título, género, año y segunda opinión sobre líneas dudosas |

Todo lo numérico (tiempos, tonos, beats) es determinista. El LLM no orquesta ni
modifica tiempos: un modelo de 4B no es fiable manipulando cientos de números y
no puede oír el audio. Su papel es el de revisor de texto, y el pipeline
funciona igual sin él (`--no-llm`).

## Flujo

```
cfg ─┬─ lead (voz) ──┬─ CREPE ───── tono + sonoridad + ataques ─┐
     │               └─ Whisper ─── palabras oídas con tiempos ─┤
     ├─ lyrics ───── silabeo ────── palabras y sílabas ─────────┼─ alineación ─ notas ─┐
     ├─ accompaniment ─ librosa ─── BPM ────────────────────────┘                      ├─ songs/<name>.txt
     │               └──────────── copia ─────────────────────────────────────────────── songs/<name>.mp3
     └─ name ─────── Qwen ───────── metadatos + revisión ──────────────────────────────┴─ work/<name>/report.md
```

## Etapas

### 0. Preparación del cfg (`pipeline.prepare`)

Lanzado sin argumentos, el script localiza la canción de `data/` y crea su cfg
(`karaokame/discover.py`). Después se completa lo que le falte al cfg. Si no tiene `lead` y
`accompaniment` pero sí `audio`, se separan con el modelo MDX23C de Ultimate
Vocal Remover (ver [vocalizame](vocalizame.md)), y de paso se extrae el vídeo o
la carátula que lleve el fichero. Si no tiene letra, se
transcribe la voz con Whisper y esa transcripción, partida en líneas por las
pausas y la puntuación, hace de letra. Si no tiene `language`, Qwen lo deduce de la
letra (sin LLM, lo detecta Whisper sobre la voz, que con voz cantada falla más). Todo lo generado se escribe en el cfg, de modo
que la siguiente ejecución no lo repite.

### 1. Tono de la voz (`karaokame/pitch.py`)

CREPE analiza la voz a 16 kHz con un valor cada 10 ms y decodificación Viterbi,
que evita saltos de octava entre frames. Un frame se considera *con voz* si la
periodicidad de CREPE supera 0,4 y su energía está a menos de 45 dB del máximo;
después se rellenan huecos de menos de 60 ms y se descartan islas de menos de
50 ms. También se detectan los ataques (*onsets*) de la voz, que se usan luego
para separar sílabas.

### 2. Palabras oídas (`karaokame/transcribe.py`)

Whisper transcribe la pista de voz con tiempos por palabra. Se le pasa el
comienzo de la letra como *prompt* para orientar el vocabulario y se desactiva
el contexto entre segmentos, que en los estribillos repetidos provoca bucles.

Después se corrige con la sonoridad de CREPE lo que Whisper suele hacer mal en
voz cantada:

- Una palabra de más de 2 s casi siempre es una palabra estirada sobre algo que
  no se oyó: se recorta al final del tramo de voz en el que empieza.
- **Segunda pasada.** Los huecos de más de 2,5 s sin palabras pero con al menos
  1 s de voz se vuelven a transcribir por separado, dando como pista las líneas
  de la letra que caen por esa zona. Así se recuperan repeticiones de estribillo
  que la primera pasada se salta.
- Se descartan las palabras de duración nula (bucles de alucinación) y, en la
  segunda pasada, las de probabilidad inferior a 0,2.

### 3. Alineación (`karaokame/align.py`)

1. **Emparejamiento.** La secuencia de palabras de la letra se alinea con la de
   Whisper mediante Needleman-Wunsch, comparando las palabras normalizadas (sin
   tildes, mayúsculas ni puntuación) con una similitud difusa. Así se toleran
   errores de transcripción ("peso" por "beso") y palabras de más o de menos.
   Las parejas con similitud ≥ 0,5 son *anclas* y heredan el tiempo de Whisper.
   Dejar una palabra de la letra sin pareja cuesta más donde no hay tiempo libre
   entre las palabras oídas; con frases repetidas, eso hace que los huecos
   caigan donde realmente caben.
2. **Interpolación.** Las palabras sin ancla se reparten, en proporción a su
   número de sílabas, sobre los tramos con voz que hay entre las anclas vecinas.
3. **Ajuste fino.** El inicio de cada palabra se pega a la transición
   silencio→voz más cercana; el final se recorta si la cola es silencio y se
   alarga mientras la voz siga sonando (notas sostenidas), sin pisar la palabra
   siguiente.
4. **Sílabas.** El tiempo de la palabra se reparte entre sus sílabas partiendo
   de un reparto uniforme y moviendo cada frontera al ataque o cambio de tono
   más cercano, si lo hay.

El silabeo (`karaokame/syllables.py`) se hace por reglas para español
(diptongos, hiatos, dígrafos, grupos consonánticos). Para otros idiomas se usan
los diccionarios de guionado de `pyphen`, que son solo una aproximación.

### 4. Notas (`karaokame/notes.py`)

- **Sinalefa.** Si una palabra acaba en vocal, la siguiente empieza por vocal y
  alguna de las dos sílabas dura menos de 160 ms, se funden en una sola nota
  ("te ha", "to en"), como en las canciones hechas a mano.
- **Tono.** Mediana del tono de CREPE en la duración de la nota, redondeada al
  semitono.
- **Melisma.** Si una sílaba contiene varios tramos de tono estable de al menos
  150 ms, se parte en varias notas; las siguientes a la primera llevan `~`.
- **Octavas.** Una nota a 9 o más semitonos de la mediana de sus vecinas se
  desplaza una octava. Una nota sin voz detectada hereda el tono de la anterior.

### 5. BPM, GAP y cuantización (`karaokame/ultrastar.py`)

El tempo se estima sobre el acompañamiento y se duplica hasta entrar en el rango
200-400. Un beat del txt dura `15 / BPM` segundos. El GAP es el instante en que
empieza la primera nota y todas las posiciones se redondean a beats relativos a
él. Toda nota dura al menos un beat y no se solapa con la anterior.

### 6. Revisión (`karaokame/llm.py`, `karaokame/pipeline.py`)

Cada línea recibe una confianza: la similitud media de sus palabras con lo
oído, rebajada cuando Whisper dudaba de la palabra y contando como 0 las
interpoladas. Una línea se marca para revisar si su confianza es inferior a
0,75 o si su ritmo es anómalo (más de 1 s o menos de 0,08 s por sílaba). Las
líneas marcadas se listan en el informe junto con lo que Whisper oyó en su
intervalo, y se envían a Qwen para que indique si lo oído corresponde a la misma
frase (`ok`, `dudosa`, `mal`). El informe avisa también de frases oídas con
claridad que no están en la letra. Qwen propone además los metadatos que no estén en el cfg; el informe
avisa de que no están verificados.

### 7. Transposición al rango (`karaokame/modulate.py`)

Si el cfg tiene `range`, se calcula el histograma de notas de la voz (tiempo
cantado en cada semitono) y se busca la transposición que deja más tiempo
cantado dentro del rango. A igualdad, gana la que menos se sale del rango y la
que menos mueve el audio.

El audio se transporta como mucho seis semitonos arriba o abajo. Cantar una
octava por encima o por debajo no requiere otro acompañamiento (y UltraStar
puntúa sin tener en cuenta la octava), así que la octava se elige aparte: una
canción de soprano para un rango de barítono se resuelve con uno o dos
semitonos y cantando una octava abajo, en vez de bajar el audio casi una
octava, que sonaría mal. Las notas del txt transportado se escriben ya en la
octava en la que se va a cantar.

Si transportar mejora menos de dos puntos porcentuales respecto a no hacerlo,
no se transporta y no se genera nada más. El informe
recoge el rango pedido, el de la voz original, la transposición elegida y el
porcentaje de tiempo que queda dentro, y avisa si es inferior al 90 %.

## Limitaciones conocidas

- La precisión de los tiempos de palabra de Whisper ronda los 100-200 ms; el
  ajuste a la sonoridad lo mejora al inicio de frase, pero no dentro de un
  legato.
- Si Whisper no oye una frase entera (coros, voz muy procesada), sus tiempos son
  interpolados y hay que corregirlos a mano. El informe las señala.
- El reparto de sílabas dentro de una palabra es heurístico.
- El veredicto de Qwen solo compara textos: una línea puede estar bien oída y
  mal colocada en el tiempo, y eso no lo detecta.
- El año y el género que propone un modelo de 4B pueden ser erróneos.
- El idioma detectado es una lengua de las que conoce Whisper: un dialecto sale
  como la lengua estándar más próxima (napolitano → `it`). Conviene comprobarlo.
- Una letra transcrita por Whisper tendrá errores (palabras mal oídas, líneas
  mal cortadas); la alineación parece perfecta porque se compara consigo misma.
- No se generan notas doradas, *freestyle*, rap, duetos ni las etiquetas de medley.
