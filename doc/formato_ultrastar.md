# Formato del txt generado

Resumen de cómo karaokame rellena el formato de UltraStar WorldParty.

## Cabecera

| Etiqueta | Origen |
|---|---|
| `#ARTIST`, `#TITLE` | cfg (`artist`, `title`) o propuesta del LLM a partir de `name` |
| `#LANGUAGE` | `language` del cfg: `es` → `Español` (también `ca`, `gl`, `eu`); el resto en inglés (`English`, `French`...) |
| `#GENRE`, `#YEAR` | cfg o propuesta del LLM; se omiten si no se conocen |
| `#CREATOR` | cfg (`creator`) u opción `--creator` |
| `#MP3` | `<name>.mp3`, la copia del acompañamiento |
| `#COVER` | `<name>.jpg`, copia del fichero `cover` del cfg, si lo hay |
| `#VIDEO` | `<name>.mp4`, copia del fichero `video` del cfg, si lo hay |
| `#BPM` | Tempo del acompañamiento, duplicado hasta el rango 200-400, con coma decimal |
| `#GAP` | Milisegundos hasta el inicio de la primera nota, con coma decimal |

No se generan `#BACKGROUND`, `#MEDLEYSTARTBEAT` ni `#MEDLEYENDBEAT`: hay que
añadirlas a mano. En la versión transportada (`<name>_X.txt`) las notas están
desplazadas, `#MP3` apunta a `<name>_X.mp3` y el título lleva `(X)` al final. El nombre del mp3 sigue el `name`
del cfg y no la convención `Artista - Título.mp3`.

## Notas

```
: 0 3 6 A
│ │ │ │ └─ sílaba
│ │ │ └─── tono: semitonos respecto al do central (C4 = 0)
│ │ └───── duración en beats
│ └─────── beat de inicio, relativo al GAP
└───────── tipo: karaokame solo genera notas normales (:)
```

- Duración de un beat: `15 / BPM` segundos. Instante de una nota:
  `GAP + beat × 15000 / BPM` ms.
- La última sílaba de cada palabra lleva un espacio detrás, salvo al final de
  la línea.
- Una sinalefa se escribe como una sola nota con las dos sílabas: `te ha`.
- Un cambio de tono dentro de una sílaba añade notas con texto `~`.
- `- N` marca el cambio de línea en el beat N, situado entre el final de una
  línea y el comienzo de la siguiente.
- `E` cierra la canción.

El fichero se escribe en UTF-8 sin BOM, con saltos de línea `\n`.
