#!/usr/bin/env python3
"""Genera doc/karaokame.pdf: formato del cfg, del txt de UltraStar y opciones del script."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer, Table, TableStyle

FONTS = Path("/usr/share/fonts/truetype/dejavu")
OUTPUT = Path(__file__).resolve().parent / "karaokame.pdf"

for name, filename in [
    ("Sans", "DejaVuSans.ttf"),
    ("Sans-Bold", "DejaVuSans-Bold.ttf"),
    ("Sans-Italic", "DejaVuSans-Oblique.ttf"),
    ("Mono", "DejaVuSansMono.ttf"),
]:
    pdfmetrics.registerFont(TTFont(name, str(FONTS / filename)))
pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans-Italic")

INK = colors.HexColor("#1f2933")
ACCENT = colors.HexColor("#1d4e89")
RULE = colors.HexColor("#c5ccd3")
SHADE = colors.HexColor("#f2f4f7")

BODY = ParagraphStyle("body", fontName="Sans", fontSize=9.5, leading=14, textColor=INK, spaceAfter=6)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.5, leading=11.5, spaceAfter=0)
HEAD = ParagraphStyle("head", parent=CELL, fontName="Sans-Bold", textColor=colors.white)
TITLE = ParagraphStyle("title", fontName="Sans-Bold", fontSize=24, leading=30, textColor=ACCENT, alignment=TA_CENTER)
SUBTITLE = ParagraphStyle("subtitle", parent=BODY, fontSize=11, leading=15, alignment=TA_CENTER, spaceAfter=18)
H1 = ParagraphStyle("h1", fontName="Sans-Bold", fontSize=15, leading=19, textColor=ACCENT, spaceBefore=16, spaceAfter=8, keepWithNext=1)
H2 = ParagraphStyle("h2", fontName="Sans-Bold", fontSize=11, leading=15, textColor=INK, spaceBefore=10, spaceAfter=5, keepWithNext=1)
CODE = ParagraphStyle(
    "code", fontName="Mono", fontSize=8.5, leading=11.5, textColor=INK, backColor=SHADE,
    borderPadding=6, leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=12,
)

WIDTH = A4[0] - 40 * mm


def p(text):
    return Paragraph(text, BODY)


def code(text):
    return Preformatted(text.strip("\n"), CODE)


def table(header, rows, widths):
    data = [[Paragraph(c, HEAD) for c in header]] + [[Paragraph(c, CELL) for c in row] for row in rows]
    t = Table(data, colWidths=[w * WIDTH for w in widths], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SHADE]),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def m(text):
    return f'<font name="Mono">{text}</font>'


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Sans", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(20 * mm, 12 * mm, "karaokame · formatos y configuración")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"{doc.page}")
    canvas.restoreState()


story = [
    Spacer(1, 10 * mm),
    Paragraph("karaokame", TITLE),
    Paragraph("Formato del cfg, formato del txt de UltraStar y configuración del script", SUBTITLE),
    p(
        "karaokame genera canciones para UltraStar WorldParty a partir de la canción completa y su letra. "
        "Usa modelos locales: MDX23C (Ultimate Vocal Remover) para separar la voz del acompañamiento, Whisper para saber cuándo se canta cada palabra, "
        "CREPE para el tono de cada nota y Qwen (servido por Ollama) para proponer los metadatos y revisar "
        "las líneas dudosas. Este documento describe lo que el script lee, lo que escribe y cómo se configura."
    ),
    # ------------------------------------------------------------------ cfg
    Paragraph("1. Fichero de configuración de la canción (.cfg)", H1),
    p(
        f"Cada canción se describe con un fichero de texto en {m('config/')}. Tiene una línea por campo con "
        f"la forma {m('clave=\"valor\"')}. Las comillas son opcionales, las líneas vacías y las que empiezan "
        f"por {m('#')} se ignoran, y las claves no distinguen mayúsculas."
    ),
    code(
        '''
name="Fenesta_che_lucive.txt"
audio="/home/roldan/karaokame/data/Fenesta_che_lucive.mp3"
lead=
accompaniment=
lyrics="/home/roldan/karaokame/data/Fenesta_che_lucive.txt"
language=
intrument=
score=
range="do3_mi4"
'''
    ),
    p(
        f"Lanzado sin argumentos, el script crea este fichero él mismo a partir de la canción que haya en "
        f"{m('data/')} (ver la sección 3). "
        "Solo hacen falta el nombre y la canción completa. El script genera el resto y lo "
        f"escribe en el propio cfg: {m('lead')}, {m('accompaniment')}, {m('language')} y, si el audio los "
        f"contiene, {m('cover')} o {m('video')}. Las demás líneas no se tocan."
    ),
    Paragraph("Campos principales", H2),
    table(
        ["Campo", "Contenido", "Notas"],
        [
            [m("name"), "Nombre base de los ficheros de salida.", f"Obligatorio. Si acaba en {m('.txt')} se quita. Con el ejemplo se generan {m('songs/Fenesta_che_lucive.txt')} y {m('.mp3')}."],
            [m("audio"), "Canción completa, con voz y música.", f"Cualquier formato con audio: mp3, wav, flac, m4a o un vídeo mp4, mkv, webm. Necesario si faltan {m('lead')} y {m('accompaniment')}: de él se separan las dos pistas."],
            [m("lead"), "Audio con la pista principal aislada: la voz.", f"De aquí salen los tiempos y los tonos. Si está vacío se genera a partir de {m('audio')}."],
            [m("accompaniment"), "Audio del acompañamiento.", f"Es lo que sonará en el karaoke y de él se calcula el BPM. Si está vacío se genera a partir de {m('audio')}."],
            [m("lyrics"), "Fichero de texto UTF-8 con la letra.", "Si está vacío, se usa como letra la transcripción de Whisper. Cada línea del fichero es una línea del karaoke; las vacías se ignoran. Conviene no pasar de 45 caracteres por línea."],
            [m("language"), "Código de idioma de dos letras.", f"{m('es')}, {m('en')}, {m('fr')}, {m('de')}, {m('it')}, {m('pt')}... Decide el silabeo y la etiqueta {m('#LANGUAGE')}. Si está vacío se deduce de la letra con el LLM (sin LLM, de la voz con Whisper)."],
            [m("range"), "Rango en el que se quiere cantar.", f"Dos notas separadas por {m('_')}, por ejemplo {m('do3_mi4')}. Si está vacío no se transporta."],
        ],
        [0.17, 0.33, 0.50],
    ),
    Spacer(1, 6),
    p(
        "Las rutas pueden ser absolutas o relativas a la carpeta del cfg. El script se detiene con un "
        f"mensaje claro si falta un campo, si un fichero no existe o si {m('lyrics')} apunta a un audio."
    ),
    Paragraph("El campo range", H2),
    p(
        f"Cada nota se escribe con su nombre en español ({m('do')}, {m('re')}, {m('mi')}, {m('fa')}, "
        f"{m('sol')}, {m('la')}, {m('si')}), una alteración opcional ({m('#')} o {m('b')}) y la octava, "
        f"con {m('do4')} como do central del piano. {m('do3_mi4')} va del do por debajo del do central al "
        f"mi por encima; {m('fa#2_sib3')} también es válido."
    ),
    p(
        "Con el rango, el script busca la transposición que deja más tiempo cantado dentro de él y genera "
        "una segunda versión de la canción. El audio se mueve como mucho seis semitonos: si hace falta "
        "más, se canta una octava por encima o por debajo, que no necesita otro acompañamiento y que "
        "UltraStar puntúa igual. Si la canción ya encaja, o transportarla apenas mejora, no se genera nada más."
    ),
    Paragraph("Campos opcionales", H2),
    table(
        ["Campo", "Efecto"],
        [
            [f"{m('artist')}, {m('title')}", f"Fijan {m('#ARTIST')} y {m('#TITLE')}. Si faltan, los propone el LLM a partir de {m('name')} y del comienzo de la letra."],
            [f"{m('genre')}, {m('year')}", f"Fijan {m('#GENRE')} y {m('#YEAR')}. Si faltan, los propone el LLM; si no los conoce, la etiqueta se omite."],
            [m("creator"), f"Fija {m('#CREATOR')}. Tiene prioridad sobre la opción {m('--creator')}."],
            [m("cover"), f"Imagen de carátula. Se copia a {m('songs/')} y da la etiqueta {m('#COVER')}. La rellena el script si el audio lleva una carátula incrustada."],
            [m("video"), f"Vídeo de fondo. Se copia a {m('songs/')} y da la etiqueta {m('#VIDEO')}. Lo rellena el script si el fichero de audio contiene una pista de vídeo."],
            [f"{m('intrument')}, {m('score')}", "Reservados para más adelante. Hoy se leen y se ignoran."],
        ],
        [0.25, 0.75],
    ),
    Spacer(1, 6),
    p(
        "Los metadatos que propone el LLM no están verificados y el informe lo avisa. Para una canción "
        "definitiva es preferible escribirlos en el cfg."
    ),
    # ------------------------------------------------------------------ txt
    Paragraph("2. Fichero txt de UltraStar", H1),
    p(
        f"El resultado es {m('songs/&lt;name&gt;.txt')}, en UTF-8 sin BOM. Tiene una cabecera de etiquetas "
        f"{m('#ETIQUETA:valor')}, una línea por nota y una {m('E')} final. Este es el comienzo del fichero "
        "generado para la canción de ejemplo (artista y título son la propuesta, errónea aquí, del LLM):"
    ),
    code(
        '''
#ARTIST:Fenesta
#TITLE:Che lucive
#LANGUAGE:Italian
#GENRE:Folk
#CREATOR:karaokame
#MP3:Fenesta_che_lucive.mp3
#COVER:Fenesta_che_lucive.jpg
#BPM:323,00
#GAP:27160,00
: 0 4 -9 Fe
: 4 7 -7 ~
: 11 9 -7 ne
: 20 10 -5 sta
...
E
'''
    ),
    Paragraph("Cabecera", H2),
    table(
        ["Etiqueta", "Significado", "Cómo la rellena karaokame"],
        [
            [m("#ARTIST"), "Intérprete de la canción.", f"Campo {m('artist')} del cfg o propuesta del LLM."],
            [m("#TITLE"), "Título de la canción.", f"Campo {m('title')} del cfg o propuesta del LLM."],
            [m("#LANGUAGE"), "Idioma principal, en inglés salvo «Español».", f"A partir de {m('language')}: {m('es')}, {m('ca')}, {m('gl')} y {m('eu')} dan «Español»; el resto, English, French, German, Italian, Portuguese."],
            [m("#GENRE"), "Género musical.", f"Campo {m('genre')} o propuesta del LLM. Se omite si no se conoce."],
            [m("#YEAR"), "Año de publicación.", f"Campo {m('year')} o propuesta del LLM. Se omite si no se conoce."],
            [m("#CREATOR"), "Autor del txt.", f"Campo {m('creator')} u opción {m('--creator')}."],
            [m("#MP3"), "Fichero de audio que suena.", f"{m('&lt;name&gt;.mp3')}, la copia del acompañamiento."],
            [m("#COVER"), "Carátula.", f"{m('&lt;name&gt;.jpg')}, copia del campo {m('cover')}. Solo si el cfg lo tiene."],
            [m("#VIDEO"), "Vídeo de fondo.", f"{m('&lt;name&gt;.mp4')}, copia del campo {m('video')}. Solo si el cfg lo tiene."],
            [m("#BPM"), "Velocidad de la rejilla de beats.", "Tempo detectado en el acompañamiento, duplicado hasta quedar entre 200 y 400. Con coma decimal."],
            [m("#GAP"), "Milisegundos hasta el beat 0.", "Instante en que empieza la primera nota. Con coma decimal."],
        ],
        [0.17, 0.33, 0.50],
    ),
    Spacer(1, 6),
    p(
        f"No se generan {m('#BACKGROUND')}, {m('#MEDLEYSTARTBEAT')} ni "
        f"{m('#MEDLEYENDBEAT')}; hay que añadirlas a mano. El nombre del mp3 sigue el {m('name')} del cfg, "
        "no la convención «Artista - Título.mp3»."
    ),
    Paragraph("Líneas de nota", H2),
    p(f"Cada nota es una línea con cinco campos separados por espacios, por ejemplo {m(': 20 10 -5 sta')}:"),
    table(
        ["Campo", "Ejemplo", "Significado"],
        [
            ["Tipo", m(":"), f"{m(':')} normal, {m('*')} dorada, {m('F')} libre sin puntuar, {m('R')} rap, {m('G')} rap dorada. karaokame solo escribe notas normales."],
            ["Beat de inicio", m("20"), "Posición de la nota, en beats desde el GAP."],
            ["Duración", m("10"), "Longitud en beats. Como mínimo 1."],
            ["Tono", m("-5"), "Semitonos respecto al do central (C4 = 0). El juego compara sin tener en cuenta la octava."],
            ["Texto", m("sta"), "Sílaba que se canta."],
        ],
        [0.18, 0.12, 0.70],
    ),
    Spacer(1, 6),
    p(
        f"Un beat del txt dura {m('15 / BPM')} segundos, es decir, un cuarto del pulso indicado en "
        f"{m('#BPM')}. El instante real de una nota es {m('GAP + beat × 15000 / BPM')} milisegundos. "
        "Con BPM 323 cada beat dura 46,4 ms."
    ),
    p(
        f"La versión transportada, {m('songs/&lt;name&gt;_X.txt')} (por ejemplo {m('_-4')}), es el mismo "
        f"fichero con el tono de todas las notas desplazado, {m('#MP3')} apuntando a "
        f"{m('&lt;name&gt;_X.mp3')} y {m('(X)')} añadido al título para distinguirla en el juego. "
        "Tiempos, BPM y GAP no cambian."
    ),
    Paragraph("Otras líneas y convenciones", H2),
    table(
        ["Elemento", "Significado"],
        [
            [m("- N"), "Cambio de línea en el beat N. Se coloca entre el final de una línea y el comienzo de la siguiente."],
            [m("E"), "Fin de la canción."],
            ["Espacio final", "La última sílaba de cada palabra lleva un espacio detrás, salvo al final de la línea. Así el juego sabe dónde separar las palabras."],
            ["Sinalefa", f"Dos vocales de palabras contiguas cantadas juntas se escriben en una sola nota: {m('go el')}, {m('te ha')}."],
            [m("~"), "Cambio de tono dentro de una sílaba: la primera nota lleva el texto y las siguientes solo la virgulilla."],
        ],
        [0.2, 0.8],
    ),
    # ------------------------------------------------------------------ script
    Paragraph("3. Configuración del script", H1),
    p("El script se lanza desde la carpeta del proyecto con el intérprete del entorno virtual:"),
    code(
        '''
.venv/bin/python bin/karaokame.py [opciones] [config/cancion.cfg ...]
'''
    ),
    p(
        f"El argumento posicional {m('configs')} es la lista de ficheros cfg a procesar. Si alguna canción "
        "falla, el script sigue con las demás y termina con código de salida 1."
    ),
    Paragraph("Sin argumentos: la carpeta data/", H2),
    p(
        f"Si no se indica ningún cfg, el script busca la canción en {m('data/')}, donde debe haber una "
        f"sola, crea su cfg en {m('config/')} y la procesa."
    ),
    table(
        ["Qué busca", "Cómo lo reconoce"],
        [
            ["La canción", f"Cualquier fichero con una pista de audio, comprobado con {m('ffprobe')} y no por la extensión: mp3, wav, flac, m4a, mp4, mkv, webm... Si trae vídeo, se extrae y se usa como {m('#VIDEO')}; si trae una carátula incrustada, como {m('#COVER')}."],
            ["La letra", f"El {m('.txt')} con el mismo nombre que la canción o, si no existe, el único {m('.txt')} de la carpeta. Si no hay ninguno, se transcribe de la voz y queda en {m('work/&lt;name&gt;/lyrics_whisper.txt')} para corregirla."],
            ["Lo que ignora", f"Los ficheros generados por los propios scripts ({m('*_V')}, {m('*_A')}, {m('*_video')}), las imágenes y cualquier fichero sin audio."],
            ["La plantilla", f"Si existe {m('config/template.cfg')}, el cfg nuevo parte de él: sirve para fijar valores comunes como {m('range')} o {m('creator')}. Un cfg que ya existe no se sobrescribe."],
            ["Varias canciones", f"Se procesa la única que aún no tenga su txt en {m('songs/')}. Si hay varias pendientes, o varias ya generadas, el script se detiene y pide dejar solo una o indicar el cfg."],
        ],
        [0.2, 0.8],
    ),
    Spacer(1, 6),
    p(
        "Para cada canción hace, por orden: separar voz y acompañamiento si faltan, transcribir la letra si "
        f"falta, detectar el idioma si falta, generar el txt y el mp3, y transportar al rango si el cfg tiene {m('range')}."
    ),
    table(
        ["Opción", "Por defecto", "Descripción"],
        [
            [m("--whisper-model"), m("medium"), f"Modelo de faster-whisper: {m('small')}, {m('medium')}, {m('large-v3')}... Los mayores oyen mejor la voz cantada, pero tardan más y ocupan más disco. Se descargan la primera vez."],
            [m("--crepe-model"), m("full"), f"{m('full')} o {m('tiny')}. {m('tiny')} es mucho más rápido y algo menos preciso en el tono."],
            [m("--device"), m("auto"), f"{m('auto')}, {m('cpu')} o {m('cuda')}. {m('auto')} usa la GPU si PyTorch la detecta y, si no, la CPU."],
            [m("--no-llm"), "desactivada", "No usa Ollama. No se proponen metadatos ni se revisan las líneas dudosas; el resto funciona igual."],
            [m("--llm-model"), m("qwen3:4b-instruct-2507-q4_K_M"), "Modelo de Ollama para metadatos y revisión."],
            [m("--ollama-url"), m("http://localhost:11434"), "Dirección del servidor de Ollama."],
            [m("--no-sinalefa"), "desactivada", "Escribe una nota por sílaba aunque dos vocales de palabras contiguas se canten juntas."],
            [m("--no-melisma"), "desactivada", f"No parte las sílabas con cambios de tono en notas {m('~')}; cada sílaba queda como una sola nota."],
            [m("--creator"), m("karaokame"), f"Valor de {m('#CREATOR')} cuando el cfg no trae el campo {m('creator')}."],
            [m("--uvr-dir"), m("ultimatevocalremovergui/"), "Carpeta con la copia de Ultimate Vocal Remover que se usa para separar la voz (modelo MDX23C-InstVoc HQ, segmento 256, solape 8)."],
            [m("--force"), "desactivada", "Ignora la caché: vuelve a ejecutar Whisper y CREPE y repite la separación de voz."],
        ],
        [0.19, 0.37, 0.44],
    ),
    Paragraph("Ejemplos", H2),
    code(
        '''
# La canción que haya en data/, con los valores por defecto
.venv/bin/python bin/karaokame.py

# Una canción, con el modelo grande de Whisper y sin LLM
.venv/bin/python bin/karaokame.py --whisper-model large-v3 --no-llm config/cancion.cfg

# Prueba rápida de la canción de data/: CREPE pequeño, una nota por sílaba
.venv/bin/python bin/karaokame.py --crepe-model tiny --no-melisma --no-sinalefa
'''
    ),
    Paragraph("Salidas y caché", H2),
    table(
        ["Fichero", "Contenido"],
        [
            [m("config/&lt;name&gt;.cfg"), "Cfg creado para la canción de data/."],
            [m("songs/&lt;name&gt;.txt"), "Canción de UltraStar."],
            [m("songs/&lt;name&gt;.mp3"), "Copia del acompañamiento."],
            [m("songs/&lt;name&gt;.jpg / .mp4"), "Carátula y vídeo, si el cfg los indica."],
            [m("songs/&lt;name&gt;_X.txt / .mp3"), "Versión transportada X semitonos, si hay rango y hace falta transportar."],
            [m("&lt;audio&gt;_V.mp3 / _A.mp3"), "Voz y acompañamiento separados, junto al fichero de audio."],
            [m("work/&lt;name&gt;/report.md"), "Informe: calidad de la alineación, avisos y líneas que conviene revisar a mano."],
            [m("work/&lt;name&gt;/whisper_&lt;modelo&gt;.json"), "Caché de las palabras oídas por Whisper."],
            [m("work/&lt;name&gt;/pitch_&lt;modelo&gt;.npz"), "Caché del análisis de tono de CREPE."],
        ],
        [0.4, 0.6],
    ),
    Spacer(1, 6),
    p(
        "La caché depende de la pista de voz, del idioma y del modelo. Mientras no cambien, las ejecuciones "
        "siguientes tardan segundos, lo que permite probar opciones como las de sinalefa o melisma sin "
        f"repetir el análisis. Con {m('--force')} se recalcula todo. Como referencia, en CPU una canción de "
        "3:40 tarda unos 7 minutos en CREPE full y entre 2 y 3 en Whisper medium."
    ),
]

doc = SimpleDocTemplate(
    str(OUTPUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm, topMargin=18 * mm, bottomMargin=20 * mm,
    title="karaokame: formatos y configuración", author="karaokame",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUTPUT)
