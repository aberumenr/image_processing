import struct


def crear_imagen(ancho, alto, color):
    return [
        [color for _ in range(ancho)]
        for _ in range(alto)
    ]


def pintar_pixel(imagen, x, y, color):
    alto = len(imagen)
    ancho = len(imagen[0])

    if 0 <= x < ancho and 0 <= y < alto:
        imagen[y][x] = color


def rectangulo(imagen, x1, y1, x2, y2, color):
    for y in range(y1, y2 + 1):
        for x in range(x1, x2 + 1):
            pintar_pixel(imagen, x, y, color)


def linea(imagen, x1, y1, x2, y2, color, grosor=1):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1

    error = dx - dy

    while True:
        radio = grosor // 2

        for py in range(-radio, radio + 1):
            for px in range(-radio, radio + 1):
                pintar_pixel(
                    imagen,
                    x1 + px,
                    y1 + py,
                    color
                )

        if x1 == x2 and y1 == y2:
            break

        error_doble = 2 * error

        if error_doble > -dy:
            error -= dy
            x1 += sx

        if error_doble < dx:
            error += dx
            y1 += sy


def punto_en_poligono(x, y, puntos):
    dentro = False
    j = len(puntos) - 1

    for i in range(len(puntos)):
        xi, yi = puntos[i]
        xj, yj = puntos[j]

        cruza = (
            (yi > y) != (yj > y)
            and x
            < (
                (xj - xi)
                * (y - yi)
                / (yj - yi + 0.000001)
                + xi
            )
        )

        if cruza:
            dentro = not dentro

        j = i

    return dentro


def poligono(imagen, puntos, color):
    minimo_x = min(x for x, y in puntos)
    maximo_x = max(x for x, y in puntos)

    minimo_y = min(y for x, y in puntos)
    maximo_y = max(y for x, y in puntos)

    for y in range(minimo_y, maximo_y + 1):
        for x in range(minimo_x, maximo_x + 1):
            if punto_en_poligono(
                x + 0.5,
                y + 0.5,
                puntos
            ):
                pintar_pixel(
                    imagen,
                    x,
                    y,
                    color
                )


def dibujar_glam_batman(imagen):
    negro = (13, 13, 18)
    gris_oscuro = (42, 44, 53)
    gris = (90, 94, 105)
    plata = (165, 170, 180)
    blanco = (240, 238, 230)
    piel = (218, 157, 120)
    rojo = (150, 20, 42)
    morado = (105, 30, 145)
    rosa = (220, 30, 135)
    amarillo = (235, 190, 45)

    # cape
    poligono(
        imagen,
        [
            (54, 58),
            (36, 72),
            (18, 119),
            (37, 105),
            (50, 121),
            (61, 96),
            (105, 96),
            (119, 121),
            (132, 104),
            (146, 117),
            (128, 70),
            (107, 57)
        ],
        negro
    )

    # legs
    rectangulo(
        imagen,
        65,
        113,
        80,
        143,
        gris_oscuro
    )

    rectangulo(
        imagen,
        84,
        113,
        99,
        143,
        gris_oscuro
    )

    # boots
    rectangulo(
        imagen,
        63,
        135,
        81,
        143,
        negro
    )

    rectangulo(
        imagen,
        83,
        135,
        101,
        143,
        negro
    )

    linea(
        imagen,
        65,
        138,
        79,
        138,
        plata,
        2
    )

    linea(
        imagen,
        85,
        138,
        99,
        138,
        plata,
        2
    )

    linea(
        imagen,
        66,
        141,
        72,
        136,
        plata
    )

    linea(
        imagen,
        72,
        136,
        79,
        141,
        plata
    )

    linea(
        imagen,
        85,
        141,
        92,
        136,
        plata
    )

    linea(
        imagen,
        92,
        136,
        99,
        141,
        plata
    )

    # torso
    poligono(
        imagen,
        [
            (57, 63),
            (108, 63),
            (113, 112),
            (52, 112)
        ],
        gris_oscuro
    )

    # left side of the vest
    poligono(
        imagen,
        [
            (61, 65),
            (76, 67),
            (70, 102),
            (57, 105)
        ],
        negro
    )

    # right side of the vest
    poligono(
        imagen,
        [
            (104, 65),
            (89, 67),
            (94, 102),
            (108, 105)
        ],
        negro
    )

    # open chest
    poligono(
        imagen,
        [
            (76, 68),
            (89, 68),
            (93, 93),
            (82, 101),
            (71, 93)
        ],
        piel
    )

    linea(
        imagen,
        82,
        75,
        82,
        94,
        rojo,
        2
    )

    # silver vest edges
    linea(
        imagen,
        62,
        67,
        72,
        96,
        plata,
        2
    )

    linea(
        imagen,
        103,
        67,
        93,
        96,
        plata,
        2
    )

    # vest studs
    for x, y in [
        (65, 72),
        (68, 82),
        (71, 92),
        (100, 72),
        (97, 82),
        (94, 92)
    ]:
        rectangulo(
            imagen,
            x - 1,
            y - 1,
            x + 1,
            y + 1,
            blanco
        )

    # left shoulder spikes
    poligono(
        imagen,
        [
            (56, 62),
            (44, 68),
            (47, 58),
            (39, 55),
            (50, 53),
            (52, 43),
            (58, 55),
            (65, 61)
        ],
        gris
    )

    # right shoulder spikes
    poligono(
        imagen,
        [
            (109, 62),
            (121, 68),
            (118, 58),
            (126, 55),
            (115, 53),
            (113, 43),
            (107, 55),
            (100, 61)
        ],
        gris
    )

    # arms
    linea(
        imagen,
        57,
        72,
        42,
        98,
        gris_oscuro,
        9
    )

    linea(
        imagen,
        108,
        72,
        120,
        101,
        gris_oscuro,
        9
    )

    # silver outline behind the cowl
    poligono(
        imagen,
        [
            (58, 21),
            (63, 3),
            (72, 20),
            (93, 20),
            (102, 3),
            (107, 21),
            (108, 59),
            (100, 68),
            (65, 68),
            (57, 59)
        ],
        gris
    )

    # black bat cowl
    poligono(
        imagen,
        [
            (60, 20),
            (64, 7),
            (70, 22),
            (95, 22),
            (101, 7),
            (105, 20),
            (105, 58),
            (99, 65),
            (66, 65),
            (60, 58)
        ],
        negro
    )

    # ear highlights
    linea(
        imagen,
        64,
        7,
        70,
        23,
        plata,
        2
    )

    linea(
        imagen,
        101,
        7,
        95,
        23,
        plata,
        2
    )

    # white glam face paint
    poligono(
        imagen,
        [
            (66, 30),
            (75, 25),
            (82, 30),
            (89, 25),
            (99, 30),
            (97, 47),
            (92, 49),
            (89, 54),
            (83, 49),
            (77, 54),
            (74, 49),
            (67, 47)
        ],
        blanco
    )

    # eye makeup
    poligono(
        imagen,
        [
            (68, 34),
            (80, 32),
            (78, 39),
            (69, 40)
        ],
        negro
    )

    poligono(
        imagen,
        [
            (85, 32),
            (98, 34),
            (96, 40),
            (87, 39)
        ],
        negro
    )

    # eye highlights
    rectangulo(
        imagen,
        72,
        35,
        76,
        36,
        gris
    )

    rectangulo(
        imagen,
        89,
        35,
        93,
        36,
        gris
    )

    # mouth
    rectangulo(
        imagen,
        73,
        48,
        93,
        59,
        piel
    )

    rectangulo(
        imagen,
        78,
        53,
        89,
        57,
        negro
    )

    rectangulo(
        imagen,
        80,
        53,
        87,
        54,
        blanco
    )

    rectangulo(
        imagen,
        81,
        57,
        87,
        58,
        rojo
    )

    # utility belt
    rectangulo(
        imagen,
        54,
        105,
        111,
        112,
        negro
    )

    for x in range(59, 108, 8):
        rectangulo(
            imagen,
            x,
            107,
            x + 2,
            109,
            plata
        )

    # yellow buckle
    rectangulo(
        imagen,
        76,
        104,
        89,
        113,
        amarillo
    )

    # bat symbol on the buckle
    poligono(
        imagen,
        [
            (77, 108),
            (80, 106),
            (82, 108),
            (86, 106),
            (89, 108),
            (86, 109),
            (83, 112),
            (80, 109)
        ],
        negro
    )

    # guitar neck
    linea(
        imagen,
        35,
        83,
        96,
        107,
        negro,
        11
    )

    linea(
        imagen,
        35,
        83,
        96,
        107,
        gris,
        7
    )

    # guitar head
    poligono(
        imagen,
        [
            (25, 74),
            (42, 78),
            (38, 91),
            (21, 84)
        ],
        gris
    )

    # tuning pegs
    rectangulo(imagen, 23, 76, 27, 78, plata)
    rectangulo(imagen, 21, 81, 25, 83, plata)
    rectangulo(imagen, 36, 86, 40, 88, plata)

    # guitar strings
    linea(
        imagen,
        27,
        78,
        99,
        105,
        blanco
    )

    linea(
        imagen,
        26,
        81,
        98,
        108,
        plata
    )

    linea(
        imagen,
        26,
        84,
        96,
        111,
        negro
    )

    # star guitar body
    poligono(
        imagen,
        [
            (102, 89),
            (108, 101),
            (124, 97),
            (117, 111),
            (130, 123),
            (112, 121),
            (106, 139),
            (98, 122),
            (82, 129),
            (91, 113),
            (78, 103),
            (96, 102)
        ],
        plata
    )

    # guitar center
    poligono(
        imagen,
        [
            (102, 99),
            (115, 106),
            (111, 120),
            (97, 118),
            (92, 107)
        ],
        gris_oscuro
    )

    # glam guitar details
    rectangulo(
        imagen,
        101,
        106,
        105,
        112,
        rosa
    )

    rectangulo(
        imagen,
        109,
        108,
        113,
        114,
        morado
    )

    # hand over the guitar
    rectangulo(
        imagen,
        90,
        99,
        99,
        108,
        negro
    )


def calcular_padding(ancho, bits=24):
    bytes_fila = (
        ancho * bits + 7
    ) // 8

    padding = (
        4 - bytes_fila % 4
    ) % 4

    return padding


def crear_cabeceras(ancho, alto):
    bits = 24
    padding = calcular_padding(ancho, bits)

    bytes_fila = ancho * 3 + padding
    image_size = bytes_fila * alto

    offset = 54
    file_size = offset + image_size

    file_header = struct.pack(
        "<2sIHHI",
        b"BM",
        file_size,
        0,
        0,
        offset
    )

    dib_header = struct.pack(
        "<IiiHHIIiiII",
        40,
        ancho,
        alto,
        1,
        bits,
        0,
        image_size,
        2835,
        2835,
        0,
        0
    )

    return file_header + dib_header


def codificar_pixeles(imagen):
    ancho = len(imagen[0])
    padding = calcular_padding(ancho)

    datos = bytearray()

    # bmp saves the rows upside down
    for fila in reversed(imagen):
        for r, g, b in fila:
            # rgb to bgr
            datos.extend([b, g, r])

        datos.extend(
            b"\x00" * padding
        )

    return datos


def guardar_bmp(imagen, nombre):
    alto = len(imagen)
    ancho = len(imagen[0])

    cabeceras = crear_cabeceras(
        ancho,
        alto
    )

    pixeles = codificar_pixeles(
        imagen
    )

    with open(nombre, "wb") as archivo:
        archivo.write(cabeceras)
        archivo.write(pixeles)

    print("bmp file created:", nombre)


def main():
    # 161 pixels gives one byte of padding
    ancho = 161
    alto = 160

    # brighter purple so the black cowl is visible
    imagen = crear_imagen(
        ancho,
        alto,
        (34, 15, 48)
    )

    # stage lights
    poligono(
        imagen,
        [
            (0, 0),
            (28, 0),
            (59, 153),
            (23, 153)
        ],
        (82, 20, 92)
    )

    poligono(
        imagen,
        [
            (132, 0),
            (160, 0),
            (139, 153),
            (103, 153)
        ],
        (82, 20, 92)
    )

    # stage floor
    rectangulo(
        imagen,
        0,
        145,
        160,
        159,
        (37, 37, 45)
    )

    dibujar_glam_batman(imagen)

    guardar_bmp(
        imagen,
        "glam_metal_batman.bmp"
    )

    padding = calcular_padding(ancho)

    print("dimensions:", ancho, "x", alto)
    print("color depth: 24 bits")
    print("character: glam metal batman")
    print("padding per row:", padding, "byte")
    print(
        "bytes per row:",
        ancho * 3 + padding
    )


if __name__ == "__main__":
    main()