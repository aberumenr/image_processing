# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:57:40 2026

@author: martin.ruiz
"""
import struct

COLORES = {
    '.': None,
    'R': (220, 20, 30),       # Rojo
    'S': (110, 30, 20),       # Rojo oscuro
    'P': (255, 190, 130),     # Piel
    'K': (20, 20, 20),        # Negro
    'B': (30, 60, 180),       # Azul
    'C': (120, 65, 30),       # Café
    'G': (30, 160, 60),       # Verde
    'Y': (240, 190, 60),      # Amarillo
    'T': (110, 60, 30)        # Café oscuro
}



mono = [
    "...GGGGGG...",
    "..GGGGGGGG..",
    "..GGGGGGGG..",
    "..YYGGGGYY..",
    ".YYPPPPPPYY.",
    ".YYPPKPPKYY.",
    "..PPPPPPPP..",
    "...PPPPPP...",
    "..GGGGGGGG..",
    ".GGGGGGGGGG.",
    ".GGGGGGGGGG.",
    "..GGG..GGG..",
    "..GGG..GGG..",
    "..TT....TT..",
    "..TT....TT..",
    "............"
]

def crear_imagen(ancho, alto):
    imagen = []
    for y in range(alto):
        fila = []
        for x in range(ancho):
            # Fondo azul cielo
            color = (92, 148, 252)
            # Suelo verde
            if y >= alto - 4:
                color = (50, 130, 45)
            fila.append(color)
        imagen.append(fila)
    return imagen

# def crear_imagen(ancho, alto):
#     imagen = []
#     for y in range(alto):
#         fila = []
#         for x in range(ancho):
#             if x == 0:
#                 color = (255, 0, 0)  # Rojo
#             elif x == 1:
#                 color = (0, 255, 0)  # Verde
#             else:
#                 color = (0, 0, 255)  # Azul
#             fila.append(color)
#         imagen.append(fila)
#     return imagen

def pintar_sprite(imagen, sprite, inicio_x, inicio_y):
    alto = len(imagen)
    ancho = len(imagen[0])
    for y, fila in enumerate(sprite):
        for x, pixel in enumerate(fila):
            color = COLORES[pixel]
            posicion_x = inicio_x + x
            posicion_y = inicio_y + y
            if color is not None:
                if (0 <= posicion_x < ancho and
                    0 <= posicion_y < alto):
                    imagen[posicion_y][posicion_x] = color
                    

def calcular_padding(ancho, bits=24):
    bytes_fila = (ancho * bits + 7) // 8
    padding = (4 - bytes_fila % 4) % 4
    return padding

def crear_cabeceras(ancho, alto):
    bits = 24
    padding = calcular_padding(ancho, bits)
    bytes_fila = ancho * 3 + padding
    image_size = bytes_fila * alto
    offset = 54
    file_size = offset + image_size
    # File Header: 14 bytes
    file_header = struct.pack("<2sIHHI",b"BM",file_size,0,0,offset)
    # DIB Header: 40 bytes
    dib_header = struct.pack("<IiiHHIIiiII",40,ancho,alto,1,bits,0,image_size,2835,2835,0,0)

    return file_header + dib_header


def codificar_pixeles(imagen):
    alto = len(imagen)
    ancho = len(imagen[0])
    padding = calcular_padding(ancho)
    datos = bytearray()
    # BMP almacena las filas desde abajo
    for fila in reversed(imagen):
        for r, g, b in fila:
            # Conversión RGB -> BGR
            datos.extend([b, g, r])
        # Agregar bytes de relleno
        datos.extend(b"\x00" * padding)

    return datos


def guardar_bmp(imagen, nombre):
    alto = len(imagen)
    ancho = len(imagen[0])
    cabeceras = crear_cabeceras(ancho, alto)
    pixeles = codificar_pixeles(imagen)
    with open(nombre, "wb") as archivo:
        archivo.write(cabeceras)
        archivo.write(pixeles)
    print("Archivo BMP creado:", nombre)


# def main():
#     ancho = 8
#     alto = 2
#     imagen = crear_imagen(ancho, alto)
#     guardar_bmp(imagen,"imagen_generada.bmp")


def main():
    ancho = 80
    alto = 32
    # Crear fondo
    imagen = crear_imagen(ancho, alto)
    pintar_sprite(imagen,mono,52,8)
    # Generar archivo BMP
    guardar_bmp(imagen,"mono.bmp")

    print("Dimensiones:", ancho, "x", alto)
    print("Profundidad: 24 bits")
    print("Personajes: mono")

if __name__ == "__main__":
    main()


