

import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from pathlib import Path


def cargar_degradacion():
    ruta = Path(__file__).resolve().parent / "Degradacion de imagen.py"
    especificacion = importlib.util.spec_from_file_location("degradacion", ruta)
    if especificacion is None or especificacion.loader is None:
        raise ImportError(f"No se pudo cargar {ruta}")
    modulo = importlib.util.module_from_spec(especificacion)
    especificacion.loader.exec_module(modulo)
    return modulo


def filtro_wiener(g, h, K=0.01):
    G = np.fft.fft2(g)
    G_centrada = np.fft.fftshift(G)

    h_ampliada = np.zeros_like(g, dtype=float)
    kh, kw = h.shape
    h_ampliada[:kh, :kw] = h
    h_ampliada = np.roll(
        h_ampliada,
        (-(kh // 2), -(kw // 2)),
        axis=(0, 1)
)
    H = np.fft.fft2(h_ampliada)
    H_centrada = np.fft.fftshift(H)

    W = np.conjugate(H_centrada) / (np.abs(H_centrada) ** 2 + K)
    F_estimada_centrada = W * G_centrada
    F_estimada = np.fft.ifftshift(F_estimada_centrada)
    imagen_filtrada = np.real(np.fft.ifft2(F_estimada))
    return np.clip(imagen_filtrada, 0, 255), G_centrada, W


ruta_imagen = Path(__file__).resolve().parent / "kitty.jpg"
imagen = plt.imread(ruta_imagen)

if imagen.ndim == 3:
    imagen = np.mean(imagen[:, :, :3], axis=2)
if imagen.max() <= 1:
    imagen = imagen * 255

degradacion = cargar_degradacion()
g, h, _, _ = degradacion.degradar_imagen(imagen)
imagen_filtrada, G_centrada, W = filtro_wiener(g, h)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.imshow(imagen, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen original f(x,y)")
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(g, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen degradada g(x,y)")
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(np.log1p(np.abs(G_centrada)), cmap="gray")
plt.title("Espectro centrado G(u,v)")
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(imagen_filtrada, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen filtrada (Wiener)")
plt.axis("off")
plt.tight_layout()
plt.show()


