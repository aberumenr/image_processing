import numpy as np
import matplotlib.pyplot as plt

def convolucion_2d(I, h):
    filas, columnas = I.shape
    kh, kw = h.shape
    pad_h = kh // 2
    pad_w = kw // 2

    # Padding de la imagen
    I_pad = np.pad(I,((pad_h, pad_h), (pad_w, pad_w)),mode="edge")
    resultado = np.zeros_like(I, dtype=float)

    for i in range(filas):
        for j in range(columnas):

            ventana = I_pad[i:i+kh,j:j+kw]

            resultado[i, j] = np.sum(ventana * h)

    return resultado


def degradar_imagen(I, sigma_blur=4, sigma_ruido=20):
    I = np.asarray(I, dtype=float)
    tamaño = int(6 * sigma_blur + 1)

    if tamaño % 2 == 0:
        tamaño += 1

    centro = tamaño // 2
    x = np.arange(-centro, centro + 1)
    X, Y = np.meshgrid(x, x)
    h = np.exp(-(X**2 + Y**2) / (2 * sigma_blur**2))
    h = h / np.sum(h)

    I_blur = convolucion_2d(I, h)
    n = np.random.normal(loc=0, scale=sigma_ruido, size=I.shape)
    g = np.clip(I_blur + n, 0, 255)

    return g, h, I_blur, n


if __name__ == "__main__":
    imagen = plt.imread("kitty.jpg")

    if imagen.ndim == 3:
        I = np.mean(imagen[:, :, :3], axis=2)
    else:
        I = imagen.astype(float)

    if I.max() <= 1:
        I = I * 255

    g, h, I_blur, n = degradar_imagen(I)

    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(I, cmap="gray")
    plt.title("Imagen original I(x,y)")
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.imshow(I_blur, cmap="gray")
    plt.title("Desenfoque h(x,y) * I(x,y)")
    plt.axis("off")
    plt.subplot(2, 2, 3)
    plt.imshow(n, cmap="gray")
    plt.title("Ruido blanco n(x,y)")
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.imshow(g, cmap="gray")
    plt.title("Imagen degradada g(x,y)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    