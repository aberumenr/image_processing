import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def nearest_neighbor_manual(image, new_width, new_height):
    original_height, original_width, channels = image.shape

    result = np.zeros(
        (new_height, new_width, channels),
        dtype=np.uint8
    )

    scale_x = original_width / new_width
    scale_y = original_height / new_height

    for new_y in range(new_height):
        for new_x in range(new_width):

            original_x = int(new_x * scale_x)
            original_y = int(new_y * scale_y)

            result[new_y, new_x] = image[
                original_y,
                original_x
            ]

    return result


def bilinear_manual(image, new_width, new_height):
    original_height, original_width, channels = image.shape

    result = np.zeros(
        (new_height, new_width, channels),
        dtype=np.uint8
    )

    scale_x = (original_width - 1) / (new_width - 1)
    scale_y = (original_height - 1) / (new_height - 1)

    for new_y in range(new_height):
        for new_x in range(new_width):
            x = new_x * scale_x
            y = new_y * scale_y

            # coords of the four neighbor pixels
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))

            x1 = min(x0 + 1, original_width - 1)
            y1 = min(y0 + 1, original_height - 1)

            dx = x - x0
            dy = y - y0

            Q00 = image[y0, x0].astype(float)
            Q10 = image[y0, x1].astype(float)
            Q01 = image[y1, x0].astype(float)
            Q11 = image[y1, x1].astype(float)

            # bilinear interpolation
            value = (
                Q00 * (1 - dx) * (1 - dy)
                + Q10 * dx * (1 - dy)
                + Q01 * (1 - dx) * dy
                + Q11 * dx * dy
            )

            result[new_y, new_x] = np.clip(
                np.round(value),
                0,
                255
            ).astype(np.uint8)

    return result

folder = Path(__file__).resolve().parent
imagePath = folder / "kitty500.jpg"

imagen = cv2.imread(str(imagePath))

if imagen is None:
    raise FileNotFoundError(f"Could not open: {imagePath}")

imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

new_width = 1000
new_height = 1000

print("Calculating nearest-neighbor interpolation...")

nearest_neighbor = nearest_neighbor_manual(
    imagen,
    new_width,
    new_height
)

print("Calculating bilinear interpolation...")

bilinear = bilinear_manual(
    imagen,
    new_width,
    new_height
)

print("Done!")

# Convert original image to grayscale for the Fourier transform
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)

# Two-dimensional Fourier transform
transformada_fourier = np.fft.fft2(imagen_gris)

# Print the complex values
print("\nFourier transform values:")
print(transformada_fourier)

print("\nFourier transform dimensions:")
print(transformada_fourier.shape)

fourier_centrada = np.fft.fftshift(transformada_fourier)

espectro_magnitud = np.log1p(
    np.abs(fourier_centrada)
)

fig, ejes = plt.subplots(1, 4, figsize=(20, 6))

ejes[0].imshow(imagen)
ejes[0].set_title(
    f"Original\n"
    f"{imagen.shape[1]} × {imagen.shape[0]} pixels"
)

ejes[1].imshow(nearest_neighbor, interpolation="none")
ejes[1].set_title(
    f"Manual nearest neighbor\n"
    f"{nearest_neighbor.shape[1]} × "
    f"{nearest_neighbor.shape[0]} pixels"
)

ejes[2].imshow(bilinear, interpolation="none")
ejes[2].set_title(
    f"Manual bilinear interpolation\n"
    f"{bilinear.shape[1]} × "
    f"{bilinear.shape[0]} pixels"
)

ejes[3].imshow(espectro_magnitud, cmap="gray")
ejes[3].set_title(
    "Fourier magnitude spectrum\n"
    f"{transformada_fourier.shape[1]} × "
    f"{transformada_fourier.shape[0]} values"
)

for eje in ejes:
    eje.axis("off")

plt.tight_layout()
plt.show()