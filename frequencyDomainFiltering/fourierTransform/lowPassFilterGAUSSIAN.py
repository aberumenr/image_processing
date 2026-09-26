import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

image_path = Path(__file__).parent / "chameleon.jpg"

image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"No se encontró la imagen: {image_path}")

rows, columns = image.shape

F = np.fft.fft2(image)
F_centered = np.fft.fftshift(F)

original_spectrum = np.log1p(np.abs(F_centered))

sigma = 50

u, v = np.ogrid[:rows, :columns]

D = np.sqrt(
    (u - rows / 2) ** 2 +
    (v - columns / 2) ** 2
)

H = np.exp(-(D ** 2) / (2 * sigma ** 2))

G_centered = F_centered * H

G = np.fft.ifftshift(G_centered)
filtered_image = np.real(np.fft.ifft2(G))

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap="gray")

plt.subplot(2, 2, 2)
plt.imshow(original_spectrum, cmap="gray")

plt.subplot(2, 2, 3)
plt.imshow(H, cmap="gray")

plt.subplot(2, 2, 4)
plt.imshow(filtered_image, cmap="gray")

plt.subplots_adjust(hspace=0.25, wspace=0.20)
plt.show()

print("Sigma:", sigma)
print("Intensidad mínima:", np.min(filtered_image))
print("Intensidad máxima:", np.max(filtered_image))