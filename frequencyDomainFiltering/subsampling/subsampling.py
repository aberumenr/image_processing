import cv2
import matplotlib.pyplot as plt
from pathlib import Path

folder = Path(__file__).resolve().parent
imagePath = folder / "Fine stripes.jpg"

print("trying to open:", imagePath)

imagen = cv2.imread(str(imagePath))

if imagen is None:
    raise FileNotFoundError(f"could not open: {imagePath}")

imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

factor = 8

subsampling = imagen[::factor, ::factor]

kernel = 25
sigma = 4

filteredImage = cv2.GaussianBlur(
    imagen,
    (kernel, kernel),
    sigmaX=sigma
)

subsampling_gaussiano = filteredImage[::factor, ::factor]

fig, ejes = plt.subplots(1, 3, figsize=(16, 7))

ejes[0].imshow(imagen)
ejes[0].set_title(
    f"original pic\n{imagen.shape[1]} × {imagen.shape[0]} pixels"
)

ejes[1].imshow(
    subsampling,
    interpolation="nearest"  
)
ejes[1].set_title(
    f"subsampling, factor {factor}\n"
    f"{subsampling.shape[1]} × {subsampling.shape[0]} pixels"
)

ejes[2].imshow(
    subsampling_gaussiano,
    interpolation="nearest"
)
ejes[2].set_title(
    f"gaussian filter\n"
    f"{subsampling_gaussiano.shape[1]} × "
    f"{subsampling_gaussiano.shape[0]} pixels"
)

for eje in ejes:
    eje.axis("off")

plt.tight_layout()
plt.show()