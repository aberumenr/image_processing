import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_gaussian_kernel(size, sigma):
    if size % 2 == 0:
        raise ValueError("kernel size needs to be odd")

    radius = size // 2
    coordinates = np.arange(-radius, radius + 1)

    kernel = np.exp(
        -(coordinates ** 2) / (2 * sigma ** 2)
    )

    kernel = kernel / np.sum(kernel)

    return kernel


def gaussian_filter_manual(image, kernel_size, sigma):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    radius = kernel_size // 2

    image_float = image.astype(np.float32)

    padded_horizontal = np.pad(
        image_float,
        ((0, 0), (radius, radius), (0, 0)),
        mode="reflect"
    )

    horizontal_result = np.zeros_like(
        image_float,
        dtype=np.float32
    )

    for k in range(kernel_size):
        horizontal_result += (
            kernel[k]
            * padded_horizontal[
                :,
                k:k + image.shape[1],
                :
            ]
        )

    padded_vertical = np.pad(
        horizontal_result,
        ((radius, radius), (0, 0), (0, 0)),
        mode="reflect"
    )

    filtered_result = np.zeros_like(
        image_float,
        dtype=np.float32
    )

    for k in range(kernel_size):
        filtered_result += (
            kernel[k]
            * padded_vertical[
                k:k + image.shape[0],
                :,
                :
            ]
        )

    return np.clip(
        filtered_result,
        0,
        255
    ).astype(np.uint8)

folder = Path(__file__).resolve().parent
image_path = folder / "zebraBatman.jpg"

imagen = cv2.imread(str(image_path))

if imagen is None:
    raise FileNotFoundError(
        f"Could not open: {image_path}"
    )

imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

factor = 4

subsampling = imagen[::factor, ::factor]


kernel_size = 13
sigma = 2

filtered_image = gaussian_filter_manual(
    imagen,
    kernel_size,
    sigma
)

filtered_subsampling = filtered_image[
    ::factor,
    ::factor
]

fig, axes = plt.subplots(1, 3, figsize=(16, 7))

axes[0].imshow(imagen)
axes[0].set_title(
    f"original image\n"
    f"{imagen.shape[1]} × {imagen.shape[0]} pixels"
)

axes[1].imshow(
    subsampling,
    interpolation="nearest"
)
axes[1].set_title(
    f"sampling without filtering\n"
    f"factor {factor}: "
    f"{subsampling.shape[1]} × "
    f"{subsampling.shape[0]} pixels"
)

axes[2].imshow(
    filtered_subsampling,
    interpolation="nearest"
)
axes[2].set_title(
    f"gaussian filter before sampling\n"
    f"{kernel_size} × {kernel_size}, "
    f"σ = {sigma}"
)

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()