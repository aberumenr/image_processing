import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def translate_set(binary_set, shift_x, shift_y):
    translated = np.zeros_like(binary_set)

    points_y, points_x = np.where(binary_set == 255)

    new_x = points_x + shift_x
    new_y = points_y + shift_y

    valid = (
        (new_x >= 0)
        & (new_x < binary_set.shape[1])
        & (new_y >= 0)
        & (new_y < binary_set.shape[0])
    )

    translated[new_y[valid], new_x[valid]] = 255

    return translated


def erosion_manual(image, element):
    height, width = image.shape
    element_height, element_width = element.shape

    radius_y = element_height // 2
    radius_x = element_width // 2

    padded = np.pad(
        image,
        ((radius_y, radius_y), (radius_x, radius_x)),
        mode="constant",
        constant_values=0
    )

    result = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            region = padded[
                y:y + element_height,
                x:x + element_width
            ]

            values = region[element == 1]

            if np.all(values == 255):
                result[y, x] = 255

    return result


def dilation_manual(image, element):
    height, width = image.shape
    element_height, element_width = element.shape

    radius_y = element_height // 2
    radius_x = element_width // 2

    padded = np.pad(
        image,
        ((radius_y, radius_y), (radius_x, radius_x)),
        mode="constant",
        constant_values=0
    )

    result = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            region = padded[
                y:y + element_height,
                x:x + element_width
            ]

            values = region[element == 1]

            if np.any(values == 255):
                result[y, x] = 255

    return result


folder = Path(__file__).resolve().parent
image_path = folder / "nemesis.png"

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        f"could not open: {image_path}"
    )

gray_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

threshold_value = 128

_, binary_image = cv2.threshold(
    gray_image,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)

height, width = binary_image.shape

# grabbing the middle part as b
x1 = width // 4
x2 = 3 * width // 4
y1 = height // 4
y2 = 3 * height // 4

B = binary_image[y1:y2, x1:x2]

# flipping both directions gives us b hat
B_hat = B[::-1, ::-1]

padding = 100

B_canvas = np.zeros(
    (
        B.shape[0] + 2 * padding,
        B.shape[1] + 2 * padding
    ),
    dtype=np.uint8
)

B_canvas[
    padding:padding + B.shape[0],
    padding:padding + B.shape[1]
] = B

shift_x = 80
shift_y = 40

B_translated = translate_set(
    B_canvas,
    shift_x,
    shift_y
)

# small b for the morphology operations
morphology_B = np.ones(
    (5, 5),
    dtype=np.uint8
)

erosion = erosion_manual(
    binary_image,
    morphology_B
)

dilation = dilation_manual(
    binary_image,
    morphology_B
)

fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10)
)

axes[0, 0].imshow(
    binary_image,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[0, 0].set_title("binary image")

axes[0, 1].imshow(
    B,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[0, 1].set_title("set B")

axes[0, 2].imshow(
    B_hat,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[0, 2].set_title(r"reflection $\hat{B}$")

axes[1, 0].imshow(
    B_translated,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[1, 0].set_title(
    rf"translation $(B)_z$, $z=({shift_x}, {shift_y})$"
)

axes[1, 1].imshow(
    erosion,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[1, 1].set_title("manual erosion — 5 × 5")

axes[1, 2].imshow(
    dilation,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[1, 2].set_title("manual dilation — 5 × 5")

for axis in axes.flat:
    axis.axis("off")

plt.tight_layout()
plt.show()