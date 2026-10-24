import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def translate_set(binary_set, shift_x, shift_y):

    # making an empty image where the translated set will go
    translated_set = np.zeros_like(binary_set)

    # getting the coordinates of all the white points in b
    points_y, points_x = np.where(binary_set == 255)

    # adding the translation vector z to every point
    new_x = points_x + shift_x
    new_y = points_y + shift_y

    # keeping only the points that are still inside the image
    valid_points = (
        (new_x >= 0)
        & (new_x < binary_set.shape[1])
        & (new_y >= 0)
        & (new_y < binary_set.shape[0])
    )

    # placing the translated points in the new image
    translated_set[
        new_y[valid_points],
        new_x[valid_points]
    ] = 255

    return translated_set


folder = Path(__file__).resolve().parent
image_path = folder / "nemesis.png"

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(
        f"could not open: {image_path}"
    )

# turning the image grayscale first
gray_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

threshold_value = 128

# making the binary image
_, binary_image = cv2.threshold(
    gray_image,
    threshold_value,
    255,
    cv2.THRESH_BINARY
)

height, width = binary_image.shape

# choosing the middle section of the image as set b
x1 = width // 4
x2 = 3 * width // 4
y1 = height // 4
y2 = 3 * height // 4

B = binary_image[y1:y2, x1:x2]

# reflecting b around its center
B_hat = B[::-1, ::-1]

# adding space around b so we can actually see it move
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

# translation vector z = (x, y)
shift_x = 80
shift_y = 40

B_translated = translate_set(
    B_canvas,
    shift_x,
    shift_y
)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(
    binary_image,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[0].set_title("binary image")

axes[1].imshow(
    B,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[1].set_title("set B")

axes[2].imshow(
    B_hat,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[2].set_title(r"reflection $\hat{B}$")

axes[3].imshow(
    B_translated,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[3].set_title(
    rf"translation $(B)_z$, $z=({shift_x}, {shift_y})$"
)

for axis in axes:
    axis.axis("off")

plt.tight_layout()
plt.show()