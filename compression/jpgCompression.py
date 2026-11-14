import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def divide_blocks(image, N=8):
    height, width = image.shape

    new_height = (height // N) * N
    new_width = (width // N) * N

    adjusted_image = image[:new_height, :new_width]

    blocks = adjusted_image.reshape(
        new_height // N,
        N,
        new_width // N,
        N
    )

    blocks = blocks.transpose(0, 2, 1, 3)

    return adjusted_image, blocks


def dct_matrix(N=8):
    A = np.zeros((N, N))

    for u in range(N):
        for x in range(N):
            C = 1 / np.sqrt(2) if u == 0 else 1

            A[u, x] = (
                0.5
                * C
                * np.cos(
                    ((2 * x + 1) * u * np.pi)
                    / (2 * N)
                )
            )

    return A


def transform_dct(blocks, N=8):
    blocks = blocks.astype(float) - 128
    A = dct_matrix(N)

    F = np.matmul(
        np.matmul(A, blocks),
        A.T
    )

    return F


def inverse_dct(F, N=8):
    A = dct_matrix(N)

    reconstructed_blocks = np.matmul(
        np.matmul(A.T, F),
        A
    )

    reconstructed_blocks += 128

    return reconstructed_blocks


def reconstruct_image(blocks):
    block_rows, block_columns, N, _ = blocks.shape

    reconstructed = blocks.transpose(
        0,
        2,
        1,
        3
    )

    reconstructed = reconstructed.reshape(
        block_rows * N,
        block_columns * N
    )

    return reconstructed

folder = Path(__file__).resolve().parent
image_path = folder / "babycat.jpg"

image = cv2.imread(
    str(image_path),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(
        f"could not open: {image_path}"
    )

adjusted_image, blocks = divide_blocks(
    image,
    N=8
)

dct_blocks = transform_dct(
    blocks,
    N=8
)

reconstructed_blocks = inverse_dct(
    dct_blocks,
    N=8
)

reconstructed_float = reconstruct_image(
    reconstructed_blocks
)

absolute_error = np.abs(
    adjusted_image.astype(float)
    - reconstructed_float
)

reconstructed_image = np.clip(
    np.round(reconstructed_float),
    0,
    255
).astype(np.uint8)

# making tiny errors visible
error_image = np.zeros(
    (
        absolute_error.shape[0],
        absolute_error.shape[1],
        3
    ),
    dtype=np.uint8
)

maximum_error = absolute_error.max()

if maximum_error > 0:
    error_image[:, :, 0] = (
        absolute_error / maximum_error * 255
    ).astype(np.uint8)

block_row = 5
block_column = 5

selected_block = blocks[
    block_row,
    block_column
]

selected_dct = dct_blocks[
    block_row,
    block_column
]

print("original size:", adjusted_image.shape)
print("block matrix shape:", blocks.shape)

print("\nselected block:")
print(selected_block)

print("\ndct of selected block:")
print(np.round(selected_dct, 2))

print("\nmaximum absolute error:", maximum_error)
print("average absolute error:", absolute_error.mean())

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes[0, 0].imshow(
    adjusted_image,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[0, 0].set_title("original image")

axes[0, 1].imshow(
    reconstructed_image,
    cmap="gray",
    vmin=0,
    vmax=255
)
axes[0, 1].set_title("image reconstructed with inverse DCT")

axes[1, 0].imshow(error_image)
axes[1, 0].set_title(
    "normalized absolute error\n"
    "black = 0, red = error"
)

axes[1, 1].imshow(
    selected_block,
    cmap="gray",
    vmin=0,
    vmax=255,
    interpolation="nearest"
)
axes[1, 1].set_title(
    f"block [{block_row}, {block_column}]\n"
    "8 × 8 pixels"
)

axes[1, 1].set_xticks(
    np.arange(-0.5, 8, 1),
    minor=True
)
axes[1, 1].set_yticks(
    np.arange(-0.5, 8, 1),
    minor=True
)
axes[1, 1].grid(
    which="minor",
    color="red",
    linewidth=0.6
)

for axis in axes.flat:
    axis.set_xticks([])
    axis.set_yticks([])

plt.tight_layout()
plt.show()