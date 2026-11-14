import numpy as np
import matplotlib.pyplot as plt

def create_image(width, height, channels=3):
    return np.zeros(
        (height, width, channels),
        dtype=np.uint8
    )

def generate_colors(width, height):
    x = np.arange(width)
    y = np.arange(height)

    X, Y = np.meshgrid(x, y)

    Xn = X / max(width - 1, 1)
    Yn = Y / max(height - 1, 1)

    R = 255 * Xn
    G = 255 * Yn
    B = 127.5 + 127.5 * np.sin(
        2 * np.pi * Xn
    )

    image = np.stack((R, G, B), axis=2)

    return np.clip(
        image,
        0,
        255
    ).astype(np.uint8)

def prepare_bmp_pixels(image):
    # change rgb to bgr
    image_bgr = image[:, :, ::-1]

    # bmp stores rows from bottom to top
    image_bgr = image_bgr[::-1]

    height, width, channels = image_bgr.shape

    bytes_per_pixel = channels * image_bgr.dtype.itemsize
    row_bytes = width * bytes_per_pixel

    # bytes needed for the row to be a multiple of 4
    padding = (4 - row_bytes % 4) % 4

    rows = []

    for row in image_bgr:
        row_bytes_data = row.tobytes()
        row_with_padding = (
            row_bytes_data + bytes(padding)
        )
        rows.append(row_with_padding)

    pixel_data = b"".join(rows)

    return pixel_data, padding

width = 513
height = 512

image_rgb = generate_colors(width, height)

pixel_data, padding = prepare_bmp_pixels(image_rgb)

print("image shape:", image_rgb.shape)
print("padding per row:", padding)
print("pixel data bytes:", len(pixel_data))

plt.imshow(image_rgb)
plt.title(
    f"generated RGB image\n"
    f"{width} × {height} pixels"
)
plt.axis("off")
plt.tight_layout()
plt.show()