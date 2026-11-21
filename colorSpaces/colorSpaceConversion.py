import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path


def normalize_rgb(rgb_image):
    if isinstance(rgb_image, Image.Image):
        rgb_image = np.asarray(
            rgb_image.convert("RGB")
        )

    rgb = np.asarray(rgb_image)

    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(
            "input needs to be an RGB image "
            "with shape (H, W, 3)"
        )

    if np.issubdtype(rgb.dtype, np.integer):
        rgb_normalized = (
            rgb.astype(np.float32) / 255.0
        )
    else:
        rgb_normalized = rgb.astype(np.float32)

    if (
        rgb_normalized.min() < 0.0
        or rgb_normalized.max() > 1.0
    ):
        raise ValueError(
            "RGB values must be in the range "
            "[0, 255] or [0, 1]"
        )

    return rgb_normalized


def rgb_to_cmy(rgb_image):
    rgb = normalize_rgb(rgb_image)

    cmy = 1.0 - rgb

    return cmy


def rgb_to_hsi(rgb_image):
    rgb = normalize_rgb(rgb_image)

    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    # intensity
    I = (R + G + B) / 3.0

    # saturation
    rgb_sum = R + G + B
    minimum = np.minimum(
        np.minimum(R, G),
        B
    )

    S = np.zeros_like(I)

    valid_sum = rgb_sum > 0

    S[valid_sum] = (
        1.0
        - (
            3.0 * minimum[valid_sum]
            / rgb_sum[valid_sum]
        )
    )

    # hue
    numerator = 0.5 * (
        (R - G) + (R - B)
    )

    denominator = np.sqrt(
        (R - G) ** 2
        + (R - B) * (G - B)
    )

    theta = np.zeros_like(I)

    valid_denominator = denominator > 0

    cosine_value = np.zeros_like(I)

    cosine_value[valid_denominator] = (
        numerator[valid_denominator]
        / denominator[valid_denominator]
    )

    cosine_value = np.clip(
        cosine_value,
        -1.0,
        1.0
    )

    theta[valid_denominator] = np.arccos(
        cosine_value[valid_denominator]
    )

    H = np.where(
        B <= G,
        theta,
        2.0 * np.pi - theta
    )

    # normalize hue from 0-2pi to 0-1
    H = H / (2.0 * np.pi)

    # hue does not matter when saturation is zero
    H[S == 0] = 0

    hsi = np.stack(
        (H, S, I),
        axis=2
    )

    return hsi


def main():
    folder = Path(__file__).resolve().parent
    image_path = folder / "pixiecloseup.JPG"

    if not image_path.exists():
        raise FileNotFoundError(
            f"could not find: {image_path}"
        )

    rgb_image = Image.open(
        image_path
    ).convert("RGB")

    rgb_array = np.asarray(rgb_image)

    cmy_image = rgb_to_cmy(rgb_array)
    hsi_image = rgb_to_hsi(rgb_array)

    print("image:", image_path.name)
    print("dimensions:", rgb_array.shape)

    print(
        "\nCMY range:",
        cmy_image.min(),
        "to",
        cmy_image.max()
    )

    print(
        "\nHue range:",
        hsi_image[:, :, 0].min(),
        "to",
        hsi_image[:, :, 0].max()
    )

    print(
        "Saturation range:",
        hsi_image[:, :, 1].min(),
        "to",
        hsi_image[:, :, 1].max()
    )

    print(
        "Intensity range:",
        hsi_image[:, :, 2].min(),
        "to",
        hsi_image[:, :, 2].max()
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6)
    )

    axes[0].imshow(rgb_array)
    axes[0].set_title("original RGB image")

    axes[1].imshow(cmy_image)
    axes[1].set_title("CMY values")

    axes[2].imshow(hsi_image)
    axes[2].set_title(
        "HSI values\n"
        "H, S and I displayed as channels"
    )

    for axis in axes:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()