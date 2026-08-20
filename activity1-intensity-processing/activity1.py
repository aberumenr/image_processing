import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# folders
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

# results
RESULTS_DIR.mkdir(exist_ok=True)

NEGATIVE_DIR = RESULTS_DIR / "negative"
LOGARITHMIC_DIR = RESULTS_DIR / "logarithmic"
GAMMA_DIR = RESULTS_DIR / "gamma"
HISTOGRAM_DIR = RESULTS_DIR / "histograms"
EQUALIZATION_DIR = RESULTS_DIR / "equalization"

NEGATIVE_DIR.mkdir(exist_ok=True)
LOGARITHMIC_DIR.mkdir(exist_ok=True)
GAMMA_DIR.mkdir(exist_ok=True)
HISTOGRAM_DIR.mkdir(exist_ok=True)
EQUALIZATION_DIR.mkdir(exist_ok=True)

# our images
image_files = [
    "xray.jpg",
    "constellation.jpg",
    "indoors.jpg",
    "foggy.jpg"
]


def negative_transformation(image):
    """
    Applies the negative transformation:
    s = 255 - r
    """
    return 255 - image

def logarithmic_transformation(image, c):
    """
    Applies the logarithmic transformation:
    s = c * log(1 + r)
    """
    image_float = image.astype(np.float32)
    transformed = c * np.log1p(image_float)

    # Keep intensity values within the valid range
    transformed = np.clip(transformed, 0, 255)

    return transformed.astype(np.uint8)

# Standard logarithmic constant
standard_c = 255 / np.log(1 + 255)

# Different values requested by the activity
logarithmic_constants = [
    standard_c * 0.5,
    standard_c,
    standard_c * 1.5
]

for filename in image_files:
    image_path = IMAGES_DIR / filename

    # grayscale first
    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if original is None:
        print(f"Could not load: {filename}")
        continue

    negative = negative_transformation(original)

    # save the image after processing
    output_path = NEGATIVE_DIR / f"{Path(filename).stem}_negative.jpg"
    cv2.imwrite(str(output_path), negative)

    # comparison between original and negative
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Original: {filename}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(negative, cmap="gray", vmin=0, vmax=255)
    plt.title("Negative")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # apply logarithmic with diff c values
    logarithmic_results = []

    for c in logarithmic_constants:
        logarithmic = logarithmic_transformation(original, c)
        logarithmic_results.append((c, logarithmic))

        output_path = (
            LOGARITHMIC_DIR
            / f"{Path(filename).stem}_log_c_{c:.2f}.jpg"
        )
        cv2.imwrite(str(output_path), logarithmic)

    # comparison between original and logarithmic transformations
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Original: {filename}")
    plt.axis("off")

    for index, (c, logarithmic) in enumerate(
        logarithmic_results,
        start=2
    ):
        plt.subplot(1, 4, index)
        plt.imshow(logarithmic, cmap="gray", vmin=0, vmax=255)
        plt.title(f"Logarithmic\nc = {c:.2f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()