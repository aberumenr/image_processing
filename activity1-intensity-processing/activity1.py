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
    return 255 - image

def logarithmic_transformation(image, c):
    image_float = image.astype(np.float32)
    transformed = c * np.log1p(image_float)

    transformed = np.clip(transformed, 0, 255)

    return transformed.astype(np.uint8)

standard_c = 255 / np.log(1 + 255)

logarithmic_constants = [
    standard_c * 0.5,
    standard_c,
    standard_c * 1.5
]

def gamma_transformation(image, gamma, c=1.0):
    normalized = image.astype(np.float32) / 255.0
    transformed = c * np.power(normalized, gamma)

    transformed = np.clip(transformed * 255, 0, 255)

    return transformed.astype(np.uint8)

gamma_values = [
    0.4,
    1.0,
    2.5
]

def histogram_equalization(image):
    return cv2.equalizeHist(image)

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

        # apply gamma transformation with diff gamma values
    gamma_results = []

    for gamma in gamma_values:
        gamma_image = gamma_transformation(original, gamma)
        gamma_results.append((gamma, gamma_image))

        output_path = (
            GAMMA_DIR
            / f"{Path(filename).stem}_gamma_{gamma:.1f}.jpg"
        )
        cv2.imwrite(str(output_path), gamma_image)

    # comparison between original and gamma transformations
    plt.figure(figsize=(16, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap="gray", vmin=0, vmax=255)
    plt.title(f"Original: {filename}")
    plt.axis("off")

    for index, (gamma, gamma_image) in enumerate(
        gamma_results,
        start=2
    ):
        plt.subplot(1, 4, index)
        plt.imshow(gamma_image, cmap="gray", vmin=0, vmax=255)
        plt.title(f"Gamma\nγ = {gamma:.1f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()