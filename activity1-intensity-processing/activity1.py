import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# folders
BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

# failsafe 4 no results folder
RESULTS_DIR.mkdir(exist_ok=True)

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


for filename in image_files:
    image_path = IMAGES_DIR / filename

    # grayscale first
    original = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if original is None:
        print(f"Could not load: {filename}")
        continue

    negative = negative_transformation(original)

    # save the image after processing
    output_path = RESULTS_DIR / f"{Path(filename).stem}_negative.jpg"
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