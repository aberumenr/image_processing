from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "images"
RESULTS_DIR = BASE_DIR / "results"

IMAGE_FILES = [
    "image1.jpg",
    "image2.jpg",
    "image3.jpg",
    "image4.jpg",
]


def load_grayscale_image(filename):
    image_path = IMAGES_DIR / filename
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            f"Could not load the image: {image_path}"
        )

    return image

def calculate_histogram(image):
    histogram = np.zeros(256, dtype=np.int64)

    height, width = image.shape

    for row in range(height):
        for column in range(width):
            intensity = image[row, column]
            histogram[intensity] += 1

    return histogram

def negative_transformation(image):
    height, width = image.shape
    negative = np.zeros_like(image)

    for row in range(height):
        for column in range(width):
            negative[row, column] = 255 - image[row, column]

    return negative

def gamma_transformation(image, gamma):
    height, width = image.shape
    result = np.zeros_like(image)

    for row in range(height):
        for column in range(width):
            original_intensity = image[row, column]
            normalized_intensity = original_intensity / 255.0

            transformed_intensity = (
                255.0 * (normalized_intensity ** gamma)
            )

            if transformed_intensity < 0:
                transformed_intensity = 0
            elif transformed_intensity > 255:
                transformed_intensity = 255

            result[row, column] = int(
                round(transformed_intensity)
            )

    return result

def display_parameter_comparison(
    original,
    processed_images,
    title
):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True
    )
    axes = axes.flatten()

    axes[0].imshow(
        original,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    for index, (parameter, image) in enumerate(
        processed_images.items(),
        start=1
    ):
        axes[index].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[index].set_title(
            rf"$\gamma={parameter}$"
        )
        axes[index].axis("off")

    figure.suptitle(title, fontsize=16)

def display_comparison(
    original,
    processed,
    original_histogram,
    processed_histogram,
    title
):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True
    )

    axes[0, 0].imshow(
        original,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    axes[0, 0].set_title("Imagen original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        processed,
        cmap="gray",
        vmin=0,
        vmax=255
    )
    axes[0, 1].set_title("Imagen procesada")
    axes[0, 1].axis("off")

    axes[1, 0].plot(
        range(256),
        original_histogram,
        color="black"
    )
    axes[1, 0].set_title("Histograma original")
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].set_xlabel("Nivel de intensidad")
    axes[1, 0].set_ylabel("Cantidad de píxeles")
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].plot(
        range(256),
        processed_histogram,
        color="black"
    )
    axes[1, 1].set_title("Histograma procesado")
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].set_xlabel("Nivel de intensidad")
    axes[1, 1].set_ylabel("Cantidad de píxeles")
    axes[1, 1].grid(alpha=0.2)

    figure.suptitle(title, fontsize=16)

def display_original_images(images):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True
    )
    axes = axes.flatten()

    for index, filename in enumerate(IMAGE_FILES):
        axes[index].imshow(
            images[filename],
            cmap="gray",
            vmin=0,
            vmax=255
        )
        axes[index].set_title(filename)
        axes[index].axis("off")

    figure.suptitle(
        "Imágenes originales",
        fontsize=16
    )


def display_original_histograms(histograms):
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(12, 8),
        constrained_layout=True
    )
    axes = axes.flatten()

    for index, filename in enumerate(IMAGE_FILES):
        axes[index].plot(
            range(256),
            histograms[filename],
            color="black",
            linewidth=1
        )
        axes[index].set_title(f"Histograma de {filename}")
        axes[index].set_xlim(0, 255)
        axes[index].set_xlabel("Nivel de intensidad")
        axes[index].set_ylabel("Cantidad de píxeles")
        axes[index].grid(alpha=0.2)

    figure.suptitle(
        "Histogramas originales",
        fontsize=16
    )

    plt.show()

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    images = {}
    histograms = {}

    for filename in IMAGE_FILES:
        image = load_grayscale_image(filename)
        histogram = calculate_histogram(image)

        images[filename] = image
        histograms[filename] = histogram

        print(
            f"{filename}: "
            f"dimensions={image.shape}, "
            f"minimum={image.min()}, "
            f"maximum={image.max()}, "
            f"pixels={image.size}, "
            f"histogram total={histogram.sum()}"
        )
        
    # negative for image4
    image4_negative = negative_transformation(images["image4.jpg"])
    image4_negative_histogram = calculate_histogram(image4_negative)

    cv2.imwrite(
        str(RESULTS_DIR / "image4_negative.jpg"),
        image4_negative
    )

    display_comparison(
        images["image4.jpg"],
        image4_negative,
        histograms["image4.jpg"],
        image4_negative_histogram,
        "Imagen 4 - Transformación negativa"
    )

    # the gamma one
    image1_gamma_values = [1.5, 2.0, 2.5]
    image1_gamma_results = {}

    for gamma in image1_gamma_values:
        result = gamma_transformation(
            images["image1.jpg"],
            gamma
        )

        image1_gamma_results[gamma] = result

        cv2.imwrite(
            str(
                RESULTS_DIR
                / f"image1_gamma_{gamma}.jpg"
            ),
            result
        )

    display_parameter_comparison(
        images["image1.jpg"],
        image1_gamma_results,
        "Imagen 1 - Comparación de transformación gamma"
    )
    display_original_images(images)
    display_original_histograms(histograms)


if __name__ == "__main__":
    main()

plt.show()