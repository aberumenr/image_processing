import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_image(image_path):
    image = plt.imread(image_path)
    image = image[:, :, :3]

    if np.issubdtype(image.dtype, np.floating):
        image = np.round(
            image * 255
        ).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    return image


def filter_none(image):
    scanlines = bytearray()

    for row in image:
        raw_bytes = row.reshape(-1)

        scanlines.append(0)
        scanlines.extend(raw_bytes.tobytes())

    return bytes(scanlines)


def filter_sub(image):
    height, width, channels = image.shape
    bytes_per_pixel = channels

    scanlines = bytearray()

    for row in image:
        raw_bytes = row.reshape(-1)
        filtered = np.zeros_like(raw_bytes)

        for i in range(len(raw_bytes)):
            x = int(raw_bytes[i])

            if i >= bytes_per_pixel:
                a = int(
                    raw_bytes[
                        i - bytes_per_pixel
                    ]
                )
            else:
                a = 0

            filtered[i] = (
                x - a
            ) % 256

        scanlines.append(1)
        scanlines.extend(filtered.tobytes())

    return bytes(scanlines)


def filter_up(image):
    height, width, channels = image.shape
    scanlines = bytearray()

    for y in range(height):
        current_row = image[y].reshape(-1)
        filtered = np.zeros_like(current_row)

        if y > 0:
            above_row = image[y - 1].reshape(-1)
        else:
            above_row = None

        for i in range(len(current_row)):
            x = int(current_row[i])

            if y > 0:
                b = int(above_row[i])
            else:
                b = 0

            filtered[i] = (
                x - b
            ) % 256

        scanlines.append(2)
        scanlines.extend(filtered.tobytes())

    return bytes(scanlines)


def filter_average(image):
    height, width, channels = image.shape
    bytes_per_pixel = channels

    scanlines = bytearray()

    for y in range(height):
        current_row = image[y].reshape(-1)
        filtered = np.zeros_like(current_row)

        if y > 0:
            above_row = image[y - 1].reshape(-1)
        else:
            above_row = None

        for i in range(len(current_row)):
            x = int(current_row[i])

            if i >= bytes_per_pixel:
                a = int(
                    current_row[
                        i - bytes_per_pixel
                    ]
                )
            else:
                a = 0

            if y > 0:
                b = int(above_row[i])
            else:
                b = 0

            predictor = (
                a + b
            ) // 2

            filtered[i] = (
                x - predictor
            ) % 256

        scanlines.append(3)
        scanlines.extend(filtered.tobytes())

    return bytes(scanlines)


def decode_scanlines(
    scanlines,
    width,
    height,
    channels=3
):
    bytes_per_pixel = channels
    bytes_per_row = width * channels

    reconstructed = np.zeros(
        (height, width, channels),
        dtype=np.uint8
    )

    position = 0

    for y in range(height):
        filter_type = scanlines[position]
        position += 1

        filtered = np.frombuffer(
            scanlines[
                position:
                position + bytes_per_row
            ],
            dtype=np.uint8
        )

        position += bytes_per_row

        raw_bytes = np.zeros(
            bytes_per_row,
            dtype=np.uint8
        )

        if y > 0:
            above_row = reconstructed[
                y - 1
            ].reshape(-1)
        else:
            above_row = None

        for i in range(bytes_per_row):
            filtered_value = int(filtered[i])

            if filter_type == 0:
                # none
                predictor = 0

            elif filter_type == 1:
                # sub
                if i >= bytes_per_pixel:
                    predictor = int(
                        raw_bytes[
                            i - bytes_per_pixel
                        ]
                    )
                else:
                    predictor = 0

            elif filter_type == 2:
                # up
                if y > 0:
                    predictor = int(
                        above_row[i]
                    )
                else:
                    predictor = 0

            elif filter_type == 3:
                # average
                if i >= bytes_per_pixel:
                    a = int(
                        raw_bytes[
                            i - bytes_per_pixel
                        ]
                    )
                else:
                    a = 0

                if y > 0:
                    b = int(above_row[i])
                else:
                    b = 0

                predictor = (
                    a + b
                ) // 2

            else:
                raise ValueError(
                    f"unsupported filter: {filter_type}"
                )

            raw_bytes[i] = (
                filtered_value + predictor
            ) % 256

        reconstructed[y] = raw_bytes.reshape(
            width,
            channels
        )

    return reconstructed


def filtered_image(
    scanlines,
    width,
    height,
    channels=3
):
    bytes_per_row = width * channels

    result = np.zeros(
        (height, width, channels),
        dtype=np.uint8
    )

    position = 0

    for y in range(height):
        # skip the filter byte
        position += 1

        row = np.frombuffer(
            scanlines[
                position:
                position + bytes_per_row
            ],
            dtype=np.uint8
        )

        position += bytes_per_row

        result[y] = row.reshape(
            width,
            channels
        )

    return result


def main():
    folder = Path(__file__).resolve().parent
    image_path = folder / "generated_rgb.png"

    if not image_path.exists():
        raise FileNotFoundError(
            f"could not find: {image_path}"
        )

    image_rgb = load_image(image_path)

    height, width, channels = image_rgb.shape

    print("image loaded:", image_path)
    print("shape:", image_rgb.shape)
    print("type:", image_rgb.dtype)
    print()

    none_scanlines = filter_none(image_rgb)
    sub_scanlines = filter_sub(image_rgb)
    up_scanlines = filter_up(image_rgb)

    average_scanlines = filter_average(
        image_rgb
    )

    reconstructed_none = decode_scanlines(
        none_scanlines,
        width,
        height,
        channels
    )

    reconstructed_sub = decode_scanlines(
        sub_scanlines,
        width,
        height,
        channels
    )

    reconstructed_up = decode_scanlines(
        up_scanlines,
        width,
        height,
        channels
    )

    reconstructed_average = decode_scanlines(
        average_scanlines,
        width,
        height,
        channels
    )

    none_filtered = filtered_image(
        none_scanlines,
        width,
        height,
        channels
    )

    sub_filtered = filtered_image(
        sub_scanlines,
        width,
        height,
        channels
    )

    up_filtered = filtered_image(
        up_scanlines,
        width,
        height,
        channels
    )

    average_filtered = filtered_image(
        average_scanlines,
        width,
        height,
        channels
    )

    print(
        "bytes per scanline:",
        width * channels + 1
    )

    print(
        "total bytes with None:",
        len(none_scanlines)
    )

    print(
        "total bytes with Sub:",
        len(sub_scanlines)
    )

    print(
        "total bytes with Up:",
        len(up_scanlines)
    )

    print(
        "total bytes with Average:",
        len(average_scanlines)
    )

    print()

    print(
        "None reconstruction correct:",
        np.array_equal(
            image_rgb,
            reconstructed_none
        )
    )

    print(
        "Sub reconstruction correct:",
        np.array_equal(
            image_rgb,
            reconstructed_sub
        )
    )

    print(
        "Up reconstruction correct:",
        np.array_equal(
            image_rgb,
            reconstructed_up
        )
    )

    print(
        "Average reconstruction correct:",
        np.array_equal(
            image_rgb,
            reconstructed_average
        )
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(17, 9)
    )

    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title("original image")

    axes[0, 1].imshow(none_filtered)
    axes[0, 1].set_title("None filter")

    axes[0, 2].imshow(
        sub_filtered,
        interpolation="nearest"
    )
    axes[0, 2].set_title("Sub filter")

    axes[0, 3].imshow(
        up_filtered,
        interpolation="nearest"
    )
    axes[0, 3].set_title("Up filter")

    axes[1, 0].imshow(
        average_filtered,
        interpolation="nearest"
    )
    axes[1, 0].set_title("Average filter")

    axes[1, 1].imshow(reconstructed_sub)
    axes[1, 1].set_title("reconstructed from Sub")

    axes[1, 2].imshow(reconstructed_up)
    axes[1, 2].set_title("reconstructed from Up")

    axes[1, 3].imshow(
        reconstructed_average
    )
    axes[1, 3].set_title(
        "reconstructed from Average"
    )

    for axis in axes.flat:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()