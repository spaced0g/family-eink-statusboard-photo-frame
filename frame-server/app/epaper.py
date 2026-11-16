"""Image scaling and classic Floyd–Steinberg dithering for Spectra 6."""

import numpy as np
from numba import njit
from PIL import Image

EPD_WIDTH = 800
EPD_HEIGHT = 480

# The order also decides ties and matches the ESP32 packing in app.py.
PALETTE_RGB = np.array(
    [(0, 0, 0), (255, 255, 255), (255, 255, 0),
     (255, 0, 0), (0, 0, 255), (0, 255, 0)],
    dtype=np.uint8,
)
PALETTE = PALETTE_RGB / 255.0


def load_scaled(image, angle, display_mode="fit"):
    """Rotate and scale an image to 800×480, cropping or adding white margins."""
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        with Image.open(image) as source:
            rgb = source.convert("RGB")
    rgb = rgb.rotate(angle, expand=True)
    aspect = rgb.width / rgb.height
    wider_than_panel = aspect > EPD_WIDTH / EPD_HEIGHT

    if display_mode == "fill":
        if wider_than_panel:
            size = (int(EPD_HEIGHT * aspect), EPD_HEIGHT)
        else:
            size = (EPD_WIDTH, int(EPD_WIDTH / aspect))
        left = (size[0] - EPD_WIDTH) // 2
        top = (size[1] - EPD_HEIGHT) // 2
        return rgb.resize(size, Image.Resampling.LANCZOS).crop(
            (left, top, left + EPD_WIDTH, top + EPD_HEIGHT)
        )

    if wider_than_panel:
        size = (EPD_WIDTH, max(1, int(EPD_WIDTH / aspect)))
    else:
        size = (max(1, int(EPD_HEIGHT * aspect)), EPD_HEIGHT)
    canvas = Image.new("RGB", (EPD_WIDTH, EPD_HEIGHT), "white")
    offset = ((EPD_WIDTH - size[0]) // 2, (EPD_HEIGHT - size[1]) // 2)
    canvas.paste(rgb.resize(size, Image.Resampling.LANCZOS), offset)
    return canvas


@njit(cache=True)
def _dither_indices(pixels, strength):
    # Like EPF's classic mode, diffuse each truncated error share into the
    # current 8-bit range before processing the next pixel.
    height, width, _ = pixels.shape
    indices = np.empty((height, width), dtype=np.uint8)

    for row in range(height):
        for col in range(width):
            best = 0
            min_distance = np.inf
            for color in range(len(PALETTE)):
                distance = 0.0
                for channel in range(3):
                    delta = pixels[row, col, channel] / 255.0 - PALETTE[color, channel]
                    distance += delta ** 2
                if distance < min_distance:
                    min_distance = distance
                    best = color

            indices[row, col] = best
            for channel in range(3):
                error = (pixels[row, col, channel] / 255.0 - PALETTE[best, channel]) * strength
                for dy, dx, weight in ((0, 1, 7), (1, -1, 3), (1, 0, 5), (1, 1, 1)):
                    y, x = row + dy, col + dx
                    if y < height and 0 <= x < width:
                        value = pixels[y, x, channel] + int(error * weight / 16 * 255)
                        pixels[y, x, channel] = min(max(value, 0), 255)

    return indices


def convert_image(image, dithering_strength=1.0):
    """Return the dithered image as an RGB uint8 array."""
    pixels = np.array(image.convert("RGB"), dtype=np.int64)
    return PALETTE_RGB[_dither_indices(pixels, float(dithering_strength))]
