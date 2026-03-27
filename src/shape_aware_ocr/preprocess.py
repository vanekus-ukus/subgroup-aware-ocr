from __future__ import annotations

import numpy as np
from PIL import Image

IMAGE_HEIGHT = 24
IMAGE_WIDTH = 94
SQUARE_AR_THRESHOLD = 1.45
PREPROCESS_MODE_LEGACY = "legacy"
PREPROCESS_MODE_STAND = "stand"
PREPROCESS_MODES = (PREPROCESS_MODE_LEGACY, PREPROCESS_MODE_STAND)


def letterbox_rgb(image: Image.Image, out_w: int, out_h: int, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        return Image.new("RGB", (out_w, out_h), fill)

    scale = min(out_w / float(w), out_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (out_w, out_h), fill)
    canvas.paste(resized, ((out_w - new_w) // 2, (out_h - new_h) // 2))
    return canvas


def is_two_line_square_like(image: Image.Image, square_ar_threshold: float = SQUARE_AR_THRESHOLD) -> bool:
    w, h = image.size
    if h < 12:
        return False
    if w / float(max(h, 1)) >= square_ar_threshold:
        return False

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    crop_y = max(1, int(round(h * 0.08)))
    crop_x = max(1, int(round(w * 0.05)))
    if (2 * crop_y) >= h or (2 * crop_x) >= w:
        return False
    gray = gray[crop_y : h - crop_y, crop_x : w - crop_x]
    if gray.size == 0:
        return False

    threshold = float(np.percentile(gray, 45.0))
    ink = (gray <= threshold).astype(np.float32)
    row_density = ink.mean(axis=1)
    if row_density.size < 8:
        return False

    kernel_size = max(3, int(round(row_density.size / 12.0)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size,), dtype=np.float32) / float(kernel_size)
    row_density = np.convolve(row_density, kernel, mode="same")

    midpoint = row_density.size // 2
    if midpoint < 3:
        return False
    top_idx = int(np.argmax(row_density[:midpoint]))
    bottom_idx = midpoint + int(np.argmax(row_density[midpoint:]))
    top_peak = float(row_density[top_idx])
    bottom_peak = float(row_density[bottom_idx])
    if top_peak < 0.08 or bottom_peak < 0.08:
        return False
    if (bottom_idx - top_idx) < int(round(row_density.size * 0.22)):
        return False

    mid_left = int(round(row_density.size * 0.40))
    mid_right = int(round(row_density.size * 0.60))
    valley = float(np.mean(row_density[mid_left:mid_right]))
    if valley > min(top_peak, bottom_peak) * 0.72:
        return False
    return True


def unfold_square_crop(image: Image.Image, square_ar_threshold: float = SQUARE_AR_THRESHOLD) -> Image.Image:
    if not is_two_line_square_like(image, square_ar_threshold=square_ar_threshold):
        return image
    width, height = image.size
    split = height // 2
    top = image.crop((0, 0, width, split))
    bottom = image.crop((0, split, width, height))
    stitched_h = max(top.height, bottom.height)
    stitched = Image.new("RGB", (top.width + bottom.width, stitched_h), (0, 0, 0))
    stitched.paste(top, (0, (stitched_h - top.height) // 2))
    stitched.paste(bottom, (top.width, (stitched_h - bottom.height) // 2))
    return stitched


def preprocess_sequence_image(
    image: Image.Image,
    out_w: int = IMAGE_WIDTH,
    out_h: int = IMAGE_HEIGHT,
    square_ar_threshold: float = SQUARE_AR_THRESHOLD,
    preprocess_mode: str = PREPROCESS_MODE_STAND,
) -> Image.Image:
    mode = str(preprocess_mode or PREPROCESS_MODE_STAND).lower()
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unsupported preprocess mode: {preprocess_mode}")
    if mode == PREPROCESS_MODE_STAND:
        return image.resize((out_w, out_h), Image.BILINEAR)
    return letterbox_rgb(unfold_square_crop(image, square_ar_threshold=square_ar_threshold), out_w=out_w, out_h=out_h)
