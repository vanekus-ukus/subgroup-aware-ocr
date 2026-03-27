from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageFilter

STYLE_PALETTES = {
    "dark_on_light": {"fg": (24, 24, 26), "bg": (236, 234, 226)},
    "light_on_black": {"fg": (246, 246, 242), "bg": (22, 23, 28)},
    "light_on_blue": {"fg": (245, 246, 240), "bg": (32, 78, 168)},
    "dark_on_yellow": {"fg": (18, 18, 20), "bg": (233, 190, 58)},
}

REAL_SAMPLES = [
    ("AB12CD", "rect", "dark_on_light"),
    ("EF34GH", "rect", "dark_on_light"),
    ("JK56LM", "rect", "light_on_blue"),
    ("NP78QR", "rect", "light_on_black"),
    ("ST90UV", "rect", "dark_on_yellow"),
    ("WX12YZ", "rect", "light_on_blue"),
    ("KA45ME", "rect", "light_on_black"),
    ("TR67AC", "rect", "dark_on_light"),
    ("MZ88PT", "rect", "dark_on_yellow"),
    ("EU19LAB", "rect", "light_on_blue"),
    ("SQ123A", "square", "light_on_blue"),
    ("QX456B", "square", "light_on_black"),
    ("RY789C", "square", "dark_on_light"),
    ("ZA321D", "square", "dark_on_yellow"),
    ("PL654E", "square", "light_on_blue"),
    ("VK987F", "square", "light_on_black"),
    ("NL246G", "square", "dark_on_light"),
    ("DE135H", "square", "dark_on_yellow"),
]

SYNTH_SQUARE_SAMPLES = [
    ("SQ123A", "light_on_black"),
    ("QX456B", "dark_on_yellow"),
    ("RY789C", "light_on_blue"),
    ("ZA321D", "dark_on_light"),
    ("PL654E", "light_on_black"),
    ("VK987F", "dark_on_yellow"),
    ("NL246G", "light_on_blue"),
    ("DE135H", "light_on_black"),
    ("SQ123A", "dark_on_yellow"),
    ("QX456B", "light_on_blue"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a tiny deterministic dataset for shape-aware OCR research")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _draw_centered(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = _font()
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    tw = text_box[2] - text_box[0]
    th = text_box[3] - text_box[1]
    x = left + max(0, ((right - left) - tw) // 2)
    y = top + max(0, ((bottom - top) - th) // 2)
    draw.text((x, y), text, font=font, fill=fill)


def _render_rect(label: str, style: str, rng: random.Random) -> Image.Image:
    palette = STYLE_PALETTES[style]
    image = Image.new("RGB", (220, 72), palette["bg"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 215, 67), outline=(0, 0, 0), width=2)
    _draw_centered(draw, (14, 16, 206, 56), label, palette["fg"])
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.25))
    return image


def _render_square(label: str, style: str, rng: random.Random) -> Image.Image:
    palette = STYLE_PALETTES[style]
    image = Image.new("RGB", (132, 132), palette["bg"])
    draw = ImageDraw.Draw(image)
    draw.rectangle((4, 4, 127, 127), outline=(0, 0, 0), width=2)
    split = max(2, len(label) // 2)
    top = label[:split]
    bottom = label[split:]
    _draw_centered(draw, (10, 20, 122, 58), top, palette["fg"])
    _draw_centered(draw, (10, 68, 122, 108), bottom, palette["fg"])
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.25))
    return image


def _save_samples(root: Path, rows: list[tuple[str, str, str]], rng: random.Random) -> list[dict[str, str]]:
    metadata = []
    for index, (label, shape, style) in enumerate(rows, start=1):
        filename = f"{index:06d}_{label}.png"
        out_path = root / filename
        if shape == "square":
            image = _render_square(label, style, rng)
        else:
            image = _render_rect(label, style, rng)
        image.save(out_path)
        metadata.append({"file": filename, "shape": shape, "class_name": style, "label": label})
    return metadata


def _write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    out_root = Path(args.out)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    real_root = out_root / "real"
    synth_root = out_root / "synth_square"
    manifests_root = out_root / "manifests"
    real_root.mkdir(parents=True, exist_ok=True)
    synth_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    real_rows = _save_samples(real_root, REAL_SAMPLES, rng)
    synth_rows = _save_samples(synth_root, [(label, "square", style) for label, style in SYNTH_SQUARE_SAMPLES], rng)

    _write_manifest(
        manifests_root / "shape_manifest.csv",
        [{"file": row["file"], "shape": row["shape"]} for row in real_rows],
        ["file", "shape"],
    )
    _write_manifest(
        manifests_root / "style_manifest.csv",
        [{"file": row["file"], "class_name": row["class_name"]} for row in real_rows],
        ["file", "class_name"],
    )
    _write_manifest(
        manifests_root / "synthetic_style_manifest.csv",
        [{"file": row["file"], "class_name": row["class_name"]} for row in synth_rows],
        ["file", "class_name"],
    )

    print(f"[INFO] Generated real samples: {len(real_rows)}")
    print(f"[INFO] Generated synthetic square samples: {len(synth_rows)}")
    print(f"[INFO] Output root: {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
