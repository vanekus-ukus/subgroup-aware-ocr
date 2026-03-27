from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from shape_aware_ocr.dataset import load_shape_map_from_manifest, load_split_map_from_manifest
from shape_aware_ocr.labels import label_from_filename_stem, normalized_match_stem

PALETTES = [
    {"name": "white_on_blue", "fg": (245, 246, 242), "bg": (28, 78, 170)},
    {"name": "white_on_black", "fg": (246, 246, 243), "bg": (23, 24, 28)},
    {"name": "white_on_green", "fg": (245, 246, 240), "bg": (27, 105, 66)},
    {"name": "white_on_red", "fg": (245, 244, 238), "bg": (145, 30, 34)},
    {"name": "black_on_yellow", "fg": (20, 20, 22), "bg": (232, 191, 48)},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a synthetic square pool from train-square samples of a fixed benchmark")
    parser.add_argument("--data-root", required=True, type=str)
    parser.add_argument("--shape-manifest", required=True, type=str)
    parser.add_argument("--split-manifest", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--count", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def jitter_rgb(rgb: tuple[int, int, int], rng: random.Random, scale: float) -> np.ndarray:
    base = np.array(rgb, dtype=np.float32) / 255.0
    per_channel = np.array([rng.uniform(1.0 - scale, 1.0 + scale) for _ in range(3)], dtype=np.float32)
    overall = rng.uniform(1.0 - scale * 0.7, 1.0 + scale * 0.7)
    return clamp01(base * per_channel * overall)


def make_soft_foreground_mask(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    p_lo = float(np.quantile(gray, 0.16))
    p_hi = float(np.quantile(gray, 0.84))
    denom = max(p_hi - p_lo, 1e-3)
    mask = clamp01((p_hi - gray) / denom)
    mask = mask ** rng.uniform(1.2, 1.85)
    mask_img = Image.fromarray(np.uint8(mask * 255.0), mode="L")
    if rng.random() < 0.85:
        mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 1.0)))
    return np.asarray(mask_img, dtype=np.float32) / 255.0


def make_texture(gray: np.ndarray, rng: random.Random) -> np.ndarray:
    blur_radius = rng.uniform(2.0, 4.0)
    low = Image.fromarray(np.uint8(gray * 255.0), mode="L").filter(ImageFilter.GaussianBlur(radius=blur_radius))
    low_arr = np.asarray(low, dtype=np.float32) / 255.0
    texture = 1.0 + (low_arr - float(low_arr.mean())) * rng.uniform(0.2, 0.35)
    return np.clip(texture, 0.78, 1.22)[..., None]


def make_gradient(h: int, w: int, rng: random.Random) -> np.ndarray:
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    grad = 1.0 + xs * rng.uniform(-0.08, 0.08) + ys * rng.uniform(-0.06, 0.06)
    return np.clip(grad, 0.90, 1.10)[..., None]


def augment_square(img: Image.Image, palette: dict[str, object], rng: random.Random) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    gray = (arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114).astype(np.float32)
    mask = make_soft_foreground_mask(gray, rng)
    texture = make_texture(gray, rng)
    gradient = make_gradient(arr.shape[0], arr.shape[1], rng)
    fg = jitter_rgb(tuple(palette["fg"]), rng, scale=0.05)
    bg = jitter_rgb(tuple(palette["bg"]), rng, scale=0.10)
    colorized = bg[None, None, :] * (1.0 - mask[..., None]) + fg[None, None, :] * mask[..., None]
    contrast_mix = rng.uniform(0.03, 0.10)
    colorized = colorized * texture * gradient
    colorized = colorized * (1.0 - contrast_mix) + arr * contrast_mix
    noise_sigma = rng.uniform(0.0, 0.010)
    if noise_sigma > 0.0:
        colorized = colorized + np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, noise_sigma, size=colorized.shape).astype(np.float32)
    out = Image.fromarray(np.uint8(clamp01(colorized) * 255.0), mode="RGB")
    return out.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.35)))


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    shape_map, _ = load_shape_map_from_manifest(Path(args.shape_manifest))
    split_map, _ = load_split_map_from_manifest(Path(args.split_manifest))

    candidates = []
    for image_path in sorted(data_root.glob('*.png')):
        match_key = normalized_match_stem(image_path.stem)
        if split_map.get(match_key) != 'train':
            continue
        if shape_map.get(match_key, -1) != 1:
            continue
        label = label_from_filename_stem(image_path.stem)
        if not label:
            continue
        candidates.append((image_path, label))
    if not candidates:
        raise RuntimeError('No train-square samples found')

    rng = random.Random(args.seed)
    chosen = []
    while len(chosen) < int(args.count):
        chosen.append(rng.choice(candidates))

    manifest_rows = []
    class_rows = []
    palette_names = [palette['name'] for palette in PALETTES]
    for idx, (src_path, label) in enumerate(chosen, start=1):
        palette = PALETTES[idx % len(PALETTES)]
        palette_name = str(palette['name'])
        out_name = f"{idx:06d}_{label}.png"
        out_path = out_root / out_name
        with Image.open(src_path) as handle:
            image = handle.convert('RGB').copy()
        aug = augment_square(image, palette=palette, rng=random.Random(args.seed + idx * 17))
        aug.save(out_path)
        manifest_rows.append({
            'file': out_name,
            'label': label,
            'source_path': str(src_path),
            'source_match_key': normalized_match_stem(src_path.stem),
            'palette': palette_name,
        })
        class_rows.append({'file': out_name, 'class_name': palette_name})

    with open(out_root / 'manifest.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['file', 'label', 'source_path', 'source_match_key', 'palette'])
        writer.writeheader()
        writer.writerows(manifest_rows)
    with open(out_root / 'style_manifest.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['file', 'class_name'])
        writer.writeheader()
        writer.writerows(class_rows)
    summary = {
        'data_root': str(data_root),
        'shape_manifest': str(args.shape_manifest),
        'split_manifest': str(args.split_manifest),
        'count': len(manifest_rows),
        'source_square_train_candidates': len(candidates),
        'palette_counts': {name: sum(1 for row in manifest_rows if row['palette'] == name) for name in palette_names},
        'seed': int(args.seed),
    }
    with open(out_root / 'summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"[INFO] Synthetic square pool: {len(manifest_rows)} images")
    print(f"[INFO] Source train-square candidates: {len(candidates)}")
    print(f"[INFO] Output root: {out_root}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
