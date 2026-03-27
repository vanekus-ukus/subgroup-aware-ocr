from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a balanced qualitative contact sheet from a benchmark root')
    parser.add_argument('--data-root', required=True, type=str)
    parser.add_argument('--shape-manifest', required=True, type=str)
    parser.add_argument('--out', required=True, type=str)
    parser.add_argument('--rect-count', type=int, default=12)
    parser.add_argument('--square-count', type=int, default=12)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    shape_manifest = Path(args.shape_manifest)
    out_path = Path(args.out)
    shape_rows = {}
    with open(shape_manifest, 'r', newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            shape_rows[row['file']] = row['shape']

    buckets = {'rect': [], 'square': []}
    for image_path in sorted(data_root.glob('*.png')):
        shape = shape_rows.get(image_path.name, '')
        if shape in buckets:
            buckets[shape].append(image_path)

    random.Random(args.seed).shuffle(buckets['rect'])
    random.Random(args.seed + 1).shuffle(buckets['square'])
    selected = buckets['rect'][: args.rect_count] + buckets['square'][: args.square_count]

    cols = 4
    thumb_w, thumb_h = 220, 90
    margin = 14
    rows = (len(selected) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * (thumb_w + margin) + margin, rows * (thumb_h + 40 + margin) + margin), (245, 245, 242))
    draw = ImageDraw.Draw(canvas)
    for idx, image_path in enumerate(selected):
        image = Image.open(image_path).convert('RGB')
        image.thumbnail((thumb_w, thumb_h))
        tile = Image.new('RGB', (thumb_w, thumb_h), (230, 230, 228))
        tile.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))
        x = margin + (idx % cols) * (thumb_w + margin)
        y = margin + (idx // cols) * (thumb_h + 40 + margin)
        canvas.paste(tile, (x, y))
        shape = shape_rows.get(image_path.name, 'unknown')
        label = image_path.stem.split('_', 1)[1] if '_' in image_path.stem else image_path.stem
        draw.text((x, y + thumb_h + 6), f'{shape} | {label[:10]}', fill=(20, 20, 20))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f'[INFO] Saved contact sheet to {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
