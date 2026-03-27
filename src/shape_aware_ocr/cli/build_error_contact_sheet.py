from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def _normalize_path(raw_path: str) -> Path:
    normalized = str(raw_path or "").replace("\\", "/").strip()
    return Path(normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a qualitative contact sheet from OCR top-error CSV")
    parser.add_argument("--errors-csv", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--top-n", type=int, default=24)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--copy-dir", type=str, default="")
    parser.add_argument("--thumb-width", type=int, default=220)
    parser.add_argument("--thumb-height", type=int, default=90)
    return parser.parse_args()


def _read_rows(path: Path, top_n: int) -> list[dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: float(row.get("norm_ed", "0") or 0.0), reverse=True)
    return rows[: max(0, int(top_n))]


def main() -> int:
    args = parse_args()
    errors_csv = Path(args.errors_csv)
    out_path = Path(args.out)
    copy_dir = Path(args.copy_dir) if args.copy_dir else None
    rows = _read_rows(errors_csv, args.top_n)
    if not rows:
        raise SystemExit(f"No rows found in {errors_csv}")

    cols = max(1, int(args.cols))
    thumb_w = max(80, int(args.thumb_width))
    thumb_h = max(40, int(args.thumb_height))
    caption_h = 54
    margin = 14
    canvas_rows = (len(rows) + cols - 1) // cols
    canvas = Image.new(
        "RGB",
        (cols * (thumb_w + margin) + margin, canvas_rows * (thumb_h + caption_h + margin) + margin),
        (245, 245, 242),
    )
    draw = ImageDraw.Draw(canvas)

    if copy_dir:
        copy_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(rows):
        image_path = _normalize_path(row.get("file", ""))
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h))
        tile = Image.new("RGB", (thumb_w, thumb_h), (230, 230, 228))
        tile.paste(image, ((thumb_w - image.width) // 2, (thumb_h - image.height) // 2))

        x = margin + (idx % cols) * (thumb_w + margin)
        y = margin + (idx // cols) * (thumb_h + caption_h + margin)
        canvas.paste(tile, (x, y))

        gt = str(row.get("gt", ""))[:18]
        pred = str(row.get("pred", ""))[:18]
        norm_ed = str(row.get("norm_ed", ""))
        sample_class = str(row.get("sample_class", ""))
        if ":" in sample_class:
            sample_class = sample_class.split(":", 1)[1]
        draw.text((x, y + thumb_h + 4), f"gt={gt}", fill=(20, 20, 20))
        draw.text((x, y + thumb_h + 20), f"pred={pred}", fill=(120, 25, 25))
        draw.text((x, y + thumb_h + 36), f"{sample_class} | ned={norm_ed}", fill=(60, 60, 60))

        if copy_dir:
            target_name = f"{idx + 1:03d}_{image_path.name}"
            tile.save(copy_dir / target_name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[INFO] Saved error contact sheet to {out_path}")
    if copy_dir:
        print(f"[INFO] Copied {len(rows)} tiles to {copy_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
