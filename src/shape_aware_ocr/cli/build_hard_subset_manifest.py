from __future__ import annotations

import argparse
import csv
from pathlib import Path

SAMPLE_CLASS_SEPARATOR = "|"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a filtered hard-subgroup manifest from benchmark CSV manifests")
    parser.add_argument("--split-manifest", required=True, type=str)
    parser.add_argument("--shape-manifest", required=True, type=str)
    parser.add_argument("--sample-class-manifest", action="append", default=[])
    parser.add_argument("--split-name", type=str, default="train")
    parser.add_argument("--shape", choices=["square", "rect", "any"], default="square")
    parser.add_argument("--include-token", action="append", default=[])
    parser.add_argument("--exclude-token", action="append", default=[])
    parser.add_argument("--score", type=float, default=1.0)
    parser.add_argument("--out", required=True, type=str)
    return parser.parse_args()


def _row_to_key(row: dict[str, str], key_cols: list[str]) -> str:
    for key_col in key_cols:
        raw = str(row.get(key_col, "")).strip()
        if not raw:
            continue
        path = Path(raw)
        return path.stem if path.suffix else raw
    return ""


def _load_simple_map(path: Path, value_columns: tuple[str, ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        key_cols = [name for name in ("match_key", "crop_rel_path", "crop_path", "output_path", "file", "path") if name in fields]
        value_col = next((name for name in value_columns if name in fields), "")
        if not key_cols:
            raise RuntimeError(f"No key column in {path}")
        if not value_col:
            raise RuntimeError(f"No value column in {path}")
        for row in reader:
            key = _row_to_key(row, key_cols)
            value = str(row.get(value_col, "")).strip()
            if key and value:
                mapping[key] = value
    return mapping


def _load_class_maps(paths: list[Path]) -> dict[str, str]:
    combined: dict[str, list[str]] = {}
    use_namespace = len(paths) > 1
    for path in paths:
        stem = path.stem.strip().lower()
        for suffix in ("_manifest", "_class", "_classes"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        namespace = stem if use_namespace else ""
        local = _load_simple_map(path, ("class_name", "sample_class", "color_class", "class", "group", "palette"))
        for key, value in local.items():
            encoded = f"{namespace}:{value}" if namespace else value
            bucket = combined.setdefault(key, [])
            if encoded not in bucket:
                bucket.append(encoded)
    return {key: SAMPLE_CLASS_SEPARATOR.join(values) for key, values in combined.items()}


def _matches_tokens(raw: str, include_tokens: list[str], exclude_tokens: list[str]) -> bool:
    tokens = {chunk.strip() for chunk in str(raw or "").split(SAMPLE_CLASS_SEPARATOR) if chunk.strip()}
    if include_tokens and not all(token in tokens for token in include_tokens):
        return False
    if exclude_tokens and any(token in tokens for token in exclude_tokens):
        return False
    return True


def main() -> int:
    args = parse_args()
    split_map = _load_simple_map(Path(args.split_manifest), ("split", "subset", "fold"))
    shape_map = _load_simple_map(Path(args.shape_manifest), ("shape", "plate_shape", "type"))
    sample_class_map = _load_class_maps([Path(p) for p in args.sample_class_manifest if p])

    split_name = str(args.split_name or "").strip().lower()
    required_shape = str(args.shape or "square").strip().lower()
    include_tokens = [str(token).strip() for token in args.include_token if str(token).strip()]
    exclude_tokens = [str(token).strip() for token in args.exclude_token if str(token).strip()]

    rows: list[dict[str, object]] = []
    for key, split_value in sorted(split_map.items()):
        if split_name and str(split_value).strip().lower() != split_name:
            continue
        if required_shape != "any" and str(shape_map.get(key, "")).strip().lower() != required_shape:
            continue
        sample_class = str(sample_class_map.get(key, ""))
        if not _matches_tokens(sample_class, include_tokens=include_tokens, exclude_tokens=exclude_tokens):
            continue
        rows.append(
            {
                "match_key": key,
                "split": split_value,
                "shape": required_shape,
                "sample_class": sample_class,
                "hard_score": float(args.score),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["match_key", "split", "shape", "sample_class", "hard_score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Hard subset manifest: {out_path}")
    print(f"[INFO] Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
