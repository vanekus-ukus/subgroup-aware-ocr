from __future__ import annotations

import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

from .dataset import iter_image_files, load_sample_class_map_from_manifest, load_shape_map_from_manifest
from .labels import label_from_filename_stem, normalized_match_stem


@dataclass(frozen=True)
class BenchmarkSample:
    image_path: Path
    match_key: str
    label: str
    shape: str
    sample_class: str


@dataclass
class BenchmarkSummary:
    out_root: str
    total_selected: int
    train_count: int
    val_count: int
    test_count: int
    rect_count: int
    square_count: int


def _copy_file(src: Path, dst: Path, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "hardlink":
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _load_samples(src_root: Path, shape_manifest: Path, sample_class_manifest: Path | None = None, min_label_len: int = 4) -> list[BenchmarkSample]:
    shape_map, _ = load_shape_map_from_manifest(shape_manifest)
    sample_class_map: dict[str, str] = {}
    if sample_class_manifest is not None and sample_class_manifest.exists():
        sample_class_map, _ = load_sample_class_map_from_manifest(sample_class_manifest)

    samples: list[BenchmarkSample] = []
    for image_path in iter_image_files(src_root):
        label = label_from_filename_stem(image_path.stem)
        if len(label) < int(min_label_len):
            continue
        match_key = normalized_match_stem(image_path.stem)
        shape_flag = shape_map.get(match_key, -1)
        if shape_flag not in (0, 1):
            continue
        samples.append(
            BenchmarkSample(
                image_path=image_path,
                match_key=match_key,
                label=label,
                shape="square" if shape_flag == 1 else "rect",
                sample_class=str(sample_class_map.get(match_key, "")),
            )
        )
    if not samples:
        raise RuntimeError("No benchmark samples matched the provided manifests")
    return samples


def _sample_by_shape(samples: list[BenchmarkSample], rect_count: int, square_count: int, seed: int) -> list[BenchmarkSample]:
    rng = random.Random(seed)
    rect_samples = [sample for sample in samples if sample.shape == "rect"]
    square_samples = [sample for sample in samples if sample.shape == "square"]
    rng.shuffle(rect_samples)
    rng.shuffle(square_samples)
    if rect_count > 0:
        rect_samples = rect_samples[: min(int(rect_count), len(rect_samples))]
    if square_count > 0:
        square_samples = square_samples[: min(int(square_count), len(square_samples))]
    selected = rect_samples + square_samples
    rng.shuffle(selected)
    return selected


def _stratified_split(samples: list[BenchmarkSample], val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[BenchmarkSample]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[BenchmarkSample]] = {}
    for sample in samples:
        key = (sample.shape, sample.sample_class or "__none__")
        groups.setdefault(key, []).append(sample)

    splits = {"train": [], "val": [], "test": []}
    for key in sorted(groups.keys()):
        group = list(groups[key])
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * float(test_ratio)))
        n_val = int(round(n * float(val_ratio)))
        if n >= 3 and test_ratio > 0 and n_test == 0:
            n_test = 1
        if n >= 3 and val_ratio > 0 and n_val == 0:
            n_val = 1
        if n_test + n_val >= n:
            overflow = (n_test + n_val) - (n - 1)
            if overflow > 0:
                reduce_val = min(overflow, n_val)
                n_val -= reduce_val
                overflow -= reduce_val
                n_test = max(0, n_test - overflow)
        splits["test"].extend(group[:n_test])
        splits["val"].extend(group[n_test : n_test + n_val])
        splits["train"].extend(group[n_test + n_val :])

    if not splits["val"] and float(val_ratio) > 0.0 and len(samples) >= 2 and splits["train"]:
        splits["val"].append(splits["train"].pop())
    if not splits["test"] and float(test_ratio) > 0.0 and len(samples) >= 3 and len(splits["train"]) >= 2:
        splits["test"].append(splits["train"].pop())
    if not splits["train"]:
        donor = "val" if len(splits["val"]) > len(splits["test"]) else "test"
        if splits[donor]:
            splits["train"].append(splits[donor].pop())
    return splits


def build_private_benchmark(
    src_root: Path,
    shape_manifest: Path,
    out_root: Path,
    sample_class_manifest: Path | None = None,
    rect_count: int = 0,
    square_count: int = 0,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy_mode: str = "copy",
    min_label_len: int = 4,
    clean: bool = False,
) -> BenchmarkSummary:
    if clean and out_root.exists():
        shutil.rmtree(out_root)
    real_root = out_root / "real"
    manifests_root = out_root / "manifests"
    real_root.mkdir(parents=True, exist_ok=True)
    manifests_root.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(
        src_root=src_root,
        shape_manifest=shape_manifest,
        sample_class_manifest=sample_class_manifest,
        min_label_len=min_label_len,
    )
    selected = _sample_by_shape(samples, rect_count=rect_count, square_count=square_count, seed=seed)
    splits = _stratified_split(selected, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    shape_rows = []
    style_rows = []
    split_rows = []
    source_rows = []
    index = 1
    for split_name in ("train", "val", "test"):
        for sample in splits[split_name]:
            filename = f"{index:06d}_{sample.label}.png"
            out_path = real_root / filename
            _copy_file(sample.image_path, out_path, copy_mode=copy_mode)
            shape_rows.append({"file": filename, "match_key": Path(filename).stem, "shape": sample.shape})
            split_rows.append({"file": filename, "match_key": Path(filename).stem, "split": split_name})
            source_rows.append(
                {
                    "file": filename,
                    "match_key": Path(filename).stem,
                    "source_path": str(sample.image_path),
                    "source_match_key": sample.match_key,
                    "label": sample.label,
                    "shape": sample.shape,
                    "sample_class": sample.sample_class,
                }
            )
            if sample.sample_class:
                style_rows.append({"file": filename, "match_key": Path(filename).stem, "class_name": sample.sample_class})
            index += 1

    def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(manifests_root / "shape_manifest.csv", ["file", "match_key", "shape"], shape_rows)
    _write_csv(manifests_root / "split_manifest.csv", ["file", "match_key", "split"], split_rows)
    _write_csv(
        manifests_root / "source_manifest.csv",
        ["file", "match_key", "source_path", "source_match_key", "label", "shape", "sample_class"],
        source_rows,
    )
    if style_rows:
        _write_csv(manifests_root / "style_manifest.csv", ["file", "match_key", "class_name"], style_rows)

    summary = {
        "out_root": str(out_root),
        "src_root": str(src_root),
        "shape_manifest": str(shape_manifest),
        "sample_class_manifest": str(sample_class_manifest) if sample_class_manifest else "",
        "rect_requested": int(rect_count),
        "square_requested": int(square_count),
        "total_selected": len(selected),
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "rect_count": sum(1 for sample in selected if sample.shape == "rect"),
        "square_count": sum(1 for sample in selected if sample.shape == "square"),
        "style_labeled_count": sum(1 for sample in selected if sample.sample_class),
        "copy_mode": copy_mode,
        "seed": int(seed),
    }
    with open(manifests_root / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return BenchmarkSummary(
        out_root=str(out_root),
        total_selected=int(summary["total_selected"]),
        train_count=int(summary["train_count"]),
        val_count=int(summary["val_count"]),
        test_count=int(summary["test_count"]),
        rect_count=int(summary["rect_count"]),
        square_count=int(summary["square_count"]),
    )
