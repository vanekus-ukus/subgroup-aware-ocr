from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

from .labels import label_from_filename_stem, normalized_match_stem
from .preprocess import IMAGE_HEIGHT, IMAGE_WIDTH, SQUARE_AR_THRESHOLD, is_two_line_square_like, letterbox_rgb, preprocess_sequence_image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SAMPLE_CLASS_SEPARATOR = "|"


@dataclass(frozen=True)
class SequenceSample:
    image_path: Path
    label: str
    encoded_label: list[int]
    match_key: str = ""
    shape_flag: int = -1
    sample_class: str = ""


@dataclass(frozen=True)
class ShapeSample:
    image_path: Path
    label: int
    source: str = "source"


class SimpleSequenceAugment:
    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    def __call__(self, image: Image.Image, sample_index: int) -> Image.Image:
        rng = random.Random(self.seed + int(sample_index) * 9973)
        if rng.random() < 0.8:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.8, 1.2))
        if rng.random() < 0.8:
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
        if rng.random() < 0.35:
            image = image.rotate(rng.uniform(-3.0, 3.0), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        return image


class SequenceDataset(Dataset):
    def __init__(
        self,
        samples: list[SequenceSample],
        preprocess_mode: str,
        square_ar_threshold: float = SQUARE_AR_THRESHOLD,
        augment: bool = False,
        augment_seed: int = 42,
    ):
        self.samples = samples
        self.preprocess_mode = preprocess_mode
        self.square_ar_threshold = float(square_ar_threshold)
        self.augment = SimpleSequenceAugment(seed=augment_seed) if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as handle:
            image = handle.convert("RGB").copy()
        if self.augment is not None:
            image = self.augment(image, sample_index=index)
        square_like = int(is_two_line_square_like(image, square_ar_threshold=self.square_ar_threshold))
        image = preprocess_sequence_image(
            image,
            out_w=IMAGE_WIDTH,
            out_h=IMAGE_HEIGHT,
            square_ar_threshold=self.square_ar_threshold,
            preprocess_mode=self.preprocess_mode,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        array = (array - 0.5) / 0.5
        shape_flag = sample.shape_flag if sample.shape_flag in (0, 1) else square_like
        return (
            torch.from_numpy(array),
            torch.tensor(sample.encoded_label, dtype=torch.long),
            torch.tensor(shape_flag, dtype=torch.long),
            str(sample.sample_class or ""),
        )


class ShapeDataset(Dataset):
    def __init__(self, samples: list[ShapeSample], img_size: int = 96, train: bool = False, seed: int = 42):
        self.samples = samples
        self.img_size = int(img_size)
        self.train = bool(train)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample.image_path) as handle:
            image = handle.convert("RGB").copy()
        if self.train:
            rng = random.Random(self.seed + index * 53)
            if rng.random() < 0.7:
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.85, 1.15))
            if rng.random() < 0.4:
                image = image.rotate(rng.uniform(-3.0, 3.0), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        image = letterbox_rgb(image, out_w=self.img_size, out_h=self.img_size)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array), torch.tensor(sample.label, dtype=torch.long)


def collate_sequences(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    target_lengths = torch.tensor([target.size(0) for target in targets], dtype=torch.long)
    shape_flags = torch.stack([item[2] for item in batch], dim=0)
    sample_classes = [item[3] for item in batch]
    return images, torch.cat(targets, dim=0), target_lengths, shape_flags, sample_classes


def iter_image_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def encode_label(label: str, char_dict: dict[str, int]) -> list[int]:
    encoded: list[int] = []
    for char in label:
        if char not in char_dict:
            raise KeyError(f"Out-of-vocabulary symbol: {char!r}")
        encoded.append(char_dict[char])
    return encoded


def load_sequence_samples(
    data_root: Path,
    char_dict: dict[str, int],
    shape_map: dict[str, int] | None = None,
    sample_class_map: dict[str, str] | None = None,
) -> list[SequenceSample]:
    shape_map = shape_map or {}
    sample_class_map = sample_class_map or {}
    samples: list[SequenceSample] = []
    for image_path in iter_image_files(data_root):
        label = label_from_filename_stem(image_path.stem)
        if not label:
            continue
        encoded = encode_label(label, char_dict)
        key = normalized_match_stem(image_path.stem)
        samples.append(
            SequenceSample(
                image_path=image_path,
                label=label,
                encoded_label=encoded,
                match_key=key,
                shape_flag=int(shape_map.get(key, -1)),
                sample_class=str(sample_class_map.get(key, "")),
            )
        )
    if not samples:
        raise RuntimeError(f"No valid image/label pairs found under {data_root}")
    return samples


def _shape_to_flag(raw: str) -> int:
    lowered = str(raw or "").strip().lower()
    if lowered.startswith("sq"):
        return 1
    if lowered.startswith("rect"):
        return 0
    return -1


def load_shape_map_from_manifest(path: Path) -> tuple[dict[str, int], int]:
    if not path.exists():
        raise FileNotFoundError(path)
    shape_map: dict[str, int] = {}
    conflicts = 0
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        shape_col = next((name for name in ("shape", "plate_shape", "type") if name in fields), "")
        key_cols = [
            name
            for name in ("match_key", "crop_rel_path", "crop_path", "output_path", "file", "path")
            if name in fields
        ]
        if not shape_col:
            raise RuntimeError(f"No shape column in {path}")
        if not key_cols:
            raise RuntimeError(f"No key column in {path}")
        for row in reader:
            flag = _shape_to_flag(row.get(shape_col, ""))
            if flag < 0:
                continue
            key = _row_to_key(row, key_cols)
            if not key:
                continue
            prev = shape_map.get(key)
            if prev is not None and prev != flag:
                conflicts += 1
            shape_map[key] = flag
    return shape_map, conflicts


def load_sample_class_map_from_manifest(path: Path) -> tuple[dict[str, str], int]:
    if not path.exists():
        raise FileNotFoundError(path)
    class_map: dict[str, str] = {}
    conflicts = 0
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        class_col = next(
            (name for name in ("class_name", "sample_class", "color_class", "class", "group", "palette") if name in fields),
            "",
        )
        key_cols = [
            name
            for name in ("match_key", "crop_rel_path", "crop_path", "output_path", "file", "path")
            if name in fields
        ]
        if not class_col:
            raise RuntimeError(f"No class column in {path}")
        if not key_cols:
            raise RuntimeError(f"No key column in {path}")
        for row in reader:
            key = _row_to_key(row, key_cols)
            if not key:
                continue
            sample_class = str(row.get(class_col, "")).strip()
            if not sample_class:
                continue
            prev = class_map.get(key)
            if prev is not None and prev != sample_class:
                conflicts += 1
            class_map[key] = sample_class
    return class_map, conflicts


def manifest_namespace(path: Path) -> str:
    stem = path.stem.strip().lower()
    for suffix in ("_manifest", "_class", "_classes"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem or path.stem.strip().lower()


def combine_sample_class_names(values: list[str]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        name = str(value or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return SAMPLE_CLASS_SEPARATOR.join(ordered)


def split_sample_class_names(raw: str) -> list[str]:
    if not raw:
        return []
    return [chunk.strip() for chunk in str(raw).split(SAMPLE_CLASS_SEPARATOR) if chunk.strip()]


def load_sample_class_maps_from_manifests(paths: list[Path] | tuple[Path, ...]) -> tuple[dict[str, str], int]:
    manifest_paths = [Path(path) for path in paths if str(path)]
    if not manifest_paths:
        return {}, 0
    use_namespace = len(manifest_paths) > 1
    combined: dict[str, list[str]] = {}
    conflicts = 0
    for manifest_path in manifest_paths:
        loaded_map, local_conflicts = load_sample_class_map_from_manifest(manifest_path)
        conflicts += int(local_conflicts)
        namespace = manifest_namespace(manifest_path) if use_namespace else ""
        for key, value in loaded_map.items():
            encoded = f"{namespace}:{value}" if namespace else value
            bucket = combined.setdefault(key, [])
            if encoded not in bucket:
                bucket.append(encoded)
    return {key: combine_sample_class_names(values) for key, values in combined.items()}, conflicts


def load_hard_keys_from_manifest(path: Path, topk: int = 0) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    scored: dict[str, float] = {}
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        key_cols = [
            name
            for name in ("match_key", "crop_rel_path", "crop_path", "output_path", "file", "path")
            if name in fields
        ]
        score_col = next((name for name in ("norm_ed", "avg_norm_ed", "hard_score", "score") if name in fields), "")
        if not key_cols:
            raise RuntimeError(f"No key column in {path}")
        for row in reader:
            key = _row_to_key(row, key_cols)
            if not key:
                continue
            try:
                score = float(row.get(score_col, "1.0")) if score_col else 1.0
            except Exception:
                score = 1.0
            previous = scored.get(key)
            if previous is None or score > previous:
                scored[key] = score
    items = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
    if topk > 0:
        items = items[:topk]
    return [key for key, _ in items]


def load_split_map_from_manifest(path: Path) -> tuple[dict[str, str], int]:
    if not path.exists():
        raise FileNotFoundError(path)
    split_map: dict[str, str] = {}
    conflicts = 0
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        split_col = next((name for name in ("split", "subset", "fold") if name in fields), "")
        key_cols = [
            name
            for name in ("match_key", "crop_rel_path", "crop_path", "output_path", "file", "path")
            if name in fields
        ]
        if not split_col:
            raise RuntimeError(f"No split column in {path}")
        if not key_cols:
            raise RuntimeError(f"No key column in {path}")
        for row in reader:
            key = _row_to_key(row, key_cols)
            if not key:
                continue
            split_name = str(row.get(split_col, "")).strip().lower()
            if not split_name:
                continue
            prev = split_map.get(key)
            if prev is not None and prev != split_name:
                conflicts += 1
            split_map[key] = split_name
    return split_map, conflicts


def _row_to_key(row: dict[str, str], key_cols: list[str]) -> str:
    raw_match_key = str(row.get("match_key", "")).strip()
    if raw_match_key:
        return normalized_match_stem(raw_match_key)
    for column in key_cols:
        raw = str(row.get(column, "")).strip()
        if raw:
            return normalized_match_stem(Path(raw.replace("\\", "/")).stem)
    return ""


def split_train_val(samples: list, val_split: float, seed: int) -> tuple[list, list]:
    if len(samples) < 2:
        return list(samples), []
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split_index = int(round(len(samples) * (1.0 - float(val_split))))
    split_index = min(max(1, split_index), len(samples) - 1)
    train_idx = set(indices[:split_index])
    train = [samples[idx] for idx in range(len(samples)) if idx in train_idx]
    val = [samples[idx] for idx in range(len(samples)) if idx not in train_idx]
    return train, val
