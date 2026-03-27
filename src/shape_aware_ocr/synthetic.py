from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .dataset import encode_label, iter_image_files
from .labels import label_from_filename_stem


def load_synthetic_pool(
    synth_root: Path,
    char_dict: dict[str, int],
    allowed_keys: set[str] | None = None,
) -> tuple[dict[str, list[tuple[Path, list[int]]]], dict[str, int]]:
    if not synth_root.exists():
        raise FileNotFoundError(synth_root)
    pool: dict[str, list[tuple[Path, list[int]]]] = defaultdict(list)
    scanned = 0
    used = 0
    bad_label = 0
    for image_path in iter_image_files(synth_root):
        scanned += 1
        label = label_from_filename_stem(image_path.stem)
        if not label:
            bad_label += 1
            continue
        key = label
        if allowed_keys is not None and key not in allowed_keys:
            continue
        try:
            encoded = encode_label(label, char_dict)
        except KeyError:
            bad_label += 1
            continue
        pool[key].append((image_path, encoded))
        used += 1
    return pool, {
        "scanned_images": scanned,
        "used_images": used,
        "bad_or_oov_labels": bad_label,
        "keys_with_candidates": len(pool),
        "allowed_keys": -1 if allowed_keys is None else len(allowed_keys),
    }


def _late_decay_ratio(base_ratio: float, decay_last: int, epoch: int, total_epochs: int) -> float:
    ratio = max(0.0, float(base_ratio))
    if ratio <= 0.0 or decay_last <= 0:
        return ratio
    decay_length = min(int(decay_last), int(total_epochs))
    start_epoch = total_epochs - decay_length + 1
    if epoch < start_epoch:
        return ratio
    if decay_length <= 1:
        return 0.0
    position = epoch - start_epoch
    alpha = float(position) / float(decay_length - 1)
    return max(0.0, ratio * (1.0 - alpha))


def epoch_synth_ratio(
    base_ratio: float,
    decay_last: int,
    epoch: int,
    total_epochs: int,
    schedule_mode: str = "late_decay",
    warmup_epochs: int = 0,
    stop_epoch: int = 0,
) -> float:
    ratio = max(0.0, float(base_ratio))
    if ratio <= 0.0:
        return 0.0

    mode = str(schedule_mode or "late_decay").strip().lower()
    if mode in {"", "late_decay"}:
        return _late_decay_ratio(ratio, decay_last=decay_last, epoch=epoch, total_epochs=total_epochs)

    if mode == "constant":
        return ratio

    warmup_epochs = max(0, int(warmup_epochs))
    stop_epoch = max(0, int(stop_epoch))

    if mode == "warmup_stop":
        if warmup_epochs <= 0:
            return 0.0
        return ratio if int(epoch) <= warmup_epochs else 0.0

    if mode == "early_decay":
        if stop_epoch <= 1:
            return 0.0
        if int(epoch) >= stop_epoch:
            return 0.0
        alpha = float(max(0, int(epoch) - 1)) / float(max(1, stop_epoch - 1))
        return max(0.0, ratio * (1.0 - alpha))

    if mode == "warmup_decay":
        if warmup_epochs > 0 and int(epoch) <= warmup_epochs:
            return ratio
        if stop_epoch <= 0:
            stop_epoch = total_epochs
        if int(epoch) >= stop_epoch:
            return 0.0
        if stop_epoch <= warmup_epochs + 1:
            return 0.0
        position = int(epoch) - warmup_epochs
        span = stop_epoch - warmup_epochs
        alpha = float(max(0, position)) / float(max(1, span))
        return max(0.0, ratio * (1.0 - alpha))

    raise ValueError(f"Unsupported synth schedule mode: {schedule_mode!r}")


def sample_synthetic_for_epoch(
    pool: dict[str, list[tuple[Path, list[int]]]],
    target_count: int,
    rng: random.Random,
) -> tuple[list[Path], list[list[int]], list[int]]:
    if target_count <= 0 or not pool:
        return [], [], []
    keys = list(pool.keys())
    if target_count <= len(keys):
        chosen_keys = rng.sample(keys, target_count)
    else:
        chosen_keys = [rng.choice(keys) for _ in range(target_count)]
    image_paths: list[Path] = []
    encoded_labels: list[list[int]] = []
    shape_flags: list[int] = []
    for key in chosen_keys:
        image_path, encoded = rng.choice(pool[key])
        image_paths.append(image_path)
        encoded_labels.append(encoded)
        shape_flags.append(1)
    return image_paths, encoded_labels, shape_flags
