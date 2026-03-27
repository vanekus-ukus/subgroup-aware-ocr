from __future__ import annotations

import json
from pathlib import Path

from .dataset import iter_image_files
from .labels import label_from_filename_stem


def build_alphabet_from_labels(labels: list[str]) -> dict:
    symbols = sorted({ch for label in labels for ch in label})
    tokens = symbols + ["<BLANK>"]
    blank_index = len(tokens) - 1
    return {
        "tokens": symbols,
        "blank_token": "<BLANK>",
        "blank_index": blank_index,
        "num_classes": len(tokens),
    }


def build_alphabet_from_root(data_root: Path) -> dict:
    labels = []
    for image_path in iter_image_files(data_root):
        label = label_from_filename_stem(image_path.stem)
        if label:
            labels.append(label)
    if not labels:
        raise RuntimeError(f"No valid labels found under {data_root}")
    return build_alphabet_from_labels(labels)


def save_alphabet(payload: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_alphabet(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_char_dict(alphabet: dict) -> tuple[dict[str, int], int, list[str]]:
    tokens = list(alphabet["tokens"])
    char_dict = {ch: idx for idx, ch in enumerate(tokens)}
    blank_index = int(alphabet["blank_index"])
    return char_dict, blank_index, tokens
