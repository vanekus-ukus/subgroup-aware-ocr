from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .alphabet import build_char_dict, load_alphabet
from .dataset import (
    SequenceDataset,
    collate_sequences,
    load_sample_class_maps_from_manifests,
    load_sequence_samples,
    load_shape_map_from_manifest,
    load_split_map_from_manifest,
    split_sample_class_names,
)
from .metrics import bootstrap_confidence_interval, ctc_greedy_decode, edit_distance, macro_group_cer, summarize_bucket, weighted_shape_cer
from .model import LPRNetTorch
from .preprocess import PREPROCESS_MODE_STAND, SQUARE_AR_THRESHOLD


@dataclass
class EvaluationConfig:
    data_root: str
    checkpoint: str
    alphabet: str
    out: str
    batch_size: int = 32
    workers: int = 0
    preprocess_mode: str = PREPROCESS_MODE_STAND
    square_ar_threshold: float = SQUARE_AR_THRESHOLD
    shape_manifest: str = ""
    sample_class_manifests: tuple[str, ...] = ()
    split_manifest: str = ""
    split_name: str = ""
    best_shape_square_weight: float = 0.65
    bootstrap_samples: int = 500
    bootstrap_seed: int = 42
    max_errors: int = 200


@dataclass
class EvaluationSummary:
    report_path: str
    errors_csv: str
    subgroup_csv: str


def _decode_to_strings(preds: list[list[int]], targets: list[list[int]], tokens: list[str]) -> tuple[list[str], list[str]]:
    pred_strings = ["".join(tokens[idx] for idx in seq if 0 <= idx < len(tokens)) for seq in preds]
    target_strings = ["".join(tokens[idx] for idx in seq if 0 <= idx < len(tokens)) for seq in targets]
    return pred_strings, target_strings


def _loader_kwargs(workers: int) -> dict:
    kwargs = {"num_workers": max(0, int(workers)), "pin_memory": torch.cuda.is_available()}
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def evaluate_ocr(config: EvaluationConfig) -> EvaluationSummary:
    out_root = Path(config.out)
    out_root.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(config.checkpoint, map_location="cpu")
    alphabet = load_alphabet(Path(config.alphabet))
    char_dict, blank_index, tokens = build_char_dict(alphabet)
    model = LPRNetTorch(num_classes=int(checkpoint["num_classes"]))
    model.load_state_dict(checkpoint["model_state_dict"])

    shape_map: dict[str, int] = {}
    if config.shape_manifest:
        shape_map, _ = load_shape_map_from_manifest(Path(config.shape_manifest))
    sample_class_map, _ = load_sample_class_maps_from_manifests(
        tuple(Path(manifest_path) for manifest_path in config.sample_class_manifests if manifest_path)
    )

    samples = load_sequence_samples(Path(config.data_root), char_dict=char_dict, shape_map=shape_map, sample_class_map=sample_class_map)
    split_conflicts = 0
    if config.split_manifest and config.split_name:
        split_map, split_conflicts = load_split_map_from_manifest(Path(config.split_manifest))
        wanted_split = str(config.split_name).strip().lower()
        samples = [sample for sample in samples if split_map.get(sample.match_key) == wanted_split]
        if not samples:
            raise RuntimeError(f"No samples found for split={wanted_split!r}")
    dataset = SequenceDataset(samples, preprocess_mode=config.preprocess_mode, square_ar_threshold=config.square_ar_threshold, augment=False)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_sequences, **_loader_kwargs(config.workers))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    bucket_overall = {"edits": 0, "chars": 0, "exact": 0, "samples": 0}
    bucket_square = {"edits": 0, "chars": 0, "exact": 0, "samples": 0}
    bucket_rect = {"edits": 0, "chars": 0, "exact": 0, "samples": 0}
    bucket_classes: dict[str, dict[str, float]] = {}
    per_sample_cer: list[float] = []
    per_sample_exact: list[float] = []
    error_rows: list[dict[str, object]] = []
    sample_index = 0

    with torch.no_grad():
        for batch_idx, (images, targets, target_lengths, shape_flags, sample_classes) in enumerate(tqdm(loader, desc="Eval", unit="batch")):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = ctc_greedy_decode(logits.log_softmax(dim=2), blank_index=blank_index)

            offset = 0
            target_lists: list[list[int]] = []
            for length in target_lengths.tolist():
                target = targets[offset : offset + length].tolist()
                target_lists.append(target)
                offset += length

            pred_strings, target_strings = _decode_to_strings(preds, target_lists, tokens)
            for local_idx, (pred, target) in enumerate(zip(preds, target_lists)):
                gt_text = target_strings[local_idx]
                pred_text = pred_strings[local_idx]
                edits = edit_distance(pred, target)
                norm_ed = float(edits) / float(max(1, len(target)))
                exact = float(pred == target)
                shape_bucket = bucket_square if int(shape_flags[local_idx].item()) == 1 else bucket_rect
                for bucket in (bucket_overall, shape_bucket):
                    bucket["edits"] += edits
                    bucket["chars"] += len(target)
                    bucket["exact"] += exact
                    bucket["samples"] += 1
                sample_class = sample_classes[local_idx].strip()
                for class_name in split_sample_class_names(sample_class):
                    bucket = bucket_classes.setdefault(class_name, {"edits": 0, "chars": 0, "exact": 0, "samples": 0})
                    bucket["edits"] += edits
                    bucket["chars"] += len(target)
                    bucket["exact"] += exact
                    bucket["samples"] += 1
                per_sample_cer.append(norm_ed)
                per_sample_exact.append(exact)
                if pred != target:
                    error_rows.append(
                        {
                            "index": sample_index,
                            "gt": gt_text,
                            "pred": pred_text,
                            "norm_ed": norm_ed,
                            "shape": "square" if int(shape_flags[local_idx].item()) == 1 else "rect",
                            "sample_class": sample_class,
                            "file": str(samples[sample_index].image_path),
                        }
                    )
                sample_index += 1

    cer, exact = summarize_bucket(bucket_overall)
    cer_square, exact_square = summarize_bucket(bucket_square)
    cer_rect, exact_rect = summarize_bucket(bucket_rect)
    weighted = weighted_shape_cer(cer_square, cer_rect, config.best_shape_square_weight, fallback=cer)
    macro = macro_group_cer(cer_square, cer_rect, fallback=cer)

    subgroup_rows = [
        {"group": "overall", "cer": cer, "exact": exact, "samples": int(bucket_overall["samples"]), "chars": int(bucket_overall["chars"])} ,
        {"group": "square", "cer": cer_square, "exact": exact_square, "samples": int(bucket_square["samples"]), "chars": int(bucket_square["chars"])} ,
        {"group": "rect", "cer": cer_rect, "exact": exact_rect, "samples": int(bucket_rect["samples"]), "chars": int(bucket_rect["chars"])} ,
    ]
    for class_name, bucket in sorted(bucket_classes.items()):
        class_cer, class_exact = summarize_bucket(bucket)
        subgroup_rows.append(
            {
                "group": f"sample_class:{class_name}",
                "cer": class_cer,
                "exact": class_exact,
                "samples": int(bucket["samples"]),
                "chars": int(bucket["chars"]),
            }
        )

    report = {
        "samples": int(bucket_overall["samples"]),
        "chars": int(bucket_overall["chars"]),
        "cer": cer,
        "exact_match": exact,
        "cer_square": cer_square,
        "cer_rect": cer_rect,
        "weighted_shape_cer": weighted,
        "macro_shape_cer": macro,
        "exact_square": exact_square,
        "exact_rect": exact_rect,
        "bootstrap": {
            "cer": bootstrap_confidence_interval(per_sample_cer, n_bootstrap=config.bootstrap_samples, seed=config.bootstrap_seed),
            "exact": bootstrap_confidence_interval(per_sample_exact, n_bootstrap=config.bootstrap_samples, seed=config.bootstrap_seed),
        },
        "checkpoint": str(config.checkpoint),
        "alphabet": str(config.alphabet),
        "data_root": str(config.data_root),
        "split_manifest": str(config.split_manifest),
        "split_name": str(config.split_name),
        "split_conflicts": int(split_conflicts),
    }

    report_path = out_root / "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    subgroup_csv = out_root / "subgroup_metrics.csv"
    with open(subgroup_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "cer", "exact", "samples", "chars"])
        writer.writeheader()
        writer.writerows(subgroup_rows)

    error_rows.sort(key=lambda row: float(row["norm_ed"]), reverse=True)
    errors_csv = out_root / "eval_errors_top.csv"
    with open(errors_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "gt", "pred", "norm_ed", "shape", "sample_class", "file"])
        writer.writeheader()
        writer.writerows(error_rows[: max(0, int(config.max_errors))])

    return EvaluationSummary(report_path=str(report_path), errors_csv=str(errors_csv), subgroup_csv=str(subgroup_csv))
