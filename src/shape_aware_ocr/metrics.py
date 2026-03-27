from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable

import numpy as np
import torch

SAMPLE_CLASS_SEPARATOR = "|"


def edit_distance(seq_a: list[int], seq_b: list[int]) -> int:
    n = len(seq_a)
    m = len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        table[i][0] = i
    for j in range(m + 1):
        table[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1, table[i - 1][j - 1] + cost)
    return table[n][m]


def ctc_greedy_decode(log_probs: torch.Tensor, blank_index: int) -> list[list[int]]:
    time_steps, batch_size, _ = log_probs.shape
    preds = log_probs.detach().cpu().argmax(dim=2)
    out: list[list[int]] = []
    for batch_idx in range(batch_size):
        prev = None
        seq = []
        for time_idx in range(time_steps):
            token = int(preds[time_idx, batch_idx].item())
            if token == blank_index:
                prev = token
                continue
            if prev is not None and token == prev:
                prev = token
                continue
            seq.append(token)
            prev = token
        out.append(seq)
    return out


def _empty_bucket() -> dict[str, float]:
    return {"edits": 0.0, "chars": 0.0, "exact": 0.0, "samples": 0.0}


def init_eval_stats() -> dict[str, object]:
    return {
        "overall": _empty_bucket(),
        "square": _empty_bucket(),
        "rect": _empty_bucket(),
        "sample_classes": defaultdict(_empty_bucket),
    }


def update_bucket(bucket: dict[str, float], pred: list[int], target: list[int]) -> None:
    edits = edit_distance(pred, target)
    bucket["edits"] += float(edits)
    bucket["chars"] += float(len(target))
    bucket["exact"] += float(pred == target)
    bucket["samples"] += 1.0


def accumulate_stats(
    stats: dict[str, object],
    pred_sequences: list[list[int]],
    target_concat: torch.Tensor,
    target_lengths: torch.Tensor,
    shape_flags: torch.Tensor,
    sample_classes: list[str] | None = None,
) -> None:
    classes = sample_classes or []
    offset = 0
    for index, length in enumerate(target_lengths.tolist()):
        target = target_concat[offset : offset + length].tolist()
        offset += length
        pred = pred_sequences[index]
        update_bucket(stats["overall"], pred, target)
        if int(shape_flags[index]) == 1:
            update_bucket(stats["square"], pred, target)
        else:
            update_bucket(stats["rect"], pred, target)
        sample_class = classes[index].strip() if index < len(classes) else ""
        if sample_class:
            for class_name in [chunk.strip() for chunk in sample_class.split(SAMPLE_CLASS_SEPARATOR) if chunk.strip()]:
                update_bucket(stats["sample_classes"][class_name], pred, target)


def summarize_bucket(bucket: dict[str, float]) -> tuple[float, float]:
    chars = float(bucket["chars"])
    samples = float(bucket["samples"])
    cer = math.nan if chars <= 0 else float(bucket["edits"]) / chars
    exact = math.nan if samples <= 0 else float(bucket["exact"]) / samples
    return cer, exact


def weighted_shape_cer(cer_square: float, cer_rect: float, square_weight: float, fallback: float) -> float:
    weight = min(max(float(square_weight), 0.0), 1.0)
    if math.isnan(cer_square) and math.isnan(cer_rect):
        return float(fallback)
    if math.isnan(cer_square):
        return float(cer_rect)
    if math.isnan(cer_rect):
        return float(cer_square)
    return (weight * float(cer_square)) + ((1.0 - weight) * float(cer_rect))


def macro_group_cer(square_cer: float, rect_cer: float, fallback: float) -> float:
    values = [value for value in (square_cer, rect_cer) if not math.isnan(value)]
    if not values:
        return float(fallback)
    return float(sum(values)) / float(len(values))


def per_sample_error_series(preds: Iterable[list[int]], targets: Iterable[list[int]]) -> tuple[list[float], list[float]]:
    cer_values: list[float] = []
    exact_values: list[float] = []
    for pred, target in zip(preds, targets):
        edits = edit_distance(pred, target)
        cer_values.append(float(edits) / float(max(1, len(target))))
        exact_values.append(float(pred == target))
    return cer_values, exact_values


def bootstrap_confidence_interval(values: list[float], n_bootstrap: int = 500, alpha: float = 0.05, seed: int = 42) -> dict[str, float]:
    if not values:
        return {"mean": math.nan, "lower": math.nan, "upper": math.nan, "samples": 0}
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(max(1, int(n_bootstrap))):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(float(sum(draw)) / float(len(draw)))
    lower_q = alpha / 2.0
    upper_q = 1.0 - (alpha / 2.0)
    return {
        "mean": float(sum(values)) / float(len(values)),
        "lower": float(np.quantile(samples, lower_q)),
        "upper": float(np.quantile(samples, upper_q)),
        "samples": n,
    }
