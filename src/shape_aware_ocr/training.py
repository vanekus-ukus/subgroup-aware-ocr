from __future__ import annotations

import csv
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

from .alphabet import build_char_dict, load_alphabet
from .dataset import (
    SequenceDataset,
    SequenceSample,
    collate_sequences,
    load_hard_keys_from_manifest,
    load_sample_class_maps_from_manifests,
    load_sequence_samples,
    load_shape_map_from_manifest,
    load_split_map_from_manifest,
    split_train_val,
)
from .metrics import (
    accumulate_stats,
    ctc_greedy_decode,
    init_eval_stats,
    macro_group_cer,
    summarize_bucket,
    weighted_shape_cer,
)
from .model import LPRNetTorch
from .preprocess import PREPROCESS_MODE_STAND, SQUARE_AR_THRESHOLD
from .synthetic import epoch_synth_ratio, load_synthetic_pool, sample_synthetic_for_epoch


@dataclass
class TrainingConfig:
    data_root: str
    out: str
    alphabet: str
    epochs: int = 40
    batch_size: int = 32
    train_workers: int = 0
    val_workers: int = 0
    dataloader_prefetch: int = 2
    dataloader_persistent_workers: bool = False
    val_split: float = 0.1
    seed: int = 42
    patience: int = 8
    min_delta: float = 1e-4
    lr: float = 7e-4
    weight_decay: float = 1e-4
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    best_by: str = "weighted_shape"
    best_shape_square_weight: float = 0.65
    preprocess_mode: str = PREPROCESS_MODE_STAND
    square_ar_threshold: float = SQUARE_AR_THRESHOLD
    shape_manifest: str = ""
    sample_class_manifests: tuple[str, ...] = ()
    split_manifest: str = ""
    square_oversample: float = 1.0
    synth_root: str = ""
    synth_ratio: float = 0.0
    synth_decay_last: int = 0
    synth_schedule_mode: str = "late_decay"
    synth_warmup_epochs: int = 0
    synth_stop_epoch: int = 0
    synth_max_per_epoch: int = 0
    synth_use_all: bool = False
    synth_anchor_mode: str = "square"
    hard_manifest: str = ""
    hard_topk: int = 0
    hard_oversample: float = 1.0
    finetune_square_oversample: float = 1.0
    finetune_hard_oversample: float = 1.0
    amp: bool = True
    plot_file: str = "train_curve.png"
    sample_class_report_file: str = "train_val_sample_class_metrics.csv"
    sample_class_print_limit: int = 6


@dataclass
class TrainingSummary:
    best_checkpoint: str
    last_checkpoint: str
    run_config: str
    train_log: str
    best_score: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:  # pragma: no cover - compatibility fallback
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _loader_kwargs(workers: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int) -> dict:
    kwargs = {"num_workers": max(0, int(workers)), "pin_memory": bool(pin_memory)}
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return kwargs


def _make_jsonable_config(config: TrainingConfig) -> dict:
    payload = asdict(config)
    payload["sample_class_manifests"] = list(config.sample_class_manifests)
    return payload


def _score_from_metrics(best_by: str, val_loss: float, val_cer: float, weighted_cer: float) -> float:
    if best_by == "loss":
        return float(val_loss)
    if best_by == "cer":
        return float(val_cer)
    return float(weighted_cer)


def _duplicate_samples(samples: list[SequenceSample], factor: float, seed: int) -> list[SequenceSample]:
    if factor <= 1.0 or not samples:
        return []
    rng = random.Random(seed)
    extra: list[SequenceSample] = []
    whole = max(0, int(factor) - 1)
    frac = max(0.0, float(factor) - int(factor))
    for _ in range(whole):
        extra.extend(samples)
    if frac > 0.0:
        count = min(len(samples), int(round(frac * len(samples))))
        if count > 0:
            extra.extend(rng.sample(samples, count))
    return extra


def _split_from_manifest(
    samples: list[SequenceSample],
    split_map: dict[str, str],
) -> tuple[list[SequenceSample], list[SequenceSample], list[SequenceSample]]:
    train_samples = [sample for sample in samples if split_map.get(sample.match_key) == "train"]
    val_samples = [sample for sample in samples if split_map.get(sample.match_key) == "val"]
    test_samples = [sample for sample in samples if split_map.get(sample.match_key) == "test"]
    return train_samples, val_samples, test_samples


def _save_plot(plot_path: Path, epochs: list[int], train_losses: list[float], val_losses: list[float], val_cers: list[float], lrs: list[float]) -> None:
    if plt is None or not epochs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, train_losses, label="train_loss")
    axes[0].plot(epochs, val_losses, label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(epochs, val_cers, color="tab:orange", label="val_cer")
    axes[1].set_title("CER")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(epochs, lrs, color="tab:green", label="lr")
    axes[2].set_title("Learning Rate")
    axes[2].legend()
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    blank_index: int,
    amp_enabled: bool,
) -> tuple[float, dict[str, object], list[dict[str, object]]]:
    model.eval()
    losses = []
    stats = init_eval_stats()
    sample_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for images, targets, target_lengths, shape_flags, sample_classes in tqdm(loader, desc="Val", unit="batch"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                time_steps, batch_size, _ = logits.shape
                log_probs = logits.log_softmax(dim=2)
                input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            losses.append(float(loss.item()))
            pred_sequences = ctc_greedy_decode(log_probs, blank_index=blank_index)
            accumulate_stats(stats, pred_sequences, targets, target_lengths, shape_flags, sample_classes)
    sample_class_metrics = []
    for name, bucket in stats["sample_classes"].items():
        cer, exact = summarize_bucket(bucket)
        sample_class_metrics.append(
            {
                "class_name": name,
                "val_cer": cer,
                "val_exact": exact,
                "val_samples": int(bucket["samples"]),
            }
        )
    sample_class_metrics.sort(key=lambda row: (-int(row["val_samples"]), str(row["class_name"])))
    mean_loss = float(np.mean(losses)) if losses else math.nan
    return mean_loss, stats, sample_class_metrics


def train_ocr(config: TrainingConfig) -> TrainingSummary:
    if config.synth_ratio < 0.0:
        raise ValueError("synth_ratio must be >= 0")
    if config.hard_oversample < 1.0:
        raise ValueError("hard_oversample must be >= 1")
    if config.finetune_square_oversample < 1.0:
        raise ValueError("finetune_square_oversample must be >= 1")
    if config.finetune_hard_oversample < 1.0:
        raise ValueError("finetune_hard_oversample must be >= 1")

    set_seed(config.seed)
    data_root = Path(config.data_root)
    out_root = Path(config.out)
    out_root.mkdir(parents=True, exist_ok=True)

    alphabet = load_alphabet(Path(config.alphabet))
    char_dict, blank_index, tokens = build_char_dict(alphabet)
    num_classes = int(alphabet["num_classes"])

    shape_map: dict[str, int] = {}
    shape_conflicts = 0
    if config.shape_manifest:
        shape_map, shape_conflicts = load_shape_map_from_manifest(Path(config.shape_manifest))

    sample_class_map, sample_class_conflicts = load_sample_class_maps_from_manifests(
        tuple(Path(manifest_path) for manifest_path in config.sample_class_manifests if manifest_path)
    )

    samples = load_sequence_samples(
        data_root=data_root,
        char_dict=char_dict,
        shape_map=shape_map,
        sample_class_map=sample_class_map,
    )
    split_conflicts = 0
    test_samples: list[SequenceSample] = []
    if config.split_manifest:
        split_map, split_conflicts = load_split_map_from_manifest(Path(config.split_manifest))
        train_samples, val_samples, test_samples = _split_from_manifest(samples, split_map)
    else:
        train_samples, val_samples = split_train_val(samples, val_split=config.val_split, seed=config.seed)
    if not val_samples:
        raise RuntimeError("Validation split is empty")
    if not train_samples:
        raise RuntimeError("Training split is empty")

    base_train_samples = list(train_samples)
    square_train_real = [sample for sample in base_train_samples if sample.shape_flag == 1]

    if config.square_oversample > 1.0:
        train_samples = train_samples + _duplicate_samples(square_train_real, config.square_oversample, seed=config.seed + 13)

    hard_keys: set[str] = set()
    hard_square_real: list[SequenceSample] = []
    if config.hard_manifest:
        hard_keys = set(load_hard_keys_from_manifest(Path(config.hard_manifest), topk=config.hard_topk))
        hard_square_real = [sample for sample in base_train_samples if sample.shape_flag == 1 and sample.match_key in hard_keys]
        train_samples = train_samples + _duplicate_samples(hard_square_real, config.hard_oversample, seed=config.seed + 29)

    synth_pool = {}
    synth_stats = {
        "scanned_images": 0,
        "used_images": 0,
        "bad_or_oov_labels": 0,
        "keys_with_candidates": 0,
        "allowed_keys": 0,
    }
    synth_anchor_keys: list[str] = []
    if config.synth_ratio > 0.0 and config.synth_root:
        real_square_keys = {sample.label for sample in base_train_samples if sample.shape_flag == 1 and sample.label}
        hard_square_label_keys = {sample.label for sample in hard_square_real if sample.label}
        anchor_mode = str(config.synth_anchor_mode or "square").strip().lower()
        if anchor_mode == "hard_square" and not hard_square_label_keys:
            raise RuntimeError("synth_anchor_mode='hard_square' requires a non-empty hard square subset")
        if config.synth_use_all or anchor_mode == "all":
            allowed = None
            synth_anchor_keys = sorted(synth_pool.keys()) if synth_pool else []
        elif anchor_mode == "hard_square":
            allowed = hard_square_label_keys
            synth_anchor_keys = sorted(hard_square_label_keys)
        else:
            allowed = real_square_keys
            synth_anchor_keys = sorted(real_square_keys)
        synth_pool, synth_stats = load_synthetic_pool(Path(config.synth_root), char_dict=char_dict, allowed_keys=allowed)
        if config.synth_use_all or anchor_mode == "all":
            synth_anchor_keys = sorted(synth_pool.keys())
        else:
            synth_anchor_keys = sorted(set(synth_anchor_keys) & set(synth_pool.keys()))

    train_loader_kwargs = _loader_kwargs(config.train_workers, pin_memory=torch.cuda.is_available(), persistent_workers=config.dataloader_persistent_workers, prefetch_factor=config.dataloader_prefetch)
    val_loader_kwargs = _loader_kwargs(config.val_workers, pin_memory=torch.cuda.is_available(), persistent_workers=config.dataloader_persistent_workers, prefetch_factor=config.dataloader_prefetch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = _build_grad_scaler(enabled=amp_enabled)

    model = LPRNetTorch(num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=blank_index, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=config.min_delta,
        min_lr=1e-6,
    )

    checkpoints_dir = out_root / "checkpoints"
    if checkpoints_dir.exists():
        shutil.rmtree(checkpoints_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoints_dir / "best.pt"
    best_cer_path = checkpoints_dir / "best_cer.pt"
    best_weighted_path = checkpoints_dir / "best_weighted_shape.pt"
    last_path = checkpoints_dir / "last.pt"
    log_path = out_root / "train_log.csv"
    class_report_path = out_root / config.sample_class_report_file
    plot_path = out_root / config.plot_file

    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "epoch",
            "lr",
            "train_loss",
            "val_loss",
            "val_cer",
            "val_cer_square",
            "val_cer_rect",
            "val_cer_weighted_shape",
            "val_cer_macro_shape",
            "val_exact",
            "val_exact_square",
            "val_exact_rect",
            "train_synth_ratio",
            "train_synth_count",
        ])

    with open(class_report_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "class_name", "val_cer", "val_exact", "val_samples"])

    run_config = _make_jsonable_config(config)
    run_config.update(
        {
            "num_classes": num_classes,
            "tokens": tokens,
            "blank_index": blank_index,
            "shape_conflicts": shape_conflicts,
            "sample_class_conflicts": sample_class_conflicts,
            "split_manifest": config.split_manifest,
            "split_conflicts": split_conflicts,
            "train_count": len(train_samples),
            "val_count": len(val_samples),
            "test_count": len(test_samples),
            "synth_stats": synth_stats,
            "base_square_train_count": len(square_train_real),
            "base_hard_square_train_count": len(hard_square_real),
            "synth_anchor_count": len(synth_anchor_keys),
        }
    )
    run_config_path = out_root / "run_config.json"
    with open(run_config_path, "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, ensure_ascii=False, indent=2)

    best_score = float("inf")
    best_val_loss = float("inf")
    best_cer = float("inf")
    best_weighted_shape = float("inf")
    epochs_no_improve = 0
    hist_epoch: list[int] = []
    hist_train_loss: list[float] = []
    hist_val_loss: list[float] = []
    hist_val_cer: list[float] = []
    hist_lr: list[float] = []

    for epoch in range(1, config.epochs + 1):
        epoch_samples = list(train_samples)
        epoch_synth_mix_ratio = 0.0
        epoch_synth_count = 0
        if synth_pool and config.synth_ratio > 0.0:
            epoch_synth_mix_ratio = epoch_synth_ratio(
                config.synth_ratio,
                config.synth_decay_last,
                epoch,
                config.epochs,
                schedule_mode=config.synth_schedule_mode,
                warmup_epochs=config.synth_warmup_epochs,
                stop_epoch=config.synth_stop_epoch,
            )
            target_count = int(round(epoch_synth_mix_ratio * len(synth_anchor_keys)))
            if config.synth_max_per_epoch > 0:
                target_count = min(target_count, int(config.synth_max_per_epoch))
            synth_paths, synth_labels, synth_flags = sample_synthetic_for_epoch(synth_pool, target_count=target_count, rng=random.Random(config.seed + 100000 + epoch))
            epoch_synth_count = len(synth_paths)
            for image_path, encoded_label, shape_flag in zip(synth_paths, synth_labels, synth_flags):
                epoch_samples.append(
                    SequenceSample(
                        image_path=image_path,
                        label="",
                        encoded_label=encoded_label,
                        shape_flag=shape_flag,
                        sample_class="synthetic_square",
                    )
                )
        if epoch_synth_mix_ratio <= 0.0 and config.finetune_square_oversample > 1.0:
            epoch_samples.extend(
                _duplicate_samples(square_train_real, config.finetune_square_oversample, seed=config.seed + 300000 + epoch)
            )
        if epoch_synth_mix_ratio <= 0.0 and config.finetune_hard_oversample > 1.0:
            epoch_samples.extend(
                _duplicate_samples(hard_square_real, config.finetune_hard_oversample, seed=config.seed + 400000 + epoch)
            )

        train_dataset = SequenceDataset(
            epoch_samples,
            preprocess_mode=config.preprocess_mode,
            square_ar_threshold=config.square_ar_threshold,
            augment=True,
            augment_seed=config.seed + epoch,
        )
        val_dataset = SequenceDataset(
            val_samples,
            preprocess_mode=config.preprocess_mode,
            square_ar_threshold=config.square_ar_threshold,
            augment=False,
            augment_seed=config.seed,
        )
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_sequences, **train_loader_kwargs)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_sequences, **val_loader_kwargs)

        print(f"\n=== Epoch {epoch}/{config.epochs} ===")
        model.train()
        train_losses = []
        for images, targets, target_lengths, _shape_flags, _sample_classes in tqdm(train_loader, desc="Train", unit="batch"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                time_steps, batch_size, _ = logits.shape
                log_probs = logits.log_softmax(dim=2)
                input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        val_loss, val_stats, sample_class_metrics = _evaluate(model, val_loader, criterion, device, blank_index, amp_enabled)
        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        val_cer, val_exact = summarize_bucket(val_stats["overall"])
        val_cer_square, val_exact_square = summarize_bucket(val_stats["square"])
        val_cer_rect, val_exact_rect = summarize_bucket(val_stats["rect"])
        val_cer_weighted = weighted_shape_cer(val_cer_square, val_cer_rect, config.best_shape_square_weight, fallback=val_cer)
        val_cer_macro = macro_group_cer(val_cer_square, val_cer_rect, fallback=val_cer)
        current_lr = float(optimizer.param_groups[0]["lr"])

        scheduler_metric = _score_from_metrics(config.best_by, val_loss, val_cer, val_cer_weighted)
        scheduler.step(scheduler_metric)

        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                epoch,
                f"{current_lr:.8f}",
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_cer:.6f}",
                "" if math.isnan(val_cer_square) else f"{val_cer_square:.6f}",
                "" if math.isnan(val_cer_rect) else f"{val_cer_rect:.6f}",
                "" if math.isnan(val_cer_weighted) else f"{val_cer_weighted:.6f}",
                "" if math.isnan(val_cer_macro) else f"{val_cer_macro:.6f}",
                "" if math.isnan(val_exact) else f"{val_exact:.6f}",
                "" if math.isnan(val_exact_square) else f"{val_exact_square:.6f}",
                "" if math.isnan(val_exact_rect) else f"{val_exact_rect:.6f}",
                f"{epoch_synth_mix_ratio:.6f}",
                epoch_synth_count,
            ])

        with open(class_report_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for row in sample_class_metrics:
                writer.writerow([
                    epoch,
                    row["class_name"],
                    "" if math.isnan(float(row["val_cer"])) else f"{float(row['val_cer']):.6f}",
                    "" if math.isnan(float(row["val_exact"])) else f"{float(row['val_exact']):.6f}",
                    int(row["val_samples"]),
                ])

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"val_CER={val_cer:.4f} (sq={val_cer_square:.4f}, rect={val_cer_rect:.4f}, w={val_cer_weighted:.4f}, macro={val_cer_macro:.4f}), "
            f"val_exact={val_exact:.4f} (sq={val_exact_square:.4f}, rect={val_exact_rect:.4f}), "
            f"synth={epoch_synth_count} (ratio={epoch_synth_mix_ratio:.3f}), lr={current_lr:.2e}"
        )
        preview_rows = sample_class_metrics[: max(0, int(config.sample_class_print_limit))]
        if preview_rows:
            preview = "; ".join(
                f"{row['class_name']}: exact={row['val_exact']:.4f}, cer={row['val_cer']:.4f}, n={row['val_samples']}"
                for row in preview_rows
            )
            print(f"[INFO] Val sample classes: {preview}")

        hist_epoch.append(epoch)
        hist_train_loss.append(train_loss)
        hist_val_loss.append(val_loss)
        hist_val_cer.append(val_cer)
        hist_lr.append(current_lr)
        _save_plot(plot_path, hist_epoch, hist_train_loss, hist_val_loss, hist_val_cer, hist_lr)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "num_classes": num_classes,
            "tokens": tokens,
            "blank_index": blank_index,
            "config": _make_jsonable_config(config),
            "epoch": epoch,
            "metrics": {
                "val_loss": val_loss,
                "val_cer": val_cer,
                "val_cer_square": val_cer_square,
                "val_cer_rect": val_cer_rect,
                "val_cer_weighted_shape": val_cer_weighted,
                "val_exact": val_exact,
            },
        }
        torch.save(checkpoint, last_path)

        current_score = _score_from_metrics(config.best_by, val_loss, val_cer, val_cer_weighted)
        improved = current_score < (best_score - config.min_delta)
        if improved:
            best_score = current_score
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(checkpoint, best_path)
            print(f"[INFO] New best by {config.best_by}: score={best_score:.6f}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] No improvement: {epochs_no_improve} epoch(s)")

        if val_cer < (best_cer - config.min_delta):
            best_cer = val_cer
            torch.save(checkpoint, best_cer_path)
            print(f"[INFO] New best by cer: score={best_cer:.6f}")

        if val_cer_weighted < (best_weighted_shape - config.min_delta):
            best_weighted_shape = val_cer_weighted
            torch.save(checkpoint, best_weighted_path)
            print(f"[INFO] New best by weighted_shape: score={best_weighted_shape:.6f}")

        if epochs_no_improve >= config.patience:
            print("[INFO] Early stopping")
            break

    return TrainingSummary(
        best_checkpoint=str(best_path),
        last_checkpoint=str(last_path),
        run_config=str(run_config_path),
        train_log=str(log_path),
        best_score=float(best_score),
    )
