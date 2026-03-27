from __future__ import annotations

import csv
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .dataset import ShapeDataset, ShapeSample, load_shape_map_from_manifest, split_train_val
from .model import ShapeClassifier


@dataclass
class ShapeClassifierConfig:
    data_root: str
    shape_manifest: str
    out: str
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.1
    seed: int = 42
    patience: int = 6
    min_delta: float = 1e-4
    img_size: int = 96
    train_workers: int = 0
    val_workers: int = 0
    balanced_sampler: bool = True
    amp: bool = True


@dataclass
class ShapeClassifierSummary:
    best_checkpoint: str
    last_checkpoint: str
    report: str


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


def _build_loader_kwargs(workers: int, pin_memory: bool) -> dict:
    kwargs = {"num_workers": max(0, int(workers)), "pin_memory": bool(pin_memory)}
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def _count_classes(samples: list[ShapeSample]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for sample in samples:
        counts[sample.label] += 1
    return counts


def _compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    preds = logits.argmax(dim=1)
    tp = int(((preds == 1) & (targets == 1)).sum().item())
    tn = int(((preds == 0) & (targets == 0)).sum().item())
    fp = int(((preds == 1) & (targets == 0)).sum().item())
    fn = int(((preds == 0) & (targets == 1)).sum().item())
    total = max(1, tp + tn + fp + fn)
    acc = (tp + tn) / total
    square_recall = tp / max(1, tp + fn)
    rect_recall = tn / max(1, tn + fp)
    balanced_acc = 0.5 * (square_recall + rect_recall)
    square_precision = tp / max(1, tp + fp)
    f1_square = (2.0 * square_precision * square_recall) / max(1e-12, square_precision + square_recall)
    return {
        "acc": float(acc),
        "balanced_acc": float(balanced_acc),
        "square_recall": float(square_recall),
        "rect_recall": float(rect_recall),
        "square_precision": float(square_precision),
        "f1_square": float(f1_square),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _load_samples(data_root: Path, shape_manifest: Path) -> list[ShapeSample]:
    shape_map, _ = load_shape_map_from_manifest(shape_manifest)
    samples: list[ShapeSample] = []
    for image_path in sorted(data_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        key = image_path.stem
        if key not in shape_map:
            from .labels import normalized_match_stem

            key = normalized_match_stem(image_path.stem)
        label = int(shape_map.get(key, -1))
        if label in (0, 1):
            samples.append(ShapeSample(image_path=image_path, label=label, source="source"))
    if not samples:
        raise RuntimeError("No samples matched the shape manifest")
    return samples


def train_shape_classifier(config: ShapeClassifierConfig) -> ShapeClassifierSummary:
    set_seed(config.seed)
    data_root = Path(config.data_root)
    out_root = Path(config.out)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_root / "checkpoints"
    if ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_samples(data_root=data_root, shape_manifest=Path(config.shape_manifest))
    train_samples, val_samples = split_train_val(samples, val_split=config.val_split, seed=config.seed)
    train_counts = _count_classes(train_samples)
    val_counts = _count_classes(val_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = _build_grad_scaler(enabled=amp_enabled)

    train_ds = ShapeDataset(train_samples, img_size=config.img_size, train=True, seed=config.seed)
    val_ds = ShapeDataset(val_samples, img_size=config.img_size, train=False, seed=config.seed)

    train_kwargs = _build_loader_kwargs(config.train_workers, pin_memory=(device.type == "cuda"))
    val_kwargs = _build_loader_kwargs(config.val_workers, pin_memory=(device.type == "cuda"))

    if config.balanced_sampler:
        weights = [1.0 / max(1, train_counts[sample.label]) for sample in train_samples]
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler, shuffle=False, **train_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, **train_kwargs)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, **val_kwargs)

    class_weights = torch.tensor(
        [len(train_samples) / (2.0 * max(1, train_counts[0])), len(train_samples) / (2.0 * max(1, train_counts[1]))],
        dtype=torch.float32,
        device=device,
    )

    model = ShapeClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, threshold=config.min_delta, min_lr=1e-6)

    best_score = -float("inf")
    epochs_no_improve = 0
    log_rows: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for images, targets in tqdm(train_loader, desc=f"Shape Train {epoch}/{config.epochs}", unit="batch"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        all_logits = []
        all_targets = []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Shape Val {epoch}/{config.epochs}", unit="batch"):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    logits = model(images)
                    loss = criterion(logits, targets)
                val_losses.append(float(loss.item()))
                all_logits.append(logits.detach().cpu())
                all_targets.append(targets.detach().cpu())

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        scheduler.step(val_loss)
        metrics = _compute_metrics(torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0))
        log_row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
        log_rows.append(log_row)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "img_size": int(config.img_size),
            "metrics": metrics,
            "epoch": epoch,
            "config": asdict(config),
        }
        torch.save(checkpoint, ckpt_dir / "last.pt")

        score = metrics["balanced_acc"]
        if score > (best_score + config.min_delta):
            best_score = score
            epochs_no_improve = 0
            torch.save(checkpoint, ckpt_dir / "best.pt")
            print(f"[INFO] New best balanced_acc={best_score:.6f}")
        else:
            epochs_no_improve += 1
            print(f"[INFO] No improvement: {epochs_no_improve} epoch(s)")
        if epochs_no_improve >= config.patience:
            print("[INFO] Early stopping")
            break

    log_path = out_root / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "acc", "balanced_acc", "square_recall", "rect_recall", "square_precision", "f1_square", "tp", "tn", "fp", "fn"])
        writer.writeheader()
        writer.writerows(log_rows)

    report_path = out_root / "shape_classifier_report.json"
    report = {
        "config": asdict(config),
        "counts": {
            "train_rect": train_counts[0],
            "train_square": train_counts[1],
            "val_rect": val_counts[0],
            "val_square": val_counts[1],
        },
        "best_score": float(best_score),
        "best_checkpoint": str(ckpt_dir / "best.pt"),
        "last_checkpoint": str(ckpt_dir / "last.pt"),
        "train_log": str(log_path),
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return ShapeClassifierSummary(best_checkpoint=str(ckpt_dir / "best.pt"), last_checkpoint=str(ckpt_dir / "last.pt"), report=str(report_path))


def predict_shape_labels(data_root: Path, checkpoint_path: Path, out_csv: Path) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    img_size = int(checkpoint.get("img_size", 96))
    model = ShapeClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    for image_path in sorted(data_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        dataset = ShapeDataset([ShapeSample(image_path=image_path, label=0)], img_size=img_size, train=False)
        tensor, _ = dataset[0]
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0))
            pred = int(logits.argmax(dim=1).item())
            prob = float(torch.softmax(logits, dim=1)[0, pred].item())
        rows.append({
            "file": str(image_path),
            "shape": "square" if pred == 1 else "rect",
            "score": prob,
        })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "shape", "score"])
        writer.writeheader()
        writer.writerows(rows)
    return out_csv
