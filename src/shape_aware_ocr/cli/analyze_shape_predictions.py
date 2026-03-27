from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from shape_aware_ocr.dataset import load_shape_map_from_manifest
from shape_aware_ocr.labels import normalized_match_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare predicted shape labels against oracle shape manifest and emit a predicted manifest")
    parser.add_argument("--predictions", required=True, type=str)
    parser.add_argument("--oracle-shape-manifest", required=True, type=str)
    parser.add_argument("--out-manifest", required=True, type=str)
    parser.add_argument("--out-report", required=True, type=str)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    oracle_map, _ = load_shape_map_from_manifest(Path(args.oracle_shape_manifest))
    pred_rows = []
    with open(args.predictions, 'r', newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_path = Path(row['file'])
            match_key = normalized_match_stem(file_path.stem)
            pred_shape = str(row['shape']).strip().lower()
            pred_flag = 1 if pred_shape.startswith('sq') else 0
            oracle_flag = int(oracle_map.get(match_key, -1))
            pred_rows.append({
                'file': file_path.name,
                'match_key': match_key,
                'shape': 'square' if pred_flag == 1 else 'rect',
                'pred_score': float(row.get('score', '0') or 0.0),
                'oracle_flag': oracle_flag,
                'pred_flag': pred_flag,
            })

    tp = sum(1 for row in pred_rows if row['pred_flag'] == 1 and row['oracle_flag'] == 1)
    tn = sum(1 for row in pred_rows if row['pred_flag'] == 0 and row['oracle_flag'] == 0)
    fp = sum(1 for row in pred_rows if row['pred_flag'] == 1 and row['oracle_flag'] == 0)
    fn = sum(1 for row in pred_rows if row['pred_flag'] == 0 and row['oracle_flag'] == 1)
    total = max(1, tp + tn + fp + fn)
    acc = (tp + tn) / total
    square_recall = tp / max(1, tp + fn)
    rect_recall = tn / max(1, tn + fp)
    balanced_acc = 0.5 * (square_recall + rect_recall)

    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['file', 'match_key', 'shape'])
        writer.writeheader()
        for row in pred_rows:
            writer.writerow({'file': row['file'], 'match_key': row['match_key'], 'shape': row['shape']})

    report = {
        'predictions': str(args.predictions),
        'oracle_shape_manifest': str(args.oracle_shape_manifest),
        'out_manifest': str(out_manifest),
        'samples': total,
        'acc': acc,
        'balanced_acc': balanced_acc,
        'square_recall': square_recall,
        'rect_recall': rect_recall,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
    }
    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"[INFO] Shape prediction report: {out_report}")
    print(f"[INFO] balanced_acc={balanced_acc:.6f}, acc={acc:.6f}, square_recall={square_recall:.6f}, rect_recall={rect_recall:.6f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
