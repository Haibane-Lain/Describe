"""Smoke test the quality gate training loop on a small subset.

Confirms the forward/backward path, threshold selection, and metric computation
work before committing to a full run.

    py -3.12 -m train.quality.gate_smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.quality.train_gate import (  # noqa: E402
    QualityDataset,
    QualityGate,
    average_precision,
    binary_metrics,
    find_threshold_for_false_accept,
    roc_auc,
)
from train.utils.device import configure_for_speed, get_device, set_seed  # noqa: E402


def main() -> int:
    set_seed(1337)
    info = get_device()
    configure_for_speed()
    print(f"device: {info.name} ({info.backend})\n")

    def load(split: str, limit: int | None = None):
        manifest = Path(f"data/manifests/vizwiz_{split}.jsonl")
        index = Path(f"data/features/vizwiz_{split}_resnet101.jsonl")
        features = np.load(f"data/features/vizwiz_{split}_resnet101.npy")
        rows = [
            json.loads(line) for line in index.open(encoding="utf-8") if line.strip()
        ]
        dataset = QualityDataset(features, rows)
        if limit:
            dataset.rows = dataset.rows[:limit]
        return dataset

    train_ds = load("train", limit=4000)
    val_ds = load("val", limit=2000)
    print(f"train {len(train_ds):,} | val {len(val_ds):,}")

    train_pos = sum(1 for r in train_ds.rows if r["unrecognizable"])
    val_pos = sum(1 for r in val_ds.rows if r["unrecognizable"])
    print(f"positives: train {train_pos:,} val {val_pos:,}")

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = QualityGate().to(info.device)
    pos_weight = torch.tensor(
        [(len(train_ds) - train_pos) / max(train_pos, 1)], device=info.device
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    losses = []
    for epoch in range(3):
        model.train()
        running, seen = 0.0, 0
        for features, targets, flaws, _ in train_loader:
            features = features.to(info.device)
            targets = targets.to(info.device)
            flaws = flaws.to(info.device)
            optimizer.zero_grad(set_to_none=True)
            logit, flaw_logits = model(features)
            loss = bce(logit, targets) + 0.5 * nn.functional.binary_cross_entropy_with_logits(
                flaw_logits, flaws
            )
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * features.size(0)
            seen += features.size(0)
        losses.append(running / max(seen, 1))
        print(f"  epoch {epoch + 1}: loss {losses[-1]:.4f}")

    check("loss decreased", losses[-1] < losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}")

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for features, targets, flaws, _ in val_loader:
            logit, _ = model(features.to(info.device))
            probs.append(torch.sigmoid(logit).cpu().numpy())
            labels.append(targets.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    ap = average_precision(probs, labels)
    auc = roc_auc(probs, labels)
    threshold = find_threshold_for_false_accept(probs, labels, 0.05)
    metrics = binary_metrics(probs, labels, threshold)

    print(f"\n  AP {ap:.4f} | AUC {auc:.4f} | threshold {threshold:.3f}")
    print(f"  FAR {metrics['false_accept_rate']:.4f} | FRR {metrics['false_reject_rate']:.4f}")

    check("AP above chance", ap > 0.15, f"{ap:.4f}")
    check("AUC above chance", auc > 0.55, f"{auc:.4f}")
    check("probability range valid", bool(probs.min() >= 0 and probs.max() <= 1))
    check("FAR bound respected", metrics["false_accept_rate"] <= 0.05,
          f"{metrics['false_accept_rate']:.4f}")

    failed = [c for c in checks if not c[1]]
    print(f"\n  {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
