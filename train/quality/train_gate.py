"""Quality gate: predict whether a photo can support a reliable description.

This is the component the project's central claim rests on. Per the proposal,
the gate is defined as **unrecognizability** prediction following the
VizWiz-QualityIssues formulation: an image is unrecognizable when a majority of
crowdworker votes say so. Perceived flaws are a separate multi-label prediction
over blur, overexposure, underexposure, framing, obstruction, rotation, and
other.

Evaluation reports **false-accept and false-reject rates separately**. A false
accept (describing an unusable image as though it were fine) is worse than a
false reject (an unnecessary retake), because the user cannot verify the
description. The threshold is therefore tuned to bound the false-accept rate.

    py -3.12 -m train.quality.train_gate --epochs 20
    py -3.12 -m train.quality.train_gate --baseline   # OpenCV rules only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.utils.device import (  # noqa: E402
    configure_for_speed,
    get_device,
    set_seed,
)

FLAW_CODES = ("BLR", "BRT", "DRK", "FRM", "NON", "OBS", "OTH", "ROT")
FLAW_NAMES = {
    "BLR": "blur",
    "BRT": "too bright",
    "DRK": "too dark",
    "FRM": "poor framing",
    "NON": "no flaw",
    "OBS": "obstructed",
    "OTH": "other",
    "ROT": "rotated",
}
MAJORITY = 3  # out of 5 crowdworker votes


class QualityDataset(Dataset):
    """Cached image features paired with recognizability and flaw labels."""

    def __init__(
        self,
        features: np.ndarray,
        rows: list[dict],
        majority: int = MAJORITY,
    ) -> None:
        self.features = features
        self.rows = []
        for row in rows:
            quality = row.get("quality")
            if not quality:
                continue
            votes = int(quality.get("unrecognizable_votes", 0))
            self.rows.append(
                {
                    "feature_row": row["feature_row"],
                    "image_id": row["image_id"],
                    # Retained so the OpenCV baseline can locate the image.
                    "file_name": row.get("file_name"),
                    "unrecognizable": votes >= majority,
                    "votes": votes,
                    "flaws": quality.get("flaws", {}),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        feature = torch.from_numpy(
            np.asarray(self.features[row["feature_row"]], dtype=np.float32)
        )
        flaw_targets = torch.tensor(
            [1.0 if row["flaws"].get(code, 0) >= MAJORITY else 0.0
             for code in FLAW_CODES],
            dtype=torch.float32,
        )
        return (
            feature,
            torch.tensor(float(row["unrecognizable"])),
            flaw_targets,
            index,
        )


class QualityGate(nn.Module):
    """Linear probe over frozen CNN features.

    The encoder is already run and cached, so the gate is a small head on top.
    A linear probe is the right first model here: it is fast, cannot overfit
    2,048 features on 23k images badly, and its decision threshold is easy to
    reason about.
    """

    def __init__(self, feature_dim: int = 2048, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.recognizable_head = nn.Linear(hidden, 1)
        self.flaw_head = nn.Linear(hidden, len(FLAW_CODES))

    def forward(self, features: torch.Tensor):
        hidden = self.trunk(features)
        return self.recognizable_head(hidden).squeeze(-1), self.flaw_head(hidden)


def find_threshold_for_false_accept(
    probabilities: np.ndarray, labels: np.ndarray, target_far: float
) -> float:
    """Pick the threshold that bounds the false-accept rate.

    Positive class is "unrecognizable". A false accept means calling an
    unrecognizable image recognizable (predicting 0 when the label is 1), so
    the gate should route to a retake unless it is confident the image is fine.
    """
    best = 0.5
    for threshold in np.linspace(0.01, 0.99, 197):
        predicted = (probabilities >= threshold).astype(int)
        unrecognizable = labels == 1
        # False accept: label unrecognizable, but we said recognizable.
        fa = int(((predicted == 0) & unrecognizable).sum())
        total_unrecognizable = int(unrecognizable.sum())
        far = fa / total_unrecognizable if total_unrecognizable else 0.0
        if far <= target_far:
            best = float(threshold)
    return best


def binary_metrics(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> dict:
    """Confusion-matrix metrics with the asymmetric error split out."""
    predicted = (probabilities >= threshold).astype(int)
    labels = labels.astype(int)

    tp = int(((predicted == 1) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # False accept: label unrecognizable (1) but predicted recognizable (0).
    false_accept_rate = fn / (fn + tp) if (fn + tp) else 0.0
    # False reject: label recognizable (0) but predicted unrecognizable (1).
    false_reject_rate = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": tp + tn + fp + fn,
    }


def average_precision(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve, computed without sklearn."""
    order = np.argsort(-probabilities)
    labels = labels[order]
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    tp = np.cumsum(labels)
    precision = tp / np.arange(1, len(labels) + 1)
    recall = tp / positives
    # Integrate with the step rule.
    ap = 0.0
    previous_recall = 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - previous_recall)
        previous_recall = r
    return float(ap)


def roc_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via rank statistics, without sklearn."""
    labels = labels.astype(int)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(probabilities)
    ranks = np.empty(len(probabilities), dtype=float)
    ranks[order] = np.arange(1, len(probabilities) + 1)
    # Average ranks for ties.
    sorted_probs = probabilities[order]
    start = 0
    for i in range(1, len(sorted_probs) + 1):
        if i == len(sorted_probs) or sorted_probs[i] != sorted_probs[start]:
            if i - start > 1:
                ranks[order[start:i]] = (start + 1 + i) / 2
            start = i
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def opencv_baseline_sweep(
    rows: list[dict], image_root: Path
) -> tuple[dict, list[dict]]:
    """Sweep blur/exposure thresholds to find the baseline's best setting.

    Comparing a learned model against one arbitrary threshold pair would be
    unfair to the rules. This evaluates a grid of thresholds and reports both
    the best operating point and the frontier, so the comparison in the paper
    is against the *best* hand-tuned rule, not a straw man.
    """
    from PIL import Image

    index: dict[str, Path] = {}
    for path in image_root.rglob("*.jpg"):
        index[path.name] = path

    variances = np.zeros(len(rows), dtype=np.float32)
    intensities = np.zeros(len(rows), dtype=np.float32)
    missing = 0
    for i, row in enumerate(rows):
        path = index.get(row.get("file_name") or "")
        if path is None:
            missing += 1
            variances[i] = 0.0
            intensities[i] = 0.0
            continue
        try:
            with Image.open(path) as image:
                grey = image.convert("L").resize((256, 256))
                array = np.asarray(grey, dtype=np.float32)
        except Exception:
            missing += 1
            continue
        laplacian = (
            -4 * array
            + np.roll(array, 1, 0) + np.roll(array, -1, 0)
            + np.roll(array, 1, 1) + np.roll(array, -1, 1)
        )
        variances[i] = float(laplacian.var())
        intensities[i] = float(array.mean())

    if missing:
        print(f"  WARNING: {missing:,} images could not be read")

    labels = np.array([1.0 if r["unrecognizable"] else 0.0 for r in rows])

    frontier: list[dict] = []
    best: dict | None = None
    for variance_threshold in (10, 20, 40, 60, 100, 150, 250, 400):
        for low, high in ((10, 245), (20, 240), (25, 235), (40, 230), (60, 220)):
            recognizable = (variances > variance_threshold) & (
                (intensities > low) & (intensities < high)
            )
            # Score = probability-like confidence that the image is recognizable.
            scores = recognizable.astype(float)
            metrics = binary_metrics(scores, labels, 0.5)
            metrics["variance_threshold"] = variance_threshold
            metrics["exposure_range"] = [low, high]
            metrics["roc_auc"] = roc_auc(scores + 0.01 * variances, labels)
            frontier.append(metrics)
            # Rank by F1, with false-accept rate as the tie-breaker since a
            # false accept is the costly error.
            key = (metrics["f1"], -metrics["false_accept_rate"])
            if best is None or key > (best["f1"], -best["false_accept_rate"]):
                best = metrics

    assert best is not None
    return best, frontier


def opencv_baseline(rows: list[dict], image_root: Path, threshold: float = 0.5) -> np.ndarray:
    """Blur + brightness rules, the comparison the proposal promises.

    Uses variance of Laplacian for blur and mean intensity for exposure, both
    computed on a downscaled greyscale image so the whole set is quick.
    """
    from PIL import Image

    index = {}
    for path in image_root.rglob("*.jpg"):
        index[path.name] = path

    scores = np.zeros(len(rows), dtype=float)
    for i, row in enumerate(rows):
        path = index.get(row["file_name"])
        if path is None:
            scores[i] = 0.0
            continue
        try:
            with Image.open(path) as image:
                grey = image.convert("L").resize((256, 256))
                array = np.asarray(grey, dtype=np.float32)
        except Exception:
            scores[i] = 0.0
            continue

        # Variance of Laplacian: low variance means little high-frequency
        # content, i.e. blurry.
        laplacian = (
            -4 * array
            + np.roll(array, 1, 0) + np.roll(array, -1, 0)
            + np.roll(array, 1, 1) + np.roll(array, -1, 1)
        )
        variance = float(laplacian.var())
        mean_intensity = float(array.mean())

        blur_ok = variance > 60.0
        exposure_ok = 25.0 < mean_intensity < 235.0
        scores[i] = 1.0 if (blur_ok and exposure_ok) else 0.0

    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the quality gate")
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--backbone", default="resnet101")
    parser.add_argument("--train-manifest", type=Path, default=Path("data/manifests/vizwiz_train.jsonl"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/manifests/vizwiz_val.jsonl"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/quality_gate"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/quality_gate"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument(
        "--target-false-accept",
        type=float,
        default=0.05,
        help="bound on the false-accept rate used to pick the decision threshold",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="evaluate the OpenCV blur/brightness rules instead of training",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    info = get_device(prefer_gpu=not args.cpu)
    configure_for_speed()
    print(f"device: {info.name} ({info.backend})\n")

    def load_split(manifest: Path):
        index_path = args.feature_dir / f"{manifest.stem}_{args.backbone}.jsonl"
        feature_path = args.feature_dir / f"{manifest.stem}_{args.backbone}.npy"
        features = np.load(feature_path)
        rows = [
            json.loads(line)
            for line in index_path.open(encoding="utf-8")
            if line.strip()
        ]
        return features, rows

    train_features, train_rows = load_split(args.train_manifest)
    val_features, val_rows = load_split(args.val_manifest)

    train_ds = QualityDataset(train_features, train_rows)
    val_ds = QualityDataset(val_features, val_rows)
    print(f"train images : {len(train_ds):,}")
    print(f"val images   : {len(val_ds):,}")

    train_pos = sum(1 for r in train_ds.rows if r["unrecognizable"])
    val_pos = sum(1 for r in val_ds.rows if r["unrecognizable"])
    print(
        f"unrecognizable: train {train_pos:,} ({100 * train_pos / max(len(train_ds), 1):.1f}%), "
        f"val {val_pos:,} ({100 * val_pos / max(len(val_ds), 1):.1f}%)"
    )
    print(
        "  NOTE: this class is heavily imbalanced, so aggregate accuracy is "
        "misleading. The metrics that matter are below.\n"
    )

    # ------------------------------------------------------------- baseline
    if args.baseline:
        print("OpenCV blur + brightness baseline (threshold sweep)")
        labels = np.array([1.0 if r["unrecognizable"] else 0.0 for r in val_ds.rows])
        image_root = Path("data/raw/vizwiz/images")
        best, frontier = opencv_baseline_sweep(val_ds.rows, image_root)

        print("\n  Best hand-tuned operating point")
        print(f"    variance threshold : {best['variance_threshold']}")
        print(f"    exposure range     : {best['exposure_range']}")
        print(f"    precision / recall : {best['precision']:.3f} / {best['recall']:.3f}")
        print(f"    F1                 : {best['f1']:.4f}")
        print(f"    FALSE ACCEPT rate  : {best['false_accept_rate']:.4f}")
        print(f"    FALSE REJECT rate  : {best['false_reject_rate']:.4f}")
        print(f"    accuracy           : {best['accuracy']:.4f}")

        args.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "best": best,
            "n_configurations": len(frontier),
            "frontier": frontier,
        }
        (args.run_dir / "baseline_metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.run_dir / 'baseline_metrics.json'}")
        return 0

    # -------------------------------------------------------------- training
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = QualityGate(
        feature_dim=args.feature_dim, hidden=args.hidden, dropout=args.dropout
    ).to(info.device)
    print(f"gate parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Weight the positive class so the imbalanced majority class does not
    # dominate; this is what makes the gate usable rather than merely accurate.
    pos_weight = torch.tensor(
        [(len(train_ds) - train_pos) / max(train_pos, 1)], device=info.device
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.run_dir / "history.jsonl"

    best_ap = -1.0
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for features, targets, flaws, _ in train_loader:
            features = features.to(info.device)
            targets = targets.to(info.device)
            flaws = flaws.to(info.device)

            optimizer.zero_grad(set_to_none=True)
            recognizable_logit, flaw_logits = model(features)
            loss = bce(recognizable_logit, targets) + 0.5 * nn.functional.binary_cross_entropy_with_logits(
                flaw_logits, flaws
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            running += float(loss.item()) * features.size(0)
            seen += features.size(0)

        scheduler.step()

        # ------------------------------------------------------- validation
        model.eval()
        probabilities, labels = [], []
        flaw_probs, flaw_targets = [], []
        with torch.no_grad():
            for features, targets, flaws, _ in val_loader:
                features = features.to(info.device)
                recognizable_logit, flaw_logits = model(features)
                probabilities.append(torch.sigmoid(recognizable_logit).cpu().numpy())
                labels.append(targets.numpy())
                flaw_probs.append(torch.sigmoid(flaw_logits).cpu().numpy())
                flaw_targets.append(flaws.numpy())

        probabilities = np.concatenate(probabilities)
        labels = np.concatenate(labels)
        flaw_probs = np.concatenate(flaw_probs)
        flaw_targets = np.concatenate(flaw_targets)

        ap = average_precision(probabilities, labels)
        auc = roc_auc(probabilities, labels)
        threshold = find_threshold_for_false_accept(
            probabilities, labels, args.target_false_accept
        )
        metrics = binary_metrics(probabilities, labels, threshold)

        record = {
            "epoch": epoch + 1,
            "train_loss": running / max(seen, 1),
            "val_ap": ap,
            "val_auc": auc,
            **{k: metrics[k] for k in (
                "threshold", "accuracy", "precision", "recall", "f1",
                "false_accept_rate", "false_reject_rate",
            )},
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        print(
            f"  epoch {epoch + 1:2d}/{args.epochs} loss {record['train_loss']:.4f} "
            f"AP {ap:.3f} AUC {auc:.3f} "
            f"FAR {metrics['false_accept_rate']:.3f} "
            f"FRR {metrics['false_reject_rate']:.3f} "
            f"thr {threshold:.2f}",
            flush=True,
        )

        payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "config": {
                "feature_dim": args.feature_dim,
                "hidden": args.hidden,
                "dropout": args.dropout,
                "flaw_codes": list(FLAW_CODES),
            },
            "metrics": record,
        }
        torch.save(payload, args.checkpoint_dir / "last.pt")
        if ap > best_ap:
            best_ap = ap
            torch.save(payload, args.checkpoint_dir / "best.pt")

    # ------------------------------------------------- final per-flaw report
    payload = torch.load(args.checkpoint_dir / "best.pt", map_location=info.device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    model.eval()
    probabilities, labels, flaw_probs, flaw_targets = [], [], [], []
    with torch.no_grad():
        for features, targets, flaws, _ in val_loader:
            features = features.to(info.device)
            recognizable_logit, flaw_logits = model(features)
            probabilities.append(torch.sigmoid(recognizable_logit).cpu().numpy())
            labels.append(targets.numpy())
            flaw_probs.append(torch.sigmoid(flaw_logits).cpu().numpy())
            flaw_targets.append(flaws.numpy())
    probabilities = np.concatenate(probabilities)
    labels = np.concatenate(labels)
    flaw_probs = np.concatenate(flaw_probs)
    flaw_targets = np.concatenate(flaw_targets)

    threshold = find_threshold_for_false_accept(
        probabilities, labels, args.target_false_accept
    )
    overall = binary_metrics(probabilities, labels, threshold)
    overall["average_precision"] = average_precision(probabilities, labels)
    overall["roc_auc"] = roc_auc(probabilities, labels)

    per_flaw = {}
    for i, code in enumerate(FLAW_CODES):
        targets = flaw_targets[:, i]
        if targets.sum() == 0:
            per_flaw[code] = {"name": FLAW_NAMES[code], "n_positive": 0}
            continue
        per_flaw[code] = {
            "name": FLAW_NAMES[code],
            "n_positive": int(targets.sum()),
            "average_precision": average_precision(flaw_probs[:, i], targets),
            "roc_auc": roc_auc(flaw_probs[:, i], targets),
        }

    result = {
        "gate": overall,
        "per_flaw": per_flaw,
        "target_false_accept": args.target_false_accept,
        "class_balance": {
            "train_unrecognizable": train_pos,
            "train_total": len(train_ds),
            "val_unrecognizable": val_pos,
            "val_total": len(val_ds),
        },
    }
    (args.run_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 62)
    print("QUALITY GATE - validation")
    print("=" * 62)
    print(f"  unrecognizability (majority of 5 votes), n={overall['n']:,}")
    print(f"    ROC AUC              : {overall['roc_auc']:.4f}")
    print(f"    average precision    : {overall['average_precision']:.4f}")
    print(f"    threshold            : {overall['threshold']:.3f}")
    print(f"    precision / recall   : {overall['precision']:.3f} / {overall['recall']:.3f}")
    print(f"    F1                   : {overall['f1']:.4f}")
    print(f"    FALSE ACCEPT rate    : {overall['false_accept_rate']:.4f}  (bound {args.target_false_accept})")
    print(f"    FALSE REJECT rate    : {overall['false_reject_rate']:.4f}")
    print(f"    accuracy (misleading): {overall['accuracy']:.4f}")
    print("\n  per-flaw (average precision / ROC AUC)")
    for code, values in per_flaw.items():
        if values.get("n_positive", 0) == 0:
            print(f"    {values['name']:12s} no positives in validation")
            continue
        print(
            f"    {values['name']:12s} AP {values['average_precision']:.3f} "
            f"AUC {values['roc_auc']:.3f}  (n={values['n_positive']:,})"
        )
    print(f"\nwrote {args.run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
