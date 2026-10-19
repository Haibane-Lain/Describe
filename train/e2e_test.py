"""End-to-end pipeline test on generated data.

Runs the real code path - manifest -> feature extraction -> vocabulary ->
training -> checkpoint -> evaluation - on synthetic images with realistic
captions, so a failure points at our code rather than at dataset downloads.

    py -3.12 -m train.e2e_test
    py -3.12 -m train.e2e_test --images 96 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Workspace root, so subprocesses can import the `train` package from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Captions with real structure, so the vocabulary and loss are meaningful.
CAPTION_POOL = [
    "a person holding a white bottle over a table",
    "a small brown dog lying on a couch",
    "a red sign with white letters on a wall",
    "an open laptop on a wooden desk",
    "a cup of coffee next to a book",
    "a blue chair in a bright room",
    "a green plant beside the window",
    "a black cat sleeping on a bed",
    "a yellow taxi on a busy street",
    "a plate of food on a kitchen table",
    "a hand holding a medicine bottle",
    "a white door with a metal handle",
]


def make_images(dest: Path, count: int, size: int = 224, seed: int = 0) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(count):
        # Each image gets a colour signature so features are not identical.
        base = rng.integers(0, 200, size=3)
        array = np.full((size, size, 3), base, dtype="uint8")
        array = np.clip(
            array + rng.normal(0, 18, size=array.shape), 0, 255
        ).astype("uint8")
        image = Image.fromarray(array, mode="RGB")
        draw = ImageDraw.Draw(image)
        draw.rectangle([20, 20, 80, 80], fill=tuple(int(v) for v in base[::-1]))
        draw.ellipse([120, 120, 200, 200], outline=(255, 255, 255), width=4)
        image.save(dest / f"syn_{i:04d}.jpg", quality=92)


def make_manifests(root: Path, count: int, val_count: int) -> tuple[Path, Path]:
    """Write train and val manifests.

    Filenames matter: train_captioner derives the feature-cache index from the
    manifest stem (``<stem>_<backbone>.jsonl``), so each manifest must be
    extracted under its own matching name.
    """
    images = root / "images"
    train_path = root / "train.jsonl"
    val_path = root / "val.jsonl"

    def write(path: Path, indices: list[int], split: str) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for i in indices:
                captions = [
                    CAPTION_POOL[(i + k) % len(CAPTION_POOL)] for k in range(5)
                ]
                handle.write(
                    json.dumps(
                        {
                            "image_id": f"syn_{i:04d}",
                            "file_name": f"syn_{i:04d}.jpg",
                            "dataset": "synthetic",
                            "split": split,
                            "captions": captions,
                            "text_detected": bool(i % 3 == 0),
                            "quality": None,
                        }
                    )
                    + "\n"
                )

    write(train_path, list(range(count)), "train")
    write(val_path, list(range(count, count + val_count)), "val")
    return train_path, val_path


def run(module: str, argv: list[str], cwd: Path) -> int:
    command = [sys.executable, "-m", module, *argv]
    print(f"\n$ {' '.join(command[1:])}", flush=True)
    import os

    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PROJECT_ROOT}{os.pathsep}{existing}" if existing else str(PROJECT_ROOT)
    )
    completed = subprocess.run(command, cwd=cwd, env=env)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end pipeline test")
    parser.add_argument("--images", type=int, default=64)
    parser.add_argument("--val-images", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--keep", action="store_true", help="keep the temp workspace")
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="describe_e2e_"))
    print(f"Workspace: {root}")
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail else ""), flush=True)

    try:
        # ---------------------------------------------------------- fixtures
        print("\n[1/5] Generating synthetic images and manifests")
        total = args.images + args.val_images
        make_images(root / "images", total)
        train_manifest, val_manifest = make_manifests(
            root, args.images, args.val_images
        )
        check("images generated", len(list((root / "images").glob("*.jpg"))) == total,
              f"{total} images")

        # ---------------------------------------------------- feature cache
        print("\n[2/5] Extracting features")
        code = run(
            "train.features.extract",
            [
                "--manifest", str(train_manifest),
                "--image-root", str(root / "images"),
                "--out-dir", str(root / "features"),
                "--batch-size", "16",
                "--backbone", "resnet101",
            ],
            cwd=root,
        )
        check("feature extraction exited 0", code == 0, f"exit {code}")

        feature_npy = root / "features" / "train_resnet101.npy"
        feature_idx = root / "features" / "train_resnet101.jsonl"
        check("feature array written", feature_npy.exists())
        if feature_npy.exists():
            arr = np.load(feature_npy)
            check("feature shape", arr.shape == (args.images, 2048), str(arr.shape))
            check("features finite", bool(np.isfinite(arr).all()))

        # ------------------------------------------------------- extraction for val
        # Train and val share the cache, so extract val images into the same dir.
        val_all = root / "val_manifest.jsonl"
        with val_all.open("w", encoding="utf-8") as out:
            for line in val_manifest.open(encoding="utf-8"):
                out.write(line)
        code = run(
            "train.features.extract",
            [
                "--manifest", str(val_all),
                "--image-root", str(root / "images"),
                "--out-dir", str(root / "features"),
                "--batch-size", "16",
                "--backbone", "resnet101",
            ],
            cwd=root,
        )
        check("val feature extraction exited 0", code == 0, f"exit {code}")

        # ------------------------------------------------------------ training
        print("\n[3/5] Training the captioner")
        code = run(
            "train.train.train_captioner",
            [
                "--manifest", str(train_manifest),
                "--val-manifest", str(val_all),
                "--feature-dir", str(root / "features"),
                "--feature-index", str(feature_idx),
                "--run-name", "e2e",
                "--checkpoint-dir", str(root / "checkpoints"),
                "--run-dir", str(root / "runs"),
                "--epochs", str(args.epochs),
                "--batch-size", "16",
                "--embed-dim", "128",
                "--hidden-dim", "128",
                "--min-freq", "1",
                "--log-every", "1000",
                "--beam-size", "3",
                "--max-len", "15",
                "--min-len", "2",
            ],
            cwd=root,
        )
        check("training exited 0", code == 0, f"exit {code}")

        best = root / "checkpoints" / "e2e" / "best.pt"
        vocab_file = root / "checkpoints" / "e2e" / "vocab.pkl"
        history = root / "runs" / "e2e" / "history.jsonl"
        check("best checkpoint written", best.exists())
        check("vocab saved beside checkpoint", vocab_file.exists())
        check("history written", history.exists())

        if history.exists():
            rows = [json.loads(line) for line in history.open(encoding="utf-8")]
            check("history has one row per epoch", len(rows) == args.epochs,
                  f"{len(rows)} rows")
            if len(rows) >= 2:
                check("train loss decreased over epochs",
                      rows[-1]["train_loss"] < rows[0]["train_loss"],
                      f"{rows[0]['train_loss']:.4f} -> {rows[-1]['train_loss']:.4f}")

        # ---------------------------------------------------------- evaluation
        print("\n[4/5] Evaluating")
        code = run(
            "train.train.evaluate",
            [
                "--checkpoint", str(best),
                "--feature-index", str(feature_idx),
                "--manifest", str(train_manifest),
                "--out", str(root / "runs" / "e2e" / "metrics.json"),
                "--pred-out", str(root / "runs" / "e2e" / "predictions.jsonl"),
                "--beam-size", "3",
                "--max-len", "15",
                "--min-len", "2",
                "--limit", "16",
            ],
            cwd=root,
        )
        check("evaluation exited 0", code == 0, f"exit {code}")

        metrics_path = root / "runs" / "e2e" / "metrics.json"
        check("metrics written", metrics_path.exists())
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            check("metrics include BLEU-4", "BLEU-4" in metrics["metrics"],
                  f"{list(metrics['metrics'])}")
            check("metrics report image count", metrics["n_images"] > 0,
                  f"n={metrics['n_images']}")

        preds_path = root / "runs" / "e2e" / "predictions.jsonl"
        check("per-image predictions written", preds_path.exists())
        if preds_path.exists():
            preds = [json.loads(line) for line in preds_path.open(encoding="utf-8")]
            check("every prediction has references",
                  all(p["references"] for p in preds))
            check("text_detected recorded for the split report",
                  any(p["text_detected"] is not None for p in preds))

        # --------------------------------------------------- stage-2 init path
        print("\n[5/5] Stage-2 initialisation (vocab change) and resume")
        code = run(
            "train.train.train_captioner",
            [
                "--manifest", str(val_all),
                "--feature-dir", str(root / "features"),
                "--feature-index", str(
                    root / "features" / "val_manifest_resnet101.jsonl"
                ),
                "--run-name", "e2e_stage2",
                "--checkpoint-dir", str(root / "checkpoints"),
                "--run-dir", str(root / "runs"),
                "--epochs", "1",
                "--batch-size", "8",
                "--embed-dim", "128",
                "--hidden-dim", "128",
                "--min-freq", "1",
                "--log-every", "1000",
                "--beam-size", "1",
                "--max-len", "10",
                "--min-len", "2",
                "--init-from", str(best),
            ],
            cwd=root,
        )
        check("stage-2 init from stage-1 checkpoint exited 0", code == 0,
              f"exit {code}")
        check("stage-2 checkpoint written",
              (root / "checkpoints" / "e2e_stage2" / "best.pt").exists())

        code = run(
            "train.train.train_captioner",
            [
                "--manifest", str(train_manifest),
                "--feature-dir", str(root / "features"),
                "--feature-index", str(feature_idx),
                "--run-name", "e2e",
                "--checkpoint-dir", str(root / "checkpoints"),
                "--run-dir", str(root / "runs"),
                "--epochs", str(args.epochs + 1),
                "--batch-size", "16",
                "--embed-dim", "128",
                "--hidden-dim", "128",
                "--min-freq", "1",
                "--log-every", "1000",
                "--beam-size", "1",
                "--max-len", "10",
                "--min-len", "2",
                "--resume", str(root / "checkpoints" / "e2e" / "last.pt"),
            ],
            cwd=root,
        )
        check("resume from last.pt exited 0", code == 0, f"exit {code}")

    finally:
        print("\n" + "=" * 66)
        failed = [c for c in checks if not c[1]]
        print(f"Result: {len(checks) - len(failed)}/{len(checks)} checks passed")
        if failed:
            print("\nFailed:")
            for name, _, detail in failed:
                print(f"  - {name} {detail}")
        else:
            print("END-TO-END TEST PASSED")
        print("=" * 66)

        if args.keep:
            print(f"\nWorkspace kept at: {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
