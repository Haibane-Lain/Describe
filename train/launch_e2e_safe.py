"""Conservative launcher for the end-to-end run.

Kept for reference. The earlier crashes were NOT fixed by
``expandable_segments`` — that setting is unsupported on this ROCm build and
printed a warning on every run. ``bench_gpu.py`` later showed batch 64 with AMP
is both the fastest (261 img/s) and modest in memory (3.95 GiB), so
``launch_coco.py`` uses that instead.

    py -3.12 -m train.launch_e2e_safe
"""

from __future__ import annotations

import os
import sys

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from train.train.train_e2e import main as train_main

    sys.argv = [
        "train_e2e",
        "--manifest", "data/manifests/flickr30k.jsonl",
        "--image-root", "data/raw/flickr30k/images",
        "--run-name", "flickr30k_e2e_safe",
        "--backbone", "resnet50", "--image-size", "224",
        "--epochs", "15",
        "--batch-size", "24",
        "--lr", "5e-4", "--cnn-lr", "1e-5", "--dropout", "0.5",
        "--max-caption-len", "32", "--scheduled-sampling", "0.3",
        "--holdout", "0.05",
        "--val-bleu-every", "1", "--val-bleu-limit", "100",
        "--val-batch-size", "8",
        "--beam-size", "2", "--min-len", "2",
        "--repetition-penalty", "1.2", "--no-repeat-ngram", "3",
        "--length-alpha", "0.7",
        "--log-every", "400",
        "--cache-limit", "6000",
    ]
    os.chdir(ROOT)
    return train_main()


if __name__ == "__main__":
    raise SystemExit(main())
