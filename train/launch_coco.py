"""Train the captioner end-to-end on COCO.

Why COCO and why this configuration:

* **Diagnosis drove this.** `train/diagnose.py` showed the model already uses
  the image (+1.21 nats when shuffled), does not repeat (0.19 repeat rate), and
  decodes fine (greedy == beam). What it does is collapse: only 134 distinct
  tokens of 7,488, with 'a' at 22%. That is undertraining, so the fix is more
  data and more steps, not a different architecture.
* **COCO is ~4x Flickr30k** (82,783 train vs 31,783 images) and is the standard
  captioning corpus.
* **Stability over speed.** `bench_gpu.py` measured batch 64 with AMP at
  261 img/s (3.95 GiB), but a sustained run at that setting died with heap
  corruption (0xc0000374) inside ntdll, and an earlier run died with an access
  violation (0xc0000005). Both are memory corruption, so this launcher keeps
  peak memory low and avoids the autocast path: batch 32, no AMP (72.5 img/s,
  3.67 GiB measured). Checkpoints are also written atomically from CPU tensors
  now, so a crash cannot corrupt the previous one.

    py -3.12 -m train.launch_coco
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from train.train.train_e2e import main as train_main

    sys.argv = [
        "train_e2e",
        "--manifest", "data/manifests/coco_train.jsonl",
        "--val-manifest", "data/manifests/coco_val.jsonl",
        "--image-root", "data/raw/coco/images",
        "--run-name", "coco_e2e",
        "--backbone", "resnet50", "--image-size", "224",
        "--epochs", "30",
        "--batch-size", "32",
        "--lr", "5e-4", "--cnn-lr", "1e-5", "--dropout", "0.4",
        "--max-caption-len", "32", "--scheduled-sampling", "0.3",
        "--val-bleu-every", "2", "--val-bleu-limit", "500",
        "--val-batch-size", "16",
        "--beam-size", "3", "--min-len", "2",
        "--repetition-penalty", "1.2", "--no-repeat-ngram", "3",
        "--length-alpha", "0.7",
        "--log-every", "500",
        "--num-workers", "6",
    ]
    os.chdir(ROOT)
    return train_main()


if __name__ == "__main__":
    raise SystemExit(main())
