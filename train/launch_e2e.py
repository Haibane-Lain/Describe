"""Launcher for the end-to-end run with a safe GPU memory configuration.

Two OOM crashes motivated this wrapper:

* Training at batch 64 with a trainable ResNet-50 peaks around 5.4 GiB, and
  beam-search validation adds a large transient on top of that.
* The failures reported "14.59 GiB free" while failing to allocate 20 MiB, which
  is the fragmentation signature, not genuine exhaustion.

So this sets ``expandable_segments`` and uses a smaller batch, which trades a
little throughput for a run that survives to completion.

    py -3.12 -m train.launch_e2e
"""

from __future__ import annotations

import os
import sys

# Must be set before torch initializes its caching allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from train.train.train_e2e import main as train_main

    sys.argv = [
        "train_e2e",
        "--manifest", "data/manifests/flickr30k.jsonl",
        "--image-root", "data/raw/flickr30k/images",
        "--run-name", "flickr30k_e2e_v2",
        "--backbone", "resnet50", "--image-size", "224",
        "--epochs", "20",
        "--batch-size", "32",
        "--lr", "5e-4", "--cnn-lr", "1e-5", "--dropout", "0.5",
        "--max-caption-len", "32", "--scheduled-sampling", "0.3",
        "--holdout", "0.05",
        "--val-bleu-every", "1", "--val-bleu-limit", "200",
        "--val-batch-size", "16",
        "--beam-size", "3", "--min-len", "2",
        "--repetition-penalty", "1.2", "--no-repeat-ngram", "3",
        "--length-alpha", "0.7",
        "--amp", "--log-every", "300",
        "--cache-limit", "8000",
    ]
    os.chdir(ROOT)
    return train_main()


if __name__ == "__main__":
    raise SystemExit(main())
