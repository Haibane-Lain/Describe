"""Higher-capacity end-to-end run.

The 512-dim decoder plateaued at 2.69 training loss and produced captions that
were semantically right but syntactically broken ("A man a shirt a and woman a
in background"). That pattern — correct content words plus repeated filler — is
a language-model capacity limit, not a vision failure, and 512 hidden units is
small for a 7,488-token vocabulary.

This doubles the decoder to 1024 dims and uses a single-layer LSTM (the
higher-capacity configuration that is still stable on this GPU). It also scores
less often, since scoring costs ~10 minutes per pass and the selection signal
does not need per-epoch resolution.

    py -3.12 -m train.launch_e2e_big
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
        "--run-name", "flickr30k_e2e_big",
        "--backbone", "resnet50", "--image-size", "224",
        "--epochs", "12",
        "--batch-size", "24",
        "--lr", "4e-4", "--cnn-lr", "1e-5", "--dropout", "0.4",
        "--embed-dim", "1024", "--hidden-dim", "1024", "--attn-dim", "1024",
        "--num-layers", "1",
        "--max-caption-len", "32", "--scheduled-sampling", "0.3",
        "--holdout", "0.05",
        "--val-bleu-every", "2", "--val-bleu-limit", "100",
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
