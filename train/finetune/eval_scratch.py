"""Evaluate the from-scratch captioner on the full validation split.

The from-scratch ablation is the baseline every fine-tuned number is measured
against, and it is the one comparison in this project that was never like-for-
like: its only surviving prediction log covers **400** validation images while
every BLIP figure is computed on 6,042. A reviewer will attack that mismatch
before anything else in the report.

This tool fixes it. It rebuilds the attention captioner from its checkpoint and
its saved vocabulary, decodes the whole validation split at beam 5, and writes a
prediction log in the same shape as the fine-tuned ones so the shared scorer and
`diagnose_predictions` can consume it unchanged.

    py -3.12 -m train.finetune.eval_scratch \
        --run-name vizwiz_ft2 --checkpoint checkpoints/vizwiz_ft2/best.pt \
        --manifest data/manifests/vizwiz_val.jsonl \
        --out runs/vizwiz_ft2/preds_vizwiz_val_full.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_scratch_model(payload: dict, device):
    """Rebuild the attention captioner from a checkpoint payload.

    Architecture values come from the checkpoint's own config rather than from
    command-line defaults, so the rebuilt model matches the saved state dict. A
    mismatch here would load silently wrong or fail on shape, and either way the
    resulting numbers would be meaningless.
    """
    from train.data.vocab import Vocab
    from train.models.attention_captioner import AttentionCaptioner
    from train.models.captioner import Captioner

    config = payload.get("config") or {}
    vocab_size = int(payload.get("vocab_size") or config.get("vocab_size"))
    if not vocab_size:
        embed = next(v for k, v in payload["model_state"].items() if k == "embed.weight")
        vocab_size = embed.shape[0]

    common = dict(
        vocab_size=vocab_size,
        feature_dim=int(config.get("feature_dim", 2048)),
        embed_dim=int(config.get("embed_dim", 512)),
        hidden_dim=int(config.get("hidden_dim", 512)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.5)),
        pad_idx=0,
    )
    if config.get("model", "attention") == "attention":
        model = AttentionCaptioner(attn_dim=int(config.get("attn_dim", 512)), **common)
    else:
        model = Captioner(**common)
    model.load_state_dict(payload["model_state"])
    return model.to(device).eval(), config, vocab_size


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate the from-scratch captioner at full split.")
    parser.add_argument("--run-name", default="vizwiz_ft2")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/vizwiz_val.jsonl"))
    parser.add_argument("--feature-index", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--length-alpha", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    from train.data.dataset import load_feature_cache
    from train.data.vocab import Vocab
    from train.utils.device import configure_for_speed, get_device

    configure_for_speed()
    info = get_device()
    device = info.device

    run_dir = Path("runs") / args.run_name
    ckpt_dir = Path("checkpoints") / args.run_name
    checkpoint = args.checkpoint or (ckpt_dir / "best.pt")
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = payload.get("config") or {}

    vocab_path = ckpt_dir / "vocab.pkl"
    if not vocab_path.exists():
        raise SystemExit(
            f"vocabulary not found: {vocab_path}\n"
            "The vocabulary must be the one saved with this run - rebuilding it would "
            "remap token ids and make the predictions garbage."
        )
    vocab = Vocab.load(vocab_path)

    feature_index = args.feature_index or Path(
        config.get("feature_index")
        or f"data/features/{args.manifest.stem}_resnet101_spatial.jsonl"
    )
    if not feature_index.exists():
        raise SystemExit(f"feature index not found: {feature_index}")

    features, cached_rows = load_feature_cache(feature_index, mmap=True)

    wanted = load_rows(args.manifest)
    if args.limit:
        wanted = wanted[: args.limit]
    by_id = {row["image_id"]: row for row in cached_rows}
    rows = []
    for row in wanted:
        cached = by_id.get(row["image_id"])
        if cached is None:
            continue
        merged = dict(cached)
        merged["captions"] = row.get("captions") or cached.get("captions") or []
        merged["file_name"] = row.get("file_name") or cached.get("file_name")
        rows.append(merged)

    model, config, vocab_size = build_scratch_model(payload, device)
    print(f"device      : {device}")
    print(f"checkpoint  : {checkpoint}  (epoch {payload.get('epoch')})")
    print(f"vocab       : {len(vocab):,} tokens (embedding {vocab_size:,})")
    print(f"features    : {feature_index.name} {tuple(features.shape)}")
    print(f"images      : {len(rows):,} (of {len(wanted):,} requested)")
    print(f"beam        : {args.beam_size}  rep-penalty {args.repetition_penalty}  "
          f"no-repeat-{args.no_repeat_ngram}  length-alpha {args.length_alpha}")
    print()

    predictions: list[dict] = []
    started = time.time()
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start : start + args.batch_size]
        batch = torch.from_numpy(
            np.stack([features[int(r["feature_row"])] for r in chunk]).astype("float32")
        ).to(device)
        with torch.no_grad():
            if args.beam_size > 1:
                token_seqs = model.beam_decode(
                    batch, vocab.start_idx, vocab.end_idx,
                    beam_size=args.beam_size,
                    max_len=args.max_len,
                    min_len=args.min_len,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram=args.no_repeat_ngram,
                    length_alpha=args.length_alpha,
                )
            else:
                token_seqs = model.greedy_decode(
                    batch, vocab.start_idx, vocab.end_idx,
                    max_len=args.max_len,
                    min_len=args.min_len,
                )
        for row, tokens in zip(chunk, token_seqs):
            predictions.append(
                {
                    "image_id": row["image_id"],
                    "file_name": row.get("file_name"),
                    "text_detected": row.get("text_detected"),
                    "prediction": vocab.decode(tokens),
                    "references": row.get("captions") or [],
                }
            )
        done = min(start + args.batch_size, len(rows))
        print(f"\r  decoded {done:,}/{len(rows):,}", end="", flush=True)

    elapsed = time.time() - started
    print(f"\n  done in {elapsed:.0f}s ({len(rows) / max(elapsed, 1e-9):.1f} img/s)")

    out = args.out or (run_dir / "preds_vizwiz_val_full.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    print(f"  written: {out} ({len(predictions):,} rows)")

    # Report immediately, so a run is never left unscored.
    from train.train.evaluate import score_predictions

    scorable = [p for p in predictions if p.get("references")]
    if scorable:
        scored = score_predictions(scorable, with_spice=False, with_meteor=False)
        core = scored.get("metrics", {})
        print(f"\n  scored on {len(scorable):,} images with references")
        for key in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr-D", "ROUGE-L"):
            if key in core:
                print(f"    {key:<10} {core[key]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
