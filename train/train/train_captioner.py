"""Train the LSTM captioner on cached CNN features.

Two-stage training, per the project plan:

    # Stage 1 - pretrain on Flickr30k
    py -3.12 -m train.train.train_captioner ^
        --manifest data/manifests/flickr30k.jsonl ^
        --run-name flickr30k --epochs 20

    # Stage 2 - fine-tune on VizWiz-Captions, initialised from stage 1
    py -3.12 -m train.train.train_captioner ^
        --manifest data/manifests/vizwiz_train.jsonl ^
        --val-manifest data/manifests/vizwiz_val.jsonl ^
        --run-name vizwiz_ft --epochs 10 --lr 1e-4 ^
        --init-from checkpoints/flickr30k/best.pt

    # Ablation - VizWiz from scratch, no Flickr30k initialisation
    py -3.12 -m train.train.train_captioner ^
        --manifest data/manifests/vizwiz_train.jsonl ^
        --val-manifest data/manifests/vizwiz_val.jsonl ^
        --run-name vizwiz_scratch --epochs 10 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.data.dataset import (  # noqa: E402
    FeatureCaptionDataset,
    load_feature_cache,
    make_collate,
)
from train.data.vocab import Vocab  # noqa: E402
from train.models.attention_captioner import AttentionCaptioner  # noqa: E402
from train.models.captioner import Captioner  # noqa: E402
from train.utils.device import (  # noqa: E402
    check_rocm_compatibility,
    configure_for_speed,
    get_device,
    set_seed,
)


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.open(encoding="utf-8") if line.strip()
    ]


def build_vocab_from(rows: list[dict], min_freq: int, max_size: int) -> Vocab:
    captions: list[str] = []
    for row in rows:
        captions.extend(row.get("captions") or [])
    if not captions:
        raise SystemExit("no captions found; check the manifest")
    return Vocab.build(captions, min_freq=min_freq, max_size=max_size)


def load_compatible_state(
    model: Captioner, checkpoint: Path, device: torch.device
) -> tuple[int, int]:
    """Load weights whose shapes match, skipping the rest.

    Stage 2 rebuilds the vocabulary on VizWiz captions, so the embedding table
    and output projection change shape. The LSTM and feature projection carry
    over; the vocabulary-specific layers are reinitialised. Returns
    (loaded, skipped) counts.
    """
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model_state", payload)

    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current and current[key].shape == value.shape
    }
    skipped = [key for key in state if key not in compatible]

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    loaded = len(compatible)
    print(f"  initialised from {checkpoint}")
    print(f"    loaded {loaded} tensors, skipped {len(skipped)} (shape or name mismatch)")
    if skipped:
        preview = ", ".join(sorted(skipped)[:4])
        print(f"    skipped e.g.: {preview}")
    return loaded, len(skipped)


@torch.no_grad()
def evaluate_loss(
    model: Captioner, loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    total, count = 0.0, 0
    for features, captions in loader:
        features = features.to(device)
        captions = captions.to(device)
        loss = model.loss(features, captions)
        total += float(loss.item()) * features.size(0)
        count += features.size(0)
    model.train()
    return total / max(count, 1)


@torch.no_grad()
def sample_captions(
    model: Captioner,
    features: torch.Tensor,
    vocab: Vocab,
    beam_size: int,
    max_len: int,
    min_len: int,
) -> list[str]:
    if beam_size > 1:
        tokens = model.beam_decode(
            features, vocab.start_idx, vocab.end_idx,
            beam_size=beam_size, max_len=max_len, min_len=min_len,
        )
    else:
        tokens = model.greedy_decode(
            features, vocab.start_idx, vocab.end_idx,
            max_len=max_len, min_len=min_len,
        )
    return [vocab.decode(seq) for seq in tokens]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the captioner")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument(
        "--feature-index",
        type=Path,
        help="feature index JSONL (default: derived from --manifest)",
    )
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--backbone", default="resnet101")
    parser.add_argument("--run-name", default="captioner")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument(
        "--model",
        choices=["lstm", "attention"],
        default="lstm",
        help="'lstm' uses a pooled feature vector; 'attention' attends over "
             "cached [C, H, W] spatial maps (requires --spatial features)",
    )
    parser.add_argument("--attn-dim", type=int, default=512)
    parser.add_argument(
        "--max-caption-len",
        type=int,
        default=32,
        help="truncate captions to this many tokens (keeps <end>). Logits cost "
             "batch x seq_len x vocab, so this bounds peak memory; Flickr30k's "
             "longest captions are ~53 tokens but its median is far lower.",
    )
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument(
        "--vocab-from",
        type=Path,
        help="reuse a saved vocab.pkl instead of rebuilding on this manifest; "
             "required to preserve embedding/output weights when fine-tuning "
             "onto a new domain",
    )
    parser.add_argument(
        "--captions-per-image", type=int, default=None,
        help="cap captions used per image (default: all)",
    )
    parser.add_argument(
        "--holdout",
        type=float,
        default=0.05,
        help="fraction of images held out for validation when --val-manifest "
             "is not given (default 0.05; use 0 to disable)",
    )
    parser.add_argument(
        "--scheduled-sampling",
        type=float,
        default=0.5,
        help="final fraction of tokens fed from the model's own predictions; "
             "0 disables scheduled sampling and uses pure teacher forcing",
    )
    parser.add_argument("--resume", type=Path, help="resume a run from a checkpoint")
    parser.add_argument(
        "--init-from", type=Path,
        help="initialise weights from another run (stage 1 -> stage 2)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int, help="use only the first N images")
    args = parser.parse_args()

    set_seed(args.seed)
    info = get_device(prefer_gpu=not args.cpu)
    applied = configure_for_speed()
    for warning in check_rocm_compatibility(info):
        print(f"WARNING: {warning}")

    print("=" * 68)
    print(f"Run          : {args.run_name}")
    print(f"Device       : {info.name} ({info.backend})")
    print(f"Manifest     : {args.manifest}")
    print("=" * 68)
    print(info.describe())

    # ------------------------------------------------------------- data
    train_rows = load_rows(args.manifest)
    if args.limit:
        train_rows = train_rows[: args.limit]

    # The attention model needs spatial maps; the LSTM model uses pooled vectors.
    cache_suffix = "_spatial" if args.model == "attention" else ""
    feature_index = args.feature_index or (
        args.feature_dir
        / f"{args.manifest.stem}_{args.backbone}{cache_suffix}.jsonl"
    )
    features, cached_rows = load_feature_cache(
        feature_index, mmap=(args.model == "attention")
    )
    print(f"Feature cache: {feature_index.name} {tuple(features.shape)}")

    # The cache index is the authoritative row list: it carries `feature_row`
    # alongside the manifest fields. Restrict it to the requested manifest.
    requested = {row["image_id"] for row in train_rows}
    train_rows = [row for row in cached_rows if row["image_id"] in requested]
    if not train_rows:
        raise SystemExit(
            f"no overlap between manifest {args.manifest} and cache {feature_index}\n"
            "Extract features for this manifest first:\n"
            f"  py -3.12 -m train.features.extract --manifest {args.manifest}"
        )

    # ------------------------------------------------------- held-out split
    # Without this, every reported number is measured on data the model trained
    # on, which makes a memorising model look excellent. The split is by image,
    # never by caption, so no image leaks across the boundary.
    if args.val_manifest:
        val_ids = {row["image_id"] for row in load_rows(args.val_manifest)}
        val_rows = [row for row in train_rows if row["image_id"] in val_ids]
        train_rows = [row for row in train_rows if row["image_id"] not in val_ids]
    elif args.holdout > 0:
        ordered = sorted(train_rows, key=lambda r: r["image_id"])
        generator = random.Random(args.seed)
        generator.shuffle(ordered)
        cut = max(1, int(len(ordered) * args.holdout))
        val_rows = ordered[:cut]
        held_out = {row["image_id"] for row in val_rows}
        train_rows = [row for row in train_rows if row["image_id"] not in held_out]
        print(
            f"Held-out     : {cut:,} images ({args.holdout:.0%}) reserved for "
            "validation, excluded from training"
        )
    else:
        val_rows = []
        print(
            "WARNING: no --val-manifest and --holdout 0, so there is NO held-out "
            "data. Reported metrics will be training metrics."
        )

    if args.vocab_from:
        # Reuse a pretrained vocabulary so the embedding table and output
        # projection keep their learned weights. Rebuilding on the new domain
        # would reinitialise both, throwing away the language model that
        # pretraining produced.
        vocab = Vocab.load(args.vocab_from)
        print(f"\nVocabulary   : {len(vocab):,} tokens (reused from {args.vocab_from})")
    else:
        vocab = build_vocab_from(train_rows, args.min_freq, args.max_vocab)
        print(f"\nVocabulary   : {len(vocab):,} tokens")

    train_dataset = FeatureCaptionDataset(
        features, train_rows, vocab,
        captions_per_image=args.captions_per_image, seed=args.seed,
        max_caption_len=args.max_caption_len,
    )
    print(f"Train pairs  : {len(train_dataset):,} "
          f"({train_dataset.num_images:,} images)")

    val_dataset = None
    if val_rows:
        val_dataset = FeatureCaptionDataset(
            features, val_rows, vocab, seed=args.seed,
            max_caption_len=args.max_caption_len,
        )
        print(f"Val pairs    : {len(val_dataset):,} "
              f"({val_dataset.num_images:,} images)")

    pad_idx = vocab.pad_idx
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate(pad_idx),
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=make_collate(pad_idx),
        )
        if val_dataset
        else None
    )

    # ------------------------------------------------------------ model
    if args.model == "attention":
        model = AttentionCaptioner(
            vocab_size=len(vocab),
            feature_dim=args.feature_dim,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            attn_dim=args.attn_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pad_idx=pad_idx,
        ).to(info.device)
    else:
        model = Captioner(
            vocab_size=len(vocab),
            feature_dim=args.feature_dim,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pad_idx=pad_idx,
        ).to(info.device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model        : {args.model}")
    print(f"Parameters   : {param_count:,}")

    start_epoch = 0
    if args.init_from:
        load_compatible_state(model, args.init_from, info.device)
    if args.resume:
        payload = torch.load(args.resume, map_location=info.device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        start_epoch = int(payload.get("epoch", 0))
        print(f"  resumed from {args.resume} at epoch {start_epoch}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.resume and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01
    )

    run_dir = args.run_dir / args.run_name
    ckpt_dir = args.checkpoint_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = ckpt_dir / "vocab.pkl"
    vocab.save(vocab_path)

    config = {
        "run_name": args.run_name,
        "model": args.model,
        "manifest": str(args.manifest),
        "val_manifest": str(args.val_manifest) if args.val_manifest else None,
        "feature_index": str(feature_index),
        "backbone": args.backbone,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "feature_dim": args.feature_dim,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "attn_dim": args.attn_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "min_freq": args.min_freq,
        "max_vocab": args.max_vocab,
        "captions_per_image": args.captions_per_image,
        "vocab_size": len(vocab),
        "seed": args.seed,
        "device": str(info.device),
        "gpu_name": info.name,
        "backend": info.backend,
        "torch": info.torch_version,
        "runtime": info.runtime_version,
        "runtime_settings": applied,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history_path = run_dir / "history.jsonl"

    # Generator for scheduled-sampling decisions, seeded for reproducibility.
    sampling_generator = torch.Generator(device=info.device).manual_seed(args.seed)

    # ----------------------------------------------------------- training
    print("\nTraining")
    if args.scheduled_sampling > 0:
        print(
            f"Scheduled sampling: ramping 0.00 -> {args.scheduled_sampling:.2f} "
            "after the first teacher-forced epoch"
        )
    best_val = float("inf")
    started = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_dataset.set_epoch(epoch)
        epoch_started = time.time()
        running, seen = 0.0, 0

        # Scheduled sampling schedule: pure teacher forcing for the first
        # epoch, then a linear ramp to --scheduled-sampling over the remainder.
        # Epoch 0 must be teacher-forced or the model has nothing to condition on.
        if args.scheduled_sampling > 0 and args.epochs > 1:
            progress = max(0, epoch - 1) / max(1, args.epochs - 2)
            sampling_ratio = args.scheduled_sampling * min(1.0, progress)
        else:
            sampling_ratio = 0.0

        for step, (feats, caps) in enumerate(train_loader):
            feats = feats.to(info.device, non_blocking=True)
            caps = caps.to(info.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = model.scheduled_sampling_loss(
                feats, caps, sampling_ratio, generator=sampling_generator
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch = feats.size(0)
            running += float(loss.item()) * batch
            seen += batch

            if (step + 1) % args.log_every == 0:
                print(
                    f"  epoch {epoch + 1:2d}/{args.epochs} "
                    f"step {step + 1:5d}/{len(train_loader)} "
                    f"loss {running / seen:.4f} "
                    f"|g| {float(grad_norm):.2f} "
                    f"lr {optimizer.param_groups[0]['lr']:.2e} "
                    f"ss {sampling_ratio:.2f}",
                    flush=True,
                )

        scheduler.step()
        train_loss = running / max(seen, 1)
        val_loss = (
            evaluate_loss(model, val_loader, info.device) if val_loader else None
        )
        elapsed = time.time() - epoch_started

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(elapsed, 1),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        message = (
            f"  epoch {epoch + 1:2d} done | train {train_loss:.4f}"
            + (f" | val {val_loss:.4f}" if val_loss is not None else "")
            + f" | {elapsed:5.1f}s"
        )
        print(message, flush=True)

        payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab_size": len(vocab),
            "config": config,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(payload, ckpt_dir / "last.pt")

        monitor = val_loss if val_loss is not None else train_loss
        if monitor < best_val:
            best_val = monitor
            torch.save(payload, ckpt_dir / "best.pt")
            print(f"    new best ({monitor:.4f}) -> {ckpt_dir / 'best.pt'}")

        if (epoch + 1) % 5 == 0:
            torch.save(payload, ckpt_dir / f"epoch_{epoch + 1:03d}.pt")

    total = time.time() - started
    print(f"\nFinished in {total / 60:.1f} min")
    print(f"Checkpoints : {ckpt_dir}")
    print(f"History     : {history_path}")

    # ------------------------------------------------------- sample output
    if len(train_dataset) >= 3:
        print("\nSample captions (train images, beam decoding)")
        idx = torch.randperm(len(train_dataset))[:3].tolist()
        sample_feats = torch.stack(
            [train_dataset[i][0] for i in idx]
        ).to(info.device)
        for i, caption in enumerate(
            sample_captions(
                model, sample_feats, vocab,
                args.beam_size, args.max_len, args.min_len,
            )
        ):
            print(f"  [{i}] {caption}")
        print("  (a smoke-level model produces degenerate text; that is expected)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
