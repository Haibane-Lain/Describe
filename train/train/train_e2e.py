"""End-to-end captioner training: encoder and decoder trained jointly.

The cached-feature pipeline trains a decoder on frozen ImageNet features, which
caps caption quality because ImageNet features are not caption features. This
script runs raw images through a trainable encoder so the visual representation
adapts to the captioning task.

Two details matter and are easy to get wrong:

* **Discriminative learning rates.** The encoder is pretrained, so it trains at
  ``--cnn-lr`` (default 1e-5), roughly 50x below the decoder. Fine-tuning a
  pretrained CNN at the decoder's rate destroys it.
* **Selection on BLEU, not loss.** Captioning cross-entropy plateaus long
  before BLEU peaks, so choosing checkpoints by validation loss stops training
  while quality is still improving.

    py -3.12 -m train.train.train_e2e --manifest data/manifests/flickr30k.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.data.vocab import Vocab  # noqa: E402
from train.features.extract import build_spatial_encoder, preprocess  # noqa: E402
from train.models.attention_captioner import AttentionCaptioner  # noqa: E402
from train.models.captioner import collate_captions  # noqa: E402
from train.train.evaluate import corpus_bleu, tokenize_simple  # noqa: E402
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
        raise SystemExit("no captions found; cannot build a vocabulary")
    return Vocab.build(captions, min_freq=min_freq, max_size=max_size)


def build_image_index(image_root: Path) -> dict[str, Path]:
    """Map file name -> path, caching the result to a JSON sidecar.

    ``rglob`` over 123k COCO images is slow and every dataset construction
    repeats it, so the index is written once next to the images and reused.
    The cache is invalidated if the directory's modification time changes.
    """
    cache_path = image_root.parent / f".{image_root.name}_index.json"
    try:
        stamp = int(image_root.stat().st_mtime)
    except OSError:
        stamp = 0

    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("mtime") == stamp:
                return {name: Path(p) for name, p in payload["index"].items()}
        except (json.JSONDecodeError, KeyError, OSError):
            pass  # fall through and rebuild

    index: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            index[path.name] = path

    try:
        cache_path.write_text(
            json.dumps(
                {"mtime": stamp, "index": {n: str(p) for n, p in index.items()}}
            ),
            encoding="utf-8",
        )
    except OSError:
        pass  # a missing cache only costs time
    return index


class ImageCaptionDataset(Dataset):
    """Raw images paired with encoded captions.

    Images are decoded once and cached in float16 under a bounded LRU cap.
    JPEG decoding was the throughput bottleneck (~78 img/s), but caching every
    image is not affordable: at 301 KB each in float16, all 30,194 Flickr30k
    images need 8.5 GB, and caching them in float32 needed ~18 GB, which
    exhausted RAM and stalled training. A bounded cache keeps most of the
    speedup without the risk.
    """

    def __init__(
        self,
        rows: list[dict],
        image_root: Path,
        vocab: Vocab,
        image_size: int = 224,
        max_caption_len: int = 32,
        captions_per_image: int | None = None,
        cache_images: bool = True,
        cache_limit: int = 8000,
    ) -> None:
        self.vocab = vocab
        self.image_size = image_size
        self.max_caption_len = max_caption_len
        self.cache_images = cache_images
        self.cache_limit = cache_limit

        self.image_index = build_image_index(image_root)

        self.pairs: list[tuple[str, str]] = []
        for row in rows:
            captions = row.get("captions") or []
            if not captions:
                continue
            selected = captions
            if captions_per_image is not None:
                selected = captions[:captions_per_image]
            for caption in selected:
                encoded = vocab.encode(caption)
                if len(encoded) > 2:
                    self.pairs.append((row["file_name"], caption))

        # OrderedDict used as an LRU so the cache stays under `cache_limit`.
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, file_name: str) -> torch.Tensor:
        """Return a preprocessed image tensor, caching in float16 with an LRU cap."""
        if self.cache_images:
            cached = self._cache.get(file_name)
            if cached is not None:
                self._cache.move_to_end(file_name)
                return cached.float()
        path = self.image_index.get(file_name)
        if path is None:
            raise FileNotFoundError(f"image not found: {file_name}")
        with Image.open(path) as handle:
            tensor = preprocess(handle, self.image_size)
        if self.cache_images and self.cache_limit > 0:
            self._cache[file_name] = tensor.to(torch.float16)
            if len(self._cache) > self.cache_limit:
                self._cache.popitem(last=False)
        return tensor

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[int]]:
        file_name, caption = self.pairs[index]
        encoded = self.vocab.encode(caption)
        if len(encoded) > self.max_caption_len:
            encoded = encoded[: self.max_caption_len - 1] + [encoded[-1]]
        return self._load_image(file_name), encoded


class PadCollate:
    """Collate images with captions padded to a common length.

    A class rather than a closure: DataLoader workers receive the collate
    function by pickling, and a locally-defined closure cannot be pickled
    ("Can't pickle local object 'make_collate.<locals>.collate'").
    """

    def __init__(self, pad_idx: int) -> None:
        self.pad_idx = pad_idx

    def __call__(self, batch: list[tuple[torch.Tensor, list[int]]]):
        images = torch.stack([item[0] for item in batch])
        captions = collate_captions([item[1] for item in batch], self.pad_idx)
        return images, captions


class EncoderDecoder(nn.Module):
    """Trainable CNN encoder feeding the attention decoder."""

    def __init__(self, decoder: AttentionCaptioner, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        features = self.encoder(images)
        return self.decoder(features, captions)

    def loss(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        sampling_ratio: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        features = self.encoder(images)
        if sampling_ratio > 0:
            return self.decoder.scheduled_sampling_loss(
                features, captions, sampling_ratio, generator=generator
            )
        return self.decoder.loss(features, captions)

    @torch.no_grad()
    def decode(
        self,
        images: torch.Tensor,
        start_idx: int,
        end_idx: int,
        beam_size: int,
        max_len: int,
        min_len: int,
        repetition_penalty: float,
        no_repeat_ngram: int,
        length_alpha: float,
    ) -> list[list[int]]:
        features = self.encoder(images)
        return self.decoder.beam_decode(
            features, start_idx, end_idx,
            beam_size=beam_size, max_len=max_len, min_len=min_len,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram=no_repeat_ngram,
            length_alpha=length_alpha,
        )


def split_rows(
    train_rows: list[dict],
    val_rows: list[dict] | None,
    holdout: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Return (train_rows, val_rows).

    ``val_rows`` is passed in already loaded, rather than being filtered out of
    ``train_rows``. Filtering only works when the validation manifest is a
    subset of the training manifest; for COCO the official train and val splits
    are disjoint, so filtering silently produced **zero** validation images and
    the run had no held-out signal at all.
    """
    if val_rows is not None:
        val_ids = {r["image_id"] for r in val_rows}
        # Guard against leakage if the two manifests do overlap.
        train_rows = [r for r in train_rows if r["image_id"] not in val_ids]
        return train_rows, val_rows
    if holdout <= 0:
        return train_rows, []
    ordered = sorted(train_rows, key=lambda r: r["image_id"])
    generator = random.Random(seed)
    generator.shuffle(ordered)
    cut = max(1, int(len(ordered) * holdout))
    held = ordered[:cut]
    held_out = {row["image_id"] for row in held}
    return [r for r in train_rows if r["image_id"] not in held_out], held


@torch.no_grad()
def score_bleu(
    model: EncoderDecoder,
    dataset: ImageCaptionDataset,
    vocab: Vocab,
    device: torch.device,
    image_size: int,
    batch_size: int,
    limit: int,
    beam_size: int,
    min_len: int,
    repetition_penalty: float,
    no_repeat_ngram: int,
    length_alpha: float,
) -> dict[str, float]:
    """Beam-search a validation subset and compute corpus BLEU."""
    model.eval()
    # Beam search expands the batch by beam_size and holds a vocab-sized logit
    # tensor per step, so scoring needs a much smaller batch than training.
    torch.cuda.empty_cache() if device.type == "cuda" else None
    # One caption per image is enough for a model-selection signal.
    seen: set[str] = set()
    sample: list[tuple[str, str]] = []
    for file_name, caption in dataset.pairs:
        if file_name in seen:
            continue
        seen.add(file_name)
        sample.append((file_name, caption))
        if len(sample) >= limit:
            break

    hypotheses: list[list[str]] = []
    references: list[list[list[str]]] = []
    for start in range(0, len(sample), batch_size):
        chunk = sample[start : start + batch_size]
        images = torch.stack([dataset._load_image(name) for name, _ in chunk])
        images = images.to(device)
        try:
            decoded = model.decode(
                images, vocab.start_idx, vocab.end_idx,
                beam_size=beam_size, max_len=30, min_len=min_len,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
                length_alpha=length_alpha,
            )
        except torch.OutOfMemoryError:
            # Fall back to greedy on this chunk rather than losing the epoch.
            print(f"\n  WARNING: beam search OOM at batch {batch_size}; "
                  "retrying greedy", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            decoded = model.decoder.greedy_decode(
                model.encoder(images), vocab.start_idx, vocab.end_idx,
                max_len=30, min_len=min_len,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram=no_repeat_ngram,
            )
        for tokens, (_, reference) in zip(decoded, chunk):
            hypotheses.append(tokenize_simple(vocab.decode(tokens)))
            references.append([tokenize_simple(reference)])

    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return corpus_bleu(hypotheses, references, max_n=4)


def save_checkpoint(payload: dict, path: Path) -> None:
    """Write a checkpoint atomically, from CPU tensors.

    Two crashes killed long runs here, both memory corruption inside ntdll
    (0xc0000374 heap corruption, 0xc0000005 access violation). One hit while
    ``torch.save`` was writing, leaving a truncated ``last.pt`` that could not
    be reloaded. So:

    * the state dict is moved to CPU first, keeping the save away from GPU
      memory operations, and
    * the file is written to a temporary path and then renamed, so an
      interrupted save cannot destroy the previous good checkpoint.
    """
    import tempfile

    cpu_payload = dict(payload)
    if "model_state" in cpu_payload:
        cpu_payload["model_state"] = {
            key: value.detach().to("cpu", copy=True)
            for key, value in cpu_payload["model_state"].items()
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        torch.save(cpu_payload, temp_path)
        temp_path.replace(path)      # atomic on the same filesystem
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train an attention captioner end-to-end on raw images"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--run-name", default="e2e")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4, help="decoder learning rate")
    parser.add_argument(
        "--cnn-lr", type=float, default=1e-5,
        help="encoder learning rate; keep well below --lr, the encoder is pretrained",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--attn-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--min-freq", type=int, default=5)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--vocab-from", type=Path)
    parser.add_argument("--max-caption-len", type=int, default=32)
    parser.add_argument("--captions-per-image", type=int)
    parser.add_argument(
        "--cache-limit", type=int, default=8000,
        help="max images held in the float16 LRU cache (~300 KB each); "
             "8000 is ~2.4 GB. 0 disables caching. Ignored when --num-workers > 0.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=6,
        help="DataLoader worker processes for image decoding. 0 decodes on the "
             "main thread, which starves the GPU (measured: GPU idle, step "
             "counter frozen, 5 CPU cores busy).",
    )
    parser.add_argument("--holdout", type=float, default=0.05)
    parser.add_argument("--scheduled-sampling", type=float, default=0.3)
    parser.add_argument("--val-bleu-every", type=int, default=2)
    parser.add_argument("--val-bleu-limit", type=int, default=200)
    parser.add_argument(
        "--val-batch-size", type=int, default=16,
        help="batch size for beam-search scoring; beam search expands the batch "
             "by beam_size and needs far less than the training batch",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--min-len", type=int, default=2)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram", type=int, default=3)
    parser.add_argument("--length-alpha", type=float, default=0.7)
    parser.add_argument("--amp", action="store_true", help="mixed precision")
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    info = get_device(prefer_gpu=not args.cpu)
    configure_for_speed()
    for warning in check_rocm_compatibility(info):
        print(f"WARNING: {warning}")
    device = info.device

    print("=" * 70)
    print(f"Run          : {args.run_name}  (end-to-end)")
    print(f"Device       : {info.name} ({info.backend})")
    print(f"Backbone     : {args.backbone} @ {args.image_size}px (trainable)")
    print("=" * 70)

    # --------------------------------------------------------------- data
    all_rows = load_rows(args.manifest)
    if args.limit:
        all_rows = all_rows[: args.limit]
    # Load the validation manifest independently: COCO's official train and val
    # splits are disjoint, so it cannot be derived by filtering the train rows.
    explicit_val = load_rows(args.val_manifest) if args.val_manifest else None
    train_rows, val_rows = split_rows(all_rows, explicit_val, args.holdout, args.seed)
    if not train_rows:
        raise SystemExit("no training rows after splitting")
    print(f"Train images : {len(train_rows):,}")
    print(f"Val images   : {len(val_rows):,}")
    if explicit_val is not None and not val_rows:
        raise SystemExit(
            f"--val-manifest {args.val_manifest} produced 0 validation images; "
            "refusing to train with no held-out signal"
        )

    if args.vocab_from:
        vocab = Vocab.load(args.vocab_from)
        print(f"Vocabulary   : {len(vocab):,} tokens (reused)")
    else:
        vocab = build_vocab_from(train_rows, args.min_freq, args.max_vocab)
        print(f"Vocabulary   : {len(vocab):,} tokens")

    train_ds = ImageCaptionDataset(
        train_rows, args.image_root, vocab,
        image_size=args.image_size,
        max_caption_len=args.max_caption_len,
        captions_per_image=args.captions_per_image,
        cache_limit=args.cache_limit,
    )
    val_ds = (
        ImageCaptionDataset(
            val_rows, args.image_root, vocab,
            image_size=args.image_size,
            max_caption_len=args.max_caption_len,
            # Do not cache validation images. The training cache already holds
            # several GB, and a second cache exhausted RAM (free memory fell to
            # 0.23 GB and training stalled). Validation is read once per scoring
            # pass, so caching buys little and costs a lot.
            cache_images=False,
        )
        if val_rows
        else None
    )
    print(f"Train pairs  : {len(train_ds):,}")

    pad_idx = vocab.pad_idx
    # Image decoding must happen off the main thread. With num_workers=0 the
    # main thread decoded every JPEG while the GPU sat idle: measured 15.77 GiB
    # of VRAM free and the step counter frozen, with the process burning five
    # CPU cores. Workers keep the GPU fed.
    #
    # Each worker would otherwise hold its own copy of the image cache, so
    # caching is disabled when workers are used and the OS page cache does the
    # job instead.
    use_workers = args.num_workers > 0
    if use_workers and train_ds.cache_images:
        train_ds.cache_images = False
        train_ds._cache.clear()
        print("Note         : image cache disabled (workers handle decoding)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=PadCollate(pad_idx), drop_last=False,
        persistent_workers=use_workers,
        prefetch_factor=4 if use_workers else None,
        pin_memory=device.type == "cuda",
    )

    # -------------------------------------------------------------- model
    encoder, feature_dim = build_spatial_encoder(args.backbone, args.image_size)
    for param in encoder.parameters():
        param.requires_grad_(True)          # the whole point of this script
    encoder = encoder.to(device)

    decoder = AttentionCaptioner(
        vocab_size=len(vocab),
        feature_dim=feature_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        attn_dim=args.attn_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=pad_idx,
    ).to(device)

    model = EncoderDecoder(decoder, encoder).to(device)
    encoder_params = sum(p.numel() for p in encoder.parameters())
    decoder_params = sum(p.numel() for p in decoder.parameters())
    print(f"Params       : encoder {encoder_params:,} + decoder {decoder_params:,}")

    start_epoch = 0
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        start_epoch = int(payload.get("epoch", 0))
        print(f"  resumed from {args.resume} at epoch {start_epoch}")
    elif args.init_from:
        payload = torch.load(args.init_from, map_location=device, weights_only=False)
        state = payload.get("model_state", payload)
        current = model.state_dict()
        compatible = {
            k: v for k, v in state.items()
            if k in current and current[k].shape == v.shape
        }
        model.load_state_dict(compatible, strict=False)
        print(f"  initialised from {args.init_from} "
              f"({len(compatible)}/{len(state)} tensors matched)")

    # Discriminative learning rates: the encoder is pretrained.
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder.parameters(), "lr": args.cnn_lr},
            {"params": decoder.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    ckpt_dir = args.checkpoint_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.run_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(ckpt_dir / "vocab.pkl")
    history_path = run_dir / "history.jsonl"

    config = vars(args).copy()
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in config.items()}
    config["model"] = "attention"
    config["feature_dim"] = feature_dim
    config["end_to_end"] = True
    # The evaluator needs to know where the raw images live for this run.
    config["image_root"] = str(args.image_root)

    print("\nTraining")
    print(f"  encoder lr {args.cnn_lr:g} | decoder lr {args.lr:g} | "
          f"batch {args.batch_size} x accum {args.grad_accum}")
    if args.scheduled_sampling > 0:
        print(f"  scheduled sampling ramps 0 -> {args.scheduled_sampling}")
    print(f"  checkpoint selection: best validation BLEU-4 "
          f"(every {args.val_bleu_every} epochs)")

    best_bleu4 = -1.0
    best_score = -1.0
    steps_per_epoch = len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, seen = 0.0, 0
        started = time.time()
        # Scheduled sampling ramps in only after a teacher-forced first epoch.
        ratio = 0.0
        if args.scheduled_sampling > 0 and epoch >= 1:
            ratio = args.scheduled_sampling * min(
                1.0, (epoch) / max(args.epochs - 1, 1)
            )

        optimizer.zero_grad(set_to_none=True)
        for step, (images, captions) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss = model.loss(
                    images, captions, sampling_ratio=ratio, generator=generator
                )
                scaled = loss / args.grad_accum

            scaler.scale(scaled).backward()

            if step % args.grad_accum == 0 or step == steps_per_epoch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running += float(loss.item()) * images.size(0)
            seen += images.size(0)

            if step % args.log_every == 0 or step == steps_per_epoch:
                elapsed = time.time() - started
                rate = seen / elapsed if elapsed else 0
                print(
                    f"  epoch {epoch + 1:2d}/{args.epochs} step {step:5d}/{steps_per_epoch}"
                    f" loss {running / max(seen, 1):.4f}"
                    f" | {rate:5.1f} img/s ss {ratio:.2f}",
                    flush=True,
                )

        scheduler.step()
        epoch_seconds = time.time() - started
        train_loss = running / max(seen, 1)

        record: dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "seconds": epoch_seconds,
            "scheduled_sampling": ratio,
        }

        # ------------------------------------------- validation BLEU
        if val_ds is not None and (epoch + 1) % args.val_bleu_every == 0:
            metrics = score_bleu(
                model, val_ds, vocab, device, args.image_size,
                batch_size=args.val_batch_size, limit=args.val_bleu_limit,
                beam_size=args.beam_size, min_len=args.min_len,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram=args.no_repeat_ngram,
                length_alpha=args.length_alpha,
            )
            record.update({f"val_{k}": v for k, v in metrics.items()})
            bleu4 = metrics.get("BLEU-4", 0.0)
            # BLEU-4 is 0 for every epoch early in training, so selecting on it
            # alone freezes the checkpoint at the first epoch evaluated. Score
            # on a weighted blend that stays discriminative while BLEU-4 is
            # still zero, and report BLEU-4 separately.
            score = (
                0.15 * metrics.get("BLEU-1", 0.0)
                + 0.25 * metrics.get("BLEU-2", 0.0)
                + 0.25 * metrics.get("BLEU-3", 0.0)
                + 0.35 * bleu4
            )
            record["selection_score"] = score
            print(
                f"  epoch {epoch + 1:2d} done | train {train_loss:.4f} | "
                f"val BLEU-1 {metrics.get('BLEU-1', 0):.4f} "
                f"BLEU-2 {metrics.get('BLEU-2', 0):.4f} "
                f"BLEU-4 {bleu4:.4f} | score {score:.4f} | {epoch_seconds:.0f}s",
                flush=True,
            )
            if score > best_score:
                best_score = score
                best_bleu4 = bleu4
                save_checkpoint(
                    {
                        "model_state": model.state_dict(),
                        "config": config,
                        "vocab_size": len(vocab),
                        "epoch": epoch + 1,
                        "val_bleu4": bleu4,
                        "selection_score": score,
                    },
                    ckpt_dir / "best.pt",
                )
                print(f"    new best (score {score:.4f}) -> {ckpt_dir / 'best.pt'}",
                      flush=True)
        else:
            print(f"  epoch {epoch + 1:2d} done | train {train_loss:.4f} | "
                  f"{epoch_seconds:.0f}s", flush=True)

        save_checkpoint(
            {
                "model_state": model.state_dict(),
                "config": config,
                "vocab_size": len(vocab),
                "epoch": epoch + 1,
            },
            ckpt_dir / "last.pt",
        )
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                {"model_state": model.state_dict(), "config": config,
                 "epoch": epoch + 1},
                ckpt_dir / f"epoch_{epoch + 1:03d}.pt",
            )

        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    print(f"\nBest validation BLEU-4: {best_bleu4:.4f}  (selection score {best_score:.4f})")
    print(f"Checkpoints: {ckpt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
