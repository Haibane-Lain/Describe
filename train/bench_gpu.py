"""Find the fastest GPU configuration that is actually stable.

Two failures remain unexplained and they cost most of the available compute:

* an OOM at batch 64 that reported "14.59 GiB free" while failing to allocate
  20 MiB (a fragmentation signature), and
* an access violation (0xC0000005) inside amdhip64_7.dll at the validation
  boundary.

Rather than burn an epoch per hypothesis, this measures memory and throughput
for each configuration, then runs the beam-search scoring path that crashed.
Reported throughput directly determines how much training fits in a day.

    py -3.12 -m train.bench_gpu
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.features.extract import build_spatial_encoder  # noqa: E402
from train.models.attention_captioner import AttentionCaptioner  # noqa: E402
from train.utils.device import configure_for_speed, get_device  # noqa: E402


def measure(batch: int, amp: bool, steps: int = 12, image_size: int = 224):
    """Return (img/s, peak_GiB) for one configuration, or None on failure."""
    device = torch.device("cuda")
    encoder, feature_dim = build_spatial_encoder("resnet50", image_size)
    for p in encoder.parameters():
        p.requires_grad_(True)
    encoder = encoder.to(device)
    decoder = AttentionCaptioner(
        vocab_size=7488, feature_dim=feature_dim, embed_dim=512, hidden_dim=512,
        attn_dim=512, num_layers=1, dropout=0.4, pad_idx=0,
    ).to(device)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    images = torch.randn(batch, 3, image_size, image_size, device=device)
    captions = torch.randint(0, 7488, (batch, 24), device=device)
    captions[:, 0] = 1

    started = time.time()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            features = encoder(images)
            loss = decoder.loss(features, captions)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 2**30

    del encoder, decoder, optimizer, images, captions
    gc.collect()
    torch.cuda.empty_cache()
    return batch * steps / elapsed, peak


def main() -> int:
    configure_for_speed()
    info = get_device()
    print(f"device: {info.name}\n")
    print("Training step throughput (higher is better)\n")
    print(f"  {'batch':>6} {'amp':>5} {'img/s':>8} {'peak GiB':>9}")
    results = []
    for batch, amp in ((32, False), (64, False), (32, True), (64, True), (96, False)):
        try:
            rate, peak = measure(batch, amp)
            results.append((batch, amp, rate, peak))
            print(f"  {batch:>6} {str(amp):>5} {rate:>8.1f} {peak:>9.2f}", flush=True)
        except torch.OutOfMemoryError as exc:
            print(f"  {batch:>6} {str(amp):>5}   OOM: {str(exc)[:60]}", flush=True)
        except Exception as exc:
            print(f"  {batch:>6} {str(amp):>5}   {type(exc).__name__}: {str(exc)[:60]}",
                  flush=True)

    if not results:
        print("\nno configuration completed")
        return 1

    best = max(results, key=lambda r: r[2])
    print(f"\nfastest stable: batch {best[0]}, amp {best[1]}, "
          f"{best[2]:.1f} img/s, {best[3]:.2f} GiB peak")

    # Beam-search scoring is where the driver crashed, so exercise it directly.
    print("\nBeam-search scoring path (the crash site)")
    encoder, feature_dim = build_spatial_encoder("resnet50", 224)
    encoder = encoder.to(info.device).eval()
    decoder = AttentionCaptioner(
        vocab_size=7488, feature_dim=feature_dim, embed_dim=512, hidden_dim=512,
        attn_dim=512, num_layers=1, dropout=0.4, pad_idx=0,
    ).to(info.device).eval()
    ok = True
    for val_batch, beam in ((8, 2), (16, 2), (16, 5), (32, 5)):
        try:
            torch.cuda.empty_cache()
            images = torch.randn(val_batch, 3, 224, 224, device=info.device)
            with torch.no_grad():
                feats = encoder(images)
                seqs = decoder.beam_decode(
                    feats, 1, 2, beam_size=beam, max_len=30, min_len=2,
                    repetition_penalty=1.2, no_repeat_ngram=3, length_alpha=0.7,
                )
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"  scoring batch {val_batch:>3} beam {beam}: OK "
                  f"({len(seqs)} seqs, peak {peak:.2f} GiB)", flush=True)
            del images, feats
        except Exception as exc:
            ok = False
            print(f"  scoring batch {val_batch:>3} beam {beam}: "
                  f"{type(exc).__name__}: {str(exc)[:60]}", flush=True)
    print(f"\n  scoring path: {'stable' if ok else 'FAILED at some setting'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
