"""Compare deployment candidates: accuracy, size, latency and peak memory.

The headline "544 MB versus 6.8 MB" is misleading, because the LoRA adapter is
**not standalone** - it holds only the 1.77 M adapter weights and requires the
full BLIP-base model alongside it. This measures the real deployment footprint of
each option on this machine so the trade can be stated honestly.

    py -3.12 -m train.finetune.deploy_compare
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CANDIDATES = [
    # label, run dir, model size, checkpoint
    ("BLIP-large, full decoder", "runs/blip_large_head", "large", "checkpoints/blip_large_head/best.pt"),
    ("BLIP-base, LoRA adapter", "runs/blip_base_lora", "base", "checkpoints/blip_base_lora/best.pt"),
    ("BLIP-base, full decoder", "runs/blip_base_head", "base", "checkpoints/blip_base_head/best.pt"),
]

N_IMAGES = 24
BEAMS = 5


def main() -> int:
    import numpy as np
    import torch

    from train.finetune.cache import CachePaths, load_cache
    from train.finetune.base import build_model, load_trainable_state_dict, prepare_model_for_checkpoint
    from train.finetune.evaluate_blip import decode_cached
    from train.utils.device import configure_for_speed, get_device
    from transformers import AutoTokenizer

    configure_for_speed()
    info = get_device()
    print(f"device: {info.name}\n")

    results = []
    for label, run_dir, size, checkpoint in CANDIDATES:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            print(f"  {label}: checkpoint missing, skipped")
            continue

        # Load the image features this model was trained against.
        tag = "large" if size == "large" else "full"
        cache_index = Path(f"data/features/vizwiz_val_{tag}_blip.jsonl")
        cache_tensor = Path(f"data/features/vizwiz_val_{tag}_blip.npy")
        if not cache_index.exists():
            print(f"  {label}: cache missing ({cache_index.name}), skipped")
            continue
        features, index = load_cache(CachePaths(index=cache_index, tensor=cache_tensor))
        from train.finetune.dataset import build_split

        rows = build_split(index, keep=N_IMAGES, sample=True, seed=0)

        model_id = ("Salesforce/blip-image-captioning-large" if size == "large"
                    else "Salesforce/blip-image-captioning-base")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        t0 = time.time()
        model, _ = build_model(model_id, with_vision=False, dtype=torch.float32, device=info.device)
        prepare_model_for_checkpoint(model, payload, verbose=False)
        load_trainable_state_dict(model, payload["model"])
        load_seconds = time.time() - t0

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Warm-up, then timed runs.
        decode_cached(model, rows[:4], features, tokenizer, info.device,
                      batch_size=8, num_beams=BEAMS, max_new_tokens=30)
        timings = []
        for _ in range(3):
            t0 = time.time()
            captions = decode_cached(model, rows, features, tokenizer, info.device,
                                     batch_size=8, num_beams=BEAMS, max_new_tokens=30)
            timings.append((time.time() - t0) / len(rows) * 1000)

        peak = torch.cuda.max_memory_allocated() / 1e6

        # Real deployment footprint: adapter alone is useless without its base.
        artifact_mb = checkpoint_path.stat().st_size / 1e6
        base_mb = 990.0 if size == "base" else 0.0
        total_mb = artifact_mb + base_mb

        results.append({
            "label": label,
            "artifact_mb": artifact_mb,
            "base_model_mb": base_mb,
            "total_footprint_mb": total_mb,
            "load_seconds": load_seconds,
            "ms_per_image": statistics.median(timings),
            "peak_gpu_mb": peak,
            "sample": captions[0] if captions else "",
        })
        print(f"  {label}")
        print(f"    artifact        : {artifact_mb:,.1f} MB" +
              (f"  + base {base_mb:,.0f} MB = {total_mb:,.1f} MB total" if base_mb else "  (self-contained)"))
        print(f"    load time       : {load_seconds:.1f} s")
        print(f"    latency         : {statistics.median(timings):.0f} ms/image (beam {BEAMS}, batch 8)")
        print(f"    peak GPU        : {peak:,.0f} MB")
        print(f"    sample          : {captions[0][:64] if captions else ''}")
        print()

        del model
        torch.cuda.empty_cache()

    out = Path("runs/deploy_compare.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
