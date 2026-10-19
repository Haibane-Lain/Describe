"""Measure BLIP fine-tuning throughput and memory on this GPU.

The question is how long fine-tuning takes and which configuration fits in
16 GB. Both need a real measurement: VRAM headroom decides the batch size,
batch size decides total time, and estimates have been wrong on this machine
before in both directions.

Three modes are measured, in the order the plan escalates through them:

``cached``
    **The one that matters.** Vision tower frozen, image features precomputed,
    text decoder (and its cross-attention) trained. This is what training
    actually runs. The vision tower is loaded only to produce the features, so
    the measured step covers the cached-feature forward and backward.

``lora``
    The same as ``cached`` but wrapping the text decoder's linear layers with
    LoRA. Reports the same step plus the adapter overhead.

``full``
    Everything trainable, raw pixels through both towers. This is the
    end-to-end ablation. It is measured so the cost of that option is a number
    rather than an opinion.

    py -3.12 -m train.bench_blip                  # base model, all modes
    py -3.12 -m train.bench_blip --model large    # BLIP-large
    py -3.12 -m train.bench_blip --modes cached --cached-batches 8 16 32
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.finetune.base import (  # noqa: E402
    build_model,
    resolve_model_id,
    text_decoder_lora_targets,
    trainable_state_dict,
)
from train.utils.device import configure_for_speed, get_device  # noqa: E402

CAPTION = "a man in a red helmet riding a small moped on a dirt road"

# Image counts for the corpora this project has on disk.
CORPORA = {
    "VizWiz train (23,431)": 23_431,
    "Flickr30k (31,783)": 31_783,
    "COCO train (82,783)": 82_783,
    "COCO + Flickr (114,566)": 114_566,
}

RESULT_ROWS: list[tuple] = []


def report(mode: str, model_label: str, batch: int, dtype_name: str, per_step: float, rate: float, peak: float, note: str = "") -> None:
    RESULT_ROWS.append((mode, model_label, batch, dtype_name, per_step, rate, peak, note))
    print(
        f"  {mode:<7} {model_label:<6} {batch:>5} {dtype_name:>6} {per_step * 1000:>9.0f} "
        f"{rate:>8.1f} {peak:>9.2f}  {note}",
        flush=True,
    )


def make_text_batch(batch: int, device, max_length: int = 32):
    """Tokenise a synthetic caption batch without needing the processor."""
    ids = torch.randint(5, 30_000, (batch, max_length), device=device)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "labels": ids.clone(),
    }


def measure(step, *, reps: int = 5) -> tuple[float, float]:
    step()  # warmup, also triggers any lazy kernel compilation
    torch.cuda.synchronize()
    started = time.time()
    for _ in range(reps):
        step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    return (time.time() - started) / reps, peak


def bench_cached(model_id: str, label: str, batches: list[int], dtypes, *, lora: bool) -> None:
    """Frozen vision tower, cached features, text decoder trained."""
    from train.finetune.train_blip import apply_manual_lora

    mode = "lora" if lora else "cached"

    for dtype_name, dtype in dtypes:
        for batch in batches:
            model = optimizer = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                model, report_info = build_model(model_id, with_vision=False, dtype=torch.float32)
                model = model.to("cuda")

                if lora:
                    apply_manual_lora(model, rank=16, alpha=32.0, dropout=0.0)
                else:
                    for parameter in model.text_decoder.parameters():
                        parameter.requires_grad_(True)

                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                model.train()
                parameters = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=1e-4)

                embeds = torch.randn(
                    batch, report_info.num_image_tokens, report_info.vision_hidden_size, device="cuda",
                    dtype=torch.float32,
                )
                text = make_text_batch(batch, "cuda")

                def step():
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=dtype, enabled=(dtype is torch.bfloat16)):
                        out = model(image_embeds=embeds, **text)
                    out.loss.backward()
                    optimizer.step()

                per, peak = measure(step)
                report(mode, label, batch, dtype_name, per, batch / per, peak, f"{trainable/1e6:.1f}M train")
            except torch.OutOfMemoryError:
                report(mode, label, batch, dtype_name, 0, 0, 0, "OOM")
            except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
                report(mode, label, batch, dtype_name, 0, 0, 0, f"{type(exc).__name__}: {str(exc)[:40]}")
            finally:
                del model, optimizer
                gc.collect()
                torch.cuda.empty_cache()


def bench_full(model_id: str, label: str, batches: list[int], dtypes, image_size: int) -> None:
    """Raw pixels through both towers, everything trainable."""
    for dtype_name, dtype in dtypes:
        for batch in batches:
            model = optimizer = images = None
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                model, report_info = build_model(model_id, with_vision=True, dtype=torch.float32)
                model = model.to("cuda")
                for parameter in model.parameters():
                    parameter.requires_grad_(True)

                trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

                images = torch.rand(batch, 3, image_size, image_size, device="cuda")
                text = make_text_batch(batch, "cuda")

                def step():
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=dtype, enabled=(dtype is torch.bfloat16)):
                        out = model(pixel_values=images, **text)
                    out.loss.backward()
                    optimizer.step()

                per, peak = measure(step, reps=3)
                report("full", label, batch, dtype_name, per, batch / per, peak, f"{trainable/1e6:.1f}M train")
            except torch.OutOfMemoryError:
                report("full", label, batch, dtype_name, 0, 0, 0, "OOM")
            except Exception as exc:  # noqa: BLE001
                report("full", label, batch, dtype_name, 0, 0, 0, f"{type(exc).__name__}: {str(exc)[:40]}")
            finally:
                del model, optimizer, images
                gc.collect()
                torch.cuda.empty_cache()


def bench_encoder(model_id: str, label: str, batches: list[int], image_size: int) -> None:
    """Vision tower throughput: this is the caching pass, run once per corpus."""
    for batch in batches:
        model = images = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model, report_info = build_model(model_id, with_vision=True, dtype=torch.float32)
            model = model.to("cuda")
            model.eval()
            images = torch.rand(batch, 3, image_size, image_size, device="cuda")

            with torch.no_grad():
                def step():
                    model.encode_image(images)

                per, peak = measure(step, reps=5)

            report("encode", label, batch, "fp32", per, batch / per, peak, "features only")
        except Exception as exc:  # noqa: BLE001
            report("encode", label, batch, "fp32", 0, 0, 0, f"{type(exc).__name__}: {str(exc)[:40]}")
        finally:
            del model, images
            gc.collect()
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark BLIP fine-tuning on this GPU.")
    parser.add_argument("--model", default="base", help="'base', 'large', or a Hub repo id")
    parser.add_argument("--modes", nargs="+", default=["cached", "lora", "full", "encode"],
                        choices=["cached", "lora", "full", "encode"])
    parser.add_argument("--cached-batches", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--full-batches", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--dtypes", nargs="+", default=["bf16", "fp32"], choices=["bf16", "fp32"])
    args = parser.parse_args()

    configure_for_speed()
    info = get_device()
    model_id = resolve_model_id(args.model)
    label = "large" if "large" in model_id else "base"
    dtypes = [(name, torch.bfloat16 if name == "bf16" else torch.float32) for name in args.dtypes]

    props = torch.cuda.get_device_properties(0)
    print(f"device : {info.name}  {props.total_memory / 2**30:.2f} GiB  ({info.backend})")
    print(f"model  : {model_id}\n")

    probe, report_info = build_model(model_id, with_vision=True, dtype=torch.float32)
    image_size = report_info.image_size
    del probe
    gc.collect()
    torch.cuda.empty_cache()

    print(f"  image size {image_size}, {report_info.num_image_tokens} image tokens, "
          f"text hidden {report_info.text_hidden_size}, "
          f"text decoder {report_info.num_parameters_text_decoder / 1e6:.1f} M params")
    print(f"  LoRA target leaves: {sorted(text_decoder_lora_targets())}\n")

    print(f"  {'mode':<7} {'model':<6} {'batch':>5} {'dtype':>6} {'ms/step':>9} {'img/s':>8} {'peak GiB':>9}  note")
    if "encode" in args.modes:
        bench_encoder(model_id, label, [16, 32, 64], image_size)
    if "cached" in args.modes:
        bench_cached(model_id, label, args.cached_batches, dtypes, lora=False)
    if "lora" in args.modes:
        bench_cached(model_id, label, args.cached_batches, dtypes, lora=True)
    if "full" in args.modes:
        bench_full(model_id, label, args.full_batches, dtypes, image_size)

    if not RESULT_ROWS:
        print("\nnothing completed")
        return 1

    usable = [row for row in RESULT_ROWS if row[5] > 0 and row[6] < props.total_memory / 2**30 * 0.92]
    if not usable:
        print("\nno configuration completed inside 92% of VRAM")
        return 1

    print("\n  Fastest configuration per mode that fits comfortably in VRAM:")
    for mode in ("encode", "cached", "lora", "full"):
        rows = [row for row in usable if row[0] == mode]
        if not rows:
            continue
        best = max(rows, key=lambda row: row[5])
        print(f"    {mode:<7} batch {best[2]:<4} {best[3]:<5} {best[5]:7.1f} img/s  ({best[6]:.2f} GiB peak)")

    print("\n  Projected wall time for one epoch (training step only):")
    print(f"    {'corpus':<26} {'cached':>9} {'lora':>9} {'full':>9}")
    for name, count in CORPORA.items():
        cells = []
        for mode in ("cached", "lora", "full"):
            rows = [row for row in usable if row[0] == mode]
            if not rows:
                cells.append("        -")
                continue
            best = max(rows, key=lambda row: row[5])
            cells.append(f"{count / best[5] / 60:8.1f}m")
        print(f"    {name:<26} {cells[0]} {cells[1]} {cells[2]}")

    print("\n  'cached' and 'lora' include the training step only; add ~10-15% for")
    print("  validation and loading. The one-off caching pass is 'encode' applied")
    print("  to the corpus, and it happens once, not per epoch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
