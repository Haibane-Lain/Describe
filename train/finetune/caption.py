"""Caption a photograph, a folder, or a batch - from the command line.

This is the practical front door to the trained captioner. Everything else in
this package either trains or measures; this is the one you run when you want to
look at a picture and see what the model says about it.

    # one photograph, default model (BLIP-large, the best one)
    py -3.12 -m train.finetune.caption myphoto.jpg

    # anything Pillow can open, including PNG and WebP
    py -3.12 -m train.finetune.caption shot.png

    # a whole folder, or a glob
    py -3.12 -m train.finetune.caption path\\to\\photos\\
    py -3.12 -m train.finetune.caption "photos\\*.jpg"

    # compare models on the same images
    py -3.12 -m train.finetune.caption myphoto.jpg --model blip_base_lora
    py -3.12 -m train.finetune.caption myphoto.jpg --all-models

    # machine-readable, for piping into something else
    py -3.12 -m train.finetune.caption photos\\ --jsonl --out caps.jsonl

Models, best first (all measured on the full 7,542-image VizWiz validation split):

    blip_large_head       BLEU-4 0.2045   544 MB   the best captioner
    blip_base_head        BLEU-4 0.1958   526 MB
    blip_base_textaware   BLEU-4 0.1891   526 MB   trained on text-bearing images only
    blip_base_lora        BLEU-4 0.1897     7 MB   adapter only, smallest artifact

A note on what this model does and does not do. It *describes* what it sees and
names objects well. It does **not** transcribe text: asked about a medicine
bottle it will say "a bottle of medicine", not the name printed on it. Reading
printed text is the OCR component's job, not this model's. Treat the output as a
description of the scene, not as an authority on any detail written on an object.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Best first. Kept here rather than discovered by scanning `checkpoints/` so the
# default is a deliberate choice and the list is stable across machines.
KNOWN_MODELS = {
    "blip_large_head": ("large", "best - BLEU-4 0.2045"),
    "blip_base_head": ("base", "BLEU-4 0.1958"),
    "blip_base_textaware": ("base", "BLEU-4 0.1891"),
    "blip_base_lora": ("base", "BLEU-4 0.1897, adapter only (7 MB)"),
}
DEFAULT_MODEL = "blip_large_head"


def expand_inputs(inputs: list[str]) -> list[Path]:
    """Turn paths, globs and directories into a sorted list of image files."""
    import glob as globlib

    found: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            found.extend(
                p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            )
            continue
        if path.is_file():
            found.append(path)
            continue
        matches = [
            Path(m) for m in globlib.glob(raw, recursive=True)
            if Path(m).is_file() and Path(m).suffix.lower() in IMAGE_SUFFIXES
        ]
        if matches:
            found.extend(sorted(matches))
        else:
            raise SystemExit(f"no image found at {raw!r}")
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for p in found:
        key = p.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def resolve_models(args) -> list[str]:
    if args.all_models:
        return list(KNOWN_MODELS)
    if args.model:
        return [args.model]
    return [DEFAULT_MODEL]


def load_captioner(name: str, device=None):
    """Load one model by run name, or by an explicit checkpoint path."""
    from train.finetune.predict import BlipCaptioner

    run_dir = Path("runs") / name
    if not run_dir.exists():
        raise SystemExit(
            f"no run directory {run_dir}. Available: {', '.join(KNOWN_MODELS)}\n"
            "If the checkpoint exists elsewhere, pass --checkpoint explicitly."
        )
    model_size = KNOWN_MODELS.get(name, ("base", ""))[0]
    return BlipCaptioner.from_run(run_dir, model=model_size, device=device)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Caption photographs with a fine-tuned BLIP captioner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("inputs", nargs="+", help="image file, directory, or glob")
    parser.add_argument("--model", default=None, choices=sorted(KNOWN_MODELS),
                        help=f"run name; default {DEFAULT_MODEL}")
    parser.add_argument("--all-models", action="store_true",
                        help="caption with every trained model, for comparison")
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="explicit checkpoint path, overriding --model")
    parser.add_argument("--num-beams", type=int, default=5,
                        help="beam width; 5 is what the reported metrics use, 1 is ~4x faster")
    parser.add_argument("--bf16", action="store_true",
                        help="bf16 autocast: about 4x faster at beam-5, but captions are not "
                             "bit-reproducible. Off by default so results are stable")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--jsonl", action="store_true", help="emit one JSON object per caption")
    parser.add_argument("--out", type=Path, default=None, help="also write predictions to this file")
    parser.add_argument("--timing", action="store_true", help="report per-image latency")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    import torch

    from train.utils.device import configure_for_speed, get_device

    configure_for_speed()
    device = torch.device("cpu") if args.cpu else get_device().device

    images = expand_inputs(args.inputs)
    models = resolve_models(args)
    if args.checkpoint:
        models = [str(args.checkpoint)]

    if not args.jsonl:
        print(f"device : {device}")
        print(f"images : {len(images)}")
        print(f"models : {', '.join(models)}")
        print()

    records: list[dict] = []
    for name in models:
        if args.checkpoint:
            from train.finetune.predict import BlipCaptioner

            captioner = BlipCaptioner.from_run(
                Path("runs") / DEFAULT_MODEL, model="large",
                checkpoint=args.checkpoint, device=device,
            )
            label = Path(args.checkpoint).parent.name
        else:
            label = name
            captioner = load_captioner(name, device=device)

        started = time.time()
        captions = captioner.caption_batch(
            [str(p) for p in images],
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            bf16=args.bf16,
        )
        elapsed = time.time() - started

        for path, caption in zip(images, captions):
            record = {
                "model": label,
                "image": str(path),
                "caption": caption.text,
                "latency_ms": round(caption.latency_ms, 1),
            }
            records.append(record)
            if args.jsonl:
                print(json.dumps(record))
            else:
                print(f"[{label}] {path.name}")
                print(f"    {caption.text}")
                if args.timing:
                    print(f"    ({caption.latency_ms:.0f} ms)")

        if not args.jsonl:
            per_image = 1000 * elapsed / max(len(images), 1)
            print(f"\n  {label}: {len(images)} image(s) in {elapsed:.1f}s "
                  f"({per_image:.0f} ms/image, beams={args.num_beams})\n")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        print(f"written: {args.out} ({len(records)} records)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
