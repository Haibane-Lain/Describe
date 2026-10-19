"""Extract and cache frozen CNN features.

The encoder runs once; every training epoch afterwards reads a cached tensor.
This is what keeps decoder training cheap, and it means the GPU is only needed
for this stage (though it helps a lot here).

    # Verify the whole extract -> cache -> load path without any download
    py -3.12 -m train.features.extract --synthetic

    # Real extraction
    py -3.12 -m train.features.extract --manifest data/manifests/flickr30k.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.utils.device import configure_for_speed, get_device  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES = {
    "resnet50": ("resnet50", "ResNet50_Weights.IMAGENET1K_V2", 2048),
    "resnet101": ("resnet101", "ResNet101_Weights.IMAGENET1K_V2", 2048),
}


def build_encoder(backbone: str) -> tuple[nn.Module, int]:
    """Frozen CNN with the classification head removed."""
    import torchvision

    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {list(BACKBONES)}")

    factory_name, weights_name, feature_dim = BACKBONES[backbone]
    factory = getattr(torchvision.models, factory_name)
    weights = getattr(torchvision.models, weights_name.split(".")[0])
    weights = getattr(weights, weights_name.split(".")[1])

    model = factory(weights=weights)
    model.fc = nn.Identity()  # [N, feature_dim]
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, feature_dim


class RestoreSpatial(nn.Module):
    """Undo the flatten in ``ResNet._forward_impl``.

    torchvision's ResNet runs ``avgpool -> torch.flatten(x, 1) -> fc``. With
    ``avgpool`` replaced by identity the flatten turns ``[B, C, H, W]`` into
    ``[B, C*H*W]``, so this module reshapes it back to ``[B, C, H, W]``.
    """

    def __init__(self, channels: int, spatial: int) -> None:
        super().__init__()
        self.channels = channels
        self.spatial = spatial

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], self.channels, self.spatial, self.spatial)


def build_spatial_encoder(
    backbone: str, image_size: int = 224
) -> tuple[nn.Module, int]:
    """Frozen CNN that returns the pre-pooling spatial feature map.

    The pooled ``[N, 2048]`` vector forces the decoder to compress an entire
    image into one vector, which loses the spatial detail needed to name
    objects and their positions. Attention needs the ``[N, 2048, 7, 7]`` map
    instead, so the global average pool is bypassed.
    """
    import torchvision

    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {list(BACKBONES)}")

    factory_name, weights_name, feature_dim = BACKBONES[backbone]
    factory = getattr(torchvision.models, factory_name)
    weights_enum = getattr(torchvision.models, weights_name.split(".")[0])
    weights = getattr(weights_enum, weights_name.split(".")[1])

    model = factory(weights=weights)
    # ResNet downsamples by 32, so a 224px input gives a 7x7 map.
    spatial = image_size // 32
    model.avgpool = nn.Identity()
    model.fc = RestoreSpatial(feature_dim, spatial)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, feature_dim


def preprocess(image: Image.Image, image_size: int) -> torch.Tensor:
    """Resize, centre-crop, normalize to an ImageNet-style tensor."""
    from torchvision import transforms

    pipeline = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return pipeline(image.convert("RGB"))


def extract_spatial_from_manifest(
    manifest: Path,
    image_root: Path,
    backbone: str = "resnet101",
    image_size: int = 224,
    batch_size: int = 32,
    out_dir: Path = Path("data/features"),
    limit: int | None = None,
    prefer_gpu: bool = True,
) -> Path:
    """Cache pre-pooling spatial feature maps for an attention decoder.

    Writes ``[N, C, H, W]`` in float16. At 224x224 with ResNet-101 that is
    ``[N, 2048, 7, 7]`` — about 200 KB per image, so 31,783 images is ~6.4 GB.
    Stored as a memory-mapped ``.npy`` so training can read rows lazily rather
    than holding it all in RAM.
    """
    records = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    if limit:
        records = records[:limit]
    if not records:
        raise SystemExit(f"manifest {manifest} contains no records")

    info = get_device(prefer_gpu=prefer_gpu)
    configure_for_speed()
    print(f"  device   : {info.name} ({info.backend})")
    print(f"  backbone : {backbone} (spatial maps, pre-pooling)")
    print(f"  records  : {len(records):,}")

    model, feature_dim = build_spatial_encoder(backbone, image_size)
    model = model.to(info.device)

    image_index = {}
    for path in image_root.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            image_index[path.name] = path

    # Determine the spatial size with one probe image.
    probe_name = records[0]["file_name"]
    probe_path = image_index.get(probe_name)
    if probe_path is None:
        raise SystemExit(f"probe image {probe_name} not found under {image_root}")
    with torch.no_grad():
        probe = preprocess(Image.open(probe_path), image_size)
        probe_map = model(probe.unsqueeze(0).to(info.device))
    _, channels, height, width = probe_map.shape
    print(f"  map size : [{channels}, {height}, {width}]")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest.stem
    array_path = out_dir / f"{stem}_{backbone}_spatial.npy"
    # Preallocate on disk via memmap so RAM stays bounded.
    store = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.float16,
        shape=(len(records), channels, height, width),
    )

    kept: list[dict] = []
    missing = 0
    batch: list[torch.Tensor] = []
    started = time.time()

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        stacked = torch.stack(batch).to(info.device)
        with torch.no_grad():
            out = model(stacked).cpu().numpy().astype(np.float16)
        for offset, values in enumerate(out):
            store[len(kept) - len(out) + offset] = values
        batch = []

    with torch.no_grad():
        for i, record in enumerate(records):
            path = image_index.get(record["file_name"])
            if path is None:
                missing += 1
                continue
            try:
                tensor = preprocess(Image.open(path), image_size)
            except Exception as exc:
                print(f"  WARNING: could not read {path.name}: {exc}")
                missing += 1
                continue
            batch.append(tensor)
            kept.append(record)
            if len(batch) == batch_size or i == len(records) - 1:
                flush()
                done = len(kept)
                if done % (batch_size * 20) < batch_size:
                    elapsed = time.time() - started
                    rate = done / elapsed if elapsed else 0
                    print(
                        f"\r    {done:,}/{len(records):,}  {rate:6.1f} img/s  "
                        f"eta {(len(records) - done) / rate / 60 if rate else 0:5.1f} min",
                        end="", flush=True,
                    )

    print()
    store.flush()
    del store

    if missing:
        print(f"  WARNING: {missing:,} images missing or unreadable")

    # Trim the array to the rows actually kept.
    full = np.load(array_path, mmap_mode="r")
    if len(kept) != full.shape[0]:
        trimmed = np.lib.format.open_memmap(
            array_path.with_suffix(".trim.npy"), mode="w+", dtype=np.float16,
            shape=(len(kept), channels, height, width),
        )
        for row in range(len(kept)):
            trimmed[row] = full[row]
        trimmed.flush()
        del trimmed, full
        array_path.with_suffix(".trim.npy").replace(array_path)

    index_path = out_dir / f"{stem}_{backbone}_spatial.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row, record in enumerate(kept):
            handle.write(json.dumps({**record, "feature_row": row}) + "\n")

    size_gb = array_path.stat().st_size / 1e9
    print(f"  maps     : {len(kept):,} x [{channels}, {height}, {width}] ({size_gb:.2f} GB)")
    print(f"  saved    : {array_path}")
    print(f"  index    : {index_path}")
    return index_path


def extract_from_manifest(
    manifest: Path,
    image_root: Path,
    backbone: str = "resnet101",
    image_size: int = 224,
    batch_size: int = 64,
    dtype: str = "float16",
    out_dir: Path = Path("data/features"),
    limit: int | None = None,
    prefer_gpu: bool = True,
) -> Path:
    """Run the encoder over every image in a manifest and cache the result."""
    records = [json.loads(line) for line in manifest.open(encoding="utf-8") if line.strip()]
    if limit:
        records = records[:limit]
    if not records:
        raise SystemExit(f"manifest {manifest} contains no records")

    info = get_device(prefer_gpu=prefer_gpu)
    configure_for_speed()
    print(f"  device   : {info.name} ({info.backend})")
    print(f"  backbone : {backbone}")
    print(f"  records  : {len(records):,}")

    model, feature_dim = build_encoder(backbone)
    model = model.to(info.device)

    image_index = {}
    for path in image_root.rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            image_index[path.name] = path

    n = len(records)
    store_dtype = np.float16 if dtype == "float16" else np.float32
    features = np.zeros((n, feature_dim), dtype=store_dtype)
    kept: list[dict] = []
    missing = 0
    batch: list[torch.Tensor] = []
    batch_rows: list[int] = []

    started = time.time()
    with torch.no_grad():
        for i, record in enumerate(records):
            path = image_index.get(record["file_name"])
            if path is None:
                missing += 1
                continue
            try:
                tensor = preprocess(Image.open(path), image_size)
            except Exception as exc:
                print(f"  WARNING: could not read {path.name}: {exc}")
                missing += 1
                continue

            batch.append(tensor)
            batch_rows.append(len(kept))
            kept.append(record)

            if len(batch) == batch_size or i == n - 1:
                stacked = torch.stack(batch).to(info.device)
                out = model(stacked).cpu().numpy().astype(store_dtype)
                for row, values in zip(batch_rows, out):
                    features[row] = values
                batch, batch_rows = [], []

                done = len(kept)
                if done % (batch_size * 10) < batch_size:
                    elapsed = time.time() - started
                    rate = done / elapsed if elapsed else 0
                    remaining = (n - done) / rate if rate else 0
                    print(
                        f"\r    {done:,}/{n:,}  {rate:6.1f} img/s  "
                        f"eta {remaining / 60:5.1f} min",
                        end="",
                        flush=True,
                    )

    print()
    if missing:
        print(f"  WARNING: {missing:,} images missing or unreadable")

    features = features[: len(kept)]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = manifest.stem
    np.save(out_dir / f"{stem}_{backbone}.npy", features)

    index_path = out_dir / f"{stem}_{backbone}.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for row, record in enumerate(kept):
            handle.write(
                json.dumps({**record, "feature_row": row}) + "\n"
            )

    size_mb = (out_dir / f"{stem}_{backbone}.npy").stat().st_size / 1e6
    print(f"  features : {features.shape} {features.dtype} ({size_mb:,.1f} MB)")
    print(f"  saved    : {out_dir / f'{stem}_{backbone}.npy'}")
    print(f"  index    : {index_path}")
    return index_path


def run_synthetic(
    out_dir: Path = Path("data/features"),
    count: int = 48,
    image_size: int = 224,
    backbone: str = "resnet101",
) -> int:
    """Exercise extract -> save -> load with generated images.

    Proves the GPU convolution path, the cache format, and the reload contract
    without downloading anything.
    """
    import tempfile

    from PIL import ImageDraw

    info = get_device()
    configure_for_speed()
    print("Synthetic feature-extraction check")
    print(f"  device  : {info.name} ({info.backend})")
    print(f"  backbone: {backbone}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images_dir = tmp_path / "images"
        images_dir.mkdir(parents=True)

        for i in range(count):
            rng = np.random.default_rng(i)
            array = (rng.random((image_size, image_size, 3)) * 255).astype("uint8")
            image = Image.fromarray(array, mode="RGB")
            ImageDraw.Draw(image).rectangle([10, 10, 60, 60], fill=(255, 0, 0))
            image.save(images_dir / f"synthetic_{i:04d}.jpg", quality=92)

        manifest = tmp_path / "synthetic.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for i in range(count):
                handle.write(
                    json.dumps(
                        {
                            "image_id": i,
                            "file_name": f"synthetic_{i:04d}.jpg",
                            "dataset": "synthetic",
                            "split": "train",
                            "captions": ["a synthetic test image"],
                            "text_detected": None,
                            "quality": None,
                        }
                    )
                    + "\n"
                )

        cache_dir = tmp_path / "cache"
        started = time.time()
        index_path = extract_from_manifest(
            manifest=manifest,
            image_root=images_dir,
            backbone=backbone,
            image_size=image_size,
            batch_size=16,
            out_dir=cache_dir,
        )
        elapsed = time.time() - started

        features = np.load(cache_dir / "synthetic_resnet101.npy")
        rows = [json.loads(line) for line in index_path.open(encoding="utf-8")]

        checks: list[tuple[str, bool, str]] = []
        checks.append(("feature array shape", features.shape == (count, 2048),
                       str(features.shape)))
        checks.append(("dtype is float16", features.dtype == np.float16,
                       str(features.dtype)))
        checks.append(("index row count matches", len(rows) == count,
                       f"{len(rows)} rows"))
        checks.append(("no NaN/Inf in features",
                       bool(np.isfinite(features).all()), ""))
        checks.append(("features are not all identical",
                       float(features.std()) > 1e-6, f"std={features.std():.4f}"))
        checks.append(("row ids align with index",
                       rows[0]["feature_row"] == 0
                       and rows[-1]["feature_row"] == count - 1, ""))
        checks.append(("throughput is plausible", elapsed < 120,
                       f"{elapsed:.1f}s for {count} images"))

        print("\n  Results")
        for name, ok, detail in checks:
            print(f"    [{'PASS' if ok else 'FAIL'}] {name}"
                  + (f" -- {detail}" if detail else ""))

        failed = [c for c in checks if not c[1]]
        print(f"\n  {len(checks) - len(failed)}/{len(checks)} checks passed")
        return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract and cache CNN features")
    parser.add_argument("--manifest", type=Path, help="manifest JSONL to process")
    parser.add_argument("--image-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--backbone", choices=list(BACKBONES), default="resnet101")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--out-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--limit", type=int, help="only process the first N records")
    parser.add_argument("--cpu", action="store_true", help="force CPU")
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="cache pre-pooling [C, H, W] maps for an attention decoder "
             "instead of pooled [C] vectors",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="self-test with generated images; needs no downloaded data",
    )
    args = parser.parse_args()

    if args.synthetic:
        return run_synthetic(
            out_dir=args.out_dir, image_size=args.image_size, backbone=args.backbone
        )

    if not args.manifest:
        parser.error("--manifest is required unless --synthetic is given")

    if args.spatial:
        extract_spatial_from_manifest(
            manifest=args.manifest,
            image_root=args.image_root,
            backbone=args.backbone,
            image_size=args.image_size,
            batch_size=min(args.batch_size, 32),
            out_dir=args.out_dir,
            limit=args.limit,
            prefer_gpu=not args.cpu,
        )
        return 0

    extract_from_manifest(
        manifest=args.manifest,
        image_root=args.image_root,
        backbone=args.backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        dtype=args.dtype,
        out_dir=args.out_dir,
        limit=args.limit,
        prefer_gpu=not args.cpu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
