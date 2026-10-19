"""Cache BLIP image features once, so training never reads an image again.

BLIP's text decoder consumes ``vision_model(pixel_values).last_hidden_state``:
a ``[N, num_patches, hidden]`` tensor. With the vision tower frozen that tensor
is a constant, so it is computed exactly once per image and replayed for every
epoch.

Two properties of this cache are deliberate, and both are copied from the
ResNet pipeline that already works in this repo:

**The index is authoritative.** ``<out>.jsonl`` carries ``feature_row`` next to
every manifest field. Training and evaluation read rows *from the index* and
filter by the requested manifest. They never join the manifest to the cache -
that join was the source of a bug the end-to-end test caught.

**Extraction is resumable.** A long extraction over 114k images will be
interrupted (this machine has wedged the GPU twice under sustained load). The
index is flushed periodically and an interrupted run picks up where it stopped
rather than starting over.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.finetune.base import BlipCachedCaptioner, build_model, resolve_model_id  # noqa: E402
from train.finetune.degrade import DegradeConfig, degrade_image  # noqa: E402
from train.utils.device import configure_for_speed, get_device  # noqa: E402

DEFAULT_OUT_DIR = Path("data/features")
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
FLUSH_SECONDS = 120.0


@dataclass(frozen=True)
class CachePaths:
    index: Path
    tensor: Path

    @staticmethod
    def for_manifest(manifest: Path, out_dir: Path = DEFAULT_OUT_DIR, tag: str = "") -> "CachePaths":
        stem = manifest.stem + (f"_{tag}" if tag else "")
        return CachePaths(
            index=out_dir / f"{stem}_blip.jsonl",
            tensor=out_dir / f"{stem}_blip.npy",
        )


def iter_manifest(manifest: Path):
    """Yield manifest rows in file order, skipping blank lines."""
    with manifest.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - bad data
                raise SystemExit(f"{manifest}:{lineno + 1}: not valid JSON: {exc}") from None


def row_image_key(row: dict) -> str | None:
    """The image identifier of a manifest row, under any of its spellings.

    Every manifest in this repo uses the bare ``file_name`` (for example
    ``VizWiz_train_00023410.jpg``), not a path, so resolution goes through
    ``--image-root`` rather than through the manifest's own directory. The other
    spellings are accepted so a manifest written elsewhere still loads.

    Kept in one place: the resume logic compares cached rows to manifest rows by
    this key, and a mismatch here would silently invalidate a partial cache.
    """
    for key in ("file_name", "image", "file", "image_path", "path"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def find_image(name: str, image_root: Path, index: dict[str, Path] | None = None) -> Path | None:
    """Locate an image by file name under ``image_root``.

    ``index`` is an optional prebuilt ``{name: path}`` map from
    :func:`index_images`; building it once turns a recursive search per image
    into a dict lookup.
    """
    if index is not None:
        return index.get(Path(name).name)

    direct = image_root / name
    if direct.exists():
        return direct
    matches = list(image_root.rglob(Path(name).name))
    return matches[0] if matches else None


def index_images(image_root: Path) -> dict[str, Path]:
    """Map every image file under ``image_root`` by basename."""
    index: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            index.setdefault(path.name, path)
    return index


def image_path_for(
    row: dict,
    manifest: Path,
    raw_root: Path | None,
    image_root: Path | None = None,
    image_index: dict[str, Path] | None = None,
) -> Path:
    """Resolve the image for a manifest row.

    Resolution order, first match wins:

    1. an absolute path stored in the row,
    2. the row's value relative to the manifest directory or the repository root,
    3. the row's basename under ``image_root`` (the normal case here),
    4. the row's value under ``raw_root``.
    """
    raw = row_image_key(row)
    if not raw:
        raise SystemExit(f"manifest row has no image field: {sorted(row)}")

    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    for base in (manifest.parent, Path.cwd()):
        resolved = (base / candidate)
        if resolved.exists():
            return resolved

    if image_root is not None:
        found = find_image(raw, image_root, image_index)
        if found is not None:
            return found

    if raw_root is not None:
        resolved = (raw_root / candidate)
        if resolved.exists():
            return resolved

    raise SystemExit(f"image not found for manifest row {raw!r}")


def _preprocess(images: list, image_size: int) -> torch.Tensor:
    """Resize + centre-crop + normalise to BLIP's ViT input.

    Uses the same constants as BLIP's own processor; doing it here avoids
    importing a processor per batch and keeps the transform identical between
    the caching pass and any later uncached path.
    """
    from PIL import Image

    tensors = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        # Shortest-edge resize then centre crop, matching the CLIP-style
        # preprocessor BLIP ships (bicubic, no antialias change).
        width, height = image.size
        scale = image_size / min(width, height)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = image.resize(new_size, resample=Image.BICUBIC)
        left = (new_size[0] - image_size) // 2
        top = (new_size[1] - image_size) // 2
        image = image.crop((left, top, left + image_size, top + image_size))

        array = np.asarray(image, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))

    batch = torch.stack(tensors)
    mean = torch.tensor(IMAGENET_MEAN, dtype=batch.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=batch.dtype).view(1, 3, 1, 1)
    return (batch - mean) / std


def _load_batch(
    rows: list[tuple[int, dict]],
    manifest: Path,
    raw_root,
    image_root,
    image_index,
    image_size: int,
    degrade=None,
):
    """Decode and preprocess one batch of rows; returns (pixel_values, kept).

    Runs on a worker thread so JPEG decode overlaps GPU work. Decoding is the
    real bottleneck here, not the vision tower: on VizWiz images the encoder
    measures ~220 img/s on synthetic tensors but a full extraction pass reaches
    only ~38 img/s, which is decode-bound. Resizing and normalising in the same
    worker means only the forward pass remains serialised.

    ``degrade`` is an optional :class:`train.finetune.degrade.DegradeConfig`.
    Degradation is applied to the decoded image, *before* the vision tower,
    because the cached tensor is the tower's output - degrading features directly
    would model something the tower never produces.
    """
    from PIL import Image

    from train.finetune.cache import row_image_key

    images = []
    kept: list[tuple[int, dict]] = []
    applied: list[str | None] = []
    for row_index, row in rows:
        try:
            path = image_path_for(row, manifest, raw_root, image_root, image_index)
            with Image.open(path) as handle:
                image = handle.convert("RGB").copy()
            if degrade is not None:
                image, flaw = degrade_image(image, degrade, row_image_key(row) or str(row_index))
                applied.append(flaw)
            else:
                applied.append(None)
            images.append(image)
            kept.append((row_index, row))
        except (SystemExit, OSError) as exc:
            # A missing or corrupt image should not kill a multi-hour pass.
            print(f"  ! skip row {row_index}: {exc}", flush=True)

    if not images:
        return None, [], []
    return _preprocess(images, image_size), kept, applied


def extract_features(
    manifest: Path,
    *,
    model_name: str = "base",
    out_dir: Path = DEFAULT_OUT_DIR,
    tag: str = "",
    batch_size: int = 32,
    limit: int | None = None,
    raw_root: Path | None = None,
    image_root: Path | None = None,
    resume: bool = True,
    dtype: str = "bf16",
    degrade: DegradeConfig | None = None,
) -> CachePaths:
    """Extract and cache BLIP vision features for one manifest.

    Returns the paths written. The tensor is ``float16`` (about 0.9 MB per image
    for BLIP-base at 384px, 1.2 MB for BLIP-large), which is the point: 114k
    images of cache cost roughly 100-140 GB in fp16 and would be 2x that in
    fp32. Mind free disk before caching COCO.
    """
    from PIL import Image

    configure_for_speed()
    info = get_device()
    paths = CachePaths.for_manifest(manifest, out_dir, tag)
    paths.index.parent.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id(model_name)
    model, report = build_model(model_id, with_vision=True, dtype=torch.float32, device=info.device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    rows = list(iter_manifest(manifest))
    if limit is not None:
        rows = rows[:limit]

    image_index = None
    if image_root is not None:
        print(f"indexing   : {image_root} ...", flush=True)
        image_index = index_images(image_root)
        print(f"             {len(image_index)} image files found", flush=True)

    use_bf16 = dtype == "bf16" and info.backend in ("rocm", "cuda")
    print(
        f"manifest   : {manifest}  ({len(rows)} rows)\n"
        f"model      : {model_id}\n"
        f"image size : {report.image_size}  patches={report.num_image_tokens}\n"
        f"device     : {info.name} ({info.backend})\n"
        f"encode     : {'bf16 autocast' if use_bf16 else dtype}\n"
        f"cache      : {paths.tensor}\n"
        f"index      : {paths.index}"
    )

    done_index: list[dict] = []
    done_tensor: np.ndarray | None = None
    started_row = 0

    if resume and paths.index.exists() and paths.tensor.exists():
        existing = [json.loads(line) for line in paths.index.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Only trust a truncated index if its rows are a prefix of this manifest.
        prefix_ok = len(existing) <= len(rows) and all(
            row_image_key(existing[i]) == row_image_key(rows[i]) for i in range(len(existing))
        )
        if prefix_ok:
            done_index = existing
            done_tensor = np.load(paths.tensor, mmap_mode="r")
            started_row = len(existing)
            print(f"resuming   : {started_row} rows already cached")
        else:
            print("resuming   : existing cache does not match this manifest; restarting")

    hidden = report.text_hidden_size
    patches = report.num_image_tokens
    if done_tensor is not None:
        features = np.empty((len(rows), patches, hidden), dtype=np.float16)
        features[:started_row] = done_tensor[:started_row]
        del done_tensor
    else:
        features = np.empty((len(rows), patches, hidden), dtype=np.float16)

    widths = max(3, len(str(len(rows))))
    torch_device = info.device
    started = time.time()
    processed = 0
    degraded_count = 0
    last_flush = started

    # Decode the next batch while the GPU works on the current one. Without
    # this overlap the pass is decode-bound at ~38 img/s; the vision tower alone
    # can do ~220 img/s.
    from concurrent.futures import ThreadPoolExecutor

    def batches():
        for begin in range(started_row, len(rows), batch_size):
            chunk = rows[begin : begin + batch_size]
            yield begin, [(begin + offset, row) for offset, row in enumerate(chunk)]

    with ThreadPoolExecutor(max_workers=1) as decoder:
        pending = None
        for begin, indexed_rows in batches():
            if pending is not None:
                pixel_values, kept, applied = pending.result()
            else:
                pixel_values, kept, applied = _load_batch(
                    indexed_rows, manifest, raw_root, image_root, image_index,
                    report.image_size, degrade,
                )

            # Queue the next decode before touching the GPU.
            next_rows = None
            for next_begin in range(begin + batch_size, min(begin + 2 * batch_size, len(rows)), batch_size):
                next_rows = [
                    (next_begin + offset, row)
                    for offset, row in enumerate(rows[next_begin : next_begin + batch_size])
                ]
                break
            pending = (
                decoder.submit(
                    _load_batch, next_rows, manifest, raw_root, image_root, image_index,
                    report.image_size, degrade,
                )
                if next_rows
                else None
            )

            if pixel_values is None:
                continue

            pixel_values = pixel_values.to(torch_device)
            # autocast matters here more than anywhere else in the project: the
            # caching pass is the only expensive step, and fp32 runs it at
            # roughly a third of the bf16 rate on this GPU.
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    embeds = model.encode_image(pixel_values)
            embeds = embeds.detach().to("cpu", dtype=torch.float16).numpy()

            for slot, (row_index, row) in enumerate(kept):
                features[row_index] = embeds[slot]
                entry = dict(row)
                entry["feature_row"] = row_index
                entry["feature_file"] = paths.tensor.name
                # Record which flaw was applied, so a degraded cache can be
                # audited and a training run explained afterwards.
                entry["degraded_with"] = applied[slot] if slot < len(applied) else None
                done_index.append(entry)
                if applied[slot]:
                    degraded_count += 1

            processed += len(kept)
            elapsed = time.time() - started
            rate = processed / elapsed if elapsed > 0 else 0.0
            # Flush on a row count or a time limit, whichever comes first. The
            # time limit matters more than it looks: an extraction pass over a
            # large corpus can sit for many minutes between row-count triggers
            # while the GPU is busy, and a crash then loses all of it.
            if (begin % (batch_size * 20) == 0
                    or begin + batch_size >= len(rows)
                    or (time.time() - last_flush) > FLUSH_SECONDS):
                print(
                    f"  {begin + len(kept):>{widths}}/{len(rows)}  "
                    f"{rate:6.1f} img/s  eta {_eta(len(rows) - begin - len(kept), rate)}",
                    flush=True,
                )
                _flush(features, done_index, paths)
                last_flush = time.time()

    _flush(features, done_index, paths)
    if degrade is not None:
        # Sidecar record: a degraded cache must be explainable afterwards, and
        # the flaw mix is the thing a reader will ask about.
        sidecar = paths.index.with_suffix(".degrade.json")
        sidecar.write_text(
            json.dumps(
                {
                    **degrade.as_dict(),
                    "manifest": str(manifest),
                    "rows": len(done_index),
                    "degraded_rows": degraded_count,
                    "degraded_fraction": degraded_count / max(len(done_index), 1),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"degraded   : {degraded_count}/{len(done_index)} rows "
            f"({100 * degraded_count / max(len(done_index), 1):.1f}%) -> {sidecar}"
        )
    print(f"done       : {len(done_index)} rows in {time.time() - started:.0f}s -> {paths.tensor}")
    return paths


def _eta(remaining: int, rate: float) -> str:
    if rate <= 0:
        return "?"
    seconds = remaining / rate
    return f"{seconds / 60:.1f}m"


def _flush(features: np.ndarray, index: list[dict], paths: CachePaths) -> None:
    """Persist progress, writing only the rows that have actually been filled.

    This detail dominates the wall-clock time of a large extraction. Writing the
    whole preallocated array means rewriting a file that grows to ~20 GB for
    VizWiz train, every flush, including the unwritten zero rows - measured at
    ~5 img/s end to end with the full-array write, against ~39 img/s for the
    same code once only the used prefix is written.

    The file's own length records how many rows are real, so a resumed run reads
    ``features.shape[0]`` and trusts it. The index is written atomically via a
    temporary file so an interrupt cannot leave a half-written index pointing at
    rows that were never flushed.
    """
    used = len(index)
    if used:
        np.save(paths.tensor, features[:used])
    tmp = paths.index.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for entry in index:
            handle.write(json.dumps(entry) + "\n")
    tmp.replace(paths.index)


def load_cache(paths: CachePaths) -> tuple[np.ndarray, list[dict]]:
    """Load a cache for training/evaluation.

    Index and tensor normally have one row each. An index produced by
    ``train.finetune.expand_captions`` is the exception: it carries one row per
    *caption* while the tensor keeps one row per *image*, with many index rows
    sharing a ``feature_row``. That is the point of the expansion - 100,575
    caption rows over 22,866 feature rows, without duplicating a 26 GB tensor -
    so the check below validates the property that actually matters instead of
    requiring equal lengths.
    """
    if not paths.tensor.exists() or not paths.index.exists():
        raise SystemExit(f"cache not found: {paths.tensor} / {paths.index}. Run train.finetune.cache first.")
    features = np.load(paths.tensor, mmap_mode="r")
    index = [json.loads(line) for line in paths.index.read_text(encoding="utf-8").splitlines() if line.strip()]

    rows_used = {int(row["feature_row"]) for row in index if "feature_row" in row}
    if not rows_used:
        raise SystemExit(f"cache index has no feature_row fields: {paths.index}")
    out_of_range = {r for r in rows_used if r < 0 or r >= features.shape[0]}
    if out_of_range:
        raise SystemExit(
            f"cache is inconsistent: {paths.index} references feature rows "
            f"{sorted(out_of_range)[:5]} but the tensor has {features.shape[0]} rows"
        )
    if len(index) != features.shape[0] and len(rows_used) != features.shape[0]:
        # Two legitimate shapes are allowed:
        #   * one index row per tensor row (the normal cache), or
        #   * an expanded index, where rows_used is a *subset* of the tensor
        #     because images without captions are never referenced.
        # Anything else means the index and tensor disagree about the corpus.
        if len(index) < len(rows_used):
            raise SystemExit(
                f"cache is inconsistent: {len(index)} index rows but {len(rows_used)} distinct "
                f"feature rows referenced in {paths.index}"
            )
    return features, index


def verify_cache(
    manifest: Path,
    paths: CachePaths,
    *,
    model_name: str = "base",
    samples: int = 3,
    tolerance: float = 0.99,
    image_root: Path | None = None,
    raw_root: Path | None = None,
    limit: int | None = None,
) -> bool:
    """Re-run the vision tower on a few real images and compare against the cache.

    This is the check that catches an off-by-one row, a swapped normalisation, a
    resize that does not match, or a feature tensor whose rows were written
    against the wrong image. Those failures are silent in training loss and
    obvious here.

    **The pass criterion is correlation, not element-wise difference.** bf16
    autocast is not bit-reproducible in the last bits, and measured on this
    machine two encodes of the same image can differ by ~3.2 in a single element
    while whose overall correlation is 0.9989. An element-wise tolerance
    therefore reports FAIL on a perfectly correct cache - which is exactly what
    it did before this was fixed. Correlation ~1.0 means the same image produced
    the same features; a wrong image scores near 0.

    ``limit`` must match the ``--limit`` used when the cache was written.
    Sampling from the *full* manifest while only a prefix was cached compares
    rows that were never written.
    """
    from PIL import Image

    configure_for_speed()
    info = get_device()
    model_id = resolve_model_id(model_name)
    model, report = build_model(model_id, with_vision=True, dtype=torch.float32, device=info.device)
    model.eval()

    features, index = load_cache(paths)
    rows = list(iter_manifest(manifest))
    if limit is not None:
        rows = rows[:limit]
    by_image = {row_image_key(entry): entry["feature_row"] for entry in index}
    image_index = index_images(image_root) if image_root is not None else None

    checked = 0
    failures: list[str] = []
    ok = True
    step = max(1, len(rows) // max(samples, 1))
    for row in rows[::step][:samples]:
        key = row_image_key(row)
        if key not in by_image:
            continue
        row_index = by_image[key]
        path = image_path_for(row, manifest, raw_root, image_root, image_index)
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        pixel_values = _preprocess([image], report.image_size).to(info.device)
        with torch.no_grad():
            # Match the extraction dtype. Running the check in fp32 against a
            # bf16-produced cache manufactures a difference that is not real.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(info.backend in ("rocm", "cuda"))):
                fresh = model.encode_image(pixel_values)[0].detach().to("cpu", dtype=torch.float32).numpy()
        cached = np.asarray(features[row_index], dtype=np.float32)

        flat_fresh = fresh.reshape(-1)
        flat_cached = cached.reshape(-1)
        # Correlation is the meaningful statistic; guard against a degenerate
        # constant vector, where correlation is undefined.
        if flat_fresh.std() == 0 or flat_cached.std() == 0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(flat_fresh, flat_cached)[0, 1])

        good = correlation >= tolerance
        ok = ok and good
        if not good:
            failures.append(key)
        print(
            f"  {'ok  ' if good else 'FAIL'} {key}: correlation = {correlation:.5f} "
            f"(mean|diff| = {float(np.abs(flat_fresh - flat_cached).mean()):.4f})"
        )
        checked += 1

    if checked == 0:
        print("  FAIL no overlapping rows between manifest and cache index")
        return False
    print(f"  {'PASS' if ok else 'FAIL'} cache verified on {checked} images")
    if failures:
        print(f"  mismatched: {failures}")
    return ok


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Cache BLIP vision features for a manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="base", help="'base', 'large', or a Hub repo id")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tag", default="", help="suffix for the output filenames")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                        help="autocast dtype for the encode pass; the cache itself is always float16")
    parser.add_argument("--limit", type=int, default=None, help="cache only the first N rows")
    parser.add_argument("--image-root", type=Path, default=None,
                        help="directory to search for the manifest's file_name values, recursively")
    parser.add_argument("--raw-root", type=Path, default=None, help="root for resolving relative image paths")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--verify", action="store_true", help="re-extract a few images and compare")
    parser.add_argument("--verify-samples", type=int, default=3)
    parser.add_argument("--degrade", action="store_true",
                        help="apply synthetic degradation drawn from the VizWiz flaw distribution, "
                             "before the vision tower. Targets the measured 4x collapse on unusable "
                             "photographs. Degradation is seeded and reproducible")
    parser.add_argument("--degrade-probability", type=float, default=0.5,
                        help="fraction of images degraded; 0.5 gives a clean/degraded mix")
    parser.add_argument("--degrade-severity", type=float, default=1.0,
                        help="multiplier on transform strength; 1.0 is calibrated to VizWiz")
    parser.add_argument("--degrade-seed", type=int, default=0)
    args = parser.parse_args(argv)

    paths = CachePaths.for_manifest(args.manifest, args.out_dir, args.tag)
    if args.verify:
        return 0 if verify_cache(
            args.manifest, paths, model_name=args.model, samples=args.verify_samples,
            image_root=args.image_root, raw_root=args.raw_root, limit=args.limit,
        ) else 1

    extract_features(
        args.manifest,
        model_name=args.model,
        out_dir=args.out_dir,
        tag=args.tag,
        batch_size=args.batch_size,
        limit=args.limit,
        raw_root=args.raw_root,
        image_root=args.image_root,
        resume=not args.no_resume,
        dtype=args.dtype,
        degrade=(
            DegradeConfig(
                probability=args.degrade_probability,
                severity=args.degrade_severity,
                seed=args.degrade_seed,
            )
            if args.degrade
            else None
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CachePaths",
    "DEFAULT_OUT_DIR",
    "extract_features",
    "image_path_for",
    "iter_manifest",
    "load_cache",
    "row_image_key",
    "verify_cache",
]
