"""Dataset acquisition.

Flickr30k is used for pretraining; VizWiz-Captions for fine-tuning and
evaluation. The original Flickr30k host (shannon.cs.illinois.edu) is frequently
unreachable, so a Hugging Face mirror is used instead.

    py -3.12 -m train.data.download --dataset flickr30k
    py -3.12 -m train.data.download --dataset vizwiz
    py -3.12 -m train.data.download --dataset all --verify
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Some hosts reject requests without a User-Agent.
_HEADERS = {"User-Agent": "describe-training/0.1 (+project dataset fetch)"}

# Allow direct execution as a script.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


FLICKR30K_EXPECTED_IMAGES = 31_783
# The official VizWiz image archives contain slightly more files than the
# caption-annotated count: the published splits are 23,431 / 7,750 / 8,000
# (39,181) but the zips hold 23,954 / 7,750 / 8,000 = 39,704 images. The extra
# ~500 train images are those whose captions were rejected or pre-canned, so
# they appear in the archives without usable captions.
VIZWIZ_EXPECTED_TRAIN = 23_954
VIZWIZ_EXPECTED_VAL = 7_750
VIZWIZ_EXPECTED_TEST = 8_000
# Images with usable captions, used by the caption manifests.
VIZWIZ_CAPTIONED_TRAIN = 23_431


@dataclass
class DatasetSpec:
    name: str
    hf_repo: str
    expected_counts: dict[str, int]
    license: str
    citation: str


SPECS: dict[str, DatasetSpec] = {
    "flickr30k": DatasetSpec(
        name="flickr30k",
        # lmms-lab-encoder/flickr30k: full-size images with the 5 captions per
        # image (columns: image, caption[], sentids[], img_id, filename).
        # Do NOT use iscc/iscc-flickr30k: it ships only 224px thumbnails and a
        # single caption per row, which is unusable for captioner training.
        hf_repo="lmms-lab-encoder/flickr30k",
        expected_counts={"train": FLICKR30K_EXPECTED_IMAGES},
        license="See original Flickr30k terms; images are Flickr user content.",
        citation=(
            "Young, P., Lai, A., Hodosh, M., & Hockenmaier, J. (2014). From image "
            "descriptions to visual denotations. TACL, 2, 67-78."
        ),
    ),
    "vizwiz": DatasetSpec(
        name="vizwiz-captions",
        # Official archives from vizwiz.org: these carry all 5 captions per
        # image plus the quality-flaw annotations. The Hugging Face mirror
        # (lmms-lab/VizWiz-Caps) is NOT sufficient: it has no train split and
        # only one caption per image.
        hf_repo="vizwiz.org (official archives)",
        expected_counts={
            "train": VIZWIZ_EXPECTED_TRAIN,
            "val": VIZWIZ_EXPECTED_VAL,
            "test": VIZWIZ_EXPECTED_TEST,
        },
        license="CC BY 4.0",
        citation=(
            "Gurari, D., Zhao, Y., Zhang, M., & Bhattacharya, N. (2020). Captioning "
            "images taken by people who are blind. ECCV."
        ),
    ),
}

VIZWIZ_IMAGE_URLS = {
    "train": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/train.zip",
    "val": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/val.zip",
    "test": "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip",
}
VIZWIZ_CAPTION_ANNOTATIONS = (
    "https://vizwiz.cs.colorado.edu/VizWiz_final/caption/annotations.zip"
)
VIZWIZ_QUALITY_ANNOTATIONS = (
    "https://vizwiz.cs.colorado.edu/VizWiz_final/image_quality/annotations.zip"
)
VIZWIZ_QUALITY_EVAL_CSV = (
    "https://vizwiz.cs.colorado.edu/VizWiz_all_answers/"
    "VizWiz_quality_issues_train_val_test.csv"
)

# COCO 2014. The official host supports HTTP range requests, so the existing
# parallel downloader applies.
COCO_URLS = {
    "train2014.zip": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014.zip": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations_trainval2014.zip": (
        "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    ),
}
# Karpathy split sizes for COCO 2014, used to verify the preparation step.
COCO_KARPATHY_TRAIN = 113_287
COCO_KARPATHY_VAL = 5_000
COCO_KARPATHY_TEST = 5_000


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num) < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PB"


def _remote_size(url: str) -> int:
    """Return the Content-Length of a URL, or 0 if unavailable."""
    import urllib.request

    request = urllib.request.Request(url, method="HEAD", headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers.get("Content-Length") or 0)


def _supports_ranges(url: str) -> bool:
    """True when the server honours HTTP Range requests."""
    import urllib.request

    request = urllib.request.Request(
        url, headers={**_HEADERS, "Range": "bytes=0-1023"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response.read(64)
            return response.status == 206
    except Exception:
        return False


def download_file_parallel(
    url: str,
    dest: Path,
    workers: int = 8,
    chunk_size: int = 8 << 20,
    force: bool = False,
) -> Path:
    """Download with parallel HTTP range requests and per-chunk resume.

    The VizWiz host serves a single connection at ~0.15 MB/s but reaches
    ~7 MB/s across eight range requests, which turns a 28-hour download into
    under an hour. Each chunk is written to its own ``.part`` file, so an
    interrupted run resumes instead of restarting.
    """
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    dest.parent.mkdir(parents=True, exist_ok=True)

    total = _remote_size(url)

    if dest.exists() and not force:
        # A killed download leaves a truncated file behind. Trusting its
        # existence produced a "BadZipFile" on extraction, so verify the size.
        actual = dest.stat().st_size
        if total and actual != total:
            print(
                f"  incomplete {dest.name}: {human_bytes(actual)} of "
                f"{human_bytes(total)} - re-downloading"
            )
            dest.unlink()
        else:
            print(
                f"  exists, skipping: {dest.name} ({human_bytes(actual)})"
            )
            return dest

    if not total:
        print("  server reports no Content-Length; falling back to single stream")
        return download_file(url, dest, force=force)

    if not _supports_ranges(url):
        print("  server does not support range requests; single stream")
        return download_file(url, dest, force=force)

    parts_dir = dest.with_suffix(dest.suffix + ".parts")
    # Guard resume against a changed remote file: if the recorded total size
    # differs, previously fetched chunks belong to a different file.
    meta = parts_dir / "total.txt"
    if parts_dir.exists():
        if not meta.exists() or meta.read_text().strip() != str(total):
            print("  discarding stale partial chunks (remote size changed)")
            for part in parts_dir.glob("*.part"):
                part.unlink()
    parts_dir.mkdir(parents=True, exist_ok=True)
    meta.write_text(str(total))

    ranges = [
        (index, start, min(start + chunk_size, total) - 1)
        for index, start in enumerate(range(0, total, chunk_size))
    ]
    print(
        f"  downloading: {dest.name} ({human_bytes(total)}) "
        f"in {len(ranges)} chunks x {workers} workers"
    )

    def fetch_chunk(spec: tuple[int, int, int]) -> tuple[int, int]:
        index, start, end = spec
        part = parts_dir / f"{index:05d}.part"
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            return index, expected

        for attempt in range(4):
            try:
                request = urllib.request.Request(
                    url, headers={**_HEADERS, "Range": f"bytes={start}-{end}"}
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    payload = response.read()
                if len(payload) != expected:
                    raise IOError(
                        f"chunk {index}: got {len(payload)} of {expected} bytes"
                    )
                part.write_bytes(payload)
                return index, expected
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(1.5 * (attempt + 1))
        return index, 0

    done = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _, size in executor.map(fetch_chunk, ranges):
            done += size
            elapsed = time.time() - started
            rate = done / elapsed if elapsed else 0
            remaining = (total - done) / rate if rate else 0
            print(
                f"\r    {100 * done / total:5.1f}%  {human_bytes(done)}"
                f" / {human_bytes(total)}  {rate / 1e6:5.2f} MB/s"
                f"  eta {remaining / 60:5.1f} min",
                end="",
                flush=True,
            )
    print()

    # Stitch the chunks together, then clean up.
    with dest.open("wb") as out:
        for index, _, _ in ranges:
            out.write((parts_dir / f"{index:05d}.part").read_bytes())
    actual = dest.stat().st_size
    if actual != total:
        raise RuntimeError(
            f"{dest.name}: assembled {actual} bytes but expected {total}"
        )

    # Remove the chunk directory. shutil.rmtree is used rather than rmdir
    # because the directory also holds the total.txt resume marker.
    import shutil

    shutil.rmtree(parts_dir, ignore_errors=True)

    elapsed = time.time() - started
    print(f"  complete: {dest.name} ({human_bytes(actual)} in {elapsed / 60:.1f} min)")
    return dest


def download_file(url: str, dest: Path, force: bool = False) -> Path:
    """Stream a URL to disk, skipping if already present."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"  exists, skipping: {dest.name} ({human_bytes(dest.stat().st_size)})")
        return dest

    print(f"  downloading: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            with dest.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100 * written / total
                        print(
                            f"\r    {pct:5.1f}%  {human_bytes(written)}"
                            f" / {human_bytes(total)}",
                            end="",
                            flush=True,
                        )
        print()
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"failed to download {url}: {exc}") from exc

    return dest


def extract_zip(archive: Path, dest_dir: Path) -> None:
    """Extract a zip archive, refusing to continue on a corrupt file."""
    import zipfile

    dest_dir.mkdir(parents=True, exist_ok=True)

    if not zipfile.is_zipfile(archive):
        size = human_bytes(archive.stat().st_size) if archive.exists() else "missing"
        raise RuntimeError(
            f"{archive} is not a valid zip ({size}). "
            "It is probably a truncated download; delete it and re-run."
        )

    print(f"  extracting: {archive.name} -> {dest_dir}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)


def download_flickr30k(
    data_root: Path, force: bool = False, limit: int | None = None
) -> None:
    """Fetch Flickr30k from Hugging Face and write images plus a caption file.

    Source: ``lmms-lab-encoder/flickr30k`` — full-size images with all five
    captions per image. Its single split is named ``test`` in the mirror, but it
    is the complete 31,783-image corpus, so it is recorded as ``train`` here
    because this project uses it for pretraining.
    """
    print("\nFlickr30k (pretraining corpus)")
    dest = data_root / "flickr30k"
    images_dir = dest / "images"
    captions_path = dest / "captions.jsonl"
    images_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print(
            "  `datasets` is not installed.\n"
            "  Install it with:  py -3.12 -m pip install datasets"
        )
        return

    repo = SPECS["flickr30k"].hf_repo
    print(f"  loading {repo} (streaming, so partial runs are resumable)")

    # Stream rather than download-then-decode: the parquet shards are ~4.4 GB
    # and streaming lets us write images as they arrive.
    dataset = load_dataset(repo, split="test", streaming=True)

    import json

    written = 0
    empty_captions = 0
    with captions_path.open("w", encoding="utf-8") as handle:
        for row in dataset:
            if limit is not None and written >= limit:
                break

            image = row.get("image")
            if image is None:
                continue

            filename = f"flickr30k_{written:06d}.jpg"
            target = images_dir / filename
            if not target.exists() or force:
                image.convert("RGB").save(target, quality=92)

            captions = _extract_captions(row)
            if not captions:
                empty_captions += 1

            handle.write(
                json.dumps(
                    {
                        "image_id": written,
                        "file_name": filename,
                        "split": "train",
                        "captions": captions,
                        "text_detected": None,
                    }
                )
                + "\n"
            )
            written += 1
            if written % 2000 == 0:
                print(f"    {written:,} images written", flush=True)

    print(f"  images  : {images_dir} ({written:,} files)")
    print(f"  captions: {captions_path}")
    if empty_captions:
        print(f"  WARNING: {empty_captions:,} rows had no captions")


def _extract_captions(row: dict) -> list[str]:
    """Pull a caption list out of whichever column the mirror provides."""
    for key in ("caption", "captions", "sentences", "raw"):
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict) and "raw" in value:
            return [str(v) for v in value["raw"]]
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict) and "raw" in item:
                    out.append(str(item["raw"]))
            if out:
                return out
    return []


def download_coco(
    data_root: Path, force: bool = False, workers: int = 8
) -> None:
    """Fetch COCO 2014 images and annotations for the Karpathy split.

    COCO is ~4x Flickr30k's image count and is the standard captioning corpus,
    which matters because the model's failure is mode collapse from
    undertraining rather than an architectural defect.

    ~18.8 GB total (train 12.6 GB, val 6.2 GB, annotations 241 MB), downloaded
    with parallel range requests.
    """
    print("\nCOCO 2014 (main training corpus)")
    dest = data_root / "coco"
    archives = dest / "archives"
    archives.mkdir(parents=True, exist_ok=True)

    for name, url in COCO_URLS.items():
        archive = download_file_parallel(
            url, archives / name, workers=workers, force=force
        )
        if name.startswith("annotations"):
            extract_zip(archive, dest / "annotations")
        else:
            # train2014.zip contains a top-level train2014/ directory, so
            # extracting into dest/images gives dest/images/train2014/...
            extract_zip(archive, dest / "images")

    print(f"  COCO ready under {dest}")


def download_vizwiz(
    data_root: Path, force: bool = False, workers: int = 8
) -> None:
    """Fetch VizWiz-Captions images plus caption and quality annotations.

    Uses parallel range requests: this host serves one connection at roughly
    0.15 MB/s but reaches ~7 MB/s across eight, which is the difference between
    a 28-hour and a sub-hour download for the ~15 GB of image archives.
    """
    print("\nVizWiz-Captions (fine-tuning corpus) and VizWiz-QualityIssues")
    dest = data_root / "vizwiz"
    archives = dest / "archives"
    archives.mkdir(parents=True, exist_ok=True)

    for split, url in VIZWIZ_IMAGE_URLS.items():
        archive = download_file_parallel(
            url, archives / f"{split}.zip", workers=workers, force=force
        )
        extract_zip(archive, dest / "images")

    caption_archive = download_file(
        VIZWIZ_CAPTION_ANNOTATIONS, archives / "caption_annotations.zip", force=force
    )
    extract_zip(caption_archive, dest / "annotations" / "caption")

    quality_archive = download_file(
        VIZWIZ_QUALITY_ANNOTATIONS, archives / "quality_annotations.zip", force=force
    )
    extract_zip(quality_archive, dest / "annotations" / "quality")

    print("\n  Annotation files expected under:")
    print(f"    {dest / 'annotations' / 'caption'}")
    print(f"    {dest / 'annotations' / 'quality'}")
    print("  Next: py -3.12 -m train.data.prepare --dataset all")


def verify(data_root: Path) -> int:
    """Report what is present against the expected counts."""
    print("\nVerification")
    problems = 0

    flickr_images = data_root / "flickr30k" / "images"
    if flickr_images.exists():
        count = sum(1 for _ in flickr_images.glob("*.jpg"))
        ok = count == FLICKR30K_EXPECTED_IMAGES
        print(
            f"  flickr30k images: {count:,} (expected {FLICKR30K_EXPECTED_IMAGES:,})"
            f" {'OK' if ok else '<-- MISMATCH'}"
        )
        problems += 0 if ok else 1
    else:
        print(f"  flickr30k images: missing at {flickr_images}")

    vizwiz_images = data_root / "vizwiz" / "images"
    if vizwiz_images.exists():
        total = sum(1 for _ in vizwiz_images.rglob("*.jpg"))
        expected = VIZWIZ_EXPECTED_TRAIN + VIZWIZ_EXPECTED_VAL + VIZWIZ_EXPECTED_TEST
        ok = total == expected
        print(
            f"  vizwiz images   : {total:,} (expected {expected:,})"
            f" {'OK' if ok else '<-- MISMATCH'}"
        )
        problems += 0 if ok else 1
    else:
        print(f"  vizwiz images   : missing at {vizwiz_images}")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description="Download project datasets")
    parser.add_argument(
        "--dataset", choices=["flickr30k", "vizwiz", "all"], default="all"
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("data/raw")
    )
    parser.add_argument("--force", action="store_true", help="re-download")
    parser.add_argument("--verify", action="store_true", help="verify then exit")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel connections for VizWiz range downloads (default 8)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Flickr30k only: stop after N images (for a fast trial run)",
    )
    args = parser.parse_args()

    data_root: Path = args.data_root
    data_root.mkdir(parents=True, exist_ok=True)

    if args.verify:
        return 1 if verify(data_root) else 0

    print(f"Data root: {data_root.resolve()}")
    if args.dataset in ("flickr30k", "all"):
        download_flickr30k(data_root, force=args.force, limit=args.limit)
    if args.dataset in ("vizwiz", "all"):
        download_vizwiz(data_root, force=args.force, workers=args.workers)

    verify(data_root)
    print("\nLicensing: VizWiz-Captions is CC BY 4.0 - attribute it in the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
