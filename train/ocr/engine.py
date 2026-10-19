"""The OCR component: one image in, text and regions out.

This is the interface ``train/docs/ocr-plan.md`` §7 asks for -- an image path or
bytes in, plain text out, with per-region boxes and confidences alongside -- and
the one the app plan expects from ``read_text()``. It is deliberately *not*
wired into the captioner: the two components run independently and their outputs
are shown to the user separately, so nothing under ``train/finetune``,
``train/train`` or ``train/models`` imports this package.

    from train.ocr import read_text
    result = read_text("photo.jpg")
    result.text              # "MEETING ROOM 204"
    result.confidence        # mean region confidence, 0.0 when nothing was found
    result.low_confidence    # True when the app should say "partially readable"
    result.regions[0].box    # four (x, y) points in the ORIGINAL image

Three behaviours are deliberate and worth knowing before reading the code:

* **The engine is EasyOCR**, because the project already chose it (see
  ``train/requirements-ocr.txt``). :class:`OcrEngine` talks to it through a
  :class:`OcrBackend` protocol so a second engine can be measured later without
  touching the scoring, which is the only part that must not drift.
* **Boxes come back in original pixel coordinates**, not the coordinates of the
  preprocessed image. ``preprocess.prepare`` returns the transform and the boxes
  are mapped back, so region weighting and the app's highlighting are correct
  even when the image was upscaled, deskewed or perspective-corrected.
* **``low_confidence`` uses the mean region confidence, not the minimum.** A
  single smudged word should not flag an otherwise readable label; that is the
  difference between the mockups' "read text" and "partially readable" states.

**Not for safety-critical decisions.** Medication names, dosages and expiry dates
are inside this component's remit, and a misread dose can cause harm. The
proposal excludes safety-critical use and this module keeps that boundary: do not
present OCR output as verified text.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.ocr.preprocess import (  # noqa: E402
    PreprocessConfig,
    PreprocessResult,
    prepare,
)

# Below this mean confidence the app should hedge ("partially readable") rather
# than read the text out flatly. Recorded in every run config so a measurement
# can be compared with the live behaviour.
LOW_CONFIDENCE = 0.35

MODEL_DIR = Path("checkpoints/easyocr")

# EasyOCR detector/recogniser defaults, spelled out because two of them are
# load-bearing for this project's dominant subtypes (dense sign and label text):
# canvas_size is the detector's working resolution and mag_ratio adds an extra
# magnification pass, both of which cost time and buy small-text recall.
DEFAULT_CANVAS_SIZE = 2560
DEFAULT_MAG_RATIO = 1.5
DEFAULT_TEXT_THRESHOLD = 0.7
DEFAULT_LINK_THRESHOLD = 0.4
DEFAULT_LOW_TEXT = 0.4
DEFAULT_BATCH_SIZE = 8


@dataclass(frozen=True)
class OcrRegion:
    """One detected text box."""

    text: str
    confidence: float
    box: tuple[tuple[int, int], ...]  # four points, ordered tl, tr, br, bl

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "box": [list(point) for point in self.box],
        }


@dataclass(frozen=True)
class OcrResult:
    """Everything OCR found in one image."""

    text: str
    regions: tuple[OcrRegion, ...]
    confidence: float
    min_confidence: float
    n_regions: int
    low_confidence: bool
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "regions": [region.as_dict() for region in self.regions],
            "confidence": round(self.confidence, 4),
            "min_confidence": round(self.min_confidence, 4),
            "n_regions": self.n_regions,
            "low_confidence": self.low_confidence,
            "meta": self.meta,
        }


class OcrBackend(Protocol):
    """The narrow seam the measurement can swap an engine behind."""

    name: str

    def read(self, image: np.ndarray) -> list[OcrRegion]: ...

    def describe(self) -> dict[str, Any]: ...


class EasyOcrBackend:
    """EasyOCR, configured for this machine.

    MIOpen is disabled through ``train.utils.device.configure_for_speed`` before
    the reader is built: the ROCm 7.2 Windows SDK is missing the rocrand headers
    its dropout kernel needs on gfx1201, and the recogniser is an LSTM. See
    ``train/docs/environment.md`` for the failure this avoids.
    """

    name = "easyocr"

    def __init__(
        self,
        *,
        lang: Sequence[str] = ("en",),
        gpu: bool | None = None,
        model_dir: Path = MODEL_DIR,
        canvas_size: int = DEFAULT_CANVAS_SIZE,
        mag_ratio: float = DEFAULT_MAG_RATIO,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        link_threshold: float = DEFAULT_LINK_THRESHOLD,
        low_text: float = DEFAULT_LOW_TEXT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        download_enabled: bool = True,
        verbose: bool = False,
    ) -> None:
        import easyocr
        import torch

        from train.utils.device import configure_for_speed

        self._settings = configure_for_speed()
        self.lang = list(lang)
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.batch_size = batch_size
        self.model_dir = Path(model_dir)

        resolved_gpu = torch.cuda.is_available() if gpu is None else bool(gpu)
        if resolved_gpu and not torch.cuda.is_available():
            print("  WARNING: gpu requested but torch.cuda is unavailable; using CPU")
            resolved_gpu = False
        if not resolved_gpu and gpu is None:
            print("  WARNING: no GPU visible to torch; OCR will run on CPU (slow)")
        self.gpu = resolved_gpu
        self.device = f"cuda:0 ({torch.cuda.get_device_name(0)})" if resolved_gpu else "cpu"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._reader = easyocr.Reader(
            self.lang,
            gpu=self.gpu,
            model_storage_directory=str(self.model_dir),
            download_enabled=download_enabled,
            verbose=verbose,
        )
        self._versions = {
            "easyocr": getattr(easyocr, "__version__", "unknown"),
            "torch": torch.__version__,
        }
        self._lock = threading.Lock()

    def read(self, image: np.ndarray) -> list[OcrRegion]:
        # Reader.readtext is not thread-safe; the FastAPI server runs one worker
        # today, but the lock makes the guarantee independent of that choice.
        with self._lock:
            raw = self._reader.readtext(
                image,
                detail=1,
                paragraph=False,
                batch_size=self.batch_size,
                canvas_size=self.canvas_size,
                mag_ratio=self.mag_ratio,
                text_threshold=self.text_threshold,
                link_threshold=self.link_threshold,
                low_text=self.low_text,
                workers=0,
            )
        regions: list[OcrRegion] = []
        for box, text, confidence in raw:
            points = tuple((int(round(float(x))), int(round(float(y)))) for x, y in box)
            regions.append(
                OcrRegion(text=str(text), confidence=float(confidence), box=points)
            )
        return regions

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "versions": self._versions,
            "device": self.device,
            "gpu": self.gpu,
            "lang": self.lang,
            "model_dir": str(self.model_dir),
            "canvas_size": self.canvas_size,
            "mag_ratio": self.mag_ratio,
            "text_threshold": self.text_threshold,
            "link_threshold": self.link_threshold,
            "low_text": self.low_text,
            "batch_size": self.batch_size,
            "runtime": getattr(self, "_settings", {}),
        }


def _reading_order(regions: list[OcrRegion]) -> list[OcrRegion]:
    """Sort regions the way a person reads them: top to bottom, then left."""
    if not regions:
        return regions
    heights = [
        max(1.0, max(p[1] for p in region.box) - min(p[1] for p in region.box))
        for region in regions
    ]
    row_height = max(statistics.median(heights) * 0.5, 1.0)

    def key(region: OcrRegion):
        y = min(point[1] for point in region.box)
        x = min(point[0] for point in region.box)
        return (int(round(y / row_height)), x)

    return sorted(regions, key=key)


class OcrEngine:
    """Preprocess, run the backend, and normalise the result."""

    def __init__(
        self,
        backend: OcrBackend,
        *,
        preprocess: PreprocessConfig | None = None,
        low_confidence_below: float = LOW_CONFIDENCE,
        pipeline: str = "none",
    ) -> None:
        self.backend = backend
        self.preprocess_config = preprocess if preprocess is not None else PreprocessConfig()
        self.low_confidence_below = low_confidence_below
        self.pipeline = pipeline
        self._cpu_fallback_used = False

    def read_text(self, image: Any) -> OcrResult:
        started = time.perf_counter()
        prepared: PreprocessResult = prepare(image, self.preprocess_config)

        try:
            regions = self.backend.read(prepared.image)
        except RuntimeError as error:  # GPU/driver failure: retry once on CPU
            if not getattr(self.backend, "gpu", False) or self._cpu_fallback_used:
                raise
            print(f"  WARNING: GPU OCR failed ({error}); retrying on CPU")
            self.backend = EasyOcrBackend(gpu=False, model_dir=self.backend.model_dir)
            self._cpu_fallback_used = True
            regions = self.backend.read(prepared.image)

        if prepared.matrix is not None:
            regions = [
                OcrRegion(
                    text=region.text,
                    confidence=region.confidence,
                    box=tuple(
                        (int(round(x)), int(round(y)))
                        for x, y in prepared.to_original(region.box)
                    ),
                )
                for region in regions
            ]

        regions = _reading_order(regions)
        confidences = [region.confidence for region in regions]
        mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        min_conf = float(min(confidences)) if confidences else 0.0
        text = "\n".join(region.text for region in regions if region.text.strip())
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return OcrResult(
            text=text,
            regions=tuple(regions),
            confidence=mean_conf,
            min_confidence=min_conf,
            n_regions=len(regions),
            low_confidence=(not text) or mean_conf < self.low_confidence_below,
            meta={
                "pipeline": self.pipeline,
                "backend": getattr(self.backend, "name", "unknown"),
                "device": getattr(self.backend, "device", "unknown"),
                "latency_ms": round(elapsed_ms, 1),
                "preprocess": prepared.flags,
            },
        )

    def read_many(self, images: Sequence[Any], *, progress: bool = True) -> list[OcrResult]:
        results: list[OcrResult] = []
        total = len(images)
        for index, image in enumerate(images, start=1):
            results.append(self.read_text(image))
            if progress and (index % 25 == 0 or index == total):
                print(f"    {index}/{total} images", flush=True)
        return results

    def describe(self) -> dict[str, Any]:
        info = self.backend.describe()
        info["pipeline"] = self.pipeline
        info["low_confidence_below"] = self.low_confidence_below
        info["preprocess"] = self.preprocess_config.as_dict()
        return info


_DEFAULT_LOCK = threading.Lock()
_DEFAULT_ENGINE: OcrEngine | None = None


def default_engine(**kwargs: Any) -> OcrEngine:
    """The shared engine the app calls. Built once; the reader load is expensive."""
    global _DEFAULT_ENGINE
    with _DEFAULT_LOCK:
        if _DEFAULT_ENGINE is None:
            backend = EasyOcrBackend(**kwargs)
            _DEFAULT_ENGINE = OcrEngine(backend, pipeline="default")
        return _DEFAULT_ENGINE


def read_text(image: Any, **backend_kwargs: Any) -> OcrResult:
    """Read printed text from ``image`` (path, bytes, PIL image, or BGR array)."""
    return default_engine(**backend_kwargs).read_text(image)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Read printed text from an image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--no-preprocess", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit the full result as JSON")
    args = parser.parse_args(argv)

    backend = EasyOcrBackend(gpu=False if args.cpu else None)
    engine = OcrEngine(
        backend,
        preprocess=None if not args.no_preprocess else PreprocessConfig(
            upscale=False, deskew=False, perspective=False, enhance=False
        ),
        pipeline="cli",
    )
    result = engine.read_text(args.image)

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(f"  device        : {result.meta['device']}")
        print(f"  regions       : {result.n_regions}")
        print(f"  mean conf     : {result.confidence:.3f}")
        print(f"  low confidence: {result.low_confidence}")
        print()
        print(result.text if result.text else "  (no text detected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
