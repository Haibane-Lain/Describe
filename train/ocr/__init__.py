"""OCR component: printed text, owned away from the captioner.

The captioner reaches only 11.7% answer hit rate on VizWiz reading questions and
five attempts to train it out of that failed, so printed text is owned here
instead (``train/docs/ocr-plan.md``). This package is the component, not a
request to build one:

    py -3.12 -m train.ocr.reading_set --split val        # the 577-image test set
    py -3.12 -m train.ocr.smoke_test_ocr                 # capability checks
    py -3.12 -m train.ocr.measure --split val            # Phase 0 measurement

    from train.ocr import read_text
    result = read_text("label.jpg")
    result.text, result.confidence, result.low_confidence

Two boundaries this package keeps on purpose:

* **The captioner does not consume OCR output** (``ocr-plan.md`` §7). The
  dependency only runs ``train/ocr -> train/finetune``, for the shared scoring
  helpers; nothing under ``train/finetune``, ``train/train`` or
  ``train/models`` imports ``train.ocr``.
* **OCR output is not verified text.** Medication names and dosages live in this
  bucket and a misread dose can cause harm; the proposal excludes
  safety-critical decisions and that exclusion applies here too.
"""

from typing import Any

# Submodules are resolved on first attribute access rather than imported here.
# Eagerly importing them would make `py -3.12 -m train.ocr.reading_set` warn that
# the module was already in sys.modules before execution -- the package __init__
# would have imported the very module runpy is about to run as __main__. Lazy
# access keeps `from train.ocr import read_text` working for the app while every
# `-m train.ocr.<tool>` invocation stays clean, which is how the rest of the
# repository invokes its tools.
_LAZY: dict[str, str] = {
    "EasyOcrBackend": "engine",
    "LOW_CONFIDENCE": "engine",
    "OcrBackend": "engine",
    "OcrEngine": "engine",
    "OcrRegion": "engine",
    "OcrResult": "engine",
    "default_engine": "engine",
    "read_text": "engine",
    "PreprocessConfig": "preprocess",
    "PreprocessResult": "preprocess",
    "load_image": "preprocess",
    "prepare": "preprocess",
    "MIN_REPORTABLE_N": "reading_set",
    "ReadingItem": "reading_set",
    "reading_items": "reading_set",
    "reading_set_stats": "reading_set",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "EasyOcrBackend",
    "LOW_CONFIDENCE",
    "MIN_REPORTABLE_N",
    "OcrBackend",
    "OcrEngine",
    "OcrRegion",
    "OcrResult",
    "PreprocessConfig",
    "PreprocessResult",
    "ReadingItem",
    "default_engine",
    "load_image",
    "prepare",
    "read_text",
    "reading_items",
    "reading_set_stats",
]
