"""Smoke test the OCR component before trusting it with a measurement.

The measurement in ``train.ocr.measure`` is only worth publishing if the parts
under it behave, so this checks each of them on synthetic images where the answer
is known: the test set is the set we think it is, preprocessing does what it
claims, coordinates survive the transform, the BGR convention is real, and the
scoring path is the project's shared one rather than a lookalike.

    py -3.12 -m train.ocr.smoke_test_ocr
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.finetune.eval_vqa import answer_covered  # noqa: E402
from train.ocr.measure import answer_hit  # noqa: E402
from train.ocr.preprocess import (  # noqa: E402
    estimate_skew_deg,
    load_image,
    prepare,
)
from train.ocr.reading_set import (  # noqa: E402
    MIN_REPORTABLE_N,
    reading_items,
    reading_set_stats,
)

FONT_CANDIDATES = (
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
)


def _font(size: int):
    from PIL import ImageFont

    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def _render(text: str, size: tuple[int, int] = (900, 220), font_size: int = 56,
            angle: float = 0.0) -> np.ndarray:
    """White canvas with black text, returned as a BGR array."""
    import cv2
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    box = draw.textbbox((0, 0), text, font=font)
    x = (size[0] - (box[2] - box[0])) // 2
    y = (size[1] - (box[3] - box[1])) // 2 - box[1]
    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    if angle:
        image = image.rotate(angle, expand=True, fillcolor=(255, 255, 255),
                             resample=Image.BICUBIC)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def normalize(text: str) -> str:
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))

    # -- the test set is the set we think it is ---------------------------
    items = reading_items("val")
    stats = reading_set_stats(items)
    check("reading set has 577 questions", len(items) == 577, f"{len(items)}")
    check("reading set has 577 distinct images",
          stats["images"] == 577 and stats["images"] == len(items), f"{stats['images']}")
    subtypes = {row["name"]: row["n"] for row in stats["subtypes"]}
    check("subtype counts sum to the set size",
          sum(subtypes.values()) == len(items) == 577, f"{sum(subtypes.values())}")
    check("sign/screen is 190 (the 188 in ocr-plan section 4 omits 2 captionless rows)",
          subtypes.get("sign / screen text") == 190, f"{subtypes.get('sign / screen text')}")
    check("six small subtypes are flagged unpublishable",
          len(stats["suppressed_subtypes"]) == 6, str(stats["suppressed_subtypes"]))
    check("every subtype has an n", all(row["n"] > 0 for row in stats["subtypes"]),
          f"{len(stats['subtypes'])} subtypes")
    check("publish flag uses the shared threshold",
          all(row["publish"] == (row["n"] >= MIN_REPORTABLE_N) for row in stats["subtypes"]),
          f"threshold {MIN_REPORTABLE_N}")

    # -- image loading and the channel convention -------------------------
    from PIL import Image

    red = Image.new("RGB", (8, 8), (255, 0, 0))
    loaded = load_image(red)
    check("load_image returns BGR, not RGB", tuple(int(v) for v in loaded[0, 0]) == (0, 0, 255),
          f"blue={loaded[0, 0, 0]}, red={loaded[0, 0, 2]}")
    gray = Image.new("L", (8, 8), 128)
    check("grayscale is widened to three channels", load_image(gray).shape == (8, 8, 3),
          str(load_image(gray).shape))

    # -- preprocessing ----------------------------------------------------
    small = _render("SMALL TEXT 42", size=(200, 60), font_size=14)
    prepared = prepare(small)
    long_side = max(prepared.flags["processed_shape"])
    check("small images are upscaled to the configured long side",
          prepared.flags["upscaled"] and long_side >= 1000, f"long side {long_side}")

    clean = _render("MEETING ROOM 204", size=(1200, 400), font_size=64)
    clean_prepared = prepare(clean)
    check("a clean, large, upright image is left alone",
          not clean_prepared.flags["upscaled"]
          and clean_prepared.flags["deskew_deg"] == 0.0
          and not clean_prepared.flags["perspective"],
          f"flags deskew={clean_prepared.flags['deskew_deg']} "
          f"perspective={clean_prepared.flags['perspective']} "
          f"upscaled={clean_prepared.flags['upscaled']}")

    rotated = _render("MEETING ROOM 204", angle=6.0)
    before = abs(estimate_skew_deg(rotated))
    rotated_prepared = prepare(rotated)
    after = abs(estimate_skew_deg(rotated_prepared.image))
    check("rotation is detected and corrected",
          rotated_prepared.flags["deskew_deg"] != 0.0 and after < 2.0 and after < before,
          f"{before:.1f} deg -> {after:.1f} deg, applied {rotated_prepared.flags['deskew_deg']}")

    # original -> processed (forward matrix) -> original (what the engine does to
    # every detected box, so a wrong direction would misplace every region).
    known_original = (100.0, 40.0)
    matrix = rotated_prepared.matrix
    check("the deskew transform is actually recorded", matrix is not None)
    if matrix is not None:
        projected = matrix @ np.array([known_original[0], known_original[1], 1.0])
        projected = projected[:2] / projected[2]
        restored = rotated_prepared.to_original([projected])[0]
        check("coordinates round-trip original -> processed -> original",
              abs(restored[0] - known_original[0]) < 1e-6
              and abs(restored[1] - known_original[1]) < 1e-6,
              f"({known_original[0]:.0f},{known_original[1]:.0f}) -> "
              f"({projected[0]:.1f},{projected[1]:.1f}) -> "
              f"({restored[0]:.1f},{restored[1]:.1f})")

    scale = prepared.flags.get("scale", 1.0)
    original_point = prepared.to_original([(100.0, 30.0)])[0]
    check("upscale maps boxes back to original pixels",
          abs(original_point[0] - 100.0 / scale) < 1.5
          and abs(original_point[1] - 30.0 / scale) < 1.5,
          f"scale {scale}: (100,30) -> ({original_point[0]:.1f},{original_point[1]:.1f})")

    # -- scoring is the project's shared rule -----------------------------
    cases = [
        ("the dog years box", "dog years", True),
        ("years later a dog", "dog years", False),
        ("small dog years later", "dog years", True),
        ("a red bike", "red", True),
    ]
    check("answer_hit agrees with eval_vqa.answer_covered",
          all(answer_hit(text, answer) == answer_covered(text, answer) == expected
              for text, answer, expected in cases))
    check("word order matters for multi-word answers",
          answer_hit("years and dog", "dog years") is False)

    # -- word-level metrics for the collected text set (proposal section 5C) --
    from train.ocr.word_metrics import cer, exact_match, normalize, wer

    check("CER is zero on identical text", cer("abc", "abc") == 0.0)
    check("CER counts one substitution in three characters", abs(cer("abc", "abd") - 1 / 3) < 1e-9,
          f"{cer('abc', 'abd'):.4f}")
    check("WER counts one wrong word in two", abs(wer("hello world", "hello there") - 0.5) < 1e-9,
          f"{wer('hello world', 'hello there'):.4f}")
    check("normalisation ignores case and punctuation",
          normalize("MLK 250ML.") == "mlk 250ml"
          and exact_match("MLK 250ML.", "mlk 250ml"),
          normalize("MLK 250ML."))

    # -- the engine itself (needs easyocr installed) ----------------------
    try:
        from train.ocr.engine import EasyOcrBackend, OcrEngine

        backend = EasyOcrBackend()
        engine = OcrEngine(backend, pipeline="smoke")
        print(f"  ... engine loaded on {backend.device}")

        result = engine.read_text(_render("MEETING ROOM 204"))
        check("known text is recovered from a clean synthetic image",
              "meeting room 204" in normalize(result.text),
              f"{normalize(result.text)!r} (conf {result.confidence:.3f})")
        check("regions come back with boxes and confidence",
              result.n_regions >= 1 and len(result.regions[0].box) == 4
              and 0.0 <= result.regions[0].confidence <= 1.0,
              f"{result.n_regions} regions")

        blank = engine.read_text(np.full((320, 480, 3), 255, dtype=np.uint8))
        check("a blank image reports no text and low confidence",
              blank.text == "" and blank.n_regions == 0 and blank.low_confidence,
              f"text={blank.text!r} low={blank.low_confidence}")
    except ImportError as error:
        check("easyocr is installed", False,
              f"{error} -- run: py -3.12 -m pip install "
              f"-c train/constraints-ocr.txt -r train/requirements-ocr.txt")

    failed = [c for c in checks if not c[1]]
    print()
    print(f"  {len(checks) - len(failed)}/{len(checks)} checks passed")
    for name, _, detail in failed:
        print(f"    FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
