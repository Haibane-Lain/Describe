"""End-to-end check of the frontend integration, without the frontend.

Exercises the pipeline the way the Flutter app does - a video clip in, the
contract map out - and asserts the properties the demo depends on:

1. all three `status` values are reachable and correct
2. the contract has exactly the keys `DescribeResult.fromMap` parses
3. `confidence` is present (the Dart type requires it) and is a placeholder
4. OCR text appears when the image contains text, tagged `text read`
5. a failure never raises - it returns a speakable message

Synthetic clips are generated from real VizWiz photographs, so this needs no
recorded footage and no running server. Run it after any change to
`train/serve.py` or the OCR/gate configs.

    py -3.12 -m train.serve_smoke
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REQUIRED_KEYS = {"status", "text", "confidence", "tags", "hint"}
VALID_STATUS = {"ok", "lowConfidence", "tooDark"}

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  - {detail}" if detail else ""))


def write_clip(path: Path, frame, frames: int = 45, jitter: bool = True) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (width, height))
    for i in range(frames):
        if jitter:
            dx, dy = int(np.sin(i / 6) * 4), int(np.cos(i / 5) * 3)
            frame_i = cv2.warpAffine(frame, np.float32([[1, 0, dx], [0, 1, dy]]), (width, height))
        else:
            frame_i = frame
        writer.write(frame_i)
    writer.release()


def main() -> int:
    import cv2

    source = Path("data/raw/vizwiz/images/val/VizWiz_val_00000001.jpg")
    if not source.exists():
        print(f"  source image missing: {source}")
        return 1

    image = cv2.imread(str(source))
    height, width = image.shape[:2]
    scale = 720 / max(height, width)
    image = cv2.resize(image, (int(width * scale), int(height * scale)))

    clips = Path(".tmp/serve_smoke")
    variants = {
        "normal": image,
        "dark": (image.astype(np.float32) * 0.02).astype(np.uint8),
        "blurred": cv2.GaussianBlur(image, (0, 0), 35),
    }
    for name, frame in variants.items():
        write_clip(clips / f"{name}.mp4", frame)
    print(f"  generated {len(variants)} synthetic clips in {clips}\n")

    from train.serve import DescribePipeline

    pipeline = DescribePipeline(verbose=False)
    print(f"  gate threshold {pipeline.gate_threshold:.3f}\n")

    outputs: dict[str, dict] = {}
    for name in variants:
        result = pipeline.describe(clips / f"{name}.mp4")
        payload = result.as_dict()
        outputs[name] = payload
        print(f"  {name:<8} -> {payload['status']:<14} {payload['text'][:64]}")

    print()
    # 1. contract shape
    for name, payload in outputs.items():
        check(f"{name}: contract has exactly the required keys",
              set(payload.keys()) == REQUIRED_KEYS,
              f"got {sorted(payload.keys())}")
        check(f"{name}: status is a valid enum value",
              payload["status"] in VALID_STATUS, payload["status"])
        check(f"{name}: text is non-empty",
              bool(payload["text"].strip()))

    # 2. statuses correct for the input
    check("normal clip is accepted",
          outputs["normal"]["status"] == "ok", outputs["normal"]["status"])
    check("near-black clip is tooDark",
          outputs["dark"]["status"] == "tooDark", outputs["dark"]["status"])
    check("blurred clip is lowConfidence",
          outputs["blurred"]["status"] == "lowConfidence", outputs["blurred"]["status"])

    # 3. hints accompany rejections, not acceptances
    for name in ("dark", "blurred"):
        check(f"{name}: hint is populated on rejection",
              bool(outputs[name]["hint"].strip()), outputs[name]["hint"])
    check("normal: hint is empty on acceptance",
          outputs["normal"]["hint"] == "")

    # 4. confidence is a placeholder, present for the Dart type
    check("confidence present and an int in all cases",
          all(isinstance(p["confidence"], int) for p in outputs.values()))

    # 5. OCR tagging is consistent with the text
    for name, payload in outputs.items():
        has_text = "Text in the image reads:" in payload["text"]
        tagged = "text read" in payload["tags"]
        check(f"{name}: 'text read' tag matches OCR presence",
              has_text == tagged, f"text={has_text} tag={tagged}")
        check(f"{name}: 'basic caption' always tagged",
              "basic caption" in payload["tags"])

    # 6. a bad path must not raise
    try:
        bad = pipeline.describe(clips / "does_not_exist.mp4")
        check("missing file returns a speakable result instead of raising",
              bool(bad.text.strip()), bad.text[:48])
    except Exception as exc:  # noqa: BLE001
        check("missing file returns a speakable result instead of raising", False, repr(exc))

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n  {passed}/{total} checks passed")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"    FAILED: {name} - {detail}")

    out = Path("runs/serve_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"passed": passed, "total": total,
                               "outputs": outputs}, indent=2), encoding="utf-8")
    print(f"  written: {out}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
