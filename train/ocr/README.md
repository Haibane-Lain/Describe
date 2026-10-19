# OCR component

> **Status: DISABLED in the integration (2026-10-19).** The component is
> currently unstable, so `train/serve.py` does **not** call it: the backend
> reports `ocr=off`, `tags` never contains `"text read"`, and `text` never
> contains a "Text in the image reads:" clause. **Nothing was removed** — this
> package, `train/configs/ocr_frozen.json` and the `checkpoints/easyocr/` weights
> are all intact. Re-enable with `py -3.12 -m train.serve --serve --ocr`.
> The measurements below describe the component when it is on.

Printed text, owned away from the captioner. The captioner reaches **11.7 %**
answer hit rate on VizWiz reading questions and four attempts to train the gap
out of it failed, so "reading" is not a modelling problem the captioner will
solve (`train/docs/ocr-plan.md`, `train/docs/captioner-experiments.md`). This
package is the component that owns it, together with the measurement that says
how much of the bucket it actually covers.

**Not for safety-critical decisions.** Medication names, dosages and expiry dates
are inside this bucket. A misread dose can cause harm; the proposal excludes
safety-critical use and that exclusion applies to everything here. Never present
OCR output as verified text.

## Measured

On the 577 VizWiz validation reading questions, with the frozen configuration in
`train/configs/ocr_frozen.json`:

| | n | Caption | OCR | Either |
|---|---|---|---|---|
| Reading bucket | 577 / 575 | 11.7% | **11.4%** | **20.3%** |

**OCR does not solve the bucket on its own — and neither does the captioner.** The
two are complementary: OCR answers 50 questions the captioner misses, the
captioner answers 51 that OCR misses, and only 16 are answered by both. The union
is the honest coverage figure. Full per-subtype tables, the near-miss analysis
(25% of misses are within half a phrase of the answer; 13% produce no text at
all), the quality-gate split and the method are in
[docs/ocr-measurement.md](../docs/ocr-measurement.md).

The engine configuration was selected on the **train** split (400 reading
questions) because the 577 is the only clean measurement of the bucket. Canvas
2560 — EasyOCR's default — cost 4.8× the time of 1600 with no measurable accuracy
difference, which is the one solid result of that screen.

```powershell
py -3.12 -m train.ocr.measure --split val --pipelines none,enhance `
    --config-file train/configs/ocr_frozen.json `
    --combined runs/blip_large_head/predictions_full_beam5.jsonl `
    --out-dir runs/ocr_easyocr --resume
```

## Install

```powershell
$env:TEMP = "$PWD\.tmp"; $env:TMP = "$PWD\.tmp"   # this shell has no writable TEMP
py -3.12 -m pip install -c train/constraints-ocr.txt -r train/requirements-ocr.txt
py -3.12 -c "import easyocr, torch; print(easyocr.__version__, torch.__version__, torch.cuda.is_available())"
py -3.12 -m train.smoke_test                       # must still be 18/18
```

The constraints file exists because `easyocr` depends on `torch`, and torch on
this machine is the ROCm build from `repo.radeon.com`, not PyPI. A resolver that
"fixes" torch replaces the GPU build with the CPU one and nothing announces it —
the exact failure `train/requirements.txt` warns about. Do not install easyocr
without `-c train/constraints-ocr.txt`.

Model weights are downloaded on first use into `checkpoints/easyocr/` (already
git-ignored): `craft_mlt_25k` (detector) and `english_g2` (recogniser).

## Use

```python
from train.ocr import read_text

result = read_text("label.jpg")        # path, bytes, PIL image, or BGR array
result.text                            # all recognised text, reading order
result.confidence                       # mean region confidence; 0.0 if none found
result.low_confidence                   # True -> say "partially readable", don't read flatly
result.regions[0].box                   # four (x, y) points in ORIGINAL image pixels
```

`read_text` builds the EasyOCR reader once and reuses it (`default_engine()`), and
serialises access with a lock because `easyocr.Reader` is not thread-safe. The
FastAPI server can therefore call it from the router as `read_text(image).text`
and `read_text(image).low_confidence`, which is what the app plan's
`read_text()` contract expects.

From the command line:

```powershell
py -3.12 -m train.ocr.engine photo.jpg
py -3.12 -m train.ocr.engine photo.jpg --json --no-preprocess
```

`OcrEngine(backend, preprocess=...)` is the explicit form, for measuring a
specific configuration:

```python
from train.ocr import EasyOcrBackend, OcrEngine
from train.ocr.preprocess import DISABLED

engine = OcrEngine(EasyOcrBackend(), preprocess=DISABLED, pipeline="raw")
engine.read_text("photo.jpg").text
```

## Interface contract

| | |
|---|---|
| **Input** | image path, bytes, PIL image, or BGR `uint8` array |
| **Output** | `OcrResult`: concatenated text in reading order, per-region boxes and confidences, mean/min confidence, `low_confidence` |
| **Coordinates** | regions are mapped back to **original** image pixels even when preprocessing upscaled, deskewed or perspective-corrected the image |
| **Channels** | arrays are **BGR**, because EasyOCR reads a 3-channel array as BGR. Passing RGB does not crash; it silently reads worse |
| **Failure** | no text found gives `text == ""`, `n_regions == 0`, `low_confidence == True` — distinguishable from text found but wrong |

The captioner never consumes this output (`ocr-plan.md` §7). The dependency only
runs one way: `train/ocr` imports scoring helpers from `train/finetune`; nothing
under `train/finetune`, `train/train` or `train/models` imports `train/ocr`.

## Measurement

```powershell
py -3.12 -m train.ocr.reading_set --split val        # the 577-image test set
py -3.12 -m train.ocr.smoke_test_ocr                 # 18 capability checks
py -3.12 -m train.ocr.measure --split val --pipelines none,enhance `
    --combined runs/blip_large_head/predictions_full_beam5.jsonl
```

Writes to `runs/ocr_easyocr/`:

| File | Contents |
|---|---|
| `regions_val_reading.jsonl` | one row per (image, pipeline): text, regions, confidences, latency, preprocess flags; append-only and resumable with `--resume` |
| `predictions_val_reading.jsonl` | `{"file_name", "prediction"}` — the repo's predictions shape, so `train.finetune.intent_table --predictions <file>` scores OCR with no new code |
| `measure_val_reading.json` | every rate with its population and n, per subtype, split by `text_detected` and by the quality gate's verdict |

### The two populations

Both are reading questions; only one can compare against the captioner.

* **A = 577** — every answerable validation reading question. The population OCR
  owns, and the one OCR's own hit rate is reported on.
* **B = 575** — A restricted to images the captioner produced a prediction for.
  The captioner's file omits two reading images (`VizWiz_val_00001985`,
  `VizWiz_val_00002169`, answers `black` and `nothing`), so any caption-vs-OCR
  comparison has to use B or it scores the caption on an empty string.

**Every published rate must name A or B.** `intent_table.json` records 575/188 for
this reason; a fresh reproduction gives 577/190. Both are right about different
populations, and the earlier handoff document labelled neither.

### The 577 is not for tuning

`measure.py` refuses engine-parameter overrides on `--split val` unless a frozen
`--config-file` is supplied. Sweeps belong on `--split train` (1,980 questions);
the frozen config is then re-measured once on the 577 and reported next to the
untuned number, never instead of it. Per-subtype rates are suppressed below
`MIN_REPORTABLE_N = 20` for the same reason `ocr-plan.md` §4 forbids them: six
subtypes have n < 20 and their rates are noise.

## Files

| File | Role |
|---|---|
| `reading_set.py` | the test set, defined once (population A) |
| `preprocess.py` | load, upscale, perspective-correct, deskew — each conditional, with the transform kept for mapping boxes back |
| `engine.py` | `read_text` / `OcrEngine` / `EasyOcrBackend`, the backend seam, reading order, confidence |
| `measure.py` | Phase 0 (OCR hit rate) and Phase 3 (caption-or-OCR), both on the shared scoring rule |
| `smoke_test_ocr.py` | synthetic-image capability checks |
| `word_metrics.py` | CER / WER / exact match for the team-collected label and document photographs (proposal §5C) |
