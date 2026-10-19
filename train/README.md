# Captioner training

Training code for the **Describe** project: a mobile image description and text
reading system for blind and low-vision users.

Architecture: a **frozen CNN encoder** with features cached to disk, plus a
**trained LSTM decoder**. Caching is the key design choice — the encoder runs
once (~8 minutes for every image in the project), and every training epoch
afterwards reads a tensor and never touches an image again.

## Status

| Stage | State |
|---|---|
| Environment verified on the RX 9070 XT | done — ROCm 7.2, GPU detected |
| Decoder correctness and speed | verified on GPU, 11/11 |
| Smoke test (loss decrease, checkpoint round-trip) | passing, 18/18 |
| Feature extraction pipeline | verified on GPU, 7/7 |
| Training loop, checkpoint, resume, stage-2 init | verified end to end, 22/22 |
| Evaluation metrics | BLEU-1–4, METEOR, ROUGE-L, CIDEr-D working; SPICE unavailable |
| Dataset download | done — Flickr30k, VizWiz, COCO on disk |
| From-scratch captioner | **abandoned as the deployed model** — BLEU-4 ~0, see [docs/status-report.md](docs/status-report.md) |
| **BLIP fine-tune (the current path)** | **working end to end** — BLEU-4 0.170 on real VizWiz, 17/17 smoke checks. See [finetune/README.md](finetune/README.md) |
| **OCR component (objective 4.4)** | **working and measured** — EasyOCR behind conditional preprocessing, measured on the 577 VizWiz reading questions the captioner cannot answer. See [ocr/README.md](ocr/README.md) and [docs/ocr-measurement.md](docs/ocr-measurement.md) |

> **The from-scratch CNN + LSTM pipeline is now the reported ablation, not the
> product.** Training a captioner from scratch did not converge to usable
> output on this data or this hardware, and the diagnosis is documented. The
> deployed captioner is a fine-tuned BLIP, in [finetune/](finetune/). Everything
> below still applies to the ablation and to the shared data and metric code.

Measured throughput: encoder **156 img/s** (all 39k images in ~8 min), beam-5
decode **12.2 ms/img** (8,000 test images in 1.6 min).

## Requirements

Use **`py -3.12`** — never `python`. The ROCm PyTorch wheels are `cp312` only.

```powershell
py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.9.1+rocmsdk20260116 True
```

> **Never run a bare `pip install torch` in this environment.** It replaces the
> ROCm build with the default CPU build and silently disables the GPU.

See [docs/environment.md](docs/environment.md) for the full verified
configuration, the MIOpen workaround for gfx1201, and how to rebuild the
environment.

## Verify before you download anything

```powershell
py -3.12 -m train.verify_decoders              # decoder correctness + speed, ~1 min
py -3.12 -m train.smoke_test                   # environment + training loop, ~1 min
py -3.12 -m train.features.extract --synthetic # encoder + cache, ~1 min
py -3.12 -m train.e2e_test                     # full pipeline on generated data, ~5 min
py -3.12 -m train.check_metrics                # which evaluation metrics work
```

Each prints a pass/fail summary and exits non-zero on failure. Run these after
any driver, ROCm, or PyTorch change.

## Full pipeline

```powershell
# 1. Data  (Flickr30k for pretraining, VizWiz for fine-tuning)
py -3.12 -m train.data.download --dataset all --workers 8
py -3.12 -m train.data.prepare --dataset all --check-leakage

# 2. Cache features  (~3.4 min Flickr30k, ~4.2 min VizWiz)
py -3.12 -m train.features.extract --manifest data/manifests/flickr30k.jsonl
py -3.12 -m train.features.extract --manifest data/manifests/vizwiz_train.jsonl
py -3.12 -m train.features.extract --manifest data/manifests/vizwiz_val.jsonl
py -3.12 -m train.features.extract --manifest data/manifests/vizwiz_test.jsonl

# 3. Stage 1 - pretrain on Flickr30k
py -3.12 -m train.train.train_captioner `
  --manifest data/manifests/flickr30k.jsonl `
  --run-name flickr30k --epochs 20

# 4. Stage 2 - fine-tune on VizWiz-Captions
py -3.12 -m train.train.train_captioner `
  --manifest data/manifests/vizwiz_train.jsonl `
  --val-manifest data/manifests/vizwiz_val.jsonl `
  --run-name vizwiz_ft --epochs 10 --lr 1e-4 `
  --init-from checkpoints/flickr30k/best.pt

# 5. Ablation - VizWiz from scratch (the domain-shift result)
py -3.12 -m train.train.train_captioner `
  --manifest data/manifests/vizwiz_train.jsonl `
  --val-manifest data/manifests/vizwiz_val.jsonl `
  --run-name vizwiz_scratch --epochs 10 --lr 1e-3

# 6. Evaluate on the test split
py -3.12 -m train.train.evaluate `
  --checkpoint checkpoints/vizwiz_ft/best.pt `
  --feature-index data/features/vizwiz_test_resnet101.jsonl `
  --manifest data/manifests/vizwiz_test.jsonl `
  --out runs/vizwiz_ft/test_metrics.json `
  --pred-out runs/vizwiz_ft/predictions.jsonl
```

## Layout

```
train/
  configs/base.yaml            default hyperparameters
  configs/ocr_frozen.json      OCR engine parameters, selected on the train split
  docs/environment.md          verified GPU/ROCm setup and known issues
  finetune/                    BLIP fine-tuning - the deployed captioner
    base.py                    cached-feature forward, checkpoint loading
    cache.py                   resumable feature extraction and verification
    train_blip.py              training loop, LoRA, checkpoints, resume
    evaluate_blip.py           batched beam search, collapse diagnostics
  ocr/                         printed text - the reading bucket's owner
    reading_set.py             the 577-image reading test set, defined once
    preprocess.py              conditional upscale / perspective / deskew
    engine.py                  read_text(): text, regions, boxes, confidence
    measure.py                 OCR hit rate and caption-or-OCR coverage
    word_metrics.py            CER / WER for the collected text set (proposal 5C)
    smoke_test_ocr.py          OCR capability gate
  utils/device.py              device resolution, MIOpen workaround, seeding
  data/
    download.py                dataset acquisition
    prepare.py                 unified manifests + leakage check
    vocab.py                   vocabulary and caption encoding
    dataset.py                 cached-feature dataset
  features/extract.py          ResNet feature extraction and caching
  models/captioner.py          LSTM decoder with greedy and beam search
  train/
    train_captioner.py         training loop, checkpoints, resume, stage-2 init
    evaluate.py                VizWiz metric set, text_detected breakdown
  smoke_test.py                environment gate
  e2e_test.py                  full-pipeline gate on generated data
```

## Design notes

**Why cached features.** The decoder is ~25M parameters and reads a `[N, 2048]`
tensor. An epoch over 31,783 images takes well under a minute once features are
cached, so the encoder is the only expensive step and it runs exactly once.

**Why `--init-from` reloads selectively.** Stage 2 rebuilds the vocabulary on
VizWiz captions, so the embedding table and output projection change shape.
`load_compatible_state` transfers the LSTM and feature projection and
reinitialises the vocabulary-specific layers, logging exactly what it skipped.
Stage-2 training logs therefore state how many tensors actually carried over.

**Why the feature cache index is authoritative.** The cache index carries
`feature_row` next to the manifest fields. Training and evaluation read rows
from the cache index and filter by the requested manifest, rather than joining
the two — joining was the source of a bug the end-to-end test caught.

**Why `text_detected` is reported.** Captioners fail hardest on images
containing text, which is exactly where the OCR branch is meant to help.
Evaluation reports metrics split by that flag so the weakness is measured
rather than hidden.

## Known limitations

- `num_workers=0` in the DataLoader: Windows spawn semantics would otherwise
  re-import the entry module. Cached features are in memory, so this is not a
  bottleneck.
- Beam search decodes one image at a time. At 8,000 test images this is minutes,
  not hours, but it is not batched.
- MIOpen is disabled on RDNA4 because of the missing-rocrand-header bug. See
  [docs/environment.md](docs/environment.md) for when to revisit this.

## Datasets

| Dataset | Use | Size | Source |
|---|---|---|---|
| Flickr30k | Pretraining | 31,783 images, 5 captions each | HF `lmms-lab-encoder/flickr30k` |
| VizWiz-Captions | Fine-tuning and evaluation | 23,431 / 7,750 / 8,000 train/val/test | official `vizwiz.org` archives |
| VizWiz-QualityIssues | Quality gate (separate track) | Same image pool, flaw + unrecognizability votes | official archives |

VizWiz-Captions is **CC BY 4.0** — attribute it in the report.

`--check-leakage` asserts that no image spans multiple splits. VizWiz-Captions
and VizWiz-QualityIssues share one image pool and partition, so mixing splits
between the two tasks is the most likely silent evaluation bug in this project.

### Why these exact sources

Two mirror choices were made after testing, and both matter:

**Flickr30k must come from `lmms-lab-encoder/flickr30k`.** The
`iscc/iscc-flickr30k` mirror contains only 224px thumbnails with a single
caption per row — unusable for captioner training. The chosen repo has
full-size images and all five captions (verified: 5 captions on 150/150 sampled
rows, images 330–500px).

**VizWiz must come from the official archives, not Hugging Face.** The
`lmms-lab/VizWiz-Caps` mirror has **no train split** and only **one caption per
image**; the other 50+ VizWiz repos on Hugging Face are VQA datasets, not
captions. The official archives are the only source of the 23,431-image
training split with all five captions.

### Why the downloader uses parallel range requests

The official VizWiz host serves a single connection at about **0.15 MB/s**,
which projects to **28 hours** for the ~15 GB of image archives. The same host
honours HTTP Range requests, and eight parallel connections reach
**~6–7 MB/s** — about 40× faster, turning the download into roughly 40 minutes.
`download_file_parallel` implements this with per-chunk `.part` files, so an
interrupted run resumes rather than restarting, and it verifies the assembled
size against the server's `Content-Length` before declaring success.

## What the real data actually contains

Verified after downloading. Three findings shape how the model must be trained
and evaluated:

**1. VizWiz test-split captions are withheld.** `caption/annotations/test.json`
has 8,000 images and **0 annotations**; train and val are complete. So
**evaluation must use the validation split**, which is the largest labelled set
available. The official test split can still be used for the quality gate,
because the unrecognizability and flaw annotations *are* published for it.

**2. Not every image has five captions.** VizWiz captions were crowd-sourced and
rejected or pre-canned ones are excluded, so the distribution is uneven:

| Split | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| train (23,431) | 565 | 732 | 914 | 1,655 | 4,775 | 14,790 |
| val (7,750) | 208 | 274 | 286 | 547 | 1,517 | 4,918 |

`FeatureCaptionDataset` skips rows with no captions, so the 565 and 208
captionless images are simply not trained on. Flickr30k is uniform: all 31,783
images have exactly 5 captions.

**3. The gate labels are heavily imbalanced.** Only **9.1%** of train images
(2,134 of 23,431) and 9.7% of val images are unrecognizable. Aggregate accuracy
will therefore look high while being meaningless — report per-flaw
precision/recall and the unrecognizability ROC, and tune the threshold against
the false-accept rate.

The image archives also hold more files than the caption-annotated counts
(39,704 vs 39,181): the extra ~500 train images are those whose captions were
rejected. They appear in the archives without usable captions, which is expected.

`text_detected` is reconstructed from the per-caption flags, since the
per-image field is absent for roughly 40% of images: **26,926 images contain
text, 12,255 do not.**
