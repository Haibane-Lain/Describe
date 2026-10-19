# How the captioner was trained

Reference for documentation. Every figure here was read from the run artifacts
(`runs/blip_large_head/run_config.json`, `history.jsonl`, the cache indexes and
the manifests) rather than recalled, so it can be checked.

The shipped model is `checkpoints/blip_large_head/best.pt`.

---

## 1. What was trained

**Base model:** `Salesforce/blip-image-captioning-large` — a pretrained
vision-language captioning model, not a model built from scratch.

| | |
|---|---|
| Total parameters | 446.3 M |
| Vision tower | 303.7 M (**frozen**, never trained) |
| Text decoder | 142.6 M (**all of it trained**) |
| Tuning mode | `head` — full text decoder, no LoRA |
| Trainable at run time | 142,599,740 (the text decoder only) |

**Why not from scratch.** An earlier attempt trained a CNN+attention+LSTM
captioner from scratch. Retrained later with *every advantage* — all 31,181
available images, 30.5 M parameters, 10 epochs — it reached BLEU-4 **0.0204**
against the fine-tuned BLIP's **0.2034**, and overfitted from epoch 3. The
from-scratch approach is the project's documented ablation, not the shipped model.

---

## 2. The dataset

Source: **VizWiz-Captions**, photographs taken by blind users with crowd-written
descriptions.

| Split | Rows | With captions | Captions held | Trained/scored on |
|---|---|---|---|---|
| train | 23,431 | 22,866 | **100,575** | **22,866 images** |
| val | 7,750 | 7,542 | 33,145 | 7,542 scored (6,042 reported) |
| test | 8,000 | **0** | — | unusable for captions (withheld) |

- 565 training images were dropped: their captions were all rejected as
  unusable. **100% of such images are annotated `unrecognizable`** by
  crowdworkers, so dropping them removes the hardest cases from training.
- Mean **4.4 captions per image** (100,575 across 22,866).
- 16,272 training images carry visible text (`text_detected`).

**The test split contributes nothing.** VizWiz withholds test captions, so it
cannot be used for caption evaluation — only for the quality gate, whose labels
are published separately.

---

## 3. The training process — two passes

The key design decision: **the vision tower is frozen, so its output is computed
once and replayed.** This is what makes fine-tuning a 446 M model feasible on a
16 GB consumer GPU.

### Pass 1 — feature extraction (one-off, ~33 min)

```
image (384×384) → frozen ViT-L vision tower → 577 × 1024 features → disk
```

- Each image becomes a `577 × 1024` float16 array = **1.18 MB**.
- 22,866 training images → **25.8 GB** cache; 7,750 val → **8.5 GB**.
- 577 tokens = 24×24 patches + 1 CLS token, at BLIP-large's native 384 px.
- Decoding is the bottleneck (JPEG decode + resize), not the GPU: ~30 img/s.
- **Resumable**: writes only the filled prefix, so an interrupted pass restarts
  where it stopped.
- **Verified**: re-extracting images and comparing gives correlation ≥ 0.99.

### Pass 2 — text decoder fine-tuning (~31 min for 8 epochs)

```
cached features → text decoder → caption tokens → cross-entropy loss
```

Only the text decoder receives gradients. Each epoch is a full pass over all
22,866 images (each image drawing one random caption from its ~4.4).

---

## 4. Hyperparameters (shipped run)

| Setting | Value |
|---|---|
| Learning rate | **3e-5** (probe-selected; 1e-4 diverged) |
| Schedule | cosine with warmup, 429 warmup steps |
| Epochs | **8** |
| Batch size | 8 |
| Gradient accumulation | 4 |
| **Effective batch** | **32** |
| Optimiser | AdamW (via the training script) |
| Precision | **bf16 autocast** |
| Max caption length | 32 tokens |
| Total steps | 5,720 (715/epoch) |
| Checkpoint selection | per-epoch, on a held-out 1,500-image subset, beam-1 |

**bf16 is not optional.** It makes the cache pass 8.5× faster and beam-5 decoding
3.9× faster. fp32 runs the caching pass at roughly a third of the bf16 rate.

**Why lr 3e-5.** Two different learning rates are involved and should not be
conflated:

- A four-point probe (`lrprobe_*`) was run for the **LoRA** configuration
  (rank 16, 1 epoch) at 1e-4 / 3e-5 / 3e-4 / 1e-3. **1e-4 won** (val blend
  0.3174 against 0.3076 / 0.3099 / 0.3040).
- The **full-decoder** runs shipped here use **3e-5**, which is the value in
  `runs/blip_large_head/run_config.json`.

The 1e-3 probe is worth recording as the divergence signature: it produced the
**lowest training loss and the worst validation score** — memorisation, not
learning.

---

## 5. Hardware and software

| | |
|---|---|
| GPU | AMD Radeon RX 9070 XT (gfx1201, RDNA4), **15.92 GiB** |
| CPU / RAM | Ryzen 7 7800X3D / 31.1 GB |
| OS | Windows 11 |
| ROCm | 7.2 (`7.2.26024-f6f897bd3d`) |
| PyTorch | 2.9.1+rocmsdk20260116 |
| transformers | 5.5.4 |
| peft | 0.21.2 (used for the LoRA variants) |
| Python | 3.12 (`py -3.12`; cp312 wheels only) |

**Two hardware constraints that shaped the design:**

1. **MIOpen is disabled** on RDNA4 under this ROCm build (missing `rocrand`
   headers break MIOpen's dropout kernel). `configure_for_speed()` handles it.
   Consequence: **vision-tower backward passes are ~126× slower than forward**
   (1.0 img/s vs 126.5 img/s), so end-to-end training of the tower costs
   **6.5 h/epoch**. This is why the tower is frozen — not a modelling preference,
   a hardware limit.
2. **`bitsandbytes` does not install on Windows ROCm**, so QLoRA / 4-bit
   quantisation was unavailable. Full-precision adapters only.

---

## 6. Measured throughput

| Operation | Rate |
|---|---|
| Cache pass (bf16, batch 32) | 220.6 img/s (synthetic) / ~30 img/s (real, decode-bound) |
| Text-decoder training (bf16, full decoder) | 230.5 img/s @ batch 16 |
| Training epoch (22,866 images) | **~230 s** |
| Beam-1 decode | 147.5 img/s |
| Beam-5 decode, bf16 | 8.9 img/s |
| Beam-5 decode, fp32 | 2.3 img/s |
| Full-split evaluation (7,542 @ beam-5) | ~15 min |
| **Live inference** | cold start 6.8 s; **~586 ms/image** warm, beam-5 |

---

## 7. Evaluation protocol

Two distinct validation sets — **do not conflate them**:

| Set | Size | Beam | Purpose |
|---|---|---|---|
| Selection subset | 1,500 (seed 0) | 1 | per-epoch checkpoint choice only |
| **Reported split** | **6,042** | 5 | **every published number** |

The 6,042 are the validation images **not** used for selection. An earlier
version of this project reported on all 7,542, which fully contained the
selection subset and was therefore optimistically biased. The ranking did not
change after the correction, but the numbers moved down slightly.

Metrics: BLEU-1..4, CIDEr-D, ROUGE-L via a shared scorer, plus collapse
diagnostics (distinct tokens, top-token share, 4-gram match) and — the metric
that matters most — **VizWiz-VQA answer coverage**: does the caption contain the
answer a plurality of crowdworkers gave?

---

## 8. Reproducing it

```powershell
# 1. extract features (one-off, ~33 min, 25.8 GB)
py -3.12 -m train.finetune.cache --manifest data/manifests/vizwiz_train.jsonl `
  --image-root data/raw/vizwiz/images --model large --batch-size 32 --tag large

# 2. fine-tune the text decoder (~31 min)
py -3.12 -m train.finetune.train_blip --run-name blip_large_head --model large --tune head `
  --cache-index data/features/vizwiz_train_large_blip.jsonl `
  --cache-tensor data/features/vizwiz_train_large_blip.npy `
  --val-cache-index data/features/vizwiz_val_large_blip.jsonl `
  --val-cache-tensor data/features/vizwiz_val_large_blip.npy `
  --epochs 8 --lr 3e-5 --batch-size 8 --grad-accum 4 --bf16 `
  --eval-batch-size 32 --eval-beams 1 --limit-val 1500 --sample-val

# 3. evaluate on the reported split
py -3.12 -m train.finetune.rescore --all
```

---

## 9. Results (selection-free 6,042 images, beam-5)

| Model | Params trained | BLEU-4 | CIDEr-D | 4-gram | answer coverage | artifact |
|---|---|---|---|---|---|---|
| Zero-shot BLIP-base | 0 | 0.1557 | 0.3518 | 24.1% | 12.9% | 990 MB |
| E1 LoRA (q/k/v) | 1.8 M | 0.1887 | 0.4806 | 35.6% | 18.9% | **6.8 MB** |
| E3 full decoder (base) | 137.9 M | 0.1957 | 0.4916 | 38.3% | 20.8% | 526 MB |
| **BLIP-large (shipped)** | **142.6 M** | **0.2034** | **0.5399** | **40.3%** | **23.5%** | 544 MB |
| From-scratch ablation (full data) | 30.5 M | 0.0204 | 0.1507 | — | — | 353 MB |

---

## 10. What was tried and rejected — do not repeat these

Five interventions were measured and all failed. Full detail in
[captioner-experiments.md](captioner-experiments.md).

| Attempt | Outcome |
|---|---|
| Training on text-bearing images only | worse on every metric |
| Synthetic degradation (blur/dark/etc.) | regressed clean images, no gain on degraded |
| Training on VizWiz-VQA answers | template collapse at epoch 1 |
| 20-epoch schedule | worse than 8; best epoch was its first |
| Training the vision tower | infeasible: 6.5 h/epoch |

**The consistent finding: every data-side intervention failed, and scaling the
model worked.** The captioner is trained to the limit of what this data and
schedule provide.

---

## 11. Known limitation to state in any write-up

The captioner **describes, it does not read.** Measured against VizWiz-VQA:

| Question intent | Answer hit rate |
|---|---|
| colour | 51.9% |
| identify object | 42.8% |
| **reading / detail** (labels, titles, dates, prices) | **11.7%** |
| — expiry / date | 0/14 |
| — label / packaging text | 8.8% |
| — sign / screen text | 9.6% |

Reading belongs to the **OCR component**, not the captioner. See
[reading-gap-handoff.md](reading-gap-handoff.md).

The model also **hedges** on unusable photographs — 75.5% of the 208 images with
no usable caption versus 1.7% of recognizable ones — but it never *refuses*:
0 of 208 produced an empty caption. That judgement is the quality gate's job.
