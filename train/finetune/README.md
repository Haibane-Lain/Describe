# Fine-tuning BLIP on BLV photography

Replaces the from-scratch CNN + LSTM captioner, which is retained as the
reported ablation. The from-scratch model is not salvageable on this schedule:
it converged at BLEU-4 ≈ 1.3e-6 while emitting 126 of its 7,488 vocabulary
tokens and matching **zero** human 4-grams (see
[../docs/status-report.md](../docs/status-report.md) §3). A pretrained captioner
removes that failure mode outright and puts the remaining effort into domain
adaptation, which is the part that was actually missing.

## The one idea that makes this fast

`BlipForConditionalGeneration.forward` is two steps:

```python
image_embeds = self.vision_model(pixel_values=pixel_values).last_hidden_state
outputs = self.text_decoder(input_ids=..., encoder_hidden_states=image_embeds, labels=...)
```

and `BlipTextLMHeadModel.forward` takes `encoder_hidden_states` directly. So
with the vision tower frozen, **its 576 × 768 output per image is computed once
and replayed for every epoch**. Training never opens a JPEG or runs a
convolution again — the same cached-feature design as the ResNet pipeline, but
on a model that works.

```python
from train.finetune.base import build_model
model, report = build_model("Salesforce/blip-image-captioning-base", with_vision=False)
# forward takes cached features:
out = model(image_embeds=cached, input_ids=ids, attention_mask=mask, labels=labels)
```

`BlipCachedCaptioner` keeps the Hub checkpoint's parameter names
(`vision_model.*`, `text_decoder.*`), so an ordinary `load_state_dict` restores
the released weights with no key remapping. The vision tower is optional and is
dropped after caching.

## Procedure

Every command uses `py -3.12`. The ROCm wheels are cp312-only.

### 0. Sanity gate

```powershell
py -3.12 -m train.finetune.smoke_test_finetune     # 16 checks, ~1 min, no network
py -3.12 -m train.bench_blip --model base          # VRAM + throughput for this GPU
```

The smoke test runs on a randomly-initialised *tiny* BLIP, so it needs no
downloads. It asserts the cached forward equals the stock forward bit-for-bit,
that LoRA is an identity at initialisation, that a checkpoint round-trips, and
that batched generation matches the reference decoder.

### 1. Cache image features (once per corpus)

```powershell
# VizWiz train (~23.4k images)
py -3.12 -m train.finetune.cache `
  --manifest data/manifests/vizwiz_train.jsonl `
  --image-root data/raw/vizwiz/images `
  --model base --batch-size 32

# VizWiz validation (~7.75k images)
py -3.12 -m train.finetune.cache `
  --manifest data/manifests/vizwiz_val.jsonl `
  --image-root data/raw/vizwiz/images `
  --model base --batch-size 32

# verify a few cached rows against a fresh forward pass
py -3.12 -m train.finetune.cache `
  --manifest data/manifests/vizwiz_train.jsonl `
  --image-root data/raw/vizwiz/images --verify
```

Outputs `data/features/<manifest>_blip.jsonl` (the authoritative index, carrying
`feature_row` beside every manifest field) and `<manifest>_blip.npy` (float16
features). Extraction is resumable: an interrupted pass continues from the last
flush instead of restarting.

### 2. Fine-tune

```powershell
py -3.12 -m train.finetune.train_blip `
  --run-name blip_base_vizwiz `
  --model base --tune lora `
  --cache-index data/features/vizwiz_train_blip.jsonl `
  --cache-tensor data/features/vizwiz_train_blip.npy `
  --val-cache-index data/features/vizwiz_val_blip.jsonl `
  --val-cache-tensor data/features/vizwiz_val_blip.npy `
  --epochs 8 --lr 3e-5 --batch-size 16 --grad-accum 2 --bf16
```

`--tune` selects what is trained, cheapest first:

| Value | Trainable | Use when |
|---|---|---|
| `lora` (default) | ~3.9 M (2.7%) | Default. `peft` LoRA on the text decoder's attention and feed-forward projections. |
| `lora-manual` | ~3.9 M | Same, with the built-in implementation, if `peft` fights this transformers version. |
| `head` | ~138 M | LoRA plateaus below target; trains the whole text decoder and cross-attention, vision still frozen. |

None of these unfreeze the vision tower. Doing so needs raw images, loses the
cache, and costs several times more per step — and this project already has
evidence that unfreezing a pretrained encoder is not automatically better
(`flickr30k_e2e_safe` scored *worse* than the frozen model it replaced).

### 3. Evaluate

```powershell
py -3.12 -m train.finetune.evaluate_blip `
  --run-dir runs/blip_base_vizwiz `
  --cache-index data/features/vizwiz_val_blip.jsonl `
  --cache-tensor data/features/vizwiz_val_blip.npy `
  --out runs/blip_base_vizwiz/val_metrics.json `
  --pred-out runs/blip_base_vizwiz/predictions.jsonl
```

Scoring goes through `train.train.evaluate.score_predictions`, the same function
the from-scratch model is scored with, so the two are comparable. The metrics
file also carries **collapse diagnostics** — distinct tokens, top-token share,
and 2-/4-gram match rate — because those are the numbers that exposed the
from-scratch failure and BLEU-4 alone did not.

## Measured on this machine

AMD Radeon RX 9070 XT (gfx1201, 15.92 GiB), ROCm 7.2, PyTorch 2.9.1, MIOpen
disabled per [../docs/environment.md](../docs/environment.md). Each row is a
single configuration measured in its own process, warm-up excluded.

### The caching pass (vision tower, one-off)

| Batch | dtype | ms/step | img/s | peak GiB |
|---|---|---|---|---|
| 16 | bf16 | 74 | 216.7 | 1.65 |
| **32** | **bf16** | **145** | **220.6** | **2.39** |
| 64 | bf16 | 2020 | 31.7 | 3.87 |
| 32 | fp32 | 1231 | 26.0 | 2.34 |

**bf16 autocast is an 8.5× speedup on this pass** (220.6 vs 26.0 img/s) and is
now the default. Out-of-the-box fp32 would have put VizWiz alone at 25 minutes
and COCO + Flickr30k at over two hours; bf16 puts the whole 114k-image cache at
roughly **9 minutes**.

Note the batch-64 collapse. It is reproducible and it is not a memory limit
(3.87 GiB of 15.92). Keep batch ≤ 32.

### The training step (cached features, vision tower not loaded)

BLIP-base, 137.9 M-parameter text decoder:

| Mode | Batch | dtype | ms/step | img/s | peak GiB | trainable |
|---|---|---|---|---|---|---|
| full text decoder | 8 | bf16 | 51 | 158.1 | 2.76 | 137.9 M |
| full text decoder | **16** | **bf16** | **69** | **230.5** | **3.31** | 137.9 M |
| full text decoder | 32 | bf16 | 188 | 169.9 | 4.66 | 137.9 M |
| full text decoder | 16 | fp32 | 512 | 31.3 | 3.15 | 137.9 M |
| LoRA (query/key/value) | 8 | bf16 | 63 | 126.6 | 1.58 | 1.8 M |
| LoRA (query/key/value) | 16 | bf16 | 50 | 317.2 | 2.22 | 1.8 M |
| **LoRA (query/key/value)** | **32** | **bf16** | **85** | **375.5** | **3.52** | **1.8 M** |

Three conclusions that set the defaults:

1. **bf16 is 7.4× faster than fp32** (230.5 vs 31.3 img/s) at essentially the
   same memory. `--bf16` is the default recommendation.
2. **LoRA is ~1.6× faster than training the whole decoder** (375.5 vs 230.5
   img/s) *and* uses less memory at a comparable batch, because the backward
   pass only traverses the adapter. Its usual advantage — smaller optimizer
   state — is not the reason here; the vision tower is already frozen and
   cached. It also produces a 1.8 M-parameter checkpoint instead of a 138 M one,
   and cannot overwrite the pretrained language model.
3. **The best batch differs by mode** (16 for the full decoder, 32 for LoRA).
   Batch 8 is consistently worse than 16, so do not shrink below 16 to save
   memory that is not scarce.

Projected training time per epoch:

| Corpus | Images | Full decoder (230 img/s) | LoRA (375 img/s) |
|---|---|---|---|
| VizWiz train | 23,431 | ~1.7 min | **~1.0 min** |
| Flickr30k | 31,783 | ~2.3 min | ~1.4 min |
| COCO train | 82,783 | ~6.0 min | ~3.7 min |
| COCO + Flickr30k | 114,566 | ~8.3 min | ~5.1 min |

Add roughly 10–15% for validation and loading. A 12-epoch VizWiz LoRA run is on
the order of **15 minutes of GPU time**, so the planned grid — cache, stage 1,
stage 2, full validation, an LR probe, and a repeat at BLIP-large — fits inside
one day of wall clock rather than the week the plan budgeted.

### Why the plan no longer treats caching as a risk

Caching was flagged in the plan as a ~2-3 hour task based on extrapolating from
the ResNet pipeline. Measured, it is ~9 minutes, and the training step is
cheaper than the ResNet+LSTM decoder was. The budget freed by that is what pays
for BLIP-large and the end-to-end ablation.

## Verified end to end on real VizWiz data

A 600-image cache built from the first 600 rows of `vizwiz_train.jsonl` and
scored on 296 real validation images, LoRA (r=16, query/key/value), 3 epochs,
lr 1e-4, batch 16 × accum 2, bf16, beam-5. Run `blip_e2e_smoke`.

| Metric | Value | From-scratch model, for scale |
|---|---|---|
| BLEU-1 | 0.5541 | 0.434 |
| BLEU-4 | **0.1704** | **1.3e-6** |
| CIDEr-D | 0.3251 | 0.077 |
| ROUGE-L | 0.3482 | 0.320 |
| distinct tokens | **197** | **126** |
| top token `a` | 22.1% | 26.9% |
| bigram match rate | 90.8% | — |
| 4-gram match rate | **30.8%** | **0.0%** |
| mean caption length | 10.7 tokens | — |
| empty predictions | 0 | — |

Real outputs:

```
PRED: a computer screen with a message on it
REF : A computer screen shows a repair prompt on the screen.

PRED: a close up of a cell phone with a blue screen in the background
REF : A white sheet of paper has blurry text on it

PRED: a close up of a computer screen with a message on it
REF : A computer screen that has a window opened up on it and has a small arrow pointing north.
```

**Read this as a smoke test, not a result.** 296 images, one learning-rate
setting, no tuning, and BLIP's pretraining data includes COCO, which overlaps
the kind of desk-and-screen photography VizWiz is full of. What it establishes
is that the pipeline is real and that the failure mode which killed the
from-scratch model is gone: 30.8% of captions now contain a 4-gram that appears
in a human reference, against 0.0% before. The numbers that belong in the report
come from the full 23,431-image train split and the full 7,750-image validation
split.

The cache was also verified independently — re-encoding real images and
comparing against the stored features gave correlations of 0.9989–0.99988.

## Defects this work already caught

Recorded because each was silent, and each would have produced a plausible but
wrong number. Every one was found by running the code against real weights or
real images; none were visible from reading it.

| Defect | What it would have reported |
|---|---|
| `build_model` left `requires_grad=True` on every parameter | `--tune lora` would have trained the entire 138 M-parameter decoder while printing "LoRA" |
| `trainable_state_dict` filtered on `tensor.requires_grad` | Checkpoints would have saved **zero** tensors; `--resume` would silently restart from the pretrained weights |
| The image-token count was computed as `patches` | 576 instead of 577; the cache wrote fine and then failed to broadcast, caught on the first real VizWiz image |
| `apply_manual_lora` matched full paths while the targets were leaf names | Wrapped **0** layers; "LoRA training" would have been full fine-tuning of nothing |
| `peft` wrapped the whole wrapper module | Its `forward` swallowed `image_embeds` and `labels`; training crashed immediately |
| `evaluate_blip` could not read peft key shapes | Loaded **0** tensors and reported the **pretrained** model's metrics under the fine-tuned run's name |
| The cache verifier compared element-wise under bf16 | Reported FAIL on a correct cache (bf16 noise reaches ~3.2 per element with correlation 0.9989) |
| The cache verifier sampled the full manifest against a prefix cache | Reported FAIL on a correct cache |

The evaluation loader now refuses to continue when it loads zero tensors,
because "silently evaluate the wrong model" is a much worse outcome than a crash.

## Verified on the full splits

Everything above was measured on a 600/296-image smoke subset. These are the
real runs: the full **23,431-image** training split and the full **7,750-image**
validation split.

### The baseline that has to be beaten

BLIP is pretrained on COCO, and VizWiz photographs are largely everyday objects,
screens, and documents. So the unadapted model is already strong, and the
fine-tune has to be judged against it rather than against zero:

| Model | Images | BLEU-1 | BLEU-4 | CIDEr-D | ROUGE-L | distinct tokens | 4-gram match |
|---|---|---|---|---|---|---|---|
| **BLIP-base, zero-shot (no training)** | 500 | 0.5741 | **0.1682** | 0.2403 | 0.3536 | 343 | 27.2% |
| From-scratch CNN+LSTM (`vizwiz_ft2`) | 400 | 0.4340 | **1.3e-6** | 0.0768 | 0.3205 | **126** | **0.0%** |

The from-scratch row is reproduced by `train.finetune.diagnose_predictions` from
its saved prediction log, so both rows come from the same diagnostic code. It
matches the numbers in [../docs/status-report.md](../docs/status-report.md)
exactly - 126 distinct tokens, `a` at 26.9%, 4-gram match 0.0% - which is the
check that the shared scorer is behaving.

**This reframes the project's captioner result.** A pretrained model with no
fine-tuning at all reaches BLEU-4 0.168, which is at the proposal's 0.15-0.25
target. The deliverable is therefore not "fine-tuning reaches the target"; it is
"fine-tuning on blind-user photography moves the model on the dimensions that
matter for this application", and those dimensions have to be chosen honestly:

* **CIDEr-D and ROUGE-L** measure grounding and overlap; zero-shot is weak at
  0.2403 CIDEr-D.
* **Distinct tokens and 4-gram match** measure whether output is real language.
  Both models pass; the from-scratch model did not.
* **Behaviour on unrecognizable photographs** is what the application actually
  needs and what no aggregate captioning metric reports. See the quality/flaw
  splits produced by `diagnose_predictions`.

### Learning-rate probe

1 epoch, 2,000 rows sampled at random from the full manifest (not the leading
rows, which are annotation-ordered), validated on 800 images, LoRA r=16:

| lr | train loss | BLEU-1 | BLEU-4 | CIDEr-D | ROUGE-L | distinct | blend |
|---|---|---|---|---|---|---|---|
| 3e-5 | 3.8164 | 0.5710 | 0.1681 | 0.4138 | 0.3713 | 642 | 0.3076 |
| **1e-4** | 3.7213 | **0.5839** | **0.1750** | 0.4383 | 0.3801 | 723 | **0.3174** |
| 3e-4 | 3.5911 | 0.5801 | 0.1673 | 0.4467 | 0.3843 | 803 | 0.3099 |
| 1e-3 | 3.4925 | 0.5687 | 0.1639 | 0.4389 | 0.3838 | 815 | 0.3040 |

**1e-4**, degrading on both sides. The 1e-3 row is the classic divergence
signature: the *lowest* training loss and the *worst* validation score, which is
why checkpoint selection cannot use loss. Note also that distinct-token count
rises monotonically with the learning rate - more aggressive adaptation uses
more of the language - so it is a diagnostic, not a score to maximise.

## Operational note: the cache flush dominated the run time

The first full cache pass ran at **5.3 img/s**, about an hour and a half for
VizWiz train. The cause was not the GPU: `_flush` wrote the entire preallocated
array on every checkpoint, which grows to ~20 GB, so each flush rewrote tens of
gigabytes of mostly-unwritten zero rows.

Writing only the filled prefix (`features[:len(index)]`) and flushing on a
120-second timer as well as a row count took the same pass to **49 img/s** -
a **10x** improvement, 7 minutes instead of 64. The throughput still degrades as
the file grows (49 img/s early, 32 img/s late), which is the remaining write
cost, and is the reason the timer exists: an interrupt should cost seconds, not
the whole pass.

### Beam width dominates evaluation cost, and bf16 dominates beam width

Per-epoch validation at beam-5 over the full split is *not affordable*, and it
took a measurement to see why. Decoding 64 cached images on this GPU:

| beams | dtype | use_cache | img/s | full 7,750-image pass |
|---|---|---|---|---|
| 1 | bf16 | yes | **147.5** | ~1 min |
| 5 | bf16 | yes | **8.9** | ~15 min |
| 5 | fp32 | yes | 2.3 | ~57 min |
| 5 | fp32 | no | 1.7 | ~74 min |

Three consequences, all of which shape how runs are configured:

1. **bf16 autocast is a 3.9x speedup on beam search** (2.3 → 8.9 img/s). Without
   it, a single full-split beam-5 evaluation is an hour, so it can only be run
   once. With it, it is 15 minutes. `decode_cached` therefore enables bf16 by
   default and `--no-bf16` exists only to quantify the difference.
2. **Per-epoch selection must use a cheaper setting.** Greedy decoding runs at
   147 img/s, so checkpoint selection uses beam-1 on a fixed 1,500-image subset
   (~10 s/epoch) and the full beam-5 pass happens once, on the chosen
   checkpoint, at the end.
3. **`use_cache` is working and matters** - disabling it costs 35%, so the KV
   cache is not being silently bypassed. Worth checking, because "the decoder is
   inexplicably slow" has several causes on this backend and this rules one out.

Training itself is not the bottleneck: a 22,866-image epoch takes **~79 seconds**
at LoRA batch 32, or 4,500 tokens/s.

### E1: LoRA r=16 on the full training split

12 epochs, lr 1e-4, batch 32, bf16, per-epoch selection by beam-1 on a fixed
1,500-image subset:

| epoch | train loss | BLEU-1 | BLEU-4 | CIDEr-D | ROUGE-L | distinct | 4-gram | blend |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.4116 | 0.6023 | 0.1497 | 0.4275 | 0.3720 | 1271 | 23.9% | 0.3015 |
| 2 | 3.1430 | 0.5856 | 0.1407 | 0.4255 | 0.3669 | 1324 | 23.7% | 0.2886 |
| 3 | 3.0809 | 0.6091 | 0.1566 | 0.4443 | 0.3768 | 1210 | 25.5% | 0.3086 |
| 6 | 2.9939 | 0.6090 | 0.1571 | 0.4598 | 0.3793 | 1226 | 26.3% | 0.3081 |
| 9 | 2.9473 | 0.6008 | 0.1549 | 0.4666 | 0.3786 | 1255 | 27.0% | 0.3042 |
| **11** | 2.9391 | 0.6055 | **0.1594** | 0.4707 | 0.3798 | 1248 | 26.7% | **0.3089** |
| 12 | 2.9385 | 0.6050 | 0.1588 | 0.4693 | 0.3792 | 1254 | 26.6% | — |

Training loss falls monotonically the whole way (3.41 → 2.94) while validation
quality **plateaus from about epoch 6**. That gap is the overfitting signature
this project has seen before, and it is why selection is on the metric rather
than on the loss: the best epoch by loss is the last one, the best by quality is
epoch 11, and the difference between them is small but real.

CIDEr-D, however, keeps improving to the final epoch (0.4275 → 0.4707), so the
model is still getting *more grounded* after BLEU stops moving. BLEU-4 at
beam-1 is a noisy statistic - it moves ±0.01 between adjacent epochs on identical
data - so per-epoch differences smaller than that should not be read as
improvements.

## Live inference and its two-path verification

`predict.py` turns a JPEG into a caption with no precomputed features, which is
what the backend needs and what nothing else in the package does.

```python
from train.finetune.predict import BlipCaptioner
c = BlipCaptioner.from_run("runs/blip_base_lora", model="base")
c.caption("photo.jpg")            # path, PIL.Image, or raw bytes
c.caption_batch(paths)            # batched
```

`verify_live.py` asserts the live path and the cached-feature path agree, in two
parts, because they are two different claims:

| check | what it proves | result |
|---|---|---|
| same fp16 features fed to both paths | prompt tokens, EOS choice, beam config, token trimming are identical | **20/20 identical** |
| real JPEG in, end to end | preprocessing, vision tower, and caching pipeline agree | **19/20** — the one difference is the fp16 feature cache vs an fp32 re-encode, which moves the encoder by ~0.03% and flips a beam |

Live latency on this GPU: **~750 ms/image** at beam-5 fp32, ~580 ms at bf16,
against ~250-430 ms for the cached path (which excludes image decode). That is
the per-stage figure proposal 5C asks for.

Two findings from building this, both of which were silent wrong answers rather
than errors:

1. **`generate()` does not switch the model to eval mode.** Decoding without
   `model.eval()` runs with dropout active. BLIP's own configs set both dropout
   probabilities to 0.0, so no project run surfaces it - but the cached path and
   the live path disagreed on 7 of 20 real images until `decode_cached` called
   `eval()` explicitly.
2. **A checkpoint loaded into a model with no adapter modules is a no-op that
   reports success.** Without `prepare_model_for_checkpoint` first, the adapter
   keys were matched against the *base* weights of the same shape, so every
   tensor was written back onto itself: `144 tensors loaded`, and the model
   decoded as the unadapted base model. `load_trainable_state_dict` now raises
   if the checkpoint carries adapter tensors and none were applied, and the
   verify harness rejects a "successful" load that contains no `lora_` keys.

## Final comparison — all four models on the same full split

Every row: the full **7,542 caption-bearing VizWiz validation images** (of 7,750;
208 have no captions), **beam-5**, bf16, scored by the shared
`score_predictions`. Identical images, identical settings, one scorer.

| Model | Params trained | BLEU-1 | BLEU-4 | CIDEr-D | ROUGE-L | distinct tokens | 4-gram match | checkpoint |
|---|---|---|---|---|---|---|---|---|
| Zero-shot BLIP-base (no training) | 0 | **0.5621** | 0.1572 | 0.3555 | 0.3607 | 1612 | 24.2% | 990 MB |
| E2 LoRA +dense | 3.9 M | 0.5989 | 0.1749 | 0.4121 | 0.3909 | 1925 | 27.3% | ~4 MB |
| E1 LoRA (query/key/value) | 1.8 M | 0.6059 | 0.1897 | 0.4828 | 0.3975 | 2409 | 35.8% | **6.8 MB** |
| **E3 full text decoder** | 137.9 M | 0.5996 | **0.1958** | **0.4945** | **0.4005** | **2591** | **38.3%** | 526 MB |
| From-scratch CNN+LSTM (ablation) | 30 M | 0.4340 | **1.3e-6** | 0.0768 | 0.3205 | **126** | **0.0%** | — |

The from-scratch row is from `runs/vizwiz_ft2` on 400 images - **the only full
prediction log that run kept**, so it is not strictly like-for-like on image
count. Its collapse is qualitative, not marginal: 126 distinct tokens against
2,591, and a 4-gram match rate of exactly zero. Re-running that evaluation on
all 7,542 images would be more rigorous and is cheap now that the code exists;
it is worth doing before publication.

### What the numbers actually say

**Fine-tuning helps, and the improvement is larger than BLEU-4 suggests.**
Against zero-shot, E3 gains +24% BLEU-4, **+39% CIDEr-D**, +11% ROUGE-L, and
+14 points of 4-gram match. BLEU-1 is slightly *lower* (0.5996 vs 0.5621 is
higher; E1's 0.6059 is highest) - unigram overlap was never the problem, which
is the same lesson the from-scratch collapse taught from the other direction.

**The full decoder beats LoRA, so the comparison the plan set out to make came
out in favour of full fine-tuning.** E3 leads E1 on BLEU-4 (0.1958 vs 0.1897),
CIDEr-D (0.4945 vs 0.4828), ROUGE-L, distinct tokens (2591 vs 2409) and 4-gram
match (38.3% vs 35.8%). The margin is real but modest, and it is confounded:
E3 ran at lr 3e-5 and E1 at 1e-4, because a full decoder at a LoRA-tuned rate
diverges. **This is not a clean parameter-efficiency ablation**, and the table
should say so. The practical trade is 526 MB against 6.8 MB - a 77x smaller
artifact for about 3% less BLEU-4, which is a defensible deployment choice
either way.

**Widening LoRA made it worse.** E2 adds the attention-output and both MLP
projections (3.9 M trainable, up from 1.8 M) and lands *below* narrow LoRA on
every quality metric, while costing slightly more per epoch (85 s vs 79 s) and
producing a checkpoint roughly twice the size. The attention projections were
already the right place to adapt. This is a negative result worth reporting:
"more LoRA" is not "better LoRA".

**Note the checkpoint column.** E3 stores 526 MB of decoder weights; E1 stores
6.8 MB of adapter. For a mobile backend serving one model that difference is
immaterial. For a report that must ship artifacts, it is not.

### Where captioning breaks — the number this project exists for

E3's full-split predictions, split by the annotation the quality gate is
*trained to predict* (`quality.unrecognizable`), joined from `vizwiz_val.jsonl`:

| Bucket | n | BLEU-1 | BLEU-4 | CIDEr-D | mean length | 4-gram match |
|---|---|---|---|---|---|---|
| all | 7542 | 0.5996 | 0.1958 | 0.4945 | 12.0 | 38.3% |
| recognizable | 6999 | **0.6313** | **0.2089** | **0.5132** | 11.9 | 40.7% |
| **unrecognizable** | 543 | **0.2442** | **0.0500** | **0.2706** | 13.5 | **7.6%** |
| no flaw | 2345 | 0.6574 | 0.2260 | 0.6286 | 11.9 | 46.8% |
| has a dominant flaw | 5197 | 0.5737 | 0.1822 | 0.4411 | 12.0 | 34.5% |
| text detected | 5536 | 0.6279 | 0.2113 | 0.4993 | 11.8 | 41.4% |
| no text detected | 2006 | 0.5256 | 0.1553 | 0.4533 | 12.4 | 29.8% |

By dominant flaw, worst first:

| Flaw | n | BLEU-1 | BLEU-4 | CIDEr-D | 4-gram match |
|---|---|---|---|---|---|
| obstructed | 51 | 0.4643 | 0.1231 | 0.3752 | 17.6% |
| too dark | 129 | 0.3978 | 0.1315 | 0.4058 | 24.8% |
| too bright | 135 | 0.4769 | 0.1485 | 0.4519 | 31.1% |
| blur | 2240 | 0.5502 | 0.1750 | 0.4329 | 33.7% |
| rotated | 480 | 0.6122 | 0.1768 | 0.4413 | 32.5% |
| poor framing | 2156 | 0.6122 | 0.1986 | 0.4819 | 37.0% |

Three things follow, and they are the most decision-relevant results here:

1. **Captioning collapses on unusable photographs**: BLEU-4 0.0500 against
   0.2089, and 4-gram match 7.6% against 40.7%. A four-fold degradation, on the
   exact images the application is built to intercept. This is the empirical
   justification for the quality gate being the product's central mechanism
   rather than a preprocessing nicety - and it is measured on the captioner's
   own outputs, not asserted.
2. **The dark, bright and obstructed buckets are the worst**, at 0.1231-0.1485
   BLEU-4, and they have only 51-135 images each. The ordering is consistent
   with the retake rules in proposal 5B - those are the conditions that already
   map to "ask for another photograph" - so the gate's action table and the
   captioner's actual weakness agree.
3. **Rotated images caption *better* than average** (0.1768, close to the
   0.1958 overall). That supports the proposal's decision to auto-correct
   rotation and never ask for a retake: the model is not especially harmed by
   it.

**Honest reading.** The mean caption length on unrecognizable images is 13.5
tokens against 11.9 on recognizable ones, and in the E1 run 11.3 in both. The
model does not go quiet on bad input - it keeps talking at slightly greater
length with much worse grounding, which is precisely the failure mode the
project's asymmetric-cost argument is about: a confident, fluent, wrong
description that the user cannot check.

## Probe: would adding Flickr30k (or COCO) to training help?

Run to settle this with a number instead of an argument. 2,000 Flickr30k images
sampled at random across the manifest (seed 0), beam-5, same scorer. **Both**
zero-shot and the fine-tuned E3 were evaluated on the *same* images, so the
comparison separates "is Flickr30k hard?" from "did the fine-tune forget it?"

| On 2,000 Flickr30k images | BLEU-1 | BLEU-4 | CIDEr-D | ROUGE-L | distinct | 4-gram | mean length |
|---|---|---|---|---|---|---|---|
| Zero-shot BLIP-base | **0.6462** | **0.2072** | **0.5468** | **0.3894** | 927 | 32.1% | 9.6 |
| E3 (fine-tuned on VizWiz) | 0.6025 | 0.1969 | 0.4078 | 0.3787 | **1252** | **42.4%** | **13.2** |

**Answer: no, adding Flickr30k would not improve VizWiz performance.** Zero-shot
BLIP already reaches CIDEr-D 0.5468 on Flickr30k - higher than any model here
reaches on VizWiz. There is no headroom for more Flickr30k data to fill.

**But the fine-tune measurably changed general-domain behaviour.** CIDEr-D fell
25% (0.5468 → 0.4078) and ROUGE-L fell 2.8%, while distinct tokens rose 35% and
mean length rose from 9.6 to 13.2 tokens. Reading the actual captions shows why
this is not simple forgetting:

```
REF   : A child in a pink dress is climbing up a set of stairs in an entry way .
  zero: a little girl standing in front of a chicken coop
  E3  : a little girl in a pink dress standing in front of a chicken house.

REF   : A girl is on rollerskates talking on her cellphone standing in a parking lot .
  zero: a woman walking down the street while talking on her cell
  E3  : a woman in a blue shirt and black shorts talking on a cell phone.

REF   : A man with black hair sits in a restaurant with a glass of beer .
  zero: two men sitting at a table in a restaurant
  E3  : a man and a woman sitting at a table in a restaurant.
```

E3 is **not worse** on these - it recovers "pink dress" that zero-shot missed and
matches the reference more closely in three of the six samples. What changed is
*style*: the fine-tune moved the model from short generic captions (9.6 tokens)
toward longer specific ones (13.2), which is exactly the behaviour that helps on
VizWiz and exactly what CIDEr-D penalises on Flickr30k's longer, differently-worded
references.

**Implications.**

1. **The VizWiz gain was a domain-style shift, not new capability.** That is
   consistent with the +24.7% BLEU-4 / +39% CIDEr-D measured earlier and with the
   val curve plateauing at epoch 6.
2. **A length-controlled comparison would be the honest way to report this.**
   CIDEr-D and BLEU are both sensitive to length, and the fine-tune changed
   length by 37%. The cross-domain numbers above should not be read as a pure
   quality regression.
3. **If a general-purpose captioner is ever wanted alongside the VizWiz one,
   Flickr30k should be added as rehearsal data** to restore the concise style.
   That is a different goal from improving this product, and it is not worth the
   GPU time now.

**Two caveats on this probe, both of which limit how much it can prove.**

- **`flickr30k.jsonl` has no test split.** Every one of its 31,783 rows is
  labelled `split: train`, so there is no clean Flickr30k holdout in this repo.
  The probe used a random sample, which is representative but not a held-out set.
- **BLIP was pretrained on Flickr30k**, so these numbers are optimistic for both
  models and partly measure style alignment with BLIP's pretraining rather than
  generalization. That weakens the absolute values but not the *comparison*,
  which is between two models on identical images.

## What users actually ask about, and whether captions answer it

Aggregate BLEU cannot express the thing this project cares about: a caption can
score well on n-gram overlap while omitting the one detail the photographer
needed. VizWiz-VQA supplies the instrument — the same photographs, the same
users, each with a spoken question and ten crowdsourced answers.

The official annotations are at
`https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip` (1.6 MB,
not the 177 MB `API.zip`), and they **do contain a training split**:

| Split | Annotations | Answerable |
|---|---|---|
| train | 20,523 | 14,991 |
| val | 4,319 | 2,934 |
| test | 8,000 | (answers withheld) |

**32% of images carry an unanswerable question** — the user's own photograph did
not support the question they asked. That is the empirical case for the quality
gate existing, from an independent source.

### Answer coverage — a metric BLEU cannot see

`train.finetune.eval_vqa` asks a blunt question of each caption: *does it contain
the answer a plurality of crowdworkers gave?*

| Model | BLEU-4 | CIDEr-D | **Answer coverage** |
|---|---|---|---|
| Zero-shot BLIP-base | 0.1572 | 0.3555 | 16.1% |
| E1 LoRA | 0.1897 | 0.4828 | 23.5% |
| **E3 full text decoder** | **0.1958** | **0.4945** | **25.9%** |
| E4 text-aware (text-bearing images only) | 0.1891 | 0.4746 | 24.5% |

Fine-tuning lifts answer coverage from 16.1% to 25.9% (**+60% relative**), a
much larger relative gain than BLEU-4 shows (+25%). The metric is informative,
and it ranks the models the same way the captioning metrics do.

### Where it fails is the detail, not the object

From `runs/blip_base_head/vqa_answer_coverage.json`:

```
Q: Can you tell me what this medicine is please?   A: 'night time'
   -> "a person is holding a bottle of medicine in their hand."
      Names the object, never reads it.

Q: What is the title of this book?                 A: 'dog years'
   -> "a magazine is laying on top of a table."
      Gets the category wrong and says nothing about the title.
```

The model reliably identifies and describes **objects** — that is the zero-shot
BLIP strength it inherited. What it cannot do is report the **specific detail**
on an object: the name, the title, the flavour, the date. That is the gap a
dataset could close, and it is a reading problem rather than an object-vocabulary
problem.

### The object audit, and a correction to it

`train.finetune.audit_objects` compares what users ask about (20,523 training
questions) against what the captioner names (7,542 val captions):

| Intent | Questions | Share |
|---|---|---|
| identify | 5,515 | 26.9% |
| reading / text | 3,504 | 17.1% |
| food | 3,450 | 16.8% |
| colour | 2,204 | 10.7% |
| count | 140 | 0.7% |
| medicine | 124 | 0.6% |

The token-level audit reported a **+10.7 percentage-point gap on "colour"**
(11.0% of questions, 0.2% of captions named the token "color"). **That figure is
an artifact and the conclusion it suggested is wrong.** The audit counts the
literal token, and a caption that says *"a pair of white tennis shoes"* never
contains the word "color". Measuring properly:

| Measure | Value |
|---|---|
| Captions naming any colour word | 3,242 / 7,542 (**43.0%**) |
| Val questions whose majority answer is a colour | 314 |
| ...where that colour appears in the caption | 195 (**62.1%**) |

**The model is not colour-weak.** The lesson generalises: a token-frequency
audit is only valid for terms that appear *in* a caption. It works for nouns
("expiration" 1.1% asked vs 0.1% named, "title" 0.8% vs 0.0%) and is invalid for
attributes that surface as a *different* word. Attribute questions must be
scored against answers, not against term frequency.

### E4 is a negative result

Restricting training to the 16,272 text-bearing images (H1/H2 in the plan) did
**not** help: BLEU-4 0.1891 vs E3's 0.1958, CIDEr-D 0.4746 vs 0.4945, answer
coverage 24.5% vs 25.9%, and 4-gram match 36.8% vs 38.3% — worse on every
metric, including the answer coverage it was built to improve. Training on text
images does not teach reading; it just removes 6,594 images' worth of signal.

**E3 stands as the best model.** Its selection-score advantage (0.3246 vs
0.3195) translated to the full split and to answer coverage, which is a useful
confirmation that per-epoch selection on a 1,500-image beam-1 subset is a usable
proxy.

## Exact image counts

Citable numbers, confirmed from the manifests and from the `train_rows` written
into every `run_config.json` at launch.

**Fine-tuning:**

| | Images |
|---|---|
| `vizwiz_train.jsonl` total rows | 23,431 |
| — dropped: no captions (all rejected) | −565 |
| **Trained on** | **22,866** |

Of those 22,866, **16,272 have `text_detected = true`** and 6,594 do not. That
split is what the E4 text-aware variant used, and it is the reason E4 exists as
a negative result rather than a hypothesis.

**Evaluation:** `vizwiz_val.jsonl` has 7,750 rows; 208 have no captions, so
**7,542 are scorable**. Two different validation sets are involved and must not
be conflated:

| | Images | Beam | Purpose |
|---|---|---|---|
| Per-epoch checkpoint selection | 1,500 (random, seed 0) | 1 | cheap enough to run every epoch |
| Reported metrics | **7,542** (all scorable) | 5 | the numbers in the tables |

All four models in the comparison were scored on the identical 7,542.

**Question answering:** the VizWiz-VQA annotations add supervision on the same
images — 20,523 train / 4,319 val, of which 14,991 train and 2,934 val are
answerable. **32% of val images carry an unanswerable question.**

**The test split contributes nothing to training or caption evaluation.**
`vizwiz_test.jsonl` has 8,000 images and 0 captions — VizWiz withholds them. It
is usable only for the quality gate, whose labels are published.

**Per-run configuration:**

| Run | Tune | Train images | Epochs | Batch × accum | lr | Best epoch |
|---|---|---|---|---|---|---|
| E1 LoRA (q/k/v) | `lora` | 22,866 | 12 | 32 × 1 | 1e-4 | 11 |
| E2 LoRA +dense | `lora` | 22,866 | 8 | 32 × 1 | 1e-4 | 3 |
| E3 full decoder | `head` | 22,866 | 8 | 16 × 2 | 3e-5 | 7 |
| E4 text-aware | `head` | 16,272 | 8 | 16 × 2 | 3e-5 | 6 |

Effective batch 32 throughout; captions truncated at 32 tokens. The learning-rate
probe used 2,000 sampled train / 800 sampled val images for 1 epoch at four rates.

**One comparison is still not like-for-like.** Every BLIP number above is on
7,542 images; the from-scratch ablation's only surviving prediction log
(`runs/vizwiz_ft2/preds_vizwiz_val.jsonl`) covers **400**. Its collapse is
qualitative (126 vs 2,591 distinct tokens, 0% vs 38.3% 4-gram match), so the
conclusion holds — but the row needs its image count stated or the evaluation
re-run.

## Dataset strategy: what the measurements say

Written after auditing available corpora, so the next dataset decision is made
against evidence rather than intuition.

### Verified available

| Dataset | Verified | Shape | Maps to | Verdict |
|---|---|---|---|---|
| VizWiz-Captions | on disk | 22,866 train / 7,542 val captioned | in-domain | baseline |
| **VizWiz-VQA** | official `Annotations.zip`, 1.6 MB | 20,523 train / 4,319 val QA | detail extraction; answer coverage | **best available lever** |
| TextCaps | `HuggingFaceM4/TextCaps` (CC-BY-4.0) | ~28k, reading-focused | text comprehension | candidate |
| COCO-Text | `howard-hou/COCO-Text` | 63,686 images, 173,589 text instances | OCR-adjacent | candidate |
| Open Food Facts | `openfoodfacts/product-database` (AGPL) | 1.7M products, **metadata only** | food packages | needs a caption-generation pipeline |
| Flickr30k / COCO | on disk | 31,783 / 123,287 | general captioning | **eliminated by measurement** |

Two negatives worth recording: the ~15 "medication" datasets on Hugging Face are
**text-only clinical benchmarks with no images**, and `flickr30k.jsonl` in this
repo **has no test split** (all 31,783 rows are `split: train`), so there is no
clean Flickr30k holdout here.

### Why object-category datasets are the wrong purchase

The audit shows the model **already names objects well**: 43% of captions name a
colour, colour answers are covered 62.1% of the time, and object identification
is the single largest question intent (26.9%) with no measured deficit. Buying
fine-grained object or attribute data would spend effort on the model's strength.

**The measured gap is detail-on-an-object, which is a reading problem:** a
medicine bottle described as *"a bottle of medicine"* without its name, a book
described as *"a magazine"* without its title. Answer coverage is 25.9% and the
failures are concentrated there.

### Ranked, by expected value

1. **VizWiz-VQA Q&A augmentation** — 14,991 answerable training pairs on the same
   photographs, the same photographers, the same objects. Convert
   `(image, question, answer)` into caption supervision by template. Highest
   relevance of anything available, and no new domain shift. The failure mode to
   guard against is the model learning to emit answer *fragments* instead of
   captions, so it runs as an ablation gated on answer coverage **and** BLEU.
2. **Synthetic degradation** — blur, dark, bright, rotation, occlusion applied
   during caching. The only intervention that targets the 4× degradation on
   unusable photographs (BLEU-4 0.0500 vs 0.2089), because no public corpus
   contains natural blind-user blur. Aimed at the gate's domain, not the
   captioner's metrics.
3. **TextCaps rehearsal** — targets the reading capability directly. Mix roughly
   1:2 with VizWiz rather than training on it alone.
4. **Open Food Facts** — real value for food packages (16.8% of question
   intent), but captions would have to be generated from product metadata and
   are not human-written. Stretch item, and must be labelled as synthetic.

**Do not** add Flickr30k or COCO: zero-shot BLIP already reaches CIDEr-D 0.5468
on Flickr30k, higher than any model here reaches on VizWiz, so there is no
headroom for more of that data to fill.

### VizWiz-VQA cannot be converted into caption training data

The obvious idea is to turn `(image, question, answer)` into caption supervision.
Inspecting the data kills it, and the numbers are worth recording so nobody
retries it:

| | Value |
|---|---|
| answerable training pairs | 14,991 |
| QA pairs **per image** | **1** (no joining problem) |
| answer length, mean / median | 1.66 / **1** word |
| answers ≤ 2 words | **86.3%** |
| most common answers | `no`, `yes`, `white`, `grey`, `black`, `blue` |
| real VizWiz caption length, mean / median | **11.50 / 10** words |

Templating these produces captions like *"a photo of no"*, *"a photo of white"*,
*"a photo of t36m"* against human captions averaging 11.5 words. Training on
that would teach answer fragments and destroy the fluency that took work to
reach. **The dataset is an excellent evaluation instrument and a poor training
corpus.**

### The one salvageable slice, and it is the interesting one

Filtering to **informative** answers — at least two words and not a bare colour —
keeps **8,742 pairs (58.3% of answerable)**. Those are not fragments; they name
the actual thing:

```
What's the name of this product?   -> basil leaves
What is in this can?               -> coca cola
What is this item?                 -> solar garden light
This item.                         -> monitor cleaning kit
What kind of drink is this?        -> dr pepper
How much money is on the table?    -> 80 cents
```

This is exactly the "specific objects that benefit our userbase" target: product
names, brands, and quantities on the objects a blind user is holding. It is the
only slice of any available dataset that supplies detail-on-an-object
supervision, and it is **8,742 images already on disk with features cached**.

It carries a real risk, which is why it is an ablation and not a plan of record:
answer-derived captions are *responses*, not descriptions, and could teach
"a photo of {noun}" as a degenerate template. **The test is whether it improves
answer coverage (25.9% baseline) without breaking fluency metrics** — a
template collapse would show up immediately as a BLEU-4 and 4-gram-match drop.
Mix at roughly 1:3 with real captions rather than training on it alone.

### Two corrections to earlier reasoning in this document

Both were caught by inspecting data rather than by argument, and both would have
wasted a training run:

1. **"The model is colour-blind."** False. The token audit reported a +10.7 point
   gap on "colour" because captions say *"white tennis shoes"*, never the word
   "color". Measured against answers, colour coverage is **62.1%**.
2. **"VizWiz-VQA is the best available training lever."** False. 86.3% of its
   answers are fragments; it is an evaluation instrument. Only the 8,742
   informative answers are usable, and only as a mixed-in ablation.

The general lesson: **a token-frequency audit is valid only for terms that appear
literally in a caption**, and a dataset's fitness must be checked by looking at
its rows, not at its description.

## Final comparison — six models, selection-free split, beam-5

**Every figure below is computed on the 6,042 validation images that were NOT used
for per-epoch checkpoint selection.** Earlier revisions of this document reported
on all 7,542 scorable images, which fully contained the 1,500-image selection
subset and was therefore optimistically biased. See "Evaluation hygiene" below.

`train.finetune.rescore --all` reproduces this table from the existing prediction
files, CPU-only, with no re-inference.

| Model | Params trained | BLEU-1 | BLEU-4 | CIDEr-D | ROUGE-L | 4-gram | **answer coverage** | checkpoint |
|---|---|---|---|---|---|---|---|---|
| Zero-shot BLIP-base | 0 | 0.5600 | 0.1557 | 0.3518 | 0.3595 | 24.1% | 12.9% | 990 MB |
| E2 LoRA +dense | 3.9 M | 0.5880 | 0.1740 | 0.4106 | 0.3763 | 27.3% | 15.7% | ~4 MB |
| E1 LoRA (q/k/v) | 1.8 M | 0.6051 | 0.1887 | 0.4806 | 0.3965 | 35.6% | 18.9% | **6.8 MB** |
| E4 text-aware | 137.9 M | 0.6088 | 0.1942 | 0.4696 | 0.3951 | 36.7% | 19.5% | 526 MB |
| E3 full decoder (base) | 137.9 M | 0.5986 | 0.1957 | 0.4916 | 0.3994 | 38.3% | 20.8% | 526 MB |
| **A5 BLIP-large, full decoder** | **142.6 M** | **0.6122** | **0.2034** | **0.5399** | **0.4076** | **40.3%** | **23.5%** | 544 MB |
| From-scratch CNN+LSTM (ablation) | 30 M | 0.4340 | **1.3e-6** | 0.0768 | 0.3205 | **0.0%** | — | — |

### Evaluation hygiene: a flaw this project had, and the fix

Per-epoch checkpoint selection needs a cheap validation pass, so it used a
1,500-image subset at beam-1. Reporting used the full 7,542 at beam-5. **The
selection subset was entirely contained inside the reported set** — the model was
partly chosen *because* it scored well on images that then appeared in the
results.

The fix is a deterministic, disjoint split, defined once in `train/finetune/holdout.py`:

| Set | Size | Use |
|---|---|---|
| selection | 1,500 (seed 0, verified unbiased: mean position 3,783 of 7,542) | checkpoint choice only |
| **reported** | **6,042** | every published number |

**The correction is small and the ranking is unchanged**, which is the useful
outcome: it means the comparisons were not artefacts of selection. Every model
loses roughly 0.001–0.005 BLEU-4 and 2–6 points of answer coverage. BLIP-large
remains best on every metric, and the headline conclusions survive:

- BLIP-large over zero-shot: BLEU-4 **+30.6%**, CIDEr-D **+53.5%**, 4-gram match
  +16.2 points, answer coverage **12.9% → 23.5%**.
- The fine-tuned models are all far above the from-scratch ablation, whose 4-gram
  match is exactly zero and which emits 126 distinct tokens against BLIP-large's
  2,891.

**BLIP-large has not plateaued.** Per-epoch selection score over its 8 epochs:
0.3368, 0.3253, 0.3360, 0.3379, 0.3339, 0.3427, 0.3431, **0.3439** — still rising
at the final epoch, with training loss still falling (2.676 → 1.939). The base
model peaked at epoch 6 of 8; this run ended on a schedule, not on convergence.

### Length is not a confound — measured

`train.finetune.length_control` rescores under a relaxed brevity penalty
(exponent 1.0 = standard BLEU, 0.0 = penalty removed). A gap that closes as the
penalty relaxes is style; one that survives is content:

| Model | exp 1.0 | exp 0.5 | exp 0.0 | hyp/ref length |
|---|---|---|---|---|
| zero-shot | 0.1897 | 0.1945 | 0.1995 | **0.95×** |
| E1 LoRA | 0.2403 | 0.2403 | 0.2403 | 1.01× |
| E3 full decoder | 0.2504 | 0.2504 | 0.2504 | 1.04× |
| E4 text-aware | 0.2491 | 0.2491 | 0.2491 | 1.02× |

**The fine-tuned models' scores do not move at all.** Their length ratio is
already ~1.0, so the brevity penalty is inert for them. It was mildly *helping*
zero-shot, whose ratio is 0.95×. **The E1 → E3 gap is content, not style.**

## Synthetic degradation: method

The captioner's measured weakness is not object vocabulary — it names objects
well — it is **unusable photographs**: on the 543 unrecognizable validation
images captioning collapses by roughly 4× (BLEU-4 0.0500 against 0.2089; 4-gram
match 7.6% against 40.7%). No public corpus contains natural blind-user blur, so
it has to be generated.

`train/finetune/degrade.py` applies the flaw types VizWiz actually annotates,
weighted by their real frequencies:

| Flaw | VizWiz positives | Sampling share |
|---|---|---|
| blur | 2,240 | 42.6% |
| poor framing | 2,156 | 41.0% |
| rotated | 480 | 9.1% |
| too bright | 135 | 2.6% |
| too dark | 129 | 2.5% |
| obstructed | 116 | 2.2% |

Verified: the observed mix tracks the prior (blur 43.1% vs 42.6%), degradation is
**deterministic per image** (seeded on file name, so a cache rebuilds
byte-for-byte), the `--degrade-probability` is honoured exactly, and each
transform produces the expected measurable change:

| Transform | Effect |
|---|---|
| dark | mean luminance 44.5 → **18.9** |
| bright | mean luminance 44.5 → **71.2** |
| blur | edge energy 0.58 → **0.35** |
| framing | edge energy 0.58 → **0.32** |

Two design decisions worth recording:

- **Degradation is applied to the image, before the vision tower.** It has to be:
  the cached tensor *is* the tower's output, so degrading features directly would
  model something the tower never produces.
- **Motion blur is built by averaging shifted copies**, not with
  `ImageFilter.Kernel`. PIL's C filter accepts only a narrow set of kernel sizes
  and rejects larger odd ones with "bad kernel size", so a 7×7 or 9×9 motion
  kernel is not reliably available. Averaging shifted copies has no size limit and
  looks more like hand-shake.

### The mixed cache

Training on degraded images alone would trade clean performance for degraded
performance, which is not a result. The acceptance criterion requires that
recognizable-image metrics **hold** while unrecognizable ones improve, so the
clean and degraded caches are concatenated into one:

| | Rows |
|---|---|
| clean cache | 23,431 |
| degraded cache (50.5% degraded) | 23,431 |
| **merged** | **46,862** |
| of which degraded | 11,831 (**25.2%** of the mix) |

That is the intended 50/50 clean-versus-mixed split. `merge_caches` streams into
a memory-mapped output rather than using `np.concatenate`, because the merged
array is 38.7 GB against this machine's 31 GB of RAM — `np.concatenate` fails
with `Unable to allocate 38.7 GiB`.

## The model already hedges on unusable photographs

The project's thesis is that the system should prefer an honest admission of
uncertainty over a confident misdescription. That behaviour **already exists in
the captioner**, and it is measurable without reference captions.

The 208 validation images with **zero** usable captions were previously excluded
from every evaluation, because BLEU needs references. They are also the hardest
images in the set: **100% of them are annotated `unrecognizable` by
crowdworkers**, against 7.2% among caption-bearing images. Measuring what the
model *says* about them needs no references.

`train.finetune.hedging_analysis` counts hedging language ("appears to be",
"blurry", "too dark", "quality issues are too severe", ...):

| Group | n | Hedged | Mean length |
|---|---|---|---|
| **208 images with no usable caption** | 208 | **75.5%** | 15.7 |
| Unrecognizable (crowdworker majority) | 543 | 33.3% | 12.7 |
| Recognizable | 6,999 | **1.7%** | 10.9 |
| dominant flaw: too dark | 129 | 30.2% | 12.2 |
| dominant flaw: too bright | 135 | 21.5% | 12.6 |
| dominant flaw: blur | 2,240 | 7.0% | 11.0 |
| dominant flaw: poor framing | 2,156 | 2.2% | 10.9 |
| dominant flaw: rotated | 480 | 1.2% | 10.8 |

**Hedging scales roughly 20× between recognizable and unrecognizable images**
(1.7% → 33.3%), and the ordering by flaw type tracks how badly each flaw destroys
information: darkness and over-exposure first, then blur, then framing and
rotation last. The model also speaks *longer* when it is unsure (10.9 → 15.7
tokens), which is the opposite of going silent.

Sample captions the model produced for photographs no crowdworker could describe:

```
appears to be a picture of a black object
a blurry image of something that is blue and white.
quality issues are too severe to recognize visual content, it's too blurry to figure out what it is.
a blurry picture of what appears to be the back of a computer screen.
```

**Why this matters.** The router's hedging rule has a real signal to key on, and
the captioner is not the confident-misdescription machine the proposal feared it
would be. The `"quality issues are too severe to recognize visual content"`
phrasing is notable — the model learned it from the caption corpus, where
crowdworkers wrote exactly that when they could not describe an image.

**Three honest caveats, which belong in any write-up:**

1. **Hedging is a lexical proxy, not calibrated confidence.** This counts phrase
   patterns; it does not show the model *knows* when it is wrong. A proper
   measurement needs token-level probabilities, which is a separate experiment.
2. **It still never declines to caption.** 0 of 208 produced an empty caption, and
   ~24% of the unusable images got an *assertive* caption. The model hedges when
   it can tell, but it has no mechanism to refuse.
3. **The 208-image group is confounded with severity.** Those images are the
   extreme tail (all unrecognizable, no captions accepted at all), so 75.5% is an
   upper bound rather than a typical rate for bad photographs.

### Result: synthetic degradation did NOT work

E7 was trained on the 50/50 clean+degraded mix (46,862 rows, 25.2% degraded),
12 epochs, base model, identical hyperparameters to E3. Evaluated on the full
7,542 at beam-5:

| Metric | E3 (baseline) | E7 (degradation) | Delta |
|---|---|---|---|
| BLEU-4 (all) | 0.1958 | 0.1902 | −0.0056 |
| CIDEr-D (all) | 0.4945 | 0.4778 | −0.0167 |
| 4-gram match (all) | 38.3% | 36.9% | −1.4 pts |
| distinct tokens | 2,591 | 2,671 | +80 |
| **recognizable BLEU-4** | **0.2089** | **0.2003** | **−0.0086** |
| **unrecognizable BLEU-4** | **0.0500** | **0.0524** | **+0.0024** |
| unrecognizable 4-gram | 7.6% | 7.4% | −0.2 pts |

**Both halves of the acceptance gate fail.** The unrecognizable bucket moved
+0.0024 BLEU-4, half the +0.0050 margin and therefore inside noise, and its 4-gram
match actually *fell*. Meanwhile the recognizable bucket regressed by 0.0086 — a
4% relative loss on the images that matter most. The mixed training traded clean
performance for no measurable gain on degraded images.

Its selection score tied E3's exactly (0.3246), which is a useful reminder that
per-epoch selection on a 1,500-image beam-1 subset **cannot** detect this kind of
change. The gate only caught it because it measured the two buckets separately.

**Why it probably failed, and why this is worth knowing:**

1. **Synthetic degradation is not real degradation.** Gaussian blur and
   brightness scaling are not what a shaky phone camera in poor light produces.
   The model learned to handle *my* augmentations, not VizWiz's actual flaws.
2. **25% of the training mix was degraded, and the clean bucket still regressed.**
   Adding a distribution shift to training cost accuracy on the original
   distribution, which is the standard augmentation trade-off appearing exactly
   as expected — it just was not repaid.
3. **The "unrecognizable" bucket is only 543 of 7,542 images (7.2%).** A 25%
   training investment aimed at 7% of the data has little leverage, and the
   `unrecognizable` label is a crowdworker judgement of whether *anyone* could
   describe the image, which is a much harder target than "somewhat degraded".

**The finding that matters more than the negative result:** BLIP-large, trained
with no degradation at all, already scores **0.0550** on the unrecognizable bucket
— better than E7's 0.0524 and 10% above E3. **Plain capacity gains beat synthetic
augmentation on this weakness.** That redirects the effort: scale the model and
the data, rather than manufacture bad photographs.

**Not yet tested, and the honest caveat:** degradation was never tried on the
BLIP-large harness, and only one severity/ratio was tried on base. A single
negative result at one configuration is not proof the approach is dead — but it
is enough to stop treating it as the priority, given that capacity already
delivers more on the same bucket.

## Analysis tools added for these experiments

| Tool | Purpose |
|---|---|
| `train.finetune.caption` | **caption a photo, folder, or glob from the command line** |
| `train.finetune.holdout` | the selection-free reporting split (1500 / 6042) |
| `train.finetune.rescore` | re-score existing predictions on the clean split, CPU-only |
| `train.finetune.eval_vqa` | answer coverage against VizWiz-VQA |
| `train.finetune.eval_by_quality` | metrics split by recognizability, with the two-sided gate |
| `train.finetune.hedging_analysis` | hedging rate versus image quality |
| `train.finetune.audit_objects` | user question demand versus caption supply |
| `train.finetune.length_control` | re-score under a relaxed brevity penalty |
| `train.finetune.degrade` | synthetic degradation from the VizWiz flaw distribution |
| `train.finetune.merge_caches` | concatenate feature caches (streams; 39 GB merge exceeds RAM) |
| `train.finetune.subset_cache` | extract a cache subset by image name, no re-encoding |
| `train.finetune.build_vqa_slice` | build the informative-answer caption slice |
| `train.finetune.eval_scratch` | evaluate the from-scratch ablation on the full validation split |
| `train.finetune.reading_gap` | per-intent answer hit rates and the component-ownership handoff |

Regression coverage is `train.finetune.smoke_test_finetune` — **22 checks**, no
network, including degradation determinism and flaw-distribution checks.

## The from-scratch ablation: why its row was not comparable

The ablation's only surviving prediction log covered **400** validation images
against the fine-tuned models' 6,042, and the mismatch was worse than it looked:
`--limit 400` restricted the *training* manifest too, so that model had been
trained on **400 images in total**, not 31,181. Its best epoch was epoch 2 with
validation loss rising from epoch 4 — textbook overfitting on 400 examples.

Its checkpoint also predates the current architecture (`hidden_seed`, 512×512)
where the model now has `image_seed` (512×2048), so it cannot be faithfully
rebuilt from its state dict. The row was therefore replaced by a clean retrain on
the **full 31,181-image trainval manifest** with the original hyperparameters
(attention model, ResNet-101 spatial features, batch 64, lr 1e-4, 2 layers,
10 epochs) — `runs/scratch_full`. This gives the from-scratch approach every
advantage, which is the fair test: if it still cannot produce fluent captions
with all the data, the conclusion no longer depends on a truncated run.

## Layout

```
train/finetune/
  base.py                   model loading, cached-feature forward, LoRA targets
  cache.py                  resumable feature extraction and verification
  dataset.py                cached-feature dataset, per-epoch caption sampling
  train_blip.py             training loop, checkpoints, resume, LoRA
  evaluate_blip.py          batched beam search, collapse diagnostics
  smoke_test_finetune.py    16 checks on a tiny synthetic BLIP
```

## Data

Splits are the official VizWiz ones, so results are comparable with the
from-scratch numbers and with published work. VizWiz test captions are
withheld, so evaluation uses the validation split (7,750 images) — the same
choice the proposal already documents.

Two properties of VizWiz that shape the run:

* **Up to five captions per image, unevenly distributed.** 565 train and 208 val
  images have no captions at all and are skipped (`build_split` drops them).
  Training samples one caption per image per epoch with a seeded RNG, so a
  multi-epoch run sees different sentences rather than the same five repeatedly.
* **~9.1% of images are unrecognizable to crowdworkers.** Those are the images
  the quality gate exists to catch, and they are the hardest captioning inputs.
  Report caption quality split by gate verdict rather than only in aggregate.

## Stability notes for this GPU

Both of these are documented failures that have already happened here, not
theoretical risks.

* **HiP driver fault `0xC0000005` in `amdhip64_7.dll`** has killed a completed
  epoch's work. Checkpoints are written every `--save-every` steps (default 500)
  with an atomic replace, and `--resume` continues from `last.pt`.
* **MIOpen is disabled on RDNA4** because the ROCm 7.2 Windows SDK is missing
  the rocrand headers its fused RNN dropout kernel needs.
  `configure_for_speed()` applies this; the BLIP path additionally exercises
  ViT convolutions and large attention tensors, so re-run `train.smoke_test`
  after any driver or ROCm change.

This sandbox's own shell cannot import `torch` without a writable `TEMP`
(`tempfile` fails at import). Set `$env:TEMP` to a real writable directory if a
run dies before printing anything.

## Deferred: the school-setting niche

The training script accepts the corpus through `--cache-index`/`--cache-tensor`,
so a school-specific corpus drops in as one more cache plus one more evaluation
split; no code change is required. The concrete decision still outstanding is
what the niche means (classroom whiteboards and handouts for a blind student,
versus text reading in general), which determines whether the corpora are
photographed in-house or taken from an existing dataset.
