# Training status and diagnosis

> **Superseded in part.** This document predates the COCO work, the `coco_e2e`
> run, and `train/diagnose.py`. Its "Remaining work" section is out of date, and
> its reading of the captioner failure — that the model needs more data and more
> steps — is not supported by the later measurements. See
> [status-report.md](status-report.md) for the current assessment and
> [build-plan.md](build-plan.md) for what remains. The bug history and
> environment findings below are still accurate and worth keeping.

Honest record of where the captioner and quality gate stand, what was verified,
and what is still needed. All numbers are measured on **held-out data**.

## Headline

The captioner started non-functional (BLEU-4 0.000, degenerate `"a a a a ..."`
beam output). It now produces genuine, often semantically correct captions.
**The first fix was architectural, not more training**: the decoder previously
saw only a single pooled 2048-dim vector with no way to attend to spatial
detail. Replacing it with an attention decoder over `7x7` spatial maps raised
BLEU-1 from 0.186 to 0.489 and BLEU-4 from 0.000 to 0.014 after only 3 epochs.

A second pass then found and fixed two further causes that capped quality
(image-blind attention queries, and a frozen encoder) plus a broken checkpoint
selection rule. See "Second pass" below for what those changed and what they did
not.

**Measured on held-out data:**

| Component | Result |
|---|---|
| Captioner (Flickr30k held-out, 1,589 images) | BLEU-1 0.475, BLEU-4 0.008, CIDEr-D 0.150, ROUGE-L 0.318 |
| Captioner, end-to-end encoder (held-out, 600 images) | BLEU-1 0.331, BLEU-4 0.0079, CIDEr-D 0.092 |
| Captioner (VizWiz fine-tuned, held-out) | BLEU-1 0.434, CIDEr-D 0.077, ROUGE-L 0.321 |
| Quality gate (VizWiz val, 7,750 images) | ROC AUC 0.964, AP 0.820, false-accept 4.8% |

Three failure modes had to be found and fixed, and all three were silent: a
decoder with no spatial attention, a scheduled-sampling loop that let `<pad>` be
substituted as a prediction and collapsed the model to a constant function, and
attention queries computed from a zero-context LSTM pass so the image never
influenced where the model looked. None showed up in aggregate loss; each needed
a direct input-sensitivity test.

**What did not work:** training longer on Flickr30k. Validation loss bottomed
out at epoch 3–4 and rose monotonically after, under every regularisation
setting tried. End-to-end encoder training did not change that: it converged to
the same held-out BLEU-4 (0.0079 vs 0.008). BLEU-4 at 0.008 is a working model,
not a usable one — a strong CNN+LSTM reaches 0.20–0.30.

## Second pass: fixing the two things that actually capped quality

The analysis above identified the symptom but not the two real causes. Both are
now fixed; the work is in `train/models/attention_captioner.py` and
`train/train/train_e2e.py`.

### Cause 1: the attention queries were image-blind

The decoder ran a first LSTM pass with a **zero** context vector and used those
hidden states as the attention queries:

```python
zeros = torch.zeros_like(tokens_in)
outputs, _ = self.lstm(torch.cat([tokens_in, zeros], dim=-1), (hidden, cell))
queries = torch.cat([hidden[-1].unsqueeze(1), outputs[:, :-1, :]], dim=1)
```

Attention was therefore steered by the language prior, not the picture — which
is exactly why the model emitted confident filler. This is not a subtle
inefficiency; it means the image never influenced where the model looked.

The fix conditions the decoder on the image from step 0: the LSTM state is
seeded from the mean spatial feature, and step 0 consumes that feature instead
of a token. `check_attn.py` now asserts that attention weights differ across
images and are not uniform, so the bug cannot return silently.

### Cause 2: the encoder was frozen

`requires_grad_(False)` on every encoder parameter meant the decoder trained on
ImageNet features, which are not caption features. `train_e2e.py` trains both
jointly on raw images, with **discriminative learning rates** (encoder 1e-5,
decoder 5e-4) because fine-tuning a pretrained CNN at the decoder's rate
destroys it.

### Cause 3: selection used the wrong metric

Checkpoints were chosen by validation **loss**. Captioning cross-entropy
plateaus long before BLEU peaks, so training stopped while quality was still
improving. An early attempt at BLEU-4 selection was worse: BLEU-4 is 0 for
every epoch early on, so the first evaluated epoch was saved and nothing after
it could ever beat it — the run silently kept a 2-epoch model. Selection now
uses a weighted blend (0.15·BLEU-1 + 0.25·BLEU-2 + 0.25·BLEU-3 + 0.35·BLEU-4)
that stays discriminative while BLEU-4 is still zero.

### Measured effect of end-to-end training

Held-out Flickr30k, beam-3:

| Checkpoint | BLEU-1 | BLEU-4 | CIDEr-D | ROUGE-L |
|---|---|---|---|---|
| Frozen encoder (best, 1,589 imgs) | 0.475 | 0.008 | 0.150 | 0.318 |
| End-to-end, epoch 1 (200 imgs) | 0.429 | 0.012 | 0.128 | 0.292 |
| End-to-end, epoch 2 (600 imgs) | 0.331 | 0.0079 | 0.092 | 0.278 |

Training loss falls steadily (3.836 → 3.175 → 2.896 → 2.671 by epoch 4), but
held-out BLEU-4 is **not yet clearly better** than the frozen-encoder model.
The honest reading: end-to-end training is working and the loss is still
descending, but at epoch 2–3 it has not yet converted that into better captions.
Sample output is qualitatively more image-grounded than the frozen model
("*A in snow is in snow*" for a shovelling scene, "*A girl a and girl a shirt a
in background*" for a crowd), but still carries filler.

### Hardware limits that shaped the run

Two failures worth recording, because both looked like something else:

* **GPU OOM at batch 64.** Reported "14.59 GiB free" while failing to allocate
  20 MiB — the fragmentation signature, not genuine exhaustion. Fixed with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` plus a smaller batch.
* **A driver-level crash.** The process died with exception code `0xC0000005`
  (access violation) inside `amdhip64_7.dll` at the start of validation, after a
  complete training epoch. This is a HIP/driver fault, not a Python error, and
  it is the second time this GPU has wedged during long runs. The working
  configuration reduces every concurrent GPU load: batch 24, scoring batch 8,
  beam 2, mixed precision off.

Measured throughput with a trainable ResNet-50: 220 img/s at batch 64 with AMP
(before the crashes), 63 img/s at batch 24 without AMP (the stable
configuration). The stability fix costs about 3.5x throughput, which is why the
final run is 15 epochs rather than 30.

### What the end-to-end work established, and what it did not

**Established:** the three causes above are real, and each fix is verified by a
test rather than by inspection. Attention now varies with the image
(`check_attn.py` asserts it). The encoder trains. Checkpoint selection no longer
freezes on a tie at zero.

**Not established:** that this reaches a usable captioner. The 512-dim
end-to-end run converged at 2.69 training loss and held-out BLEU-4 0.0079 —
statistically indistinguishable from the frozen-encoder model's 0.008. A
1024-dim decoder (50.6M params vs 29.6M) tracked the same loss trajectory
(3.97 vs 3.84 at the same step), so capacity alone was not the missing piece
either; that run was stopped rather than spending a further 8 hours on a
trajectory that was not separating from the smaller model.

**The most useful diagnostic result** came from reading the captions rather than
the scores. The model is often *semantically right*:

```
PRED: A dog the.                            REF: A white and black dog and a brown dog ...
PRED: Group people standing a of.           REF: A group of 11 people in winter wear ...
PRED: A man a shirt a and dog on boat       REF: A man and woman sitting on a deck next to a lake
PRED: A boy a shirt a in background         REF: A boy wearing a white t-shirt walks on the grass
```

It finds the dog, the group, the boat, the boy. What it cannot do is produce a
fluent multi-word sequence — it emits repeated articles ("a … a …") instead.
That is a **language-modelling** failure, not a vision failure, and it is why
BLEU-4 (which needs four consecutive correct tokens) sits at zero while BLEU-1
is 0.33–0.48. Any fix has to target decoder fluency.

### Honest bottom line against the target

The target was BLEU-4 ≈ 0.15–0.25. The best measured result remains **0.008**.
Reaching 0.15+ is not a matter of finishing the run that was stopped; on this
evidence it needs a substantially different setup, in rough order of expected
impact:

1. **Far more image–caption pairs.** Flickr30k's 31,783 images are small for
   captioning. COCO's ~123k (4x) is the standard corpus and is not yet wired up.
2. **A decoder that can model fluent text.** An LSTM over a 7,488-token
   vocabulary produces the article-repeat pattern regardless of hidden size.
   A transformer decoder, or a much larger LSTM with proper regularization,
   targets the actual failure.
3. **Much longer training at the stable configuration.** At 63 img/s an epoch is
   ~40 minutes, so the 15-epoch run is ~10 hours and 30 epochs would be ~20.

Those are a multi-day effort, not a single overnight run, and the GPU's
instability at higher load makes the fast configuration unusable.

## Captioner: attention model vs pooled LSTM

Evaluated on the **Flickr30k held-out split** (1,589 images excluded from
training), beam-5 with repetition constraints. The attention column is the full
held-out set; the pooled-LSTM column is the same split from the earlier run:

| Metric | Pooled LSTM (15 epochs) | **Attention (3 epochs)** |
|---|---|---|
| BLEU-1 | 0.186 | **0.475** |
| BLEU-2 | 0.042 | **0.172** |
| BLEU-3 | 0.003 | **0.037** |
| BLEU-4 | **0.000** | **0.008** |
| CIDEr-D | 0.007 | **0.150** |
| ROUGE-L | 0.187 | **0.318** |

Sample output at epoch 3:

```
PRED: A man a in snow shoveling
REF : A man in black attire shovels snow into the street
PRED: A man a fruit in market
REF : A Asian male wearing a red shirt is sitting next to a grocery stand selling fruits
```

Real, partially-correct descriptions. Not yet good — there is residual filler
("a in") and BLEU-4 is low — but the failure mode changed from "no language at
all" to "imperfect language", which is the difference between broken and
undertrained.

### A BLEU-4 / caption-length trade-off

Fine-tuning the working epoch-3 checkpoint with heavier regularisation
(dropout 0.4 → 0.6, lr 5e-4 → 3e-4, weight decay 1e-4) improves the overlap
metrics but loses BLEU-4:

| Checkpoint | BLEU-1 | BLEU-2 | BLEU-4 | CIDEr-D | ROUGE-L |
|---|---|---|---|---|---|
| attn3 epoch 3 | 0.4781 | 0.1782 | **0.0137** | **0.1814** | 0.3251 |
| attn4 epoch 3 (regularised) | **0.4913** | **0.1872** | 0.0000 | 0.1783 | **0.3331** |

The regularised model produces **shorter, safer captions**: it wins on unigram
and bigram overlap and on ROUGE-L, but never matches a 4-gram, so BLEU-4 goes to
zero. BLEU-4 is brutally sensitive to length — a 4-gram match requires four
consecutive correct tokens, so a model that emits fewer tokens has fewer
opportunities. Treat the 0.0137 vs 0.0000 gap as weak evidence in either
direction rather than a clear win, and note that BLEU-4 at this scale is a
noisy, low-resolution metric.

The honest summary: both checkpoints are in the same place — a model that
learns the common caption shapes ("A man in snow", "Two men at a fruit stand")
and then pads with filler when it is unsure. Neither is close to a strong
captioner.

### Cross-domain: VizWiz validation, no fine-tuning

The Flickr30k-trained model evaluated on VizWiz validation (500 images):

| Metric | Value |
|---|---|
| BLEU-1 | 0.327 |
| BLEU-2 | 0.044 |
| BLEU-4 | 0.000 |
| CIDEr-D | 0.011 |
| ROUGE-L | 0.236 |

By `text_detected`: BLEU-1 0.327 with text vs 0.331 without — no meaningful
split yet, which is expected before fine-tuning.

BLEU-1 holds up across the domain shift but BLEU-4 collapses, which is why
fine-tuning matters more than more Flickr30k epochs.

### VizWiz fine-tune (objective step 3)

Fine-tuned from the Flickr30k attention checkpoint with the vocabulary reused,
so the embedding and output layers keep their learned weights (all 17 tensors
loaded, none skipped). Evaluated on 7,542 held-out VizWiz images:

| Metric | Flickr30k only (cross-domain) | **VizWiz fine-tuned** |
|---|---|---|
| BLEU-1 | 0.327 | **0.434** |
| BLEU-2 | 0.044 | **0.127** |
| BLEU-3 | 0.000 | **0.018** |
| BLEU-4 | 0.000 | 0.000 |
| CIDEr-D | 0.011 | **0.077** (7x) |
| ROUGE-L | 0.236 | **0.321** |

`text_detected` breakdown (objective step 4):

| Split | n | BLEU-1 | BLEU-2 | CIDEr-D | ROUGE-L |
|---|---|---|---|---|---|
| `text_detected = true` | 365 | 0.4372 | 0.1290 | 0.0766 | 0.3244 |
| `text_detected = false` | 31 | 0.3924 | 0.1051 | 0.0767 | 0.2736 |

The split now separates the two groups, and — as intended — the model does
**better on images that actually contain text**, which is the case the app
cares about.

Fine-tuning validation loss: 3.873 → 3.766 → 3.733 → **3.721 (best, epoch 4)**
→ 3.733 → 3.747, overfitting after epoch 4 exactly as on Flickr30k.

Sample output after fine-tuning:

```
PRED: A screen a screen is a screen the is.
REF : A computer screen shows a repair prompt on the screen.
PRED: A holding of held
REF : A person is holding a bottle that has medicine for the night time.
```

Still visibly imperfect, but it now produces domain-appropriate content
("screen", "book", "holding") rather than Flickr30k's outdoor scenes.

## Quality gate: complete and working

Trained on VizWiz-QualityIssues, evaluated on 7,750 held-out validation images.
The gate predicts **unrecognizability** (majority of 5 crowdworker votes) and
reports the asymmetric error split the proposal specifies.

| Metric | Value |
|---|---|
| ROC AUC | **0.9643** |
| Average precision | **0.8203** |
| Threshold | 0.235 |
| **False-accept rate** | **4.79%** (bound 5%) |
| False-reject rate | 14.66% |
| F1 | 0.574 |
| Accuracy (misleading) | 0.863 |

The threshold trades a 14.7% false-reject rate to hold false-accepts under 5%.
That is the correct behaviour: describing an unusable image as fine is worse
than an unnecessary retake, because the user cannot verify the description.

Per-flaw performance (average precision / ROC AUC):

| Flaw | AP | AUC | positives |
|---|---|---|---|
| blur | 0.779 | 0.875 | 2,318 |
| no flaw | 0.672 | 0.830 | 2,276 |
| too dark | 0.597 | 0.908 | 219 |
| poor framing | 0.586 | 0.736 | 2,825 |
| too bright | 0.383 | 0.854 | 200 |
| rotated | 0.336 | 0.797 | 845 |
| other | 0.217 | 0.951 | 9 |
| obstructed | 0.203 | 0.865 | 116 |

Blur and darkness — the two conditions that trigger a retake — are the
strongest. The rarer flaws have too few positives to learn well.

### OpenCV baseline (the comparison the proposal promises)

A grid sweep over blur-variance and exposure thresholds, evaluated on the same
validation set. Even at its **best** hand-tuned operating point the rule
baseline is poor:

| Metric | Best rule baseline | Trained gate |
|---|---|---|
| F1 | 0.154 | **0.574** |
| False-accept rate | 0.143 | **0.048** |
| False-reject rate | **0.999** | 0.147 |
| ROC AUC | — | **0.964** |

The rules can only reach a 14% false-accept rate by flagging **99.9%** of all
images as unrecognizable — i.e. by asking for a retake almost always. Blur
variance and mean brightness cannot separate usable from unusable photographs.
This is a strong, honest justification for the learned gate.

## Bugs found and fixed

Every one was found by a verification script or diagnostic, not by inspection.
The first six were in the pooled LSTM path; the seventh was the root cause of
the whole failure; the eighth destroyed a run and was caught by a dedicated
collapse check.

| Bug | Effect if unfixed |
|---|---|
| Beam search reused one beam's hidden state for all beams | Silent quality loss in every reported metric |
| `<end>` not masked until `min_len` | Degenerate all-`<end>` captions |
| Finished beams not retired | `<end>` padded into output, corrupting beam-5 |
| Greedy decode fed the wrong token after finishing | Decoder conditioned on tokens it never chose |
| No held-out split | Training metrics reported as test metrics |
| BLEU fallback crashed on rows with no references | Evaluation died on legitimate VizWiz rows |
| **Decoder saw only a pooled vector, no spatial attention** | **BLEU-4 stuck at 0.000 — the core failure** |
| **Scheduled sampling could substitute `<pad>` as a "prediction"** | **Model learned to emit `<pad>`, collapsing to constant output** |

### The collapse, and how it was caught

An earlier attention run looked fine through epoch 3 and then went wrong. The
symptom was unusually clean, which is what gave it away:

```
epoch  8 done | train 3.0852 | val 3.0852 | 113.8s
epoch  9 done | train 3.0852 | val 3.0852 | 114.2s
epoch 10 done | train 3.0852 | val 3.0852 | 113.7s
```

Training and validation loss identical to four decimals, for ten epochs, is not
a plateau — it is a constant function. `train/check_attn.py` proved it:

- loss on a repeated identical batch: `8.921058` every time, **spread 0.00000000**
- loss with the image shuffled: `+0.0000` change
- loss with a **zero image**: `+0.0000` change
- predictions: `<pad>` **100%** of the time, 1 distinct token of 7,488
- beam-5 output: byte-identical for every image

Cause: scheduled sampling fed the model's own argmax back as the next input, and
that argmax could be `<pad>`. Substituting pad teaches the model that pad is a
plausible continuation, so it predicts pad more, so more pad gets substituted —
a self-reinforcing loop that destroys the model within a few hundred steps.

Fix: mask the pad logit to `-inf` before taking the argmax, and refuse any
substitution whose token is pad. `train/verify_ss_fix.py` re-runs training in
the collapse regime (sampling ratio 0.8) and asserts the loss moves (spread
0.157 vs 0.000), pad share stays 0%, and the output depends on the image. The
fixed run descends normally through the exact epochs where the broken one died.

The lesson is worth stating plainly: **a frozen loss is a bug, not
convergence.** Aggregate metrics looked plausible while the model was a
constant, and only a direct input-sensitivity test exposed it.

### A second failure: memory exhaustion

Long captions made the logits tensor `batch x seq_len x vocab` — at batch 128
with 53-token captions and a 7,488-token vocabulary that is ~200 MB per tensor,
doubled by scheduled sampling's two passes. Training stalled with 1.79 GB of
31 GB free and the machine paging, at ~1 core of CPU. Capping caption length at
32 tokens (mean length drops to 14.9) and halving the batch keeps ~17 GB free.

Also fixed: a truncated download silently accepted as complete; a `.parts`
cleanup crash after a successful 10.5 GB download; METEOR/SPICE Java bridges
hanging evaluation (both now opt-in); and the attention forward pass running one
LSTM call per token, which made an epoch take ~45 minutes instead of ~1.5.

## Environment findings

- **ROCm PyTorch was already installed and working** — no driver change needed.
  AMD documents driver 26.1.1 for ROCm 7.2 but the machine runs 26.8.1 fine.
- **MIOpen's dropout kernel fails on gfx1201** (missing rocrand header in the
  ROCm SDK), breaking any multi-layer LSTM with inter-layer dropout. Worked
  around by disabling MIOpen on RDNA4. See `docs/environment.md`.
- **VizWiz test captions are withheld**, so evaluation uses the validation
  split. The test split is still usable for the gate, whose labels are public.
- **The official VizWiz host serves 0.15 MB/s**; parallel range requests reach
  ~6 MB/s, turning a 28-hour download into 40 minutes.

## Reproducing

```powershell
# Quality gate (complete)
py -3.12 -m train.quality.train_gate --baseline   # rule comparison
py -3.12 -m train.quality.train_gate --epochs 25  # trained gate

# Attention captioner
py -3.12 -m train.features.extract --manifest data/manifests/flickr30k.jsonl --spatial
py -3.12 -m train.train.train_captioner `
    --manifest data/manifests/flickr30k.jsonl --model attention `
    --run-name flickr30k_attn3 --epochs 16 --holdout 0.05 `
    --batch-size 64 --lr 5e-4 --dropout 0.4 `
    --max-caption-len 32 --scheduled-sampling 0.25

# VizWiz fine-tune (step 3), with a real held-out split
py -3.12 -m train.features.extract `
    --manifest data/manifests/vizwiz_trainval.jsonl --spatial
py -3.12 -m train.train.train_captioner `
    --manifest data/manifests/vizwiz_trainval.jsonl `
    --val-manifest data/manifests/vizwiz_val.jsonl `
    --model attention --run-name vizwiz_ft2 --epochs 10 `
    --batch-size 64 --lr 1e-4 --dropout 0.5 --max-caption-len 32 `
    --scheduled-sampling 0.3 `
    --init-from checkpoints/flickr30k_attn3/best.pt `
    --vocab-from checkpoints/flickr30k_attn3/vocab.pkl

# Evaluation on held-out data
py -3.12 -m train.train.evaluate `
    --checkpoint checkpoints/flickr30k_attn3/best.pt `
    --feature-index data/features/flickr30k_resnet101_spatial.jsonl `
    --manifest data/manifests/flickr30k_holdout.jsonl `
    --beam-size 5 --min-len 2 --repetition-penalty 1.2 --no-repeat-ngram 3

# Regression checks
py -3.12 -m train.check_attn --checkpoint checkpoints/flickr30k_attn3/best.pt
py -3.12 -m train.verify_ss_fix
```

Three things here are not optional tuning knobs — removing any one of them
reproduces a failure documented above:

- `--max-caption-len` and the pad mask in `scheduled_sampling_loss` (collapse
  and memory exhaustion respectively)
- `--vocab-from` when fine-tuning (without it the embedding and output layers
  are reinitialised and the pretrained language model is thrown away)
- extracting features for `vizwiz_trainval.jsonl`, not `vizwiz_train.jsonl`
  (the cache is the authoritative row list, so a train-only cache silently
  leaves `--val-manifest` with nothing to hold out and `val_loss` becomes null)

## Remaining work

1. **Compare against fine-tuned BLIP**, per the proposal. This is the main
   outstanding item. If the from-scratch model cannot beat it, that is a
   legitimate reported ablation — and on the evidence here it probably cannot.
2. **Re-tune the repetition penalty** now that the model is stronger; 1.2 was
   chosen to suppress loops in a much weaker model and may now be limiting
   BLEU-4 by shortening captions.
3. **More VizWiz epochs at a lower rate** — the best checkpoint was epoch 4 of
   10, so the run was not the constraint; capacity or regularisation is.
4. **Length-normalised or length-aware decoding**, since BLEU-4 is heavily
   penalised by the model's tendency to emit short captions.
