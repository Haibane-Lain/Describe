# Captioner and quality gate: honest status report

Consolidated assessment of where the project stands against the concept
proposal's performance targets, written from the artifacts on disk rather than
from prior summaries. All numbers here were re-derived from the saved metrics
files and prediction logs, or read directly from the code.

Companion document: [build-plan.md](build-plan.md) for the remaining scope.

---

## 1. Headline

> **Superseded for the captioner.** This section's captioner and BLIP rows
> describe the state before the fine-tuned model existed. The deployed captioner
> is now a fine-tuned BLIP-large: **BLEU-4 0.2034 on 6,042 validation images**,
> against the from-scratch model's 1.3e-6, and against a zero-shot BLIP baseline
> of 0.1557. Full comparison and per-quality breakdown in
> [../finetune/README.md](../finetune/README.md). The from-scratch analysis in
> §3 below remains accurate and is now the project's reported ablation - the
> evidence for why the method changed.
>
> **All captioner figures are computed on a selection-free split.** Per-epoch
> checkpoint selection uses a 1,500-image subset, so those images are excluded
> from every reported number, leaving 6,042 of the 7,542 scorable validation
> images. Earlier revisions of this report cited figures computed on all 7,542,
> which contained the selection subset and were therefore optimistically biased.
> See "Evaluation hygiene" in [../finetune/README.md](../finetune/README.md).

**The quality gate is complete and meets its stated requirement. The
from-scratch captioner does not, and the evidence says more training data would
not have fixed it.**

| Component | Requirement (proposal §5C) | Measured | Status |
|---|---|---|---|
| Quality gate | Bounded false-accept rate, false-reject reported alongside | FA **4.79%** (bound 5%), ROC AUC **0.964**, FR 14.7% | **Met** |
| Captioner (deployed) | BLEU-4 ≈ 0.15–0.25 on VizWiz val | **BLEU-4 0.2034** on 6,042 selection-free images; CIDEr-D 0.5399; 4-gram match 40.3%; answers 23.5% of VizWiz-VQA questions | **Met** |
| Captioner (from-scratch ablation) | — | BLEU-4 **1.3e-6**; 126 distinct tokens; 0.0% 4-gram match | **Reported as ablation** |
| Zero-shot BLIP baseline | "the like-for-like comparison" (§4.3) | BLEU-4 0.1572, CIDEr-D 0.3555 | **Measured** |
| OCR + preprocessing (§4.4) | Word-level correctness on text images | Component built and measured on the 577 reading questions: **OCR 11.4%**, captioner 11.7%, **together 20.3%**. Word-level CER/WER on purpose-collected text photographs still not collected | **Partially met** |
| Router + composer (§4.5) | Decision accuracy with asymmetric cost | Not started | **Not started** |
| Guided capture (§4.6) | Best-frame selection | Not started | **Not started** |
| Mobile app (§4.1) | React Native / Expo, screen-reader operable | HTML mockups only | **Not started** |
| End-to-end eval + user study (§4.7) | Held-out set, metric suite, BLV participants | Not started | **Not started** |

Objectives 2 and 3 are now both met, and the captioner is the single largest
change since this report was written. Note what the fine-tune did **not** fix:
BLIP describes text, it does not transcribe it, so objective 4.4 (OCR) remains
entirely separate and must not be credited to this result.

Objective 4.4 has since been built and measured — see
[ocr-measurement.md](ocr-measurement.md). The result is worth stating plainly
because it is not the result the plan hoped for: **OCR reads the bucket at 11.4%
and the captioner at 11.7%, and neither is close to solving it.** They do fail on
different images, so the combined coverage is 20.3%, and that union — not OCR
alone — is the honest figure for the reading bucket. Word-level correctness on
purpose-collected label and document photographs (proposal §5C) is the part of
4.4 still outstanding, and the photographs are a team collection task.

---

## 2. What is verified and working

### 2.1 Quality gate

Source: `runs/quality_gate/metrics.json`, `runs/quality_gate/baseline_metrics.json`,
checkpoint `checkpoints/quality_gate/best.pt` (2.1 MB, MobileNetV2).

Trained on VizWiz-QualityIssues, evaluated on **7,750 held-out validation
images**, predicting *unrecognizability* by majority of 5 crowdworker votes:

| Metric | Value |
|---|---|
| ROC AUC | 0.9643 |
| Average precision | 0.8203 |
| Threshold | 0.235 |
| **False-accept rate** | **4.79%** (target bound 5%) |
| False-reject rate | 14.66% |
| F1 | 0.574 |
| TP / TN / FP / FN | 715 / 5,973 / 1,026 / 36 |

The threshold correctly trades a 14.7% false-reject rate to hold false-accepts
under 5%, which is the behaviour the proposal specifies: describing an unusable
image as fine is worse than an unnecessary retake because the user cannot verify
the description.

Per-flaw AP / AUC:

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

Blur and darkness — the two conditions that trigger a retake — are the strongest,
which is the right ordering. The rare flaws have too few positives to learn.

The OpenCV comparison the proposal promises was run, and the result is a strong
justification for the learned gate: at its **best** hand-tuned operating point
the rule baseline reaches F1 0.154 and a 14.3% false-accept rate only by
flagging **99.9%** of all images as unrecognizable. Blur variance and mean
brightness cannot separate usable from unusable photographs.

**Honest caveats.** Average precision of 0.820 with 9.7% positives is a solid but
not spectacular result; F1 0.574 means roughly 6 in 10 retake requests are
unnecessary (that is the deliberate cost trade). The per-flaw heads for
`obstructed` (116 positives), `too bright` (200) and `other` (9) are too weakly
supported to carry the proposal's per-condition routing table on their own. The
gate is a trustworthy *unrecognizability* detector; the finer-grained
condition-to-action routing in proposal §5B is only partly supported by it.

### 2.2 Evaluation and training infrastructure

Not a deliverable in itself, but real and worth stating because it constrains
what is cheap to do next:

* Cached-feature and end-to-end training paths, both with held-out splits.
* BLEU-1–4, ROUGE-L, CIDEr-D working via `pycocoevalcap`; METEOR opt-in (hangs
  above ~1,000 images); SPICE unavailable on modern JDKs — correctly reported as
  unavailable rather than as zero.
* Leakage check across VizWiz-Captions and VizWiz-QualityIssues, which share one
  image pool and partition.
* Eight real bugs found and fixed, each by a verification script rather than by
  inspection. The two most serious — a decoder with no spatial attention, and a
  scheduled-sampling loop that could substitute `<pad>` and collapse the model to
  a constant — are documented in `docs/training-status.md`.

That discipline is the project's strongest asset. The problem is that the
diagnosis it produced in the last round was wrong.

---

## 3. The captioner: measured, not inferred

### 3.1 Best available checkpoints

| Checkpoint | Eval set | BLEU-1 | BLEU-4 | CIDEr-D | ROUGE-L |
|---|---|---|---|---|---|
| `flickr30k_attn3/best.pt` | Flickr30k holdout, 1,589 img | 0.475 | 0.0077 | 0.150 | 0.318 |
| `flickr30k_attn/best.pt` | Flickr30k holdout, 400 img | 0.489 | 0.0137 | 0.169 | 0.329 |
| `vizwiz_ft2/best.pt` | VizWiz val, 400 img | 0.434 | **1.3e-6** | 0.077 | 0.320 |
| `flickr30k_e2e_safe/best.pt` | Flickr30k holdout, 600 img | 0.331 | 0.0079 | 0.092 | 0.278 |

Sources: `runs/*/eval_*.json`, `runs/flickr30k_e2e_safe/probe_ep2_600.json`.

Note the direction of travel: the **end-to-end** model, trained with a trainable
encoder and the fixes from the second pass, scores *worse* than the frozen-encoder
attention model on every metric (BLEU-1 0.331 vs 0.475). That is not what
"the encoder was the bottleneck" predicts.

### 3.2 The decisive measurement

Metrics alone understate the problem. Re-deriving statistics directly from the
saved prediction logs:

| | `vizwiz_ft2` (best) | `flickr30k_attn3` (best) | `flickr30k_e2e_safe` |
|---|---|---|---|
| Distinct tokens emitted | **126** of 7,488 vocab | 578 | 110 |
| Reference vocabulary size | 2,179 | — | — |
| Reference mass covered by emitted tokens | 63.4% | — | — |
| Most common token | `a` at **26.9%** | `a` at 23.5% | `a` at 26.0% |
| Top-10 tokens' share of output | 77.3% | 61.7% | 73.3% |
| Captions containing any 4-gram | 91.2% | 100% | 91.0% |
| …of those, **matching a reference 4-gram** | **0.0%** | **0.1%** | **0.0%** |
| Adjacent-repeat rate | 0.057 | 0.085 | 0.047 |

Predicted vs reference unigram frequency, `vizwiz_ft2`:

| token | predicted % | reference % |
|---|---|---|
| a | 26.92 | 11.89 |
| screen | 11.98 | 2.93 |
| of | 8.88 | 3.61 |
| is | 5.99 | 1.97 |
| on | 5.85 | 3.25 |
| with | 4.58 | 3.27 |
| the | 3.68 | 3.30 |

Sample outputs:

```
PRED: A screen a screen is a screen the is.
REF : A computer screen shows a repair prompt on the screen.

PRED: A holding of held
REF : A person is holding a bottle that has medicine for the night time.

PRED: A of book the is the of is.
REF : A library book with pictures of two dogs on the cover on a wooden table.
```

**Interpretation.** The model emits 126 distinct words out of a 7,488-token
vocabulary, and **not one four-word sequence it produces appears in any human
reference**. It is not stuttering — the adjacent-repeat rate is low — it is
emitting a high-frequency unigram bag with a few image-specific nouns attached
(`screen`, `book`, `yellow`). `a` is over-represented by 2.3× relative to the
reference distribution, which is what a model that has learned the marginal
token distribution and little else would produce.

This is why BLEU-4 is ~0 while BLEU-1 is 0.43: unigram overlap survives, and
nothing above it does. It is a **phrase-structure failure**, not a repetition
failure, not a length failure, and not an inference/decoding failure.

### 3.3 The collapse is in training, not at inference

The 126-token figure comes from **teacher-forced** predictions — the pass in
which the ground-truth previous token is supplied to the decoder. So the collapse
is present in the training-time conditional distribution itself. Exposure bias,
beam search, and the repetition penalty are all downstream of it and cannot be
the cause.

### 3.4 The "undertraining" diagnosis does not follow

`train/launch_coco.py` states the reasoning that committed the project to the
current run:

> "What it does is collapse: only 134 distinct tokens of 7,488, with 'a' at 22%.
> That is undertraining, so the fix is more data and more steps, not a different
> architecture."

The premise is correct and matches this report's measurements. The conclusion
does not follow from it. A model that has seen 31,783 images × 5 captions and
still emits 126 words has a broken learning signal, not an insufficient one.

Corroborating evidence that more epochs is the one thing already ruled out:

| Run | Val loss trajectory | Reading |
|---|---|---|
| `flickr30k_ss` (pooled, 15 ep) | 4.162 → **3.641 (ep 7)** → 3.673 (ep 15) | overfits after ep 7 |
| `vizwiz_ft2` (10 ep) | 3.873 → **3.721 (ep 4)** → 3.824 (ep 10) | overfits after ep 4 |
| `flickr30k_e2e` (7 ep) | 3.912 → 2.395, still descending | loss descending, BLEU flat at 0 |

Both completed runs converged and then overfit at low quality. The end-to-end run
shows that a *descending* training loss is fully compatible with a *flat, near-zero*
BLEU-4 — so "the loss is still going down" is not evidence that the run will
converge to a usable model.

### 3.5 Status of the current COCO run

`runs/coco_e2e/history.jsonl` — 3 epochs logged, `checkpoints/coco_e2e/` holds
`best.pt` and `last.pt`:

| epoch | train loss | seconds | val BLEU-1 | val BLEU-4 |
|---|---|---|---|---|
| 1 | 3.237 | 5,285 | — | — |
| 2 | 2.782 | 5,468 | 0.443 | 0.0 |
| 3 | 2.600 | 5,470 | — | — |

Configured for 30 epochs at ~91 min/epoch (`train/launch_coco.py`), so roughly
**45 hours** total. Its epoch-2 validation BLEU-1 (0.443) is no better than the
Flickr30k attention model's 0.489 on a comparable subset. COCO is ~4× the data and
would be expected to raise BLEU-1; it does not address 0% 4-gram match.

**Recommendation: stop it.** The run is not testing the hypothesis that the
evidence now identifies.

---

## 4. Defects found in the code

These are findings from reading the source, not hypotheses. They are recorded
because two of them plausibly explain the collapse, and because the third is why
the wrong diagnosis was reached.

### 4.1 Attention context is never fed to the LSTM during training, but is at inference

`train/models/attention_captioner.py`

Teacher-forced pass (used for every training step):

```python
# lines 174-177
zeros = torch.zeros_like(tokens_in)
outputs, _ = self.lstm(
    torch.cat([tokens_in, zeros], dim=-1), (hidden, cell)
)
```

The comment at line 172-173 states the intent: *"The context half is zero here;
attention context is added afterwards and does not feed back, so this is exact."*
It is not exact. The inference path does feed it back:

```python
# lines 139-141
context_embedded = self.context_proj(context)
lstm_input = torch.cat([embedded, context_embedded], dim=-1).unsqueeze(1)
output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
```

The LSTM's input weights are therefore trained entirely on inputs whose second
512 dimensions are zero, and at test time receive a non-zero `context_proj(context)`
for every token. The hidden state used as the attention query also diverges for
the same reason. This is a first-order train/inference mismatch.

It also makes the batched teacher-forced attention approximate rather than exact:
attention queries are computed from hidden states that saw a zero context, while
the logits they feed are combined with a real context vector.

### 4.2 Scheduled sampling substitutes the wrong position

`train/models/attention_captioner.py`, `scheduled_sampling_loss` (lines 245-266):

```python
predictions = torch.cat([ <one pad column>, prediction ], dim=1)   # [B, T]
...
sampled = torch.where(replace, predictions, sampled)               # position t <- predictions[t]
```

`forward()` returns T-1 positions and `predictions` is front-padded with one pad
slot, after which position `t` is replaced by `predictions[t]` — the prediction
*for* position t, not the prediction *from* t-1. Feeding the current target's own
prediction back is closer to label leakage than to scheduled sampling, and it is
not the procedure the docstring describes ("Position t is replaced with the
model's prediction from step t-1").

### 4.3 The diagnostic that justified the COCO run is broken

`train/diagnose.py` (lines 154-155):

```python
same = (predictions[:, 1:] == predictions[:, :-1]) & mask[:, 1:]
repeat_rate = float(same.sum().item()) / max(int(mask[:, 1:].sum().item()), 1)
```

The comparison is between positions t and t+1, but the mask is `mask[:, 1:]`,
which is aligned to the *targets* at those positions, not to the comparison. The
mask is offset by one relative to the thing it masks.

More importantly, the verdict rule is not diagnostic of the failure mode:

```python
fluent = repeat_rate < 0.25
print(f"   -> {'fluent-ish' if fluent else 'REPETITIVE (language-model failure)'}")
```

The collapsed model passes this test trivially — measured adjacent-repeat rate is
0.047–0.085, far below 0.25 — because a model that emits a unigram bag with a few
nouns does not stutter. It just says nothing. The script's own other outputs
(134 distinct tokens of 7,488, `a` at 22%) contradict the verdict it printed, and
that verdict is what committed the project to a 45-hour run.

**Fix direction:** the verdict should key on vocabulary collapse — distinct tokens
emitted as a fraction of vocabulary, and the top-token share — not on
adjacent-token repetition. A model emitting 126 of 7,488 tokens is collapsed by
definition regardless of how smooth its token-to-token transitions are.

---

## 5. Root cause: leading hypothesis, explicitly unverified

The collapse is consistent with **the image seed swamping the token signal** in
the LSTM input.

`context_proj(mean_feature)` is injected as a full 512-dimensional half of every
LSTM input (line 167-170 of `attention_captioner.py`), computed from a
2048-dimensional ResNet feature. Token embeddings are initialised
`uniform_(-0.1, 0.1)` (line 95). If the projected seed has a substantially larger
magnitude than the embedding, the LSTM's input at every position is dominated by
a near-constant image vector, the token identity is hard to recover from it, and
the model's best available strategy is to predict the marginal token
distribution — which is precisely the observed behaviour (126 distinct tokens,
`a` at 2.3× its natural rate, 0% 4-gram match).

**This is a hypothesis, not a finding.** Verifying it requires loading a
checkpoint and comparing the norm of `context_proj(mean_feature)` against the
norm of `embed(token)`, then running a token-sensitivity probe: hold the image
fixed, vary the previous token, and measure whether the next-token distribution
changes. If it does not change, the token signal is being swamped and the defect
is a fixable initialisation/scaling problem in the forward pass. If it does
change, the architecture is a dead end for this deadline.

**This check was not run.** It should be the first action taken, because it is the
only thing that distinguishes "fixable in an afternoon" from "switch models."

---

## 6. Other issues worth recording

* **No BLIP baseline exists.** Proposal §4.3 and §5A require a BLIP model
  fine-tuned on VizWiz-Captions as the like-for-like comparison, and §7 of
  `docs/training-status.md` lists it as "the main outstanding item." There is no
  BLIP code, no dependency, and no results file anywhere in the tree. This is the
  single most valuable missing artifact, and §7 below proposes promoting it from
  baseline to deployed model.
* **`docs/training-status.md` is stale.** It does not mention the COCO work, the
  `coco_e2e` run, the collapse measurements, or `train/diagnose.py`'s
  conclusion. Its "Remaining work" list is superseded by this report.
* **Checkpoint disk usage is 11.97 GB** across 85 files, much of it from
  smoke-test and abandoned runs (`alloc_smoke`, `oom_smoke`, `safe_smoke`,
  `ss_smoke`, `attn_smoke`, `coco_smoke`, `e2e_probe`, `e2e_smoke`, `_save_test`).
  Worth pruning before the COCO and BLIP runs compete for space.
* **BLEU-4 on 400 images is a very noisy statistic.** Several reported figures
  (0.0137 vs 0.0000 in `docs/training-status.md`, and the 1.3e-6 here) come from
  400-image subsets. Given that the true 4-gram match rate is ~0, the honest
  statement is "BLEU-4 is at or indistinguishable from zero," not a ranking among
  near-zero values.

---

## 7. Recommendation

**Status: items 1-5 below have been carried out.** The COCO run was stopped, the
fine-tune was built and run, and the from-scratch model is now the reported
ablation. What follows is the original recommendation, kept because it records
the reasoning that led here.

1. **Stop the COCO run** and reclaim the disk. It is not testing the hypothesis
   the evidence identifies.
2. **Run the token-sensitivity probe** (§5). It costs ~15 minutes and decides
   whether the from-scratch model is salvageable.
3. **In parallel, start the BLIP fine-tune on VizWiz-Captions.** This is
   already-promised work, not new scope, and it is the fastest route to a
   captioner that can actually be spoken to a user.
4. **Keep the from-scratch model as the reported ablation.** The proposal's
   objective 3 explicitly anticipates an ablation quantifying domain shift. A
   documented from-scratch-underperforms-pretrained result, with the collapse
   analysis in §3 as the explanation, is a legitimate and honest research
   contribution — and it is a far stronger deliverable than an unexplained
   0.008.
5. **Re-aim the captioning evaluation.** For the product, the metric that matters
   is whether captions are fluent and grounded enough to be spoken, which the
   gate's false-accept rate already protects. Keep BLEU-4 for comparability with
   published work, but do not let it be the only reported number.

**Bottom line (updated).** This now produces a working quality gate and a
working captioner. The two fit together better than expected: the captioner
degrades by a factor of four on the images the gate is trained to intercept
(BLEU-4 0.0500 on unrecognizable images against 0.2089 on recognizable ones,
4-gram match 7.6% against 40.7%), which is the empirical case for the gate being
the product's central mechanism. The remaining objectives — OCR, router, app,
end-to-end evaluation — are untouched by this and remain the critical path.

**Still to do before publication:** the from-scratch ablation was only ever
evaluated on 400 images while every BLIP number is on 7,542. Re-running that
evaluation on the full split is cheap now and would make the table strictly
like-for-like.
