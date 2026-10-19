# Captioner experiments: what was tried and what it showed

Three interventions beyond the original four-model comparison, each designed to
close a specific measured gap. **All three failed.** That is the headline, and it
is more useful than a success would have been: it says the captioner's remaining
weakness is not a data-augmentation problem.

Read alongside [../finetune/README.md](../finetune/README.md), which holds the
model comparison, the evaluation-hygiene correction, and the hedging analysis.

---

## 0. The evaluation correction that came first

Before these experiments, a flaw was found in every previously published number.
Per-epoch checkpoint selection used a 1,500-image validation subset, and that
subset was **entirely contained** in the 7,542 images being reported on. The
model was partly chosen because it scored well on images that then appeared in
the results.

The fix is a deterministic disjoint split (`train/finetune/holdout.py`):
1,500 images for selection, **6,042 for reporting**. Every figure below is on the
6,042. The correction was small and the model ranking did not change.

| Model | BLEU-4 | CIDEr-D | 4-gram | answer coverage |
|---|---|---|---|---|
| Zero-shot BLIP-base | 0.1557 | 0.3518 | 24.1% | 12.9% |
| E1 LoRA (6.8 MB) | 0.1887 | 0.4806 | 35.6% | 18.9% |
| E3 full decoder (base) | 0.1957 | 0.4916 | 38.3% | 20.8% |
| **BLIP-large** | **0.2034** | **0.5399** | **40.3%** | **23.5%** |

---

## 1. E4 — text-aware training (before this log)

Restricting training to the 16,272 text-bearing images, on the hypothesis that
text images teach reading.

**Result: worse on every metric**, including the answer coverage it targeted
(19.5% vs E3's 20.8%). Removing 6,594 images' worth of signal is not teaching;
it is just removing data. Detailed in the README.

---

## 2. E7 — synthetic degradation

**Hypothesis.** The captioner collapses ~4× on unusable photographs (BLEU-4
0.0500 vs 0.2089). No public corpus contains natural blind-user blur, so generate
it: apply the flaw types VizWiz annotates (blur, dark, bright, rotation,
occlusion, framing), weighted by their real frequencies, before the vision tower.

**Method.** 23,431 clean features + 23,431 features from a 50.5%-degraded cache,
merged into 46,862 rows (**25.2% degraded**, the intended 50/50 clean-vs-mixed
split). Base model, 12 epochs, all hyperparameters identical to E3.

**Result — both halves of the acceptance gate fail:**

| Metric | E3 baseline | E7 | Delta |
|---|---|---|---|
| BLEU-4 (all) | 0.1958 | 0.1902 | −0.0056 |
| CIDEr-D (all) | 0.4945 | 0.4778 | −0.0167 |
| 4-gram match | 38.3% | 36.9% | −1.4 pts |
| **recognizable BLEU-4** | **0.2089** | **0.2003** | **−0.0086 (regressed)** |
| **unrecognizable BLEU-4** | **0.0500** | **0.0524** | **+0.0024 (inside noise)** |
| unrecognizable 4-gram | 7.6% | 7.4% | −0.2 pts |

It cost 4% relative accuracy on the images that matter most and bought nothing
measurable on the images it targeted. Its selection score tied E3's exactly
(0.3246) — a reminder that per-epoch selection on a beam-1 subset **cannot**
detect this; only the two-bucket gate caught it.

**Why it failed.** Synthetic Gaussian blur and brightness scaling are not what a
shaky phone in poor light produces, so the model learned *my* augmentations, not
VizWiz's flaws. And a 25% training investment aimed at 7.2% of the data (543 of
7,542 images) has little leverage.

**The more important observation:** BLIP-large, trained with **no degradation at
all**, scores **0.0550** on the unrecognizable bucket — better than E7's 0.0524.
Plain capacity beat synthetic augmentation on the same weakness.

---

## 3. E8 — the VizWiz-VQA informative slice

**Hypothesis.** The model identifies objects but cannot report the specific
detail on them: *"a bottle of medicine"* without the name, *"a magazine"* without
the title. VizWiz-VQA supplies detail-on-an-object supervision on our own domain.
Filtering to informative answers (≥2 words, not a bare colour) yields **5,999
image–object pairs**: *"basil leaves"*, *"coca cola"*, *"solar garden light"*,
*"80 cents"*.

**Method.** The 5,879 extractable pairs (120 are among the 565 captionless training
images and so have no cached features) merged with the clean large cache —
29,310 rows, **20.5% synthetic**, the intended ~1:3 mix. BLIP-large, 8 epochs.

**Result — template collapse at epoch 1:**

| | BLEU-4 | CIDEr-D | blend |
|---|---|---|---|
| BLIP-large baseline, epoch 1 | 0.1842 | 0.5472 | 0.3368 |
| **E8, epoch 1** | **0.1122** | **0.3625** | **0.2218** |

The model learned the template, exactly as the risk assessment predicted:

```
clean ref : Red packaging for a vanilla flavor snack pack pudding cup.
E8 pred   : a photo of peanut butter pudding          <- template, and wrong

clean ref : The inside of a school or hospital's reception area.
E8 pred   : a photo of office desk                    <- template

clean ref : light haired dog with black collar looking out of the opening
E8 pred   : a photo of yellow lab                    <- template

clean ref : a sealed cd of Gwen Stefani on someone's lap
E8 pred   : a children's book is on top of a person's lap.  <- fluent, wrong
```

Training was stopped at epoch 1: the fluency gate had already failed by 39%
relative on BLEU-4.

**Two things this establishes.** First, **answers are not descriptions** — a
captioner trained on crowdworker answers learns to answer, and the template is
the cheapest way to do it. Second, and more subtly, it *did* learn object names
("peanut butter pudding" for a pudding cup) — so the signal is real, and it is
the *packaging* that destroys fluency. No amount of filtering fixes that, because
the problem is the form of the supervision, not its content.

**VizWiz-VQA's value is as an evaluation instrument, not a training corpus.**
Answer coverage (25.9% → 23.5% on the clean split) is the most decision-relevant
metric this project has, and it cost nothing to obtain.

---

## 4. E9 — a 20-epoch schedule (Phase 1a)

**Hypothesis.** The 8-epoch BLIP-large run had not converged: its per-epoch
selection score was still rising at the final epoch (0.3427 → 0.3431 → 0.3439) and
training loss was still falling (2.676 → 1.939). Unlike the base model, which
peaked at epoch 6, it was stopped by its *schedule*, not by convergence. So more
epochs might help.

An earlier attempt was abandoned because GPU contention made it slower than the
8-epoch run, and its partial curve was confounded — a 20-epoch cosine keeps the
learning rate high for longer, so it *should* trail early. This run resolves that
with nothing else on the GPU.

**Method.** BLIP-large, 20 epochs, lr 3e-5, identical data and hyperparameters to
the 8-epoch run. Total steps logged as 14,300 against the 8-epoch run's 5,720, so
the schedules are auditable. Epochs 1–9 reproduced the 8-epoch run's numbers
**exactly** (loss 2.7766 / 2.3489 / 2.2406 / 2.1404 / 2.0696 / 2.0020 / 1.9298 /
1.8744 / 1.8099), confirming a controlled comparison.

**Result — the longer schedule is worse, on every metric:**

| | best selection score | which epoch | final epoch |
|---|---|---|---|
| 8-epoch schedule | **0.3439** | 8 (its last) | 0.3439 |
| 20-epoch schedule | 0.3378 | **1 (its first)** | 0.3180 |

On the selection-free 6,042-image split at beam-5:

| Model | BLEU-1 | BLEU-4 | CIDEr-D | 4-gram | answer coverage |
|---|---|---|---|---|---|
| **8-epoch (shipped)** | 0.6122 | **0.2034** | **0.5399** | 40.3% | **23.5%** |
| 20-epoch schedule | 0.6137 | 0.2021 | 0.5208 | 40.5% | 22.7% |

**The shape of the curve is the finding.** The 20-epoch run's best epoch is its
*first*, and score declines thereafter while training loss falls to 1.4971 and
validation stays flat near 0.33. The longer cosine keeps the learning rate
elevated for 20 epochs, so the model never anneals into a sharp minimum — it
memorises rather than converges.

**Conclusion: 8 epochs with this schedule is the right budget.** The apparent
headroom was an artefact of reading a rising selection score as under-training,
when it was a schedule still annealing. This permanently closes the "more epochs"
question and saves the ~2 h per attempt.

### A hardware note found along the way

The run initially measured **1,014 s/epoch** against the 8-epoch run's 233 s —
4.35× slower with identical data and an identical loss curve. The cause was not
the run: nine exited-but-unreaped python processes were still holding ~35 GB of
commit charge, leaving 0.1 GB of free RAM against a 25.8 GB memory-mapped cache,
so the run thrashed. Terminating them (0 MB working set each) reclaimed ~49 GB of
commit, and epoch time fell 1,014 s → 572 s → **241 s**, i.e. faster than the
original 233 s train-only figure once the cache stayed resident.

Worth recording because it is a property of this Windows ROCm setup: **check for
zero-working-set python zombies before benchmarking**, or a memory-starved run
will look like a regression in the model rather than in the machine.

## 5. E10 — caption expansion, one row per caption (Phase 2)

**Hypothesis.** The captioner holds **100,575 captions across 22,866 images**
(mean 4.4 each), but each epoch draws only *one* random caption per image, so
each individual caption is seen only once or twice across an 8-epoch run — thin
exposure for a rare word. Writing one index row per caption would give every
caption equal weight.

**Method.** `train.finetune.expand_captions` wrote a 100,575-row index. Crucially
it **does not duplicate the feature tensor**: every row keeps the same
`feature_row`, and the dataset indexes the shared array, so 100,575 rows point
into 22,866 feature slots. The expanded index is **49.8 MB** where duplicating the
tensor would have needed ~117 GB. The dataset was extended to honour a
`_caption_single` flag so the single caption is used rather than re-sampled
(otherwise the expansion would silently do nothing). 9 integrity checks pass,
including that two rows sharing a `feature_row` yield identical image features
and different captions.

**Result — a genuine trade-off, not a clean win or loss:**

| Model | BLEU-1 | BLEU-4 | CIDEr-D | 4-gram | **answer coverage** | distinct | mean len |
|---|---|---|---|---|---|---|---|
| **8-epoch (shipped)** | 0.6122 | 0.2034 | **0.5399** | **40.3%** | **23.5%** | **2634** | 11.99 |
| 20-epoch schedule | 0.6137 | 0.2021 | 0.5208 | 40.5% | 22.7% | 2309 | 11.93 |
| **expanded captions** | **0.6301** | **0.2112** | 0.5298 | 39.0% | 21.9% | 2275 | 11.32 |

The expanded run has the **highest BLEU-4 and the worst 4-gram match, fewest
distinct tokens, shortest captions, and lowest answer coverage** — a suspicious
combination that looked like n-gram gaming. It is not: `length_control` shows the
brevity penalty is inert for both models (length ratio 1.01 vs 1.04), and the
BLEU-4 lead survives to exponent 0.0. **The gap is content, not length.**

**What is actually happening.** Training on every caption makes the model produce
the *most typical* description of an image rather than one of its varied
descriptions. The references are multiple human captions, so matching the common
phrasing helps BLEU-4 — while the model emits **fewer distinct tokens** (2275 vs
2634) and **answers fewer real questions** (21.9% vs 23.5%).

**Verdict: fails the pre-registered criterion, which required answer coverage to
improve *and* fluency not to regress.** Coverage fell 1.6 points. BLEU-4 improved
3.8%, so the outcome is mixed rather than cleanly negative — but the project's
stated goal is answering blind users' questions, and on that measure the shipped
8-epoch model is better. **The shipped model is unchanged.**

**A confound worth stating.** The expanded run took 25,144 total steps against the
baseline's 5,720, so its cosine schedule decayed more slowly and its LR never
fully annealed. That means this run does not cleanly isolate "more caption
diversity" from "more optimisation steps". A matched-step control (the expanded
index at ~2 epochs, ≈6,286 steps) would separate them. Given that the expanded run
already loses on the metric that matters, the control is not worth 16 minutes
unless the BLEU-4 gain is wanted for its own sake.

## 6. What actually worked

| Intervention | Verdict |
|---|---|
| Fine-tuning at all (zero-shot → E3) | **BLEU-4 +25%, answer coverage +61% relative** |
| Scaling capacity (base → BLIP-large) | **BLEU-4 +4%, CIDEr-D +10%, answer coverage +13% relative** |
| LoRA instead of full decoder | E1 is 6.8 MB vs 526 MB for ~4% less BLEU-4 — the deployment trade |
| E4 text-aware subset | worse on everything |
| E7 synthetic degradation | regressed recognizable, no gain on unrecognizable |
| E8 VQA informative slice | template collapse |
| E9 20-epoch schedule | worse than 8 epochs on every metric; best epoch was the first |

**Every data-side intervention failed, and extending the schedule failed too.
Scaling the model worked.** That is a consistent signal, not four coincidences.
Capacity and the existing 22,866 human captions are doing the work — and the
captioner is now trained to the limit of what this schedule and this data can
give it.

## 7. Recommended next step

**The training budget is settled: 8 epochs, BLIP-large, full decoder.** Four
data-side interventions and one schedule extension have all failed to beat it.
The model is trained to the limit of what this data and this schedule can give,
and no further tuning of what is already on disk is likely to move it.

The honest position: the captioner is **usable and the project's central claim is
supported** — it beats the from-scratch ablation by an order of magnitude on
BLEU-4 even when that ablation is given all 31,181 images, it answers 23.5% of
real user questions, and it already hedges ~20× more often on unusable
photographs.

Further gains require something this project does not currently have:

| Lever | Status |
|---|---|
| A stronger vision encoder | untested; the frozen tower is 68% of the model |
| More human-written captions | the only untried data source; every attempt to synthesise more signal from what is on disk failed |
| Training the vision tower | **closed** — 6.5 h/epoch on this hardware |
| More epochs | **closed** — measured worse (E9) |
| Better question answering | **not the captioner's job** — reading is 11.7% hit rate and belongs to OCR |

The two things that would actually change the project's outcome are therefore not
modelling tasks: **confirming the OCR component covers the reading bucket**
(see [reading-gap-handoff.md](reading-gap-handoff.md)), and **deciding the
deployment trade** between the 544 MB BLIP-large and the 6.8 MB LoRA adapter.

