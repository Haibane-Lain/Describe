# OCR component: plan for the reading bucket

Prepared for the OCR team. This document hands over a measured problem, a
ready-made test set, and a scoring protocol — not a request to "do OCR".

> **Status: the Phase 0 measurement now exists.** See
> [ocr-measurement.md](ocr-measurement.md) and the component in
> [`../ocr/`](../ocr/README.md). On the 577 reading questions OCR alone reaches
> **11.4%**, the captioner **11.7%**, and the two together **20.3%** — they fail
> on *different* images, which is the finding that matters: printed text is
> **not** currently solved by either component. §9 criteria 1, 2, 4 and 5 are
> met; criterion 3 is met with the numbers in that document.
>
> **One correction to this document.** §3 counts **577** images and §4 counts
> **575** questions, and they are different populations, which this document did
> not say. The captioner's prediction file omits two reading images
> (`VizWiz_val_00001985`, `VizWiz_val_00002169`), so §4's table — including its
> 188 sign/screen row — is a **population-B** table (rows with a caption). The
> full bucket is **577** rows and **190** sign/screen. Every rate in
> ocr-measurement.md names its population for this reason.

---

## 1. What is being handed over, and why

The captioner has been measured against **VizWiz-VQA**: for every answerable
question a blind user asked, does the caption contain the answer a plurality of
crowdworkers gave?

| Question intent | Answer hit rate | n |
|---|---|---|
| colour | 52.0% | 344 |
| identify object | 44.0% | 804 |
| other | 21.6% | 1,185 |
| **reading / detail** | **11.7%** | **575** |
| count | 0.0% | 15 |
| **all questions** | **29.3%** | **2,923** |

(Reproduce with `py -3.12 -m train.finetune.intent_table`. All intents are
classified and scored in a single pass, so the rows sum to the total. Two earlier
breakdowns existed in this project and disagreed — 2,923 against 3,048 — because
they classified and scored differently. The table above is the consistent one.)

**Reading is the captioner's worst bucket by a wide margin**, and it is not a
tuning problem. Four attempts to improve it by changing the training data all
failed, and the architectural fix — training the vision encoder — costs
**6.5 h/epoch** on this hardware and is therefore closed.

**Ownership has been agreed: the OCR component owns printed text.** The captioner
will continue to describe scenes and identify objects; it will not be expected to
read labels, titles, or dates.

**The captioner must not be judged on this bucket, and the OCR component must not
be assumed to cover it — it has not been measured yet.** That measurement is the
first deliverable below.

---

## 2. The test set (ready to use, no preparation needed)

| | |
|---|---|
| **Split** | VizWiz-VQA **validation** |
| **Images** | **577** (one question per image — no multi-question images) |
| **On disk** | `data/raw/vizwiz/images/val/` — all 577 verified present |
| **Annotations** | `data/raw/vizwiz/archives/VQA_Annotations.zip` → `val.json` |
| **Ground truth** | majority answer from ~10 crowdworkers per question |
| **Captioner baseline** | **11.7%** hit rate |

For tuning (not evaluation), the **train** split has **1,980** answerable reading
questions over 1,980 images.

**The test set must not be used for tuning.** It is the only clean measurement of
the reading bucket that exists, and it is small.

---

## 3. The first deliverable: measure OCR on the same 577 images

Do not start with an accuracy target. Start by producing a number for the same
population, so the project can state which component covers which intent.

**Protocol**

1. Run OCR on each of the 577 validation images.
2. Concatenate all text OCR extracts from an image into one string.
3. Ask, exactly as the captioner is scored: **does the extracted text contain the
   majority answer?**
4. Report the hit rate per subtype (table in §4) and overall.

**Scoring rules** — match the captioner's scoring so the two are comparable:

- Tokenise to lowercase alphanumerics.
- Drop stopwords; compare the remaining terms **in order** (substring on the
  joined token string), so `dog years` is not credited by a page containing
  "years" and "dog" separately.
- Multi-word answers must appear as a contiguous phrase.
- Case, punctuation and whitespace are ignored.

A reference implementation is in
[`train/finetune/reading_gap.py`](../finetune/reading_gap.py) — reuse
`tokenize`, `majority_answer` and `answer_in_caption` rather than reimplementing,
or the two components will not be measured on the same basis.

**Expected first result: below 11.7% is plausible.** OCR operating on
unconstrained blind-user photographs is a hard setting. If the first number is
low, that is a finding, not a failure — it tells the project the reading bucket is
currently covered by *nothing*.

---

## 4. Where the difficulty actually is

The 575 failures are not uniform. Measured baselines (population B — the 575 of
the 577 whose images have a captioner prediction; the full-bucket table with
OCR's measured numbers beside these is in
[ocr-measurement.md](ocr-measurement.md) §3):

| Subtype | n | Captioner hit rate | What OCR must extract |
|---|---|---|---|
| **sign / screen text** | 188 | 9.6% | menu items, screen messages, signage |
| **label / packaging text** | 136 | 8.8% | product names, ingredients, weight |
| other reading | 130 | 13.8% | mixed |
| product / brand name | 52 | 13.5% | brand, product name |
| book / media title | 28 | 17.9% | title, author, track names |
| phone / number | 16 | 25.0% | phone numbers, digits |
| expiry / date | 14 | **0.0%** | dates, batch codes |
| price / cost | 5 | 40.0% | prices |
| instructions | 4 | 25.0% | dosage, directions |
| medicine / dosage | 1 | 0.0% | drug name, dose |
| address / location | 1 | 0.0% | street, city |
| **total** | **575** | **11.7%** | |

**Priority order by volume:** sign/screen text and label/packaging text together
are **324 of 575 (56%)** of the bucket. Both are dense, small, often
low-contrast text photographed at an angle.

**Note the sample sizes.** Six subtypes have **n < 20** — phone/number (16),
expiry/date (14), price/cost (5), instructions (4), medicine/dosage (1) and
address/location (1). **Do not build features around them and do not publish
per-subtype rates for them**; they are noise. Even the captioner's 0% on expiry
dates is n=14 and should be treated as indicative rather than precise. The two
large subtypes are the only ones worth engineering against.

---

## 5. Concrete failure examples

These are real captioner outputs from the test set. The last column is what an
OCR pass would need to return.

| User asked | Crowdworker answer | Captioner said | OCR must extract |
|---|---|---|---|
| What is the expiration date on this bus pass? | `jul 08 11` | a can of food is on top of a table. | jul 08 11 |
| Can you tell me when this bus pass expires? | `10 apr 13` | a white and black ticket on top of a wooden table. | 10 apr 13 |
| What's the name of the drink? | `irn bru` | a bottle of soda is on top of a table. | irn bru |
| Can you tell me what this bottle says? | `warm vanilla sugar` | a person is holding a bottle of vanilla sugar body lotion. | warm vanilla sugar |
| Can you tell me what the error message is on the Windows screen? | `starting windows` | a computer monitor is on top of a desk. | starting windows |
| What is the title of this book? | `dog years` | a cd case with a picture of two dogs walking in the snow. | dog years |
| Can you see what pill bottle this is and what the name is? | `catalyst` | a person is holding a can of catalun in their hand. | catalyst |
| Can you tell me the price of this computer please? | `12999.01` | a laptop is on top of a table. | 12999.01 |

**The pattern is consistent:** the captioner identifies the *object* correctly and
never reads the *text on it*. It says "a bottle of vanilla sugar body lotion"
when asked what the bottle says — the answer is inside its own description, but it
cannot commit to the printed words.

---

## 6. Suggested phases

### Phase 0 — measurement (highest priority, no tuning)
Run the §3 protocol on all 577 images. Produce per-subtype and overall hit rates.
**This is the deliverable that unblocks the report**, because it lets the project
state which component owns each intent with evidence.

### Phase 1 — target the two dominant subtypes
Sign/screen text (188) and label/packaging text (136). These are 56% of the
bucket. Character-level accuracy on dense small text is the core difficulty.

### Phase 2 — tune on the 1,980 training questions
Only if Phase 0 shows headroom. Use the train split, never the 577.

### Phase 3 — joint verification
Re-measure the combined system: for each of the 575 questions, is the answer in
**either** the caption **or** the OCR output? That combined number is the honest
statement of coverage, and it is what the project should report.

---

## 7. Interface contract

Keep it simple and avoid coupling:

- **Input:** an image path or image bytes.
- **Output:** plain text — all recognised text concatenated, ideally with
  per-region bounding boxes and per-region confidence if available.
- **The captioner does not consume OCR output.** The two components run
  independently and their outputs are presented to the user separately. Do not
  build a pipeline where the captioner conditions on OCR text — that would require
  re-validating the captioner, and the reading bucket is already owned.

Bounding boxes and confidences are worth emitting even if unused now: they let the
combined evaluation weight regions, and they support the quality gate's
"unreadable" case.

---

## 8. Known risks

| Risk | Why it matters | Handling |
|---|---|---|
| **OCR output must not be trusted for safety-critical decisions** | medication names and dosages are in this bucket | the proposal already excludes safety-critical decisions; keep that boundary explicit |
| **Unanswerable questions are mixed in** | 32% of VizWiz val images carry an unanswerable question; some "reading" questions cannot be answered from the photograph at all | score only `answerable == 1` rows, and expect a ceiling below 100% |
| **Ground truth is noisy** | the "answer" is a crowdworker majority, not verified text | treat single-digit differences as noise; n is small |
| **Blurred or angled text** | blind-user photographs frequently cannot be read by anyone | the quality gate (FA 4.79%, AUC 0.964) already routes unusable images; coordinate so OCR is not blamed for images the gate should have caught |
| **The 577 set is small** | per-subtype rates are unreliable below ~50 samples | report overall plus the two large subtypes; do not publish rates for n<20 |

---

## 9. What "done" looks like

| # | Criterion |
|---|---|
| 1 | OCR hit rate measured on the same 577 images, same scoring, reported per subtype |
| 2 | A combined (caption **or** OCR) coverage number for the 575 reading questions |
| 3 | The project can state, with a measured number, which component owns each of the five question intents |
| 4 | No per-subtype claim published on fewer than 20 samples |
| 5 | The captioner is not evaluated on the reading bucket, and the OCR component is not assumed to cover it |

---

## 10. Reference material in this repo

| File | What it gives you |
|---|---|
| [`reading-gap-handoff.md`](reading-gap-handoff.md) | the generated per-intent analysis and failure examples |
| [`../finetune/reading_gap.py`](../finetune/reading_gap.py) | the scoring implementation to reuse |
| [`captioner-experiments.md`](captioner-experiments.md) | why four attempts to teach the captioner to read failed |
| [`training-process.md`](training-process.md) | the captioner's training, for context on what it can and cannot do |
| `data/raw/vizwiz/archives/VQA_Annotations.zip` | `val.json` — the 577 questions and their answers |
