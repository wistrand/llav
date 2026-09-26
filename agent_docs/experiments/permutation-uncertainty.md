# Experiment: does disagreement under option permutations predict wrong answers?

The claim under test: a model's disagreement with itself when the same options are listed in different orders
predicts when its decision should not be trusted, beyond what its confidence already says. If true, it would
be an uncertainty signal that needs no training and no extra model, only more forward passes over a shared
state. Harness: `scripts/perturb.py` (`run`, `analyze`). Raw results live in `local/results/perturb/`
(gitignored).

## Hypotheses

Fixed before the runs:

- H1 (primary): at equal compute, permutation spread improves error detection beyond the best signal from the
  same passes, the averaged answer's confidence or margin. Supported only if the 95% bootstrap interval of the
  AUROC gain is above 0, with the combination fitted on the other sources (leave one source out), because
  llav's criteria are written at run time.
- H2: among confident answers (0.9 and up), those that change under some order are wrong more often than
  stable ones. Needs 100 or more such answers per model.
- H3: any gain grows with how confusable the options are.
- H0 (expected from the pilot): spread is a costlier stand-in for confidence.

## Why the pilot shapes the test

Rotations of the caller's order on 728 existing questions (Pilot, below) gave the averaged answer's confidence
AUROC 0.886 and spread 0.878, and the two together 0.884. Once K passes are paid for, the averaged answer's
confidence is the fair baseline, not the single-pass confidence, so H1 is stated against it. The plan is built
to confirm or overturn that, and to find any narrower effect (H2, H3) that survives.

## Design

### Variants per question

| Variant                                   | Count                 | Purpose                                                                                                                     |
|-------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Base order, repeated                      | 3                     | Noise floor: identical prompts, so any spread is numeric                                                                    |
| Cyclic rotations of the base order        | all other, at most 13 | Comparable with the pilot                                                                                                   |
| Random full permutations                  | 6                     | Rotations are a narrow family of permutations                                                                               |
| Swapped Yes/No order (nouls)              | 2 orders              | Planned: brings in nouls as a 2-option case; needs a choice-shaped noul                                                     |
| Reversed levels (scores)                  | 2 orders              | Planned, analysed apart from H1: whether reordering a scale is the confound                                                 |
| Reworded question, 3 templates per source | 3                     | Planned control: is permutation special, or does any meaning-preserving change work as well                                 |
| Answer letters 1/2/3 instead of A/B/C     | per order             | Phase 2, in a research script against llama-server directly; never through llav's API, and `PROMPT_VERSION` does not change |

- The base order is drawn at random per question (seeded by the question id), so a dataset's own option order
  is never the reference.
- At most 16 questions per request (the native helper's batch). Runs use llama-server without the helper:
  its batched decode adds spread to identical prompts (Run 1).
- Questions whose single-letter keys fix the displayed order (`_align_letters`) cannot be permuted and are left
  out of the analysis.
- Nouls and scores are skipped by the current harness: llav fixes a noul's Yes, No order, and permuting a
  scale breaks its order.

### Signals

- One pass (base order): top probability, top-two margin, entropy, candidate mass.
- K passes: the averaged answer's top probability and margin; spread, the mean total-variation distance of
  each order's answer from the average; the share of orders whose answer differs from the most common one.
- Planned: the same spread measures over the reworded variants.
- Each signal is scored against two targets: a wrong base-order answer and a wrong averaged answer.

### Metrics and statistics

- AUROC and AUPRC with wrong answers as the positive class; area under the risk-coverage curve; error rate at
  80, 90 and 95% coverage.
- H1 gain: AUROC of a logistic model on [averaged confidence, spread] minus AUROC on [averaged confidence]
  alone, both fitted leave one source out, with a paired 95% bootstrap interval stratified by source.
- H2: wrong-answer rate of confident answers (0.9 and up) that change under some order against stable ones,
  with and without controlling for confidence.
- H3: the H1 gain per source against option count and against the averaged answer's margin (an interaction
  term in the logistic model).
- A hand audit of 50 confident answers that change and are wrong, because Banking77 and HelpSteer2 have noisy
  labels and some "errors" will be label mistakes.

### Data

- Target: about 5,000 choice questions per model with at least 1,000 wrong answers, enough for 100 or more
  confident answers that change under some order (H2).
- Existing sources (`evaluate.py fetch`): AG News, DBpedia, Banking77 card intents, SemIf's authored set; BoolQ
  and PubMedQA once nouls can be permuted; Yelp and HelpSteer2 only in the score analysis.
- New sources with confusable options, fetched the same way (standard library, evenly spread samples):
  - full Banking77, with 10 to 20 candidates per question: the gold intent and its nearest neighbours;
  - CLINC150, same construction;
  - TREC question types;
  - MNLI, 3-way;
  - a coarse GoEmotions subset.
- A gold-option-removed set: not for H1, but the only fair test of candidate mass, which the current sets never
  exercise (their minimum mass is 0.99).

### Models

The pinned models in `scripts/fetch-model.sh`: qwen3.5-4b, qwen3.5-2b, granite-4.2-3b, granite-4.0-h-tiny,
smollm3-3b; Muse Glimmer 30B optionally. One model family would only describe that family.

### Compute

About 5,000 questions and 12 to 22 variants each, roughly 60,000 to 100,000 passes per model. Run 1 took
699 s for 500 questions through llama-server on the RTX PRO 4000, so about 2 hours per model; five models
about half a GPU day.

### Decision points

1. Noise floor (Run 1): done. llama-server is deterministic; the helper adds spread. Use llama-server.
2. Full run on Qwen3.5-4B. If H1 is clearly negative, do not spend the other models on H1; spend them on H2 and
   the reworded-question control, which decide whether anything permutation-specific is left.
3. Framing:
   - H1 supported: permutation disagreement is a training-free error signal beyond confidence.
   - H0: averaging over permutations gives a better answer and a better confidence, and disagreement adds
     nothing beyond that. Useful to a system builder: what to compute and what to skip.
   - H2 alone: a narrow result on brittle confident answers, reported with its size and coverage.

### Threats

- Training contamination (AG News and DBpedia are likely in training data): affects accuracy more than the
  signal-error relation; reported per source.
- Label noise: the hand audit.
- Rotating an ordinal scale breaks it: scores stay out of H1.
- Numeric noise: the identical-repeat control and the llama-server path.

## Results

### Pilot, existing `shifts` data (2026-09-24)

Qwen3.5-4B Q8_0, RTX PRO 4000, native helper, cyclic rotations of the caller's order only. 728 nominal choice
questions (SemIf candidate selection and Yelp left out).

- AUROC for a wrong caller-order answer: confidence 0.860, averaged confidence 0.886, spread 0.878, averaged
  confidence plus spread 0.884. Spread added nothing to the averaged answer's confidence.
- Confident answers that changed under rotation: 7 of 17 wrong (41%), against 24 of 556 stable (4.3%),
  Yelp excluded.

### Run 1: noise floor and first H1 check (2026-09-25)

Same model and box; 500 choice questions from the local eval sets (`--limit 125` per file), both readout
paths. 460 questions after leaving out the 40 fixed-order ones.

- The native helper crashed on the first Banking77 question (`GGML_ASSERT(n_tokens_all <= cparams.n_batch)`:
  12 suffixes, 2,108 tokens, against its `n_batch` of 2048) and llav fell back to llama-server, so only the
  125 AG News questions went through the helper. The helper bug is separate from the experiment.
- Noise floor, AG News: identical prompts through llama-server give identical probabilities (spread
  0.00000); through the helper's batched decode, mean spread 0.0019, max 0.019, about 5% of the spread
  permutations cause (0.036), and no answer changes. The full run uses llama-server only.
- Permuted orders, all 460 questions: mean spread 0.053, and 14% of questions change their answer under
  some order.
- H1: adding spread to the averaged answer's confidence changes held-out AUROC by +0.002
  [-0.003, +0.008] for the base-order answer and -0.007 [-0.014, -0.002] for the averaged answer. No gain.
- H2: confident answers that changed under some order: 1 of 19 wrong (5.3%), against 10 of 345 stable
  (2.9%). The pilot's 41% did not replicate. Both counts are too small to decide; the difference in setup
  (random base order here, the dataset's order in the pilot) is a candidate cause (inference).

### Run 2: full run, Qwen3.5-4B (2026-09-25)

Qwen3.5-4B Q8_0, RTX PRO 4000, llama-server only (no helper), `perturb.py run` defaults. 6,992 choice questions
from `local/data/perturb` (`evaluate.py fetch --rows 1000`), 6,944 after leaving out 48 fixed-order ones; 95
minutes. Analysis in `local/results/perturb/full-analysis.txt`.

- Noise floor: identical prompts give identical probabilities on every question. Permuted orders change the
  answer on 28.7% of questions.
- How often the answer changes tracks how confusable the options are, not how many there are: DBpedia (14
  options) 3.0%, AG News (4) 8.7%, MNLI (3) 12.6%, CLINC150 (12) 29.4%, TREC fine types (2 to 17) 33.4%,
  Banking77 (12) 40.5%, GoEmotions (12) 74.5%.
- Averaging over orders cuts wrong answers from 1,498 to 1,384 (7.6%), and its confidence detects errors
  better than one order's: AUROC 0.880 against 0.863 for the base-order answer.
- H1 not supported. Adding spread to the averaged answer's confidence changes held-out AUROC by -0.0015
  [-0.0022, -0.0006] for the base-order answer and -0.0018 [-0.0023, -0.0013] for the averaged answer; with
  the averaged margin instead, +0.0013 [-0.0004, +0.0033] and -0.0006 [-0.0020, +0.0009]. Spread alone
  (0.866) is below the averaged confidence (0.880). Once the orders are paid for, their disagreement adds
  nothing to what their averaged answer says.
- H2 supported. Of base-order answers with confidence 0.9 and up, those that change under some order are
  wrong 20.0% of the time (57 of 285), stable ones 5.4% (228 of 4,231). It holds within confidence bands and
  per source:

  | Base-order confidence | Stable: wrong     | Changes: wrong |
  |-----------------------|-------------------|----------------|
  | 0.90 to 0.99          | 10.5% (147/1,395) | 21.9% (51/233) |
  | 0.99 to 0.999         | 3.4% (70/2,062)   | 12.8% (6/47)   |
  | 0.999 and up          | 1.4% (11/774)     | 0 of 5         |

  | Source (confidence 0.9 and up) | Stable: wrong  | Changes: wrong |
  |--------------------------------|----------------|----------------|
  | Banking77                      | 6.7% (35/522)  | 17.8% (16/90)  |
  | Banking77 card intents         | 5.2% (16/307)  | 24.4% (11/45)  |
  | CLINC150                       | 1.9% (10/540)  | 12.5% (7/56)   |
  | GoEmotions                     | 25.4% (32/126) | 41.7% (10/24)  |
  | TREC fine types                | 6.0% (16/265)  | 13.6% (6/44)   |
  | MNLI                           | 6.9% (43/624)  | 5 of 10        |
  | DBpedia                        | 0.6% (6/963)   | 1 of 9         |
  | AG News                        | 8.4% (69/823)  | 1 of 4         |

- H2 adjusted (`perturb.py h2`, output in `local/results/perturb/full-h2.txt`): logistic model of a wrong
  answer on a restricted cubic spline of logit(confidence) (4 knots), the flip, and source; risk ratio and
  difference by standardization; 95% intervals from 200 resamples within source.

  | Flip definition                      | Flipped | Raw RR | Adjusted OR       | Adjusted RR       | Adjusted RD             | MH OR (source x decile) |
  |--------------------------------------|--------:|-------:|-------------------|-------------------|-------------------------|-------------------------|
  | Any order (all rotations and perms)  |     285 |   3.71 | 2.20 [1.52, 3.22] | 1.92 [1.43, 2.57] | +0.052 [+0.025, +0.086] | 2.41 [1.64, 3.54]       |
  | Fixed budget (base + 6 permutations) |     175 |   5.02 | 3.39 [2.20, 5.11] | 2.66 [1.93, 3.47] | +0.094 [+0.054, +0.139] | 3.84 [2.52, 5.85]       |

  Adjustment roughly halves the raw effect, and what is left is well away from 1. Leaving out any one
  source keeps the adjusted OR between 2.07 and 2.39 (any order) and 2.97 and 4.27 (fixed budget). Per source,
  the fixed-budget OR is above 1 in all six sources with 10 or more flips (2.07 to 8.76), with intervals above
  1 in five (Banking77: [0.89, 4.85]). The fixed budget is the better definition as well as the fairer one:
  counting every rotation gives many-option questions more chances to flip, and those extra flips carry less
  information.
- What predicts a flip (fixed budget, all answers, logistic with source): per standard deviation, the averaged
  margin -3.47 log-odds, log(option count) +0.02. Option count explains nothing once margin and source are in.
  The averaged margin is computed from the same orders and a flip lowers it, so this is partly circular; a
  margin independent of the orders (similarity of the option descriptions) would settle it.
- The H2 flag is narrow: it marks 6.3% of confident answers and catches 57 of 285 confident errors (20%).
  Averaging over orders fixes 3 of those 57, so it says "do not trust this", not what the right answer is.
- Candidate mass is the weakest signal (AUROC 0.828); none of these questions lacks a fitting option, so this
  says nothing about the case it exists for.
- GoEmotions is wrong 63.5% of the time and changes answer on 74.5% of questions; its single labels on
  overlapping emotions are likely noisy (inference, pending the hand audit).

### Audit of confident flipped errors (2026-09-25): inconclusive

The 57 base-order answers with confidence 0.9 and up that change under some order and are wrong were coded by
hand (model error; options overlap; gold likely wrong; text does not decide). Two passes, one by Claude and
one by a human, agreed on 49% of cases (Cohen's kappa 0.23): the categories were not sharp enough to code
reliably, and the human pass was set aside. What the audit showed anyway: the flipped errors are a mix of
clear model errors and items whose label is contested, and a single gold label cannot separate the two. Nor
would a larger audit of flipped errors alone: the question is whether flipped confident errors differ from
stable ones, and that needs human judgement on hundreds of blind cases.

Two checks that need no new labels:

- Sources with cleaner labels show the larger effect, the opposite of what label noise predicts. Fixed-budget
  flip, confidence 0.9 and up, Mantel-Haenszel odds ratio over source x confidence decile: CLINC150, MNLI,
  DBpedia, SemIf and AG News together 6.65 [3.04, 14.56] (53 flipped, 26.4% wrong; 3,040 stable, 4.2%);
  Banking77, its card intents, GoEmotions and TREC together 3.17 [1.92, 5.22] (122 flipped, 27.9%; 1,301
  stable, 8.3%).
- Published multi-annotator data measures each item's ambiguity directly: ChaosNLI (100 annotations per item
  for 1,599 MNLI dev items) and GoEmotions' per-rater rows. See the next section.

### Against human label distributions (2026-09-25)

Two sources come with per-item human votes, so an item's ambiguity is measured: ChaosNLI (Nie et al., 2020;
100 annotators on 1,599 MNLI dev items, fetched as `chaos_mnli` from the tasksource mirror, the label being
the 100-vote majority) and GoEmotions (3 to 5 raters per comment, the per-rater CSVs from Google's bucket,
joined to the 1,000 items already run). `perturb.py human`; outputs `human-chaos.txt` and
`human-goemotions.txt` in `local/results/perturb/`. ChaosNLI deliberately sampled contested items: its
100-vote majority differs from the original MNLI label on 508 of 1,599, and only 289 items reach 80%
agreement. An item is "contested" below when its majority label got under 60% of the votes.

| Measure                                                      | ChaosNLI (1,599)           | GoEmotions (1,000)         |
|--------------------------------------------------------------|----------------------------|----------------------------|
| Contested items                                              | 604 (38%)                  | 154 (15%)                  |
| AUROC for a contested item: fixed-budget flip                | 0.511                      | 0.554                      |
| AUROC for a contested item: spread / 1-conf / 1-conf_avg     | 0.541 / 0.532 / 0.547      | 0.555 / 0.548 / 0.564      |
| Confident answers (0.9 and up) that flip, fixed budget       | 21 of 661                  | 18 of 150                  |
| Of those, on contested items: flipped / stable               | 66.7% / 33.6%              | 5.6% / 9.8%                |
| Human share of the model's answer: flipped / stable          | 0.48 / 0.60                | 0.47 / 0.64                |
| H2 on items with 80%+ agreement: flipped / stable wrong      | 2 of 5 / 8 of 140          | 2 of 7 / 7 of 63           |
| Jensen-Shannon distance to the human distribution (bits): base order / averaged | 0.164 / 0.142 | 0.507 / 0.479         |

- Neither a flip nor any confidence measure tracks human disagreement: every AUROC for a contested item is
  near 0.5 on both sources. The model's uncertainty and the annotators' are different things here, in line
  with ChaosNLI's own finding that models do not reproduce human label distributions.
- Confident flips are rare on these sources (21 and 18), so the H2-on-unanimous test is underpowered; its
  point estimates go the same way as the full run. What can be said: on both sources humans agree with the
  model's answer less when it flips (0.48 against 0.60, 0.47 against 0.64). On ChaosNLI two thirds of the
  flipped confident answers sit on contested items, on GoEmotions fewer than the stable ones do. So
  "flips mark contested labels" is not a general explanation of H2; together with the larger effect on the
  clean sources (previous section), the weight of evidence is that confident flips mark model errors, with
  NLI a partial exception.
- Averaging over orders moves the model's probabilities toward the human distribution on both sources
  (13% and 6% lower Jensen-Shannon distance), without changing which answer humans prefer on average (share
  0.522 both ways on ChaosNLI). This is a gain that label noise cannot manufacture: the target is the whole
  human distribution, not one label. Comparison with ChaosNLI's published model numbers is pending a check
  of their distance definition and base.

### Cross-model runs (2026-09-25; Muse Glimmer 30B and gpt-oss-20b queued 2026-09-26)

The other pinned models, same data and settings as Run 2, run in sequence on the box (`models.sh`, in
`local/results/perturb/box/`; outputs and analyses in `local/results/perturb/models/`). Fixed-budget flip
throughout: the base order and the same 6 random permutations for every question. Three model families.

| Measure                                                     | Qwen3.5-4B          | Qwen3.5-2B          | Granite 4.2 3B      | Granite 4.0 H Tiny  | SmolLM3-3B          |
|-------------------------------------------------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| Wrong, base order                                           | 21.6%               | 34.8%               | 32.7%               | 34.7%               | 35.7%               |
| Wrong after averaging over orders (relative change)         | 19.9% (-8%)         | 27.9% (-20%)        | 28.8% (-12%)        | 26.9% (-22%)        | 29.6% (-17%)        |
| Answers that change under some order                        | 24.0%               | 58.4%               | 44.0%               | 59.7%               | 55.1%               |
| Error-detection AUROC, single-order / averaged confidence   | 0.863 / 0.880       | 0.799 / 0.836       | 0.755 / 0.807       | 0.787 / 0.840       | 0.804 / 0.841       |
| Spread AUROC                                                | 0.866               | 0.827               | 0.804               | 0.816               | 0.823               |
| Candidate mass AUROC                                        | 0.828               | 0.725               | 0.566               | 0.679               | 0.741               |
| H1: spread added to averaged confidence, held-out AUROC gain | -0.002 [-0.002, -0.001] | -0.000 [-0.002, +0.001] | +0.003 [+0.001, +0.005] | -0.004 [-0.005, -0.002] | -0.002 [-0.004, -0.000] |
| Confident answers (0.9+) that flip                          | 175 of 4,516 (4%)   | 1,084 of 3,310 (33%) | 2,193 of 5,972 (37%) | 1,809 of 4,160 (43%) | 1,119 of 3,668 (31%) |
| Wrong: flipped / stable                                     | 27.4% / 5.5%        | 29.1% / 5.5%        | 51.3% / 13.6%       | 34.5% / 6.3%        | 34.8% / 6.8%        |
| Adjusted risk ratio (spline of confidence + source)         | 2.66 [1.93, 3.47]   | 2.82 [2.35, 3.49]   | 1.80 [1.64, 1.97]   | 3.48 [2.95, 4.25]   | 3.40 [2.82, 4.00]   |
| Adjusted risk difference                                    | +0.094              | +0.132              | +0.160              | +0.210              | +0.195              |
| Mantel-Haenszel OR over source x confidence decile          | 3.84 [2.52, 5.85]   | 4.19 [3.22, 5.45]   | 2.46 [2.10, 2.87]   | 6.13 [4.86, 7.72]   | 4.97 [3.94, 6.27]   |
| Per-source MH OR, sources with 10+ flips                    | 2.1 to 8.8 (6)      | 2.4 to 8.9 (7)      | 2.6 to 14.6 (7 of 8); MNLI 0.59 | 2.7 to 17.6 (8 of 8) | 2.0 to 15.9 (8 of 8) |
| ChaosNLI: distance to human distribution, base / averaged   | 0.164 / 0.142       | 0.211 / 0.120       | 0.341 / 0.242       | 0.160 / 0.108       | 0.194 / 0.118       |
| ChaosNLI, 80%+ human agreement: flipped / stable wrong      | 2 of 5 / 8 of 140   | 39.2% (51) / 17.2% (29) | 61% (132) / 55% (103), at chance on NLI | 16.7% (24) / 10.3% (78) | 12.1% (58) / 7.1% (28) |

- All three results hold on every model: averaging removes 8 to 22% of errors and adds 0.02 to 0.05 AUROC;
  spread adds between -0.004 and +0.003 AUROC beyond the averaged confidence; a confident answer that
  flips is wrong 1.8 to 3.5 times as often as a stable one after adjustment, with every per-source odds
  ratio above 1 except Granite 4.2 on MNLI.
- The smaller models flip far more (55 to 60% of answers against 24% for the 4B) and gain more from
  averaging (17 to 22% fewer errors against 8%). Their confident-flip rates are alike: 5 to 7% wrong when
  stable, 27 to 35% when flipped. Granite 4.2 3B is the outlier at 14% and 51%: it is overconfident
  everywhere, so the flag's ratio is lower and its absolute difference (16 points) is not.
- A precondition, found on Granite 4.2 3B and MNLI: at chance on a task (54% wrong), the flip runs the
  other way (odds ratio 0.59 [0.43, 0.81]). A guessing model flips everywhere, so a flip stops meaning
  anything. Leaving MNLI out raises its adjusted odds ratio from 2.83 to 4.63.
- Against human distributions (ChaosNLI): averaging lowers every model's distance, by 13% (4B) to 43% (2B),
  and puts three of the small models below the single-order 4B. On near-unanimous items the flipped
  confident answers are wrong more often than the stable ones for all four models with enough cases (the
  2B most clearly, 39% against 17%); Granite 4.2 is at chance on NLI and says nothing.
- Candidate mass separates errors on the Qwens (0.83, 0.73) but barely on Granite 4.2 (0.57): its mass
  stays on the letters right or wrong. As a general error signal it is the weakest; its job is the
  no-fitting-option case, untested here.
- H1 detail: against the averaged margin rather than confidence, spread reaches +0.011 [+0.007, +0.014] on
  the 2B and +0.009 on Granite 4.2; confidence stays the better base signal, and the statement is "at most
  a few thousandths beyond it".

### PriDe as the comparison arm, offline (2026-09-26)

PriDe (Zheng et al., 2024) estimates the model's prior over option ids as the geometric mean of the observed
id probabilities over a question's cyclic rotations, averaged over a small sample, and divides it out of the
others' base-order answers. Every run here holds every rotation, so `perturb.py pride` computes it exactly:
a 5% sample per source and option count for the prior, the rest evaluated (6,508 questions per model).
Output in `local/results/perturb/pride.txt`.

| Wrong answers, base / PriDe / averaged over orders | Qwen3.5-4B          | Qwen3.5-2B          | Granite 4.2 3B      | Granite Tiny        | SmolLM3-3B          |
|----------------------------------------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| All 6,508                                          | 21.7 / 23.0 / 20.0% | 34.5 / 30.3 / 28.6% | 32.8 / 32.1 / 29.2% | 34.5 / 32.6 / 27.4% | 35.8 / 34.3 / 30.5% |
| AUROC, base / PriDe / averaged confidence          | 0.864 / 0.860 / 0.867 | 0.797 / 0.797 / 0.824 | 0.756 / 0.786 / 0.787 | 0.789 / 0.799 / 0.827 | 0.804 / 0.807 / 0.818 |

- Averaging beats PriDe on every model. PriDe helps the four small models a little and makes the 4B worse
  (21.7% to 23.0% wrong).
- Why it hurts the 4B: its decisions are position-neutral (AG News argmax share by position 0.26, 0.25,
  0.25, 0.25; mean probability 0.27, 0.25, 0.24, 0.24), but position A never gets a tiny probability (below
  0.001 in 16% of rotations against 49 to 57% for the others). PriDe's log-space estimator reads that floor
  as a prior of 0.73 on A and divides it out, pushing decisions toward later positions. The id bias of this
  readout sits in the tail of the distribution, where it does not affect decisions; PriDe's multiplicative
  model corrects it anyway. The 2B has decision-level position bias (MNLI argmax share 0.54, 0.33, 0.13),
  and there PriDe does help, though less than averaging.

### Reworded-criterion control (2026-09-26)

Is boundary crossing specific to the option order, or does any meaning-preserving change of the question
flag the same answers? Qwen3.5-4B on the RTX 3090 box (native helper), the 8 public sources with three
hand-written paraphrases of each criterion (`data/rewords.json`; SemIf's per-item criteria have
none), 6,848 questions: the base order, 6 random permutations, and the base order under each paraphrase.
`perturb.py run --reword`; output `local/results/perturb/text/reword.jsonl`, analysis in `reword-analysis.txt`.

| Answer changes under                     | All answers (6,848) | Wrong  | Confidence 0.9+ (4,437) | Wrong  |
|------------------------------------------|--------------------:|-------:|------------------------:|-------:|
| neither                                  |               5,057 |   9.3% |                   4,254 |   5.5% |
| a paraphrase only                        |                 131 |  33.6% |                      12 |  16.7% |
| an option order only                     |               1,092 |  54.2% |                     166 |  24.7% |
| both                                     |                 568 |  67.1% |                       5 |  20.0% |

AUROC for a wrong answer: order flip 0.763, paraphrase flip 0.617; order spread 0.865, paraphrase spread
0.829.

- Option order changes answers far more often than paraphrase does: 24% of questions against 10%, and among
  confident answers 171 against 17. Three paraphrases against six permutations is not an equal budget, but
  the gap is an order of magnitude among confident answers.
- The two are not the same signal. The largest problem cell is answers that survive every paraphrase and
  change under an order (1,092, 54% wrong): the question is understood the same way each time and the
  decision still sits on a boundary. Paraphrase-only flips exist (131, 34% wrong) but are rarer and weaker.
- Read together with the flip effect: an order flip marks a decision-boundary event of the option
  representation, not a fragile reading of the criterion. That is the "specific to the decision
  representation" outcome of the paper plan's 2x2.

### Generated-text arm (2026-09-26)

"My Answer is C" and "Look at the Text" report that first-token probabilities disagree with generated answers
on instruction-tuned models (mismatch over 60%) and that generated answers are more order-robust. Test: the
same 6,944 questions, base order and the same 6 random permutations, answered by generation through
llama-server directly (`perturb.py text`: greedy, 32 tokens, thinking off, the first generated token's
alternatives recorded), Qwen3.5-4B on the RTX 3090. Two prompts: "bare", llav's own, which asks for a lone
letter; "text", which asks for the letter, a colon and the option's description. Answers parsed from the text
(a leading letter, else the longest option key or description the text contains). Outputs
`local/results/perturb/text/text-{bare,text}.jsonl`, analyses in `text-*-analysis.txt`.

| Measure                                                   | bare prompt      | text prompt      | readout (Run 2) |
|-----------------------------------------------------------|------------------|------------------|-----------------|
| Parsed as a letter                                        | 100.0%           | 99.8%            |                 |
| Generated answer = first-token argmax of the same call    | 100.0%           | 100.0%           |                 |
| Generated answer = llav's readout answer, same order      | 99.2%            | 95.1%            |                 |
| Wrong, base order                                         | 21.7%            | 22.1%            | 21.6%           |
| Wrong, majority vote over the 7 orders                    | 20.1% (-7%)      | 20.6% (-7%)      | 19.9% (-8%, averaged) |
| Answer changes under some order                           | 24.1%            | 25.5%            | 24.0%           |
| Wrong when the text answer flips / is stable              | 59.6% / 9.7%     | 58.5% / 9.7%     |                 |
| Wrong (text answer) when the readout flips / is stable    | 59.1% / 9.9%     | 59.2% / 10.4%    |                 |

By the generated first token's probability, wrong-answer rate stable / flipped (bare prompt): 0.9 to 0.97
13.1% / 31.6% (733, 95); 0.97 to 0.99 6.7% / 28.3% (773, 46); 0.7 to 0.9 24.1% / 51.0% (639, 402).

- On this model and prompt the first-token readout and the generated answer are the same thing: identical
  in every call, and the same as llav's separate readout in 99.2% of cases (the 0.8% are numeric near-ties
  between two runs). The mismatch those papers describe comes from models that open with "Sure" or refuse;
  with thinking off and a prompt that asks for a lone letter, Qwen3.5-4B does neither. The "text" prompt,
  which invites a longer answer, still starts it with the letter 99.8% of the time.
- Generated answers are not more order-robust: they change under some order on 24 to 26% of questions,
  the readout's 24%.
- The flip effect is a property of the decision, not of the readout: a generated answer that changes
  under some order is wrong 6 times as often as one that does not, and the readout's flips predict the
  generated answer's errors exactly as well as its own. The confidence-band pattern (Figure 1) repeats with
  the generated token's probability as the confidence.
- Majority vote over orders of generated answers removes 7% of errors, the readout's averaging 8%.
- Cost: generation took 3,689 s for the bare prompt and 7,067 s for the text prompt on 8 parallel slots,
  against about 95 minutes for the readout over 3 times as many orders through llama-server, and far less
  through the helper.

## Open

- A gold-option-removed set for candidate mass, in matched pairs.
- PriDe: done, see above; the K-orders timing table remains.
- VariErr NLI's 500 re-annotated MNLI items as the audit target: error against valid variation, already
  labelled.
- The margin-predicts-flips result uses the margin from the same orders; a margin independent of them
  (similarity of the option descriptions) would settle whether confusability drives flipping.
- Whether the pilot's confident-flip error rate (41%, from the dataset's own option order) was inflated by
  that order.
