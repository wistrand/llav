# Paper outline: Order Shouldn't Matter

Working title: *Order Shouldn't Matter: Marginalizing Option Order in Direct Decision Readouts from Small
Language Models*. Drafted 2026-09-26 from the results in
[permutation-uncertainty.md](permutation-uncertainty.md) and the literature in
[related-work.md](related-work.md). Every figure and table below names its data file; numbers already in
hand are filled in, arms not yet run are marked TODO. Target: a Findings paper (ACL or EMNLP) as it stands,
main track with the two TODO arms.

## Thesis

For decisions whose criteria and options are defined at request time and read directly from a model's
next-token probabilities, the order in which the options are listed is a nuisance variable. Averaging over
it improves the decision and its probabilities; the size of the disagreement between orders carries no
information beyond the averaged confidence; but a decision that changes under an equivalent order is wrong
2 to 4 times as often as one that does not, at every confidence level.

## Contributions

1. A measurement of order sensitivity on runtime-defined decisions: 24 to 60% of decisions change under
   some order across five models; sensitivity tracks the confusability of the options, not their number.
2. Permutation marginalization evaluated against both single labels and 100-annotator label distributions:
   8 to 22% fewer errors, +0.02 to +0.05 error-detection AUROC, 13 to 43% closer to human distributions.
3. A null: permutation spread adds -0.004 to +0.003 AUROC beyond the averaged confidence, a per-item test of
   the permutation-variance bias metric of Guda et al.
4. Boundary crossing as a warning sign: an answer that changes under an equivalent order is wrong 2 to 4
   times as often at the same confidence, adjusted for confidence (spline) and task, replicated per source,
   across three model families, and on items 80% or more of 100 humans agree on. With its precondition (the
   model is above chance on the task) and its limits (no correction, no tracking of human disagreement).
5. The cost model: K orders cost K suffix decodes over one shared prefix, not K reads of the text.

## Sections

### 1. Introduction

The routing example from the README (billing / technical / sales). The claim that order shouldn't matter,
the numbers showing it does, the three findings. One paragraph on why runtime-defined decisions differ
from benchmark MCQ: the options are the caller's, often near-synonyms, and there is no training set.

### 2. Related work

From [related-work.md](related-work.md), in this order: order sensitivity and selection bias (Pezeshkpour
and Hruschka; Zheng et al. PriDe; Wei et al.; Gupta et al.; the 2026 label-free strategies paper's flip
rate); permutation averaging and its cost (Guda et al. workshop and arXiv; PINE as the architectural
route); first-token readout against text answers ("My Answer is C", "Look at the Text"); consistency as
uncertainty (test-time augmentation; self-consistency vote agreement; SCOPE's null); human label variation
(ChaosNLI, Distributed NLI, VariErr, Plank). Two sentences each; the positioning paragraph from the end of
related-work.md.

### 3. Setting and readout

- The decision types (noul, choice, score) and the prompt: SemIf's `direct-options-v1`, letters A to Z, one
  forward pass, softmax over the declared letters (prompt.py, questions.py). Thinking disabled.
- Candidate mass, defined, with one sentence: it is not used as an error signal here (Table 6 shows why).
- Data: Table 1.
- Models: Qwen3.5-4B and 2B, Granite 4.2 3B, Granite 4.0 H Tiny (7B MoE, about 1B active), SmolLM3-3B,
  all Q8_0 GGUF through llama-server, deterministic (noise floor 0.0000).
- Perturbations: a random base order per question (seeded), 3 identical repeats, 6 random permutations,
  every cyclic rotation; the fixed budget (base + 6 permutations) for all per-item flip statistics, the
  reason stated (more rotations give many-option questions more chances to flip). Fixed-order questions
  (single-letter keys) excluded.

### 4. Order sensitivity (Figure 2, Table 2)

How often answers change, per source and model; the confusability result: flips are predicted by the
averaged top-two margin (-3.5 log-odds per SD on the 4B, -4.2 on the 2B) and not by option count (+0.02,
+0.13). The circularity caveat (the margin comes from the same orders) and the DBpedia-against-Banking77
contrast as the plain illustration.

### 5. Marginalizing over orders (Table 3, Figure 3)

Errors before and after; AUROC of single-order against averaged confidence; ChaosNLI distances. The
observation that the gain is largest for the weakest models. TODO arm: PriDe on the same items as the
cheaper comparison.

### 6. Disagreement is not an uncertainty score (Table 4)

H1 as pre-registered: held-out logistic combination, leave-one-source-out, paired bootstrap within source.
Five models, all near zero. The margin variant (+0.011 on the 2B) reported and dismissed as a few
thousandths. Relate to Guda et al.'s bias metric.

### 7. Boundary crossing is (Figure 1, Table 5)

The confidence-band figure; the adjusted models (spline of logit confidence + source; risk ratio and
difference by standardization; Mantel-Haenszel over source x decile; leave one source out; per source). The
ChaosNLI near-unanimous subset. The precondition (Granite 4.2 on MNLI, odds ratio 0.59). What it does not
do: averaging fixes 3 of 57 flipped confident errors on the 4B; flips do not predict human disagreement
(AUROC 0.51 to 0.55).

### 8. Cost (Table 7)

Shared-prefix batched readout: one prefix decode, K x Q suffixes in one batch. Timings from performance.md
and the helper; cite BaQCKV for the construction. TODO: the K-orders timing on the RTX PRO 4000.

### 9. Limitations

Small models only; one readout (TODO arm: generated text); labels noisy in three sources, the ChaosNLI
subset as the defence; the flag is narrow on the best model (4% of confident answers) and broad on weak
ones; the pilot's inflated 41% from the dataset's own order, as a cautionary note on reference orders.

### 10. Conclusion

The sentence: marginalize the order to improve the decision; do not read the disagreement's size as
confidence; but when a decision crosses to a different answer under an equivalent order, treat it as a
warning.

## Figures

### Figure 1 (the headline): wrong-answer rate by confidence, stable against flipped, per model

Five small panels in a row, one per model. x: base-order confidence band (0.5-0.7, 0.7-0.9, 0.9-0.97,
0.97-0.99, 0.99-0.999, 0.999-1). y: share of answers that are wrong. Two lines: stable under the fixed
budget, and flipped. Point size or a label for n. Data: `local/results/perturb/figure1-data.txt`
(regenerate with the script in that file's header once written into `perturb.py`). The values:

| Model              | 0.5-0.7 stable / flipped | 0.7-0.9      | 0.9-0.97     | 0.97-0.99    | 0.99-0.999   | 0.999-1      |
|--------------------|--------------------------|--------------|--------------|--------------|--------------|--------------|
| Qwen3.5-4B         | 39 / 63                  | 24 / 49      | 13 / 35      | 7 / 24       | 4 / 7 (n=27) | 1 / (n=4)    |
| Qwen3.5-2B         | 18 / 59                  | 18 / 52      | 10 / 37      | 7 / 26       | 4 / 18       | 1 / 0 (n=32) |
| Granite 4.2 3B     | 41 / 69                  | 29 / 68      | 39 / 67      | 23 / 58      | 20 / 58      | 12 / 42      |
| Granite 4.0 H Tiny | 20 / 67                  | 15 / 59      | 12 / 45      | 10 / 44      | 6 / 33       | 3 / 13       |
| SmolLM3-3B         | 16 / 67                  | 16 / 58      | 13 / 45      | 9 / 35       | 5 / 23       | 4 / 12       |

Caption point: the flipped line is above the stable line in every band on every model; the 0.9 cutoff
in the analysis was a choice, the effect is not confined to confident answers.

### Figure 2: order sensitivity by source and model

Grouped bars: share of questions whose answer changes under the fixed budget, sources on x sorted by the
4B's value (DBpedia 3% to GoEmotions 75%), one bar per model. Data: `analyze` output per model, or a
`changes_fixed` per-source pass. Caption: DBpedia's 14 options flip least; Banking77's 12 flip most; option
count does not explain it (Table 2).

### Figure 3: what averaging does to the probabilities (ChaosNLI)

Left: Jensen-Shannon distance to the 100-annotator distribution, base order against averaged, five models
(paired dots with an arrow). Right: the same two conditions as a reliability-style plot, model probability
of the human-majority label against human share, for the 4B. Data: `human-chaos.txt` per model
(`local/results/perturb/`, `local/results/perturb/models/`). TODO: check ChaosNLI's own base and
distance-against-divergence convention before adding BERT and RoBERTa reference points.

### Figure 4: H1 as a forest plot

Five models x two targets (base-order wrong, averaged wrong): the held-out AUROC gain of spread over the
averaged confidence with 95% intervals, all within +-0.005 of zero. Beside it, for scale, the gain of the
averaged confidence over the single-order confidence (+0.017 to +0.053). Data: `analyze` outputs.

## Tables

### Table 1: data

| Source                  | Type       | Options | Questions | Origin                                        |
|-------------------------|------------|--------:|----------:|-----------------------------------------------|
| AG News                 | topic      |       4 |     1,000 | HF fancyzhx/ag_news, test, spread sample       |
| DBpedia                 | entity type |     14 |     1,008 | HF fancyzhx/dbpedia_14, test                   |
| Banking77 card intents  | intent     |      13 |       520 | legacy-datasets/banking77, the 13 card intents |
| Banking77               | intent     |      12 |     1,000 | all 77 intents, gold + 11 nearest by name      |
| CLINC150                | intent     |      12 |       820 | clinc/clinc_oos plus, oos excluded, same construction |
| TREC fine               | question type | 2-17 |       500 | SetFit/TREC-QC, fine types within the coarse group |
| MNLI                    | NLI        |       3 |     1,000 | nyu-mll/glue, validation_matched               |
| GoEmotions              | emotion    |      12 |     1,000 | single-label items among 12 overlapping emotions |
| SemIf authored          | evidence, rules |  3 |        96 | authored144 minus candidate selection (fixed order) |
| ChaosNLI (MNLI)         | NLI        |       3 |     1,599 | tasksource mirror, 100 labels per item          |

6,944 questions after excluding 48 fixed-order ones; ChaosNLI separately. Note the three sources with
noisy labels (Banking77, GoEmotions, TREC) and the ChaosNLI subset as the check.

### Table 2: flips against confusability and option count

Logistic model of a fixed-budget flip on standardized averaged margin, log option count, and source, per
model: the two coefficients. In hand for the 4B (-3.47, +0.02) and 2B (-4.21, +0.13); Granite and SmolLM3
from their `h2.txt` last line. Plus the per-source flip rates with option counts (Figure 2's data).

### Table 3: marginalization

Per model: wrong base / wrong averaged / relative change; AUROC single / averaged; ChaosNLI JSD base /
averaged; human share of the answer base / averaged. All in the cross-model table of
permutation-uncertainty.md. TODO column: PriDe on the same items.

### Table 4: H1

Per model, for both targets and both bases (averaged confidence, averaged margin): the held-out AUROC
gain with interval. In hand for all five (`analyze` outputs).

### Table 5: the flip effect, adjusted

Per model: flips (count, share of confident answers); wrong flipped / stable; raw RR; adjusted OR, RR, RD
with intervals; MH OR; leave-one-source-out range; per-source OR range with the count of sources above 1;
the ChaosNLI 80%+ agreement cell. All in the cross-model table. A second panel: the same for the any-order
flip, to show the fixed budget is the stronger and fairer definition.

### Table 6: candidate mass is not an error signal

AUROC of 1-mass per model (0.83, 0.73, 0.57, 0.68, 0.74) against 1-confidence, with the sentence that no
item here lacks a fitting option. Goes in Limitations or an appendix; keeps the reviewer from asking.

### Table 7: cost

Per request: one prefix decode plus K x Q suffix decodes; measured seconds for K = 1, 3, 7, 14 on the RTX
PRO 4000 with and without the helper. TODO: measure; performance.md has K = 1.

## Appendix

- The exact prompt and an example request and response.
- The noise floor (Run 1): identical prompts, both readout paths, the helper's 0.0019 spread.
- The audit that failed (two raters, kappa 0.23) and why the ChaosNLI subset replaced it.
- The pilot with the dataset's own order (41% against 4.3%) as a warning about reference orders.
- Per-source versions of Table 5 for all models.

## TODO before submission, in order

1. Generated-text arm: done on the 4B (permutation-uncertainty.md). Generated answers equal the first-token
   readout in 99.2% of calls, flip as often under order, and flip-marked ones are wrong 6 times as often;
   Section 7 is about decisions. The "Look at the Text" objection becomes a paragraph in Section 3 (thinking
   off, a lone-letter prompt) with the table as an appendix. A small model is optional.
2. Reworded-criterion control: done 2026-09-26 (permutation-uncertainty.md): order flips are ten times as
   frequent as paraphrase flips among confident answers and carry more information; the paraphrase-stable,
   order-flip cell is the largest problem cell. Section 7 gets the 2x2 as Table 5b.
3. PriDe arm for Table 3: done, offline; averaging beats it on every model and it hurts the 4B.
4. K-orders timing for Table 7: done 2026-09-26 (performance.md, "Asking a question in several orders"):
   linear in K, 27 ms per extra order through the helper and 48 ms through llama-server.
5. Move the figure-1 computation into `perturb.py` (a `bands` subcommand) so the data file regenerates.
6. Check ChaosNLI's JSD convention; add reference points to Figure 3 if comparable.
