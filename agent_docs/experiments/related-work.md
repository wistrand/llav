# Related work for the permutation experiment

Collected 2026-09-25 by web search; abstracts read for the papers marked (read), the rest known from search
summaries and titles only, so check before citing. Grouped by what each line of work settles and what it
leaves for [permutation-uncertainty.md](permutation-uncertainty.md).

## Option order and selection bias in multiple choice

- Pezeshkpour and Hruschka, NAACL Findings 2024 ([arXiv 2308.11483](https://arxiv.org/abs/2308.11483)):
  accuracy gaps of 13 to 75 points across option orders. Conjecture: sensitivity arises when the model is
  torn between its top 2 or 3 options and position bias breaks the tie; majority vote over reorderings gains
  up to 8 points. Our margin result (a flip is predicted by the averaged top-two margin, not by option count)
  quantifies that conjecture.
- Zheng et al., ICLR 2024, "Large Language Models Are Not Robust Multiple Choice Selectors"
  ([arXiv 2309.03882](https://arxiv.org/abs/2309.03882)): selection bias is mostly token bias on the option
  ids (A, B, C). PriDe estimates that prior from permuted content on 2 to 20% of the test items and divides
  it out of the rest, label-free, cheaper than full or cyclic permutation, and raises accuracy (Llama-30B on
  MMLU 53.1 to 56.4). Our pilot found the position picked stays near 1/K under rotation, so llav's
  instability looks like near-tie noise rather than the additive id prior PriDe removes; PriDe is the
  natural comparison arm.
- Wei et al., ACL Findings 2024, "Unveiling Selection Biases": order and token sensitivity together.
- Gupta et al., 2024 ([arXiv 2406.19470](https://arxiv.org/abs/2406.19470), read): every model tested drops
  on MMLU when options are shuffled; they propose reporting the share answered right by chance.
- "Accuracy and Order Sensitivity Diverge Under Label-Free Strategies"
  ([arXiv 2608.11947](https://arxiv.org/abs/2608.11947), 2026, read): six models (GPT-4.1 mini, Gemini 2.5
  Flash, Llama 3.1 8B, Qwen 2.5 7B), MMLU and ARC, 1,000 items each. Cyclic permutation with majority voting
  improves 5 of 6 models on each benchmark; removing positional influence does not reliably raise accuracy.
  Defines a per-question flip rate (at least two distinct answers across permutations) as an order-sensitivity
  metric, but uses it to measure debiasing, not as a per-item error signal.
- Guda, Francis, Ashungafac, Joe-Wong and Busogi, "Rethinking Selection Bias in LLMs: Quantification and
  Mitigation using Efficient Majority Voting", ICLR 2025 workshop on uncertainty and hallucination (read, PDF
  in ~/Downloads), extended as "Quantifying and Mitigating Selection Bias in LLMs: A Transferable LoRA
  Fine-Tuning and Efficient Majority Voting Approach" ([arXiv 2511.21709](https://arxiv.org/abs/2511.21709),
  Nov 2025). Three things overlap with this experiment:
  - Their bias metric is the mean over options of the variance of that option's probability across
    permutations, label-free. Per item that is our `spread` up to the choice of distance. They use it as a
    dataset-level bias score; they do not test it against per-item errors, which is what H1 does, and H1
    says it carries nothing beyond the averaged answer's confidence.
  - Their mitigation is the averaged probability over k permutations (they call it majority voting), our
    averaged answer. They report the bias metric of the averaged output as 0.00, which holds by
    construction: an average over permutations is order-invariant. Accuracy gains on Qwen2.5-3B-Instruct,
    Phi-2 and Llama 3.2 3B over TeleQnA, MedMCQA and QASC are large (QASC 0.63 to 0.94 for Phi-2, 0.72 to
    0.86 for Llama), far above our 7.6% error reduction on Qwen3.5-4B and closer to the 26% seen so far on
    Qwen3.5-2B; smaller and older models are more order-sensitive.
  - BaQCKV caches the KV state of the question and context once and appends each permutation's option
    block with an adjusted mask; token savings (k-1)|Q+C| / (k|Q+C+O|), 90% in their runs, inference time
    down 51 to 89%. llav's native helper is the same construction (decode the prefix once, copy it to one
    sequence per question, decode all suffixes in one batch) built independently; they are prior work and
    should be cited for it. Their savings assume one question's permutations; llav also shares the prefix
    across different questions about one state.
- Wang et al., NeurIPS 2024, "Eliminating Position Bias of Language Models: A Mechanistic Approach" (PINE,
  [arXiv 2407.01100](https://arxiv.org/abs/2407.01100)): attributes position bias to causal attention and
  RoPE; bidirectional attention between segments makes inference order-invariant without training. An
  architectural alternative to averaging; not available through llama-server.
- Training-based debiasing: permutation-aware GRPO ([arXiv 2603.21016](https://arxiv.org/abs/2603.21016)),
  teacher-student permutation debiasing ([arXiv 2403.13590](https://arxiv.org/pdf/2403.13590)). Out of scope
  for a runtime that takes any GGUF.

- Open Jev reproductions, surveyed in [comparisons.md](../comparisons.md#the-open-decision-model-ecosystem):
  reflex (frozen Qwen3.5-4B, Evidence/Criterion prompt) averages two option orders inside one pass and
  reports hard-tier accuracy 0.658 to 0.685 and ECE 0.086 to 0.081 for the 4B, 0.703 to 0.766 for a 27B, and
  that "scattered readings across orders predict correctness"; prior art for both the averaging result and a
  version of H2, without the adjusted test. decider notes that reversing the order of packed questions
  changes up to 12% of answers. JevBench's diagnostic found 5% decisive flips for Jev itself in four orders.

## First-token readout against text answers

- Wang et al., 2024, "'My Answer is C': First-Token Probabilities Do Not Match Text Answers in
  Instruction-Tuned Language Models" ([arXiv 2402.14499](https://arxiv.org/abs/2402.14499), read): first-token
  and text answers disagree on over 60% of items for heavily tuned models, from "Sure" openers and refusals;
  they warn against first-token evaluation alone.
- "Look at the Text: Instruction-Tuned Language Models are More Robust Multiple Choice Selectors than You
  Think" ([arXiv 2404.08382](https://arxiv.org/abs/2404.08382), read): text answers are more robust to option
  reordering than first-token probabilities, even PriDe-debiased ones. The strongest counter-position to
  llav's readout, and the reason a generated-text arm belongs in the readout comparison.
- Output prefilling for first-token predictions ([arXiv 2505.15323](https://arxiv.org/abs/2505.15323)):
  forces a valid first token. llav's template-end prefix for Muse (` to=user<|message|>`) is the same move.
- Both papers describe the failure llav's candidate mass measures per question: the first token is not an
  answer letter. llav's startup probe refuses models where it never is.

## Scoring label tokens and calibrating them

- Holtzman et al., EMNLP 2021, "Surface Form Competition"
  ([arXiv 2104.08315](https://arxiv.org/abs/2104.08315)): option texts split probability with paraphrases of
  themselves; domain-conditional PMI corrects it. Letter labels sidestep competition between option texts;
  the mass left on non-letter tokens is what candidate mass reports.
- Zhao et al., ICML 2021, "Calibrate Before Use" ([arXiv 2102.09690](https://arxiv.org/abs/2102.09690)):
  content-free inputs expose label bias; contextual calibration gains up to 30 points. Batch Calibration
  (Zhou et al., 2023, [arXiv 2309.17249](https://arxiv.org/abs/2309.17249)) does it from the batch. Both are
  candidates for the calibration comparison in the paper plan; llav's per-type temperature is the simplest
  arm.

## Consistency under perturbation as uncertainty

- Test-time augmentation for aleatoric uncertainty (Ayhan and Berens, 2018; Wang et al., Neurocomputing
  2019, [arXiv 1807.07356](https://arxiv.org/abs/1807.07356)): variance of the prediction over input
  transformations as uncertainty, the classical analogue of permutation spread. Our H1 result is that, for
  option order, the averaged prediction's own confidence carries that information and the variance adds none.
- Self-consistency vote agreement as confidence: SCOPE
  ([arXiv 2602.13110](https://arxiv.org/pdf/2602.13110)) reports vote-agreement AUROC at or below chance
  for selective judging in 5 of 6 settings, in line with our null for spread and the share of orders that
  change the answer.
- Flip-Flop Consistency (ACL 2026, [arXiv 2510.14242](https://arxiv.org/abs/2510.14242)): trains toward the
  majority answer over prompt variations. Cycles of Thought
  ([arXiv 2406.03441](https://arxiv.org/html/2406.03441v1)): stable explanations as confidence.
- Not found: a per-item test of whether an argmax change under option permutation predicts a wrong answer
  beyond the model's confidence, adjusted for confidence and task. The H2 analysis appears to be new; state it
  as "we found no prior test", not as absence.

## Human label distributions

- Nie et al., EMNLP 2020, ChaosNLI ([ACL Anthology](https://aclanthology.org/2020.emnlp-main.734.pdf)): 100
  labels per item for 4,645 SNLI, MNLI and abductive NLI items; models evaluated by JSD and KL to the human
  distribution. Their BERT-large and RoBERTa-large MNLI numbers (JSD about 0.31) are in their leaderboard; the
  base of the logarithm and whether they report distance or divergence must be checked before comparing our
  0.164 and 0.142.
- Zhou, Nie and Bansal, ACL Findings 2022, "Distributed NLI": ensembles, MC dropout, recalibration and
  distribution distillation to predict human distributions.
- "Capturing Label Distribution: A Case Study in NLI" ([arXiv 2102.06859](https://arxiv.org/html/2102.06859))
  and later work note that calibration can lower JSD without modelling ambiguity, a caution for reading our
  averaging gain: it lowers JSD, and the paper should say whether it also tracks per-item human entropy (our
  contested-item AUROC says it does not).
- Weber-Genzel et al., ACL 2024, VariErr NLI ([arXiv 2403.01931](https://arxiv.org/abs/2403.01931), read):
  separates annotation error from valid label variation on 500 re-annotated MNLI items through explanations
  and validity judgements; automatic error detection and GPT-4 both fall short of humans. This is the audit
  our two-rater pass could not do; VariErr's items are the right target for it, with no new labelling.
- "Decoupling the Effect of Chain-of-Thought Reasoning: A Human Label Variation Perspective"
  ([arXiv 2601.03154](https://arxiv.org/abs/2601.03154), read): with CoT, the top answer is set by the
  reasoning text and the ranking of alternatives by model priors. Relevant to a generated-text arm: text
  readout changes the argmax, not the distribution.
- "A Rose by Any Other Name" (ACL Findings 2025): LLM explanations approximate human judgement distributions.
  Plank, 2022, "The 'problem' of human label variation": the case that a single gold label is the wrong
  target where humans disagree.

## Constrained decoding

- Chavan, Sept 2026, "Constrained Decoding Eliminates Structural Failures in Small LLMs but Reveals a
  Scale-Dependent Semantic Gap" ([arXiv 2609.23742](https://arxiv.org/abs/2609.23742), read): five 0.6B to 4B
  models, 14 structured-output tasks; constrained decoding takes schema validity to 100% but content accuracy
  keeps a scale-dependent gap. A letters-only readout is the most constrained output there is; the readout
  comparison in the plan measures the same structural-against-semantic split.

## Where the experiment sits

Known before this work: order sensitivity is large; averaging over orders (cyclic or full) raises accuracy;
the id prior can be estimated and removed (PriDe); prefix sharing makes permutations cheap (BaQCKV); text
answers can be more order-robust than first-token readout. What the experiment adds, subject to the runs:
runtime-defined criteria rather than benchmark questions; the head-to-head of averaged confidence against
permutation spread at equal compute (spread adds nothing); the confident-flip effect adjusted for confidence
and task, and its interpretation against human vote distributions rather than single labels; and the
averaged distribution's distance to human distributions. The open counter-position to answer is "Look at
the Text": a generated-text arm on the same items.
