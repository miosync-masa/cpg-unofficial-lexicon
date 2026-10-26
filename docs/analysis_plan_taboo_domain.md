# Analysis plan: taboo domain and dictionary recording (exploratory)

This plan was written and committed on 2026-10-26, before any taboo_domain coding was run.

This analysis is exploratory: it was not part of the earlier plans, and its result is reported as such.

## Why

Whether a covert meaning is listed in a general dictionary may depend not only on how widely the meaning is used,
but also on the editorial policy of dictionaries, which tend to leave out taboo senses. If so, items whose covert
meaning lies in a taboo domain should be in a general dictionary less often.

## Prediction

Items with a taboo covert meaning have a lower share of `general_dictionary` (recording status) than items without.

## 1. Coding

- Coders: the same three as in the earlier coding runs, with the same settings: `gpt-5.6-sol` (no temperature),
  `gemini-3.1-pro-preview` (temperature 0), `claude-opus-5-5` (no temperature). A fresh process per model, batches
  of 10, a JSONL log of every call, resume on rerun.
- Items: all 217 items of `data/blind_items.csv`. The 6 items in `data/exclusions.csv` are coded but left out of the
  tallies.
- Input: the host (`host_original`, `host_romanized`), the covert meaning (`covert_target`), the tradition and the
  language. No pull, no recording status, no source, no other coder's label.
- Output: exactly one `taboo_domain` per item (codebook `coding/codebooks/codebook_taboo.md`):
  `sex` / `excretion` / `death_illness` / `insult` / `drugs_crime` / `none`. The codebook says: if more than one
  applies, choose the one at the centre of the covert meaning.
- Consensus: the majority label (≥ 2 of 3); `unresolved` when all three differ.
- Refusals: a batch whose 3 attempts all fail is resent item by item (up to 3 attempts per item). The log keeps every
  refused request, its batch and its items.
- Pilot: the first 10 items per model. If the codebook text causes no problem, the full run follows.

## 2. Analysis set

The 211 items of `results/items_final.csv` (217 minus the 6 exclusions), with the consensus `recording_status` of the
recording coding. Items with `UNRESOLVED` recording status or `unresolved` taboo domain are left out and counted.
`taboo` = 1 when the consensus `taboo_domain` is not `none`, else 0. `general` = 1 when the recording status is
`general_dictionary`, else 0 (`cultural_reference`, `unrecorded`).

## 3. Tallies and tests

1. Crosstab of `taboo` (yes / no) × recording status (`general_dictionary` / `cultural_reference` / `unrecorded`).
2. Main comparison: share of `general_dictionary` among taboo items minus the share among non-taboo items; 95% CI
   by Newcombe's method (Wilson score intervals, hybrid); Fisher's exact test on the 2 × 2 table of `taboo` ×
   `general`, one-sided in the predicted direction (taboo lower), with the two-sided p beside it.
3. Auxiliary, for the grouping by tradition: logistic mixed model `general ~ taboo + (1 | tradition)` (raw
   tradition labels). statsmodels has no maximum-likelihood fit for this model; it is fitted with
   `BinomialBayesMixedGLM.fit_vb` (variational Bayes, default priors), and the posterior mean, posterior SD and
   mean ± 1.96 SD of the `taboo` coefficient are reported. If the fit does not converge, this is written and no other
   model is fitted in its place.
4. Per domain (6 labels): n and the share of `general_dictionary`.
5. Agreement: Fleiss' κ of `taboo_domain` (6 labels) and of `taboo` (binary), 3 coders, all 217 items.
6. Refusals per domain: for each consensus label, the number of items that were in at least one refused request,
   per coder, and the number of items resent one by one.

## 4. Japanese items (dictionary check)

For the Japanese items with a Daijisen judgement (`dictionary_check/daijisen_judgments.csv`, 41 headwords): crosstab
of `taboo` × whether the covert meaning is in a numbered sense (`番号つきの語義`), with the three levels of `判定`
beside it. Descriptive only.

## Reading

- Difference < 0 with one-sided Fisher p < .05: the prediction holds.
- Difference < 0, not significant: in the predicted direction, not shown.
- Difference ≥ 0: the prediction does not hold.

## Output

`coding/scripts/run_taboo_coding.py` → `coding/results/taboo/taboo_<coder>.csv` (+ `.jsonl`);
`analysis/analyze_taboo.py` → `results/taboo_results.json`, `results/taboo_results.md`.
