# Analysis plan: pull by form type (exploratory)

Fixed on 2026-10-26, before the pull was split by form type. The commit that adds this file is the record of that.
This analysis is exploratory: it was not part of the earlier plans, and its result is reported as such.

## Why

On an existing form whose surface meaning is in ordinary use, most of the text that contains the form uses it in
its surface meaning, so the share of covert uses that the embedding can pick up is small. On a transparent
combination (a new combination of existing elements made for the covert meaning, e.g. 草泥马), more of the text that
contains the form carries the covert meaning. The pull should therefore be smaller on existing forms.

## Prediction

`transparent_combination` items have a larger pull than `existing_form` items.

## Data

- Form type: the v1.1 consensus `form_type` (`results/items_final.csv`). The main sets contain only
  `existing_form` and `transparent_combination` items; any other label (including `UNRESOLVED`) is left out and
  counted.
- Pulls: `embeddings/results/nnpm_DE_<model>.csv` for the three models (OpenAI text-embedding-3-large, BAAI/bge-m3,
  gemini-embedding-001).
  - Condition C: the main set, `pull_C_k10` (n = 94).
  - Condition D: the items of the main set with 10 kept condition D controls, `pull_D_k10` (n = 67).
- Tradition: the raw `tradition` label of the pull file, as in H-C1 and H-D1.

## Analysis

For each condition (C, D) and each model:

1. Descriptives by form type: n, mean pull, SD.
2. Mixed model `pull ~ form_type + (1 | tradition)`, REML, `existing_form` as the reference level. The coefficient of
   `transparent_combination`: b, SE, z, one-sided p (b > 0), 95% Wald CI.
3. Holm correction across the three models within each condition.

If a fit does not converge, this is recorded and the result is given with that note; no other model is fitted in
its place.

## Reading

- Positive coefficient with Holm-corrected p < .05: the prediction holds in that model.
- Positive coefficient, not significant: in the predicted direction, not shown.
- Coefficient 0 or negative: the prediction does not hold.

The number of models in each row is reported per condition. Raw means are not compared across models, because the
models differ in cosine scale.

## Output

`analysis/analyze_form_type.py` → `results/form_type_pull.json`, `results/form_type_pull.md`.
