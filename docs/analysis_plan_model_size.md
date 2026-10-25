# Analysis plan: model size and pull (multilingual-e5, three sizes)

Fixed on 2026-10-25, before any embedding of this analysis was computed. The commit that adds this file is the
record of that.

## Why

The pull differed between the three embedding models used so far; bge-m3 gave the weakest pull, and its verdicts
were sensitive to single items entering or leaving a set. The author's hypothesis: the amount of text a model has
learnt from (the range of its knowledge) changes how far it picks up covert meanings, because a covert meaning
appears only in text written inside the community. Explaining the weakest model after the fact would be a post hoc
account. This analysis therefore uses one model family whose members differ only in size, and fixes predictions
that can fail before anything is computed.

## Models

| name | Hugging Face | parameters (model card) |
|---|---|---|
| small | `intfloat/multilingual-e5-small` | about 118 million |
| base | `intfloat/multilingual-e5-base` | about 278 million |
| large | `intfloat/multilingual-e5-large` | about 560 million |

- The three share the family and the training recipe and differ in size.
- Recorded: the revision (commit hash) of each model and the versions of sentence-transformers, transformers and
  torch.
- Input prefix: as the model card asks for symmetric similarity tasks, every string (host, covert target, control)
  gets the prefix `"query: "`. The same for the three sizes.
- Embeddings are L2-normalised; similarity is the cosine.
- Parameter count: counted on the loaded model (sum of the sizes of all parameter tensors of the
  SentenceTransformer) and compared with the model card. `params` below is the counted number.
- Run on CPU, batch size 16; the vectors are a cache outside git (`embeddings/results/e5/`, ignored).

## Items and controls

- No new controls. The existing control strings are embedded with the three sizes.
- Host and target strings as in condition C (`embeddings/controls/clean_hosts.csv` clean host; the target string of
  the condition C pipeline).
- **Main analysis: condition D** (controls generated without the covert target; `data/controls_D.json`, the first 10
  kept). Main set: the 67 items of the condition C main set with 10 kept condition D controls.
- **Secondary analysis: condition C** (`embeddings/controls/controls_clean.csv`, the first 10 kept). Main set: the 94
  items (T3 exclusions applied, `low_control` = 0, 10 kept controls).
- The keep rules of the controls (R1–R4, `low_control`, k = 10) do not depend on the embedding model, so the sets are
  the same for the three sizes.
- pull_X = sim(host, target) − mean over the 10 controls of sim(control, target), X = C, D (as in
  `embeddings/scripts/nnpm_target_blind.py`).
- Tradition: the tradition label of the pipeline, as in every earlier analysis.

## Measures

The similarity scales differ between models, so raw pulls are not compared across models.

- **Standardised pull**: `pull_std = pull / SD_model(pull)` (the mean is not subtracted). The SD (ddof = 1) is
  computed per model on the main set of the condition.
- **Effect size per model**: `d = mean(pull) / SD(pull)` (= the mean of pull_std).
- For reference, per size: the mixed model of the earlier analyses, `pull ~ 1 + (1 | tradition)`, raw labels,
  REML: intercept, its 95% CI, z.

## Hypotheses (fixed before computing)

**H-S1 (main)**: the larger the model, the larger the standardised pull.

- The data of the three sizes are stacked (one row per item and size). Mixed model
  `pull_std ~ log10(params) + (1 | item) + (1 | tradition)`, REML. Items are nested in traditions; in statsmodels:
  `mixedlm("pull_std ~ log10_params", groups=tradition, re_formula="1", vc_formula={"item": "0 + C(item_id)"})`.
- Test: coefficient of `log10(params)` > 0, one-sided Wald z, α = .05. Reported: coefficient, SE, z, one-sided p,
  95% CI (two-sided Wald).
- Also described: whether d is ordered small < base < large.

**H-S2 (secondary)**: the size effect is stronger for items that are harder to infer from outside.

- `outsider_inference` (T2 consensus) in two groups: `hard` = 1 for `inferable_with_effort` and `not_inferable`,
  0 for `inferable`; items with `UNRESOLVED` are left out (condition D: 5 items; condition C: 9 items).
- Model `pull_std ~ log10(params) * hard + (1 | item) + (1 | tradition)`, the same fitting as H-S1; pull_std as
  above (SD from the full main set of the condition).
- Test: interaction coefficient > 0, one-sided, α = .05.
- `outsider_inference` has low agreement between coders (kappa 0.44) and the `inferable` group is small (condition
  D: 17 items); the result is treated with caution.

**Both conditions**: H-S1 and H-S2 are computed for condition D (main) and condition C (secondary). The
interpretation below rests on condition D; condition C is reported beside it.

**Fitting problems**: if the model with both random effects does not converge or the tradition variance is
estimated at the boundary with a failed fit, the tradition component is dropped (`(1 | item)` only) and this is
reported. No other change of the model is made after seeing the data.

## Interpretation (fixed in advance)

| result (condition D) | in the paper |
|---|---|
| H-S1 slope positive and significant, and d ordered small < base < large | "within one model family, larger models pick up covert meanings more"; mentioned as **one possible** reason for the weak pull of bge-m3 (bge-m3 belongs to another family and is not placed on this line) |
| slope positive but not significant, or the order of d partly reversed (also: significant slope with d not in order) | "a tendency, not established"; the low power of three sizes is stated |
| slope near 0 or negative | "model size does not explain it"; size is not used to explain the weak pull of bge-m3 |

H-S2 is reported as it comes out, whatever H-S1 shows.

## Reference (no test)

The same standardisation (d = mean / SD on the main set) applied to the three earlier models (OpenAI
text-embedding-3-large, BAAI/bge-m3, gemini-embedding-001) in conditions D and C, for comparison only.

## Output

- `embeddings/results/nnpm_e5_<size>.csv` (per item: pull_C_k10, pull_D_k10, sim_host_target, simctrl_C, simctrl_D),
  `embeddings/results/e5_meta.json` (revisions, parameter counts, library versions); vectors in
  `embeddings/results/e5/` (ignored by git).
- `results/model_size_e5.{json,md}`: per size n, mean pull, SD, d, mixed-model intercept, 95% CI, z; H-S1 and H-S2
  coefficient, SE, z, one-sided p, 95% CI; the reference d of the three earlier models.
- Macros through `analysis/make_numbers.py`: `PeSmall`, `PeBase`, `PeLarge` (parameters); `DeSmallD`, `DeBaseD`,
  `DeLargeD`, `DeSmallC`, `DeBaseC`, `DeLargeC` (d); `HSoneB`, `HSoneZ`, `HSoneP`, `HSoneCI` and `HStwoB`, `HStwoZ`,
  `HStwoP`, `HStwoCI` (condition D), the same with a final `C` for condition C.
