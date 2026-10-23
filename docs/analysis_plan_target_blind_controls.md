# Analysis plan: control words generated without the covert target (conditions D and E)

Fixed on 2026-10-23, before any condition D or E control word was generated or any pull was computed. The commit
that adds this file is the record of that.

## Why

The stored control words (`data/control_cache.json`, condition C) were generated with a prompt that contained the
line `EXCLUSION: Do NOT include any word associated with: "<covert target>"`. The controls are therefore the
host's near neighbours *minus* words related to the covert target. This can lower the similarity between the
controls and the target and so inflate the pull. Conditions D and E build controls without using the covert
target at all. The results are reported whatever they are.

## 1. Items

- The 94 items of the condition C main set (H-C1): T3 exclusions applied, at least 10 kept controls in condition C.
- In condition D or E, an item with fewer than 10 controls after cleaning is removed (`low_control`), as in
  condition C. The number removed is reported.

## 2. Control words

### Condition D (primary): LLM, without the covert target

- Prompt: the v1 prompt (`legacy/v1_submission/scripts/archaeology/cpg_archaeology.py`, `generate_controls()`)
  with the two `EXCLUSION` lines removed; every other character is the same. The inputs (host string and
  language) are those of the v1 run, taken from the keys of `data/control_cache.json`, so that the exclusion
  instruction is the only difference from the stored controls.
- Model `gpt-5.2`, temperature 0.3, `max_completion_tokens` 300, user message only. If `gpt-5.2` is not
  available, the model used is recorded and reported; a different model means a difference between C and D can
  no longer be attributed to the exclusion instruction alone.
- One request per item. Requests that fail with an API error are retried up to 3 times; an empty or short
  response is not regenerated.
- Post-processing as in v1: lines shorter than 80 characters, leading numbers and bullets stripped, the host
  string removed, the first 10 words kept.
- Then the cleaning rules R1–R4 of condition C (`embeddings/scripts/build_clean.py`), with the condition C clean
  host: R1 identical to the host, R2 contains the host or is contained in it, R3 different script from the clean
  host, R4 duplicate.
- Every prompt, full response and time is logged (JSONL).

### Condition E (secondary, exploratory): nearest words in each embedding space, no LLM

- Vocabulary: the top 50,000 words of `wordfreq` 3.1.1 (`top_n_list`, wordlist `best`) for the item's language;
  the whole list where it is shorter (Hindi, Persian, Vietnamese). Words without a letter are dropped.
- Language mapping (item language label → wordfreq code), fixed here: a label maps only if it names exactly one
  modern language that wordfreq covers. `en` and `English (primarily American English; global adoption)` → en;
  `ja` → ja; `hi` → hi; `fa` → fa; `fr` → fr; `zh` and `Mandarin Chinese (普通话)` → zh; `Russian` → ru;
  `Arabic` → ar; `Vietnamese` → vi; `Polish` → pl; `Finnish` → fi. Every other label is not mapped: historical
  stages (Old English, Old Norse, Old Irish, Ottoman Turkish, Classical Nahuatl, colonial Quechua, Colonial Yucatec
  Maya, Classic Maya, Old Russian oral), labels naming several languages (`Finnish / Proto-Finnic`,
  `Polish / Russian / Ukrainian`), the emoji item, and languages wordfreq does not cover (Amharic, Yoruba, isiZulu,
  Twi, Javanese, Thai, Belarusian, Māori, contemporary Quechua). Items with an unmapped language are left out of
  condition E; the languages and item counts are reported.
- For each item and each embedding model: cosine similarity between the clean host and every vocabulary word,
  in descending order; R1–R4 applied in that order; the first 10 kept words are the controls. The controls may
  differ between models. The covert target is not used.

## 3. Analysis

- Pull, the three embedding models (`text-embedding-3-large`, `gemini-embedding-001` with
  `task_type = SEMANTIC_SIMILARITY`, `BAAI/bge-m3`), the clean host and the k = 10 rule are those of condition C:
  pull = sim(clean host, target) − mean sim(control, target) over the 10 controls.
- Mixed model `pull ~ 1 + (1 | tradition)` with the raw tradition labels, REML, one-sided z test of the
  intercept, Holm across the three models within each hypothesis. 3 models with Holm p < .05: supported; 1–2:
  partly supported; 0: not supported.
- Hypotheses:
  - **H-D1**: in condition D, the mean pull of the main set is greater than 0.
  - **H-D2**: H-D1 on the items whose first-run consensus `cpg_valid` = yes.
  - Condition E: the same two tests, exploratory.
- Diagnostic: for the same items, the mean of mean sim(controls, target) in conditions C, D and E, per model.
  This shows how far the exclusion instruction moved the controls away from the target.

## 4. Interpretation, fixed in advance

| result | treatment in the paper |
|---|---|
| H-D1 supported in all three models | the exclusion instruction did not create the pull; the effect sizes in C and D are compared |
| H-D1 supported in some models | reported as is; the claim is limited to the supported models |
| H-D1 supported in no model | the claim is reconsidered; the result is reported in the main text |

## 5. Output

- `results/target_blind_controls.json` and `.md`: `condition, hypothesis, model, n, n_groups, intercept, z,
  p_holm, verdict`, and the diagnostic table.
- Control lists: `data/controls_D.json`, `data/controls_E_<model>.json`.
- Macros in `paper/numbers.tex` through `analysis/make_numbers.py`.
