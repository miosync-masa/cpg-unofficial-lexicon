# Analysis plan: recording status (hypotheses H1–H3)

This plan was written on 2026-10-21 and fixed before the recording-status coding was run. It was added to the
repository on 2026-10-26.

The text below is the plan as fixed, shortened to the hypotheses, the procedure and the decision rules. File paths
are given in the current layout of this repository.

## Why

The first 3-model coding rejected 65 of 213 items as "not CPG" because the covert meaning is already a sense in a
**general dictionary** (French *baiser*, *coucher*; Sanskrit *kāma*). Items recorded only in **cultural or
specialist references** (kakekotoba in poetic-diction dictionaries, osechi symbolism in folklore guides) were
accepted. The "official lexicon" therefore depends on the type of record. This run codes that explicitly, and
separates CPG on existing forms from slang neologisms (novel forms with no surface meaning).

## 1. Hypotheses (fixed before any new data are seen)

- **H1 (recording gradient):** among items on an existing form, mean pull is ordered
  `general_dictionary` ≥ `cultural_reference` > `unrecorded`, and all three are > 0.
- **H2 (unrecorded CPG is detectable):** `unrecorded` items on an existing form have pull > 0 in at least two of
  the three embedding models.
- **H3 (neologisms are not CPG):** items with `form_status = novel_form` are excluded from all CPG claims; their
  pull is reported separately and descriptively only.

Every hypothesis is reported as supported, not supported, or mixed. No hypothesis is added or changed after the run.

## 2. Inputs

- `data/blind_items.csv`: 217 items, shuffled, no codes, no pulls.
- The 213 items with the earlier pulls (`pull_index` OpenAI, `pull_bge_B`, `pull_gem_B`), tradition and the earlier
  consensus codes (`results/items_with_consensus.csv`).

## 3. Codebook (`coding/codebooks/codebook_recording.md`)

The coder judges only the fields below.

### `form_status` (exactly one)
- `existing_form`: the host is an ordinary word, phrase, number, emoji, or fixed expression that already had a
  surface meaning before the covert meaning attached (e.g. 草 "grass" → "lol"; *corn* → porn; "Netflix and chill";
  304; 海老).
- `novel_form`: the host was created for the covert meaning and has no independent surface meaning: a coinage,
  blend, respelling, or abbreviation (e.g. *seggs*, *unalive*, メダカブス).

### `recording_status` (exactly one; judge the covert meaning, not the surface meaning)
- `general_dictionary`: the covert meaning is listed as a sense, or a clearly marked usage (e.g. "euphemism",
  "slang", "vulgar"), in a standard general-language dictionary of that language (e.g. Kōjien / Daijirin,
  Le Robert, Oxford/Merriam-Webster, Monier-Williams, Lane, Xiàndài Hànyǔ Cídiǎn).
- `cultural_reference`: not in a general dictionary's sense list, but documented in specialist or cultural
  references: poetic-diction or rhetoric dictionaries, symbol and folklore dictionaries, scholarly commentaries
  and editions, ethnographies, encyclopaedias, or established guides to customs.
- `unrecorded`: not documented in either kind of source as far as the coder knows; known mainly through use within
  the community (online communities, oral tradition, in-group speech).

### `recording_evidence`
The **type** of source that supports the label (e.g. "general dictionary sense", "poetic-diction dictionary",
"folklore guide", "scholarly edition", "online usage only"). A specific title is named only if the coder is certain
it exists and contains this item; otherwise the type only. No invented titles.

### `recording_confidence`
`1` guess, `2` fairly sure, `3` sure.

### Output
JSON array, one object per item, same order as input, keys exactly:
`item_id, form_status, recording_status, recording_evidence, recording_confidence`.

## 4. Coding run

- Three coders, the same providers and model IDs as the first coding run: `gpt-5.6-sol`, `gemini-3.1-pro-preview`,
  `claude-opus-5-5`.
- A fresh process per model, batches of 10, a JSONL log of every call, resume on rerun.
- Whether temperature was applied is logged per model.
- A 10-item pilot per model first; the JSON is checked before the full run.

## 5. Aggregation

- Fleiss' κ and % unanimous for `form_status` and `recording_status`.
- Consensus label = majority (≥ 2 of 3); otherwise `UNRESOLVED`.
- Crosstab of consensus `recording_status` against the earlier consensus `cpg_valid` (expected:
  `general_dictionary` ≈ `cpg_valid = no`).

## 6. Analysis

The consensus labels are merged with the 213 items with pulls on `item_id`. For each of `pull_index`, `pull_bge_B`,
`pull_gem_B`:

1. **Descriptives** by `form_status`, and by `recording_status` within `existing_form`: n, mean, SD, % positive,
   one-sided one-sample t and Wilcoxon p.
2. **Mixed model** on `existing_form` items, resolved labels only:
   `pull ~ C(recording_status, Treatment('unrecorded')) + (1 | tradition)`; b, SE, p for each level.
3. **Ordered trend test** for H1: `unrecorded = 0, cultural_reference = 1, general_dictionary = 2`;
   mixed model `pull ~ level + (1 | tradition)`; slope and one-sided p (positive).
4. **H2 test:** the `unrecorded ∧ existing_form` subset: one-sample tests plus the mixed-model intercept; the number
   of the three models that give p < .05.
5. **Sensitivity:** 1–4 repeated with only the items where all three coders agree on `recording_status`.
6. **Multiple comparisons:** Holm correction across the three models within each hypothesis test.

Because the models differ in cosine scale, models are compared by % positive and test statistics, not raw means.

## 7. Report

- One table per hypothesis with all three models side by side; a verdict per hypothesis.
- κ values, label distributions, the crosstab against `cpg_valid`.
- The list of `novel_form` items (for the "slang, not CPG" boundary discussion).

## 8. Acceptance checks

- 217 items coded by each model, 0 failed rows (rerun until clean), every call in the JSONL logs.
- Counts reconcile: items analysed = 213 minus `UNRESOLVED` labels, stated explicitly.
- Every number in the report is produced by the analysis script; no hand-typed statistics.
- Scripts rerun end to end give identical outputs (fixed seeds; the LLM calls are not rerun for this check).
