<!-- Spec as given (verbatim). Paths are those of the working folder at the time; see docs/README.md for the current layout. -->

# Spec: Recording-status coding and analysis (CPG revision, v1.0, 2026-10-21)

Hand this file to Claude Code together with the `cpg_revision/` folder.

## 0. Why

The first 3-model coding rejected 65/213 items as "not CPG" because the covert meaning is already a sense in a
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

Report every hypothesis as supported, not supported, or mixed. Do not add or change hypotheses after the run.

## 2. Inputs (all exist)

- `coding/blind_items.csv` — 217 items, shuffled, no codes, no pulls. Use as is.
- `out/items_with_consensus.csv` — 213 items with `pull_index` (OpenAI), `pull_bge_B`, `pull_gem_B`, tradition,
  and the earlier consensus codes.
- `coding/run_llm_coding.py` and `coding/aggregate_codes.py` — reuse their structure (batching, validation,
  retries, JSONL log, resume).

## 3. New codebook: `coding/codebook_recording.md`

The coder judges only the fields below. Put this text in the codebook (English):

### `form_status` (exactly one)
- `existing_form` — the host is an ordinary word, phrase, number, emoji, or fixed expression that already had a
  surface meaning before the covert meaning attached (e.g. 草 "grass" → "lol"; *corn* → porn; "Netflix and chill";
  304; 海老).
- `novel_form` — the host was created for the covert meaning and has no independent surface meaning: a coinage,
  blend, respelling, or abbreviation (e.g. *seggs*, *unalive*, メダカブス).

### `recording_status` (exactly one; judge the covert meaning, not the surface meaning)
- `general_dictionary` — the covert meaning is listed as a sense, or a clearly marked usage (e.g. "euphemism",
  "slang", "vulgar"), in a standard general-language dictionary of that language (e.g. Kōjien / Daijirin,
  Le Robert, Oxford/Merriam-Webster, Monier-Williams, Lane, Xiàndài Hànyǔ Cídiǎn).
- `cultural_reference` — not in a general dictionary's sense list, but documented in specialist or cultural
  references: poetic-diction or rhetoric dictionaries, symbol and folklore dictionaries, scholarly commentaries
  and editions, ethnographies, encyclopaedias, or established guides to customs.
- `unrecorded` — not documented in either kind of source as far as you know; known mainly through use within the
  community (online communities, oral tradition, in-group speech).

### `recording_evidence`
The **type** of source that supports the label (e.g. "general dictionary sense", "poetic-diction dictionary",
"folklore guide", "scholarly edition", "online usage only"). Name a specific title only if you are certain it
exists and contains this item; otherwise give the type only. Never invent titles.

### `recording_confidence`
`1` guess, `2` fairly sure, `3` sure.

### Output
JSON array, one object per item, same order as input, keys exactly:
`item_id, form_status, recording_status, recording_evidence, recording_confidence`.

## 4. Runner: `coding/run_recording_coding.py`

- Copy `run_llm_coding.py`; change the codebook path, the field list, and `validate()` to the enums above.
- Same providers and model IDs as the first run: `gpt-5.6-sol`, `gemini-3.1-pro-preview`, `claude-opus-5-5`.
- Fresh process per model, batches of 10, JSONL log of every call, resume on rerun, `--limit` pilot.
- Log whether temperature was applied (expect: Gemini yes, others no — same as before).
- Outputs: `coding/results/recording_{gpt,gemini,claude}.csv` (+ `.jsonl`).

Run a 10-item pilot per model first and check the JSON before the full run.

## 5. Aggregation: `coding/aggregate_recording.py`

- Fleiss' κ and % unanimous for `form_status` and `recording_status`.
- Consensus label = majority (≥ 2 of 3); otherwise `UNRESOLVED`.
- Crosstab of consensus `recording_status` against the earlier consensus `cpg_valid` (expect `general_dictionary`
  ≈ `cpg_valid = no`).
- Output `coding/results/recording_consensus.csv`, `coding/results/recording_agreement.json`.

## 6. Analysis: `analyze_recording.py` → `out/recording_results.json` + `out/recording_results.md`

Merge `recording_consensus.csv` into `out/items_with_consensus.csv` on `item_id` (213 items with pulls).

For each of `pull_index`, `pull_bge_B`, `pull_gem_B`:

1. **Descriptives** by `form_status`, and by `recording_status` within `existing_form`:
   n, mean, SD, % positive, one-sided one-sample t and Wilcoxon p.
2. **Mixed model** on `existing_form` items, labels resolved only:
   `pull ~ C(recording_status, Treatment('unrecorded')) + (1 | tradition)`; report b, SE, p for each level.
3. **Ordered trend test** for H1: code `unrecorded = 0, cultural_reference = 1, general_dictionary = 2`;
   mixed model `pull ~ level + (1 | tradition)`; report slope and one-sided p (positive).
4. **H2 test:** `unrecorded ∧ existing_form` subset: one-sample tests plus mixed intercept; count how many of the
   three models give p < .05.
5. **Sensitivity:** repeat 1–4 using only items where all three coders agree on `recording_status`.
6. **Multiple comparisons:** Holm correction across the three models within each hypothesis test.

Because models differ in cosine scale, compare models by % positive and test statistics, not raw means.

## 7. Report (`out/recording_results.md`)

- One table per hypothesis with all three models side by side; verdict per hypothesis.
- κ values, label distributions, the crosstab against `cpg_valid`.
- List of `novel_form` items (these go to the "slang, not CPG" boundary discussion).
- Append a section 12 to `verified_numbers.md` with the headline numbers.

## 8. Acceptance checks

- 217 items coded by each model, 0 `failed` rows (rerun until clean), every call in the JSONL logs.
- Counts reconcile: items analysed = 213 minus `UNRESOLVED` labels, stated explicitly.
- Every number in the report is produced by the analysis script; no hand-typed statistics.
- Scripts rerun end to end produce identical outputs (fixed seeds; LLM calls are not rerun for this check).
