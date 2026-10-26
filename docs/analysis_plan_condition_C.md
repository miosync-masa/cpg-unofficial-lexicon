# Analysis plan: condition C (hypotheses H-C1, H-C2)

This plan was written on 2026-10-22 and fixed before condition C was computed. It was added to the repository on
2026-10-26.

The text below is the plan as fixed, shortened to the hypotheses, the procedure and the decision rules, and
translated from Japanese. File paths are given in the current layout of this repository. Order of work: the host
check (section 1) first, then condition C (sections 2–5).

Common rules:
- Every number is produced by a script; nothing is copied by hand.
- Nothing is filled in by guess. An item that cannot be judged is left empty or `UNRESOLVED`, with the reason.
- Every decision that departs from the plan is recorded.
- Fixed random seed 20261022. Every LLM and embedding API call is logged in JSONL.

## 1. Host check and exclusions (before condition C)

1. `SEA_012` is excluded (its host is a placeholder string `[technique-level item …]`).
2. The `host_original` and `host_romanized` of all 217 items are checked mechanically for the same kind of problem.
   An item that meets any of the following is **listed as a candidate** (not excluded automatically):
   - an explanation in square brackets `[ ]` or in parentheses (e.g. `(visual diphrasis — no spoken form)`)
   - an explanation in English (e.g. `Ritual address to landscape entities`)
   - several forms separated by `/` or `;` (e.g. `unalive / corn / … / grape`)
   - whole lines of a song or poem (a sentence of more than 20 characters, or one with a line break)
3. Candidate list `data/exclusion_candidates.csv`: `item_id, host_original, host_romanized, problem_type,
   proposed_action`, with `proposed_action` one of `exclude` / `use_first_form` / `keep` / `ask_author`; unclear
   cases are `ask_author`.
4. The fixed exclusion list `data/exclusions.csv` (`item_id, reason`) starts with `SEA_012` only. The other
   candidates wait for the author's decision.

## 2. Clean hosts (`embeddings/controls/clean_hosts.csv`)

Background: the control lists contain the host itself (54 cases) or words that contain the host (15 cases, e.g.
苹果 → 红苹果); 62 hosts carry English notes (e.g. `pingguo_apple`, `Hodam (glutton / fat stomach)`); hosts and
controls in different scripts distort the result (seen with bge-m3).

- The clean host is the majority (≥ 2 of 3) of the `clean_host` given by the three coders of the first coding run.
- Without a majority, `host_original` is used and `clean_host_source = fallback_original` is recorded.
- Columns: `item_id, clean_host, clean_host_source, script` (`script` = Latin / CJK / Arabic / Cyrillic / Devanagari
  / Ethiopic / Javanese / Other, by Unicode range).

## 3. Control cleaning (`embeddings/controls/controls_clean.csv`)

The following rules are applied **in order** to the controls of each item (`data/control_cache.json`).

| rule | content |
|---|---|
| R1 identical | the control equals clean_host, host_original or host_romanized (ignoring case, spaces and tone marks) → dropped |
| R2 overlap | the control contains clean_host, or clean_host contains the control (both at least 2 characters) → dropped |
| R3 script | the control is in a different script from clean_host → dropped |
| R4 duplicate | the same control more than once → one is kept |

- Columns: `item_id, control, kept, rule_dropped`.
- Remaining controls per item in `embeddings/controls/controls_summary.csv` (`n_original, n_kept,
  n_dropped_R1..R4`).
- Items with fewer than 5 remaining controls are flagged `low_control`. **No new controls are made** (outside this
  plan).

## 4. Condition C

- Host = clean_host; controls = the cleaned controls (at most 10, in the original order).
- All three embedding models:
  - `text-embedding-3-large` (OpenAI, 3072 dimensions)
  - `BAAI/bge-m3` (local)
  - `gemini-embedding-001` (`task_type = SEMANTIC_SIMILARITY`)
- Sensitivity in k: k = 3, 5, 10 (an item with fewer than k remaining controls is left out of the analysis at that
  k).
- Output per model: `item_id, tradition, clean_host, n_controls_kept, sim_host_target, pull_C_k3, pull_C_k5,
  pull_C_k10`.

## 5. Hypotheses (fixed before the run)

- **H-C1:** with the exclusions of section 1 applied, on the items that are not `low_control`, the mean condition C
  pull (k = 10) is greater than 0.
  Test: in each of the three models, a one-sided test of the intercept of the mixed model
  `pull ~ 1 + (1 | tradition)`. The number of models with Holm-corrected p < .05 is reported;
  **all three models: "supported"; one or two models: "partly supported"; none: "not supported".**
- **H-C2:** the same test on the items with `cpg_valid = yes` in the first-run consensus only (decision rule as for
  H-C1).

## 6. Analyses to report (all three models side by side)

- Item counts at each step (all → section 1 exclusions → `low_control` excluded → analysed), stated to reconcile.
- Condition B (earlier) against condition C: mean, % positive, t, mixed-model z.
- Descriptives by recording status (`recording_status`), by direction (R/E/S) and by primary channel.
- Tests by tradition (n ≥ 3) with BH correction.
- Sensitivity: with `low_control` items included; at k = 3 and k = 5.
- Item-level rank correlation between models (Spearman).

## 7. Acceptance checks

- Embeddings complete in all three models; strings that could not be embedded are listed.
- Item counts reconcile at every step (exclusions, `low_control` and `UNRESOLVED` counts stated).
- Everything except the LLM and embedding API calls rerun from the start gives identical outputs.
- The report ends with a list of the points that need a decision (including the `ask_author` items).
