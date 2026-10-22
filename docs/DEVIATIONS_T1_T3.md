# Deviations and judgement calls — T1–T4 (instruction sheet v1.0, 2026-10-22)

Every place where the work departs from, or had to interpret, the instruction sheet. Counts quoted here are
copied from script output files named next to them.

Paths are those of the working folder at the time of the work; `docs/README.md` maps them to the current layout.

## Inputs

1. **Paths.** The sheet names `coding/results/codes_{gpt,gemini,claude}.csv` and
   `coding/results/consensus_codes.csv`. The first-run files are at `coding/codes_{gpt,gemini,claude}.csv` and
   `coding/agreement/consensus_codes.csv`; those were used. Repository = `$REPO`, a local copy of
   this repository at commit 22b916a (the copy already used by every earlier run; a fresh clone was not used).

## T3

2. **Detection heuristics.** The sheet lists four kinds of problem. The script turns them into nine mechanical
   rules (`t3/check_hosts.py` docstring and `t3/README.md`): the list of English descriptive words, the 20-character
   limit applied per `/`-segment, and the treatment of `—` and ` and `/` + ` are my choices.
3. **proposed_action rules** are also my choices (see `t3/README.md`). A parenthetical gloss alone is proposed as
   `keep` because T1's clean host already drops it; a single saying longer than 20 characters is `keep` (the saying
   is the host); verse split into lines is `ask_author`.

## T1

4. **Clean-host majority** compares the three coders' strings after normalisation (NFKC, case-fold, typographic
   apostrophes as `'`, whitespace collapsed, Arabic harakat removed), not as raw strings; otherwise e.g.
   `K'intu`/`k'intu` or vowelled/unvowelled Arabic would count as disagreement. The spelling of the first agreeing
   coder (gpt, gemini, claude order) is kept. Result: 208 majority, 9 fallback_original (`t1/clean_hosts.csv`).
5. **R1/R2 comparison key** removes case, all whitespace, and all combining marks (tone marks and every other
   diacritic) after the normalisation above. Hyphens and other punctuation are not removed, so `three oh four`
   is not identical to the romanized host `three-oh-four`.
6. **R3 is not applied when the clean host has no letters** (NET_006 `304`, NET_002 `🍆 🍑`): a digit or emoji host
   has no script to match, and applying R3 literally would drop every control. These two hosts and the three
   Thai/Korean hosts fall in the sheet's `Other` category; their kept controls were checked by eye and are in the
   same script as the host (Thai, Korean) or are English words and emoji (NET_002, NET_006).
7. **Counts differ from the sheet's background figures** (54 identical, 15 containing). With the rules applied to
   the clean host: R1 67, R2 269, R3 40, R4 1 of 2,125 controls; 1,748 kept (`t1/build_clean.py` output). Most R2
   drops are compounds of the host (`tupai` → `tupai hitam`, `数の子` → `数の子漬け`).
8. **k = 10 needs 10 kept controls.** Following the sheet ("an item with fewer than k kept controls is left out of
   that k"), pull_C_k10 exists only for the 93 main-set items with 10 kept controls (of 187). Because this halves
   the sample, `nnpm_replicate.py` also writes **pull_C_kall** (mean over all kept controls, up to 10). It is
   reported as an additional sensitivity column and does not enter any hypothesis verdict.
9. **Condition B for comparison** is `pull_B_k10` from the earlier runs (`embed/out/nnpm_<model>.csv`: host in
   original script, uncleaned controls). The bge-m3 condition-B file was supplied with the toolkit; bge-m3
   condition C was run locally after installing `sentence-transformers` 6.2.0 into `.venv`.
10. **Mixed models group on the pipeline tradition labels** (`cpg_archaeology_results.csv`, as in all earlier
    analyses), not the long names in `blind_items.csv`.
11. **Embedding logs and vectors.** Condition C logs every embedding request (texts and error) to
    `embed/out/nnpm_C_<model>.jsonl` and saves the vectors to `nnpm_C_<model>_vectors.npz`. Earlier A/B runs were
    not re-run and have no such log. No string failed to embed (`nnpm_C_<model>_failed.csv` are empty).

## T2

12. **Runner.** `coding/run_v1_1_coding.py` was derived from `coding/run_recording_coding.py`, which is
    `run_llm_coding.py` with the column-order fix plus strict key validation and a readable error when Claude
    returns no text (refusal). Settings as before: gpt-5.6-sol and claude-opus-5-5 without temperature (they reject
    it), gemini-3.1-pro-preview at temperature 0. The 10-item pilot is in `coding/results/pilot_v1_1/`.
13. **Codebook text.** `coding/codebook_v1_1.md` is the sheet's English text verbatim, with a three-line preamble
    (what a host and a covert target are; the two-storey idea) and the output-key line.
14. **Agreement.** "各項目の Fleiss の κ" was read as κ per coded field (κ is not defined for one item); per-item
    unanimity flags are in `v1_1_consensus.csv`.
15. **Focus items** for the old/new cross-table: the six kennings coded `novel_form` in the recording run
    (NEU_001, 002, 003, 005, 011, 012), NET_004 草泥马 and NET_005 五月三十五日.

## T4

16. **File names.** `t4/` already holds the dictionary check, so the integrated analysis is `analyze_final.py` in
    the toolkit root. `out/items_final.csv` holds the 216 items left after T3 (all items, including those
    without controls or with low_control, flagged by columns).
17. **Section 12 writer.** `analyze_recording.py` used to drop everything after section 12 of
    `verified_numbers.md` when rerun; it now replaces only section 12, so section 13 survives. Its output is
    otherwise byte-identical (checked by diff).

## Run notes (added after the runs)

18. **Refusals.** Claude refused the batch starting at NET_001 twice (`stop_reason=refusal`) and coded it on the
    third attempt of the same batch. Gemini returned an invalid `confidence` for the batch starting at T_045 twice
    and was valid on the third attempt. All attempts are in the `.jsonl` logs. No item ended as `failed`, so no
    one-item-at-a-time rerun was needed (the sheet asks for one when a batch is refused; here the refused batch
    was not left uncoded).
19. **Holm bug found and fixed before reporting.** The first draft of `analyze_final.py` fed rounded p-values to
    Holm and turned p = 0.0000 into 1.0. Both analysis scripts now apply Holm to unrounded p-values. The reported
    numbers are from the fixed scripts.
20. **Offline rerun.** `rerun_offline.sh` reruns T3 → T1 → condition-C pulls (from the saved vectors,
    `--from-vectors`) → T1 analysis → T2 aggregation → T4 without any API call. Two consecutive runs gave
    byte-identical output for all 17 output files (checked with shasum).
21. **R2 needs two characters on both sides**, as the sheet says, so NET_002 keeps the single-emoji controls
    `🍆` and `🍑` although each is half of the host `🍆 🍑`.

## Author decisions of 2026-10-22 (`DECISIONS_2026-10-22.md`)

22. **How the decisions are entered.** The decision on each of the 37 ask_author / use_first_form items is written by
    hand in `t3/author_decisions.csv` (decision and reason per item). `t3/check_hosts.py` checks that the file covers
    exactly those 37 items and writes `exclusions.csv` from it; `t1/build_clean.py` takes `first_form` as the clean
    host for adopted items. Rerunning both scripts reproduces everything.
23. **My readings of decision 1 (ask_author, 22).**
    - Excluded: MAYA_005 and QCH_006, which have no host form. NET_003 is excluded because it lists several different words.
    - NEU_006 (`beinhjálmr / valr used for 'horse'`) is also excluded as "several different words listed". It was not named in the decision.
    - The 8 utterance hosts are the 4 Exeter Book riddles NEU_007–010 and the verse lines EEU_003, EEU_013, EEU_014 and QCH_001. The T2 consensus already has host_unit = utterance for all 8, so no label was changed. `analyze_final.py` asserts this.
    - Kept and listed as matching no case (10): EEU_010, EEU_015, MAYA_004, MET_005, NET_002, NET_007, T_068, OCE_004, SEA_007, SEA_015. MET_005 is close to "no host form" but is not named in the decision.
24. **My readings of decision 2 (use_first_form, 15).**
    - Adopted (8): EEU_011, EEU_012, MET_012, NEU_001, NEU_012, NEU_017, QCH_004, SEA_009. Only 3 of these change the clean host; for the other 5 the majority value already was the first form.
    - The 3 changed clean hosts are EEU_011 `oranie` and NEU_001 `hronrád` (both were fallback_original), and SEA_009 `sasmitaning tembang`.
    - SEA_009 changes script from Javanese (the majority value was the Javanese-script form in the gloss) to Latin. Its 10 Latin controls, all dropped by R3 before, are now kept.
    - MET_012 stays in Arabic script and stays low_control.
    - Excluded (1): QCH_003 Ukhu Pacha / Hanaq Pacha (lower world / upper world).
    - Not adopted as unclear (6): AFR_015, NEU_003, NEU_004 and SEA_008 (two different expressions for the same target), MAYA_003 (two words for cave), MAYA_006 (meanings not given). These keep their T1 majority clean host.
25. **Re-embedding.** Condition C was recomputed for all items with the three models, not only for the three changed hosts. 0 failed strings in each model.
    - The re-run overwrote the first condition-C vector files. The `.jsonl` logs were appended to, not overwritten:
      each holds both runs (34 calls each). This corrects an earlier version of this note (repository
      reorganisation, 2026-10-22).
    - For the items whose host and controls did not change, the pulls are the same as in the first run: per-tradition means agree to 4 decimals except for the Old Norse kenning group, which contains NEU_001.
26. **Why the z values fell while the verdicts held.**
    - The k = 10 set lost QCH_006 (excluded; OpenAI pull 0.348) and gained EEU_011 (−0.112) and SEA_009 (0.102).
    - NEU_001's pull changed with its new clean host.
    - The verdicts are compared, with the per-model numbers, in `out/final_results.md`. That comparison reads `out/t1_results_before_decisions_2026-10-22.json` (and `out/final_results_before_decisions_2026-10-22.json`), the outputs saved just before the decisions were applied.
27. **Decision 5 makes the scope test equal to all_main.** After T3 every item has form_type existing_form or transparent_combination; the only respelling, NET_003, is excluded, and no form_type is UNRESOLVED. So cpg_scope = yes for all 211 items, and the exploratory cpg_scope_yes test is identical to all_main. The first definition is kept as `cpg_scope_two_storey` for description only.
28. **Second round of decisions (2026-10-22).** The author approved items 23–24 as they stand: the exclusion of NEU_006, the 10 kept and listed items, and the 6 items not adopted. The author also accepted SEA_009 and MET_012 as they are.
    - Because the cpg_scope = yes test is identical to all_main, `analyze_final.py` no longer tabulates it and writes a one-line note instead. The JSON still holds it, with `cpg_scope_yes_test_identical_to_all_main`.
29. **Post-hoc exclusions.** The exclusions of 2026-10-22 (QCH_006 and the others) were decided after the first condition-C results had been seen.
    - The before/after table of H-C1/H-C2 in `out/final_results.md` is SI material and must be kept.
    - Its inputs `out/*_before_decisions_2026-10-22.json` are fixed files and must not be regenerated.
30. **Backups before overwriting.** `rerun_offline.sh` now copies `out/` to `backups/out_<date_time>/` before it runs. Any other rerun that overwrites outputs (for example an embedding run) should do the same. The state of 2026-10-22 is committed in git with the tag `rev-2026-10-22`.
