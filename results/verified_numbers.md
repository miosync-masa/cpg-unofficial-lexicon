# CPG — Verified Numbers (recomputed 2026-10-21)

Source of truth: `analysis/reproduce_stats.py` → `results/verified_stats.json`, `results/per_tradition.csv`, `results/items_with_metadata.csv`.
Input: this repository as submitted (tag `v1-language-sciences-2026-03`, commit 22b916a). No API calls; stored per-item similarities only.

Status key: ✅ matches submitted paper · ⚠️ differs · ❌ cannot be traced to repository data · 🆕 new analysis

## 1. Dataset

| Item | Submitted paper | Recomputed | Status |
|---|---|---|---|
| Items coded | 217 | 217 (103 Phase 1 + 114 Phase 2) | ✅ |
| Items with pull index | 213 | 213 (excluded: MAYA_005, MET_014, NET_003, OCE_002) | ✅ |
| Traditions | 53 | 72 distinct tradition labels; 22 with n ≥ 3 (153 items) | ⚠️ label harmonisation needed |
| Languages | 15 | 48 distinct language labels (unharmonised) | ⚠️ |
| Regions | 8 | 13 under the scheme in the script; must be fixed and stated | ⚠️ |
| Controls per item | 10 | 10 (a few items 9); LLM-generated; control embeddings not stored | ✅ / note |

## 2. P1 — Overall pull

| Statistic | Submitted | Recomputed | Status |
|---|---|---|---|
| Mean pull | +0.057 | +0.0565, SD 0.102, 95% CI [0.043, 0.070] | ✅ |
| % positive | 74.6% | 74.6% | ✅ |
| One-sample t | 8.06 | 8.055, d = 0.55 | ✅ |
| Sign test | p < .001 | p < .001; Wilcoxon p < .001 | ✅ |
| Mixed model (pull ~ 1 + (1\|tradition)) | — | intercept +0.062, SE 0.010, z = 6.29, 95% CI [0.042, 0.081] | 🆕 |
| Tradition-mean test (72 means) | — | mean +0.064, 83% positive, t = 6.54 | 🆕 |
| Cluster bootstrap CI (by tradition) | — | [0.034, 0.079] | 🆕 |

**Verdict:** P1 survives every cluster-aware check. This is the backbone.

## 3. Per-tradition tests (22 traditions, n ≥ 3)

| Statistic | Submitted | Recomputed | Status |
|---|---|---|---|
| Significant, uncorrected p < .05 | 10/22 | 10/22 | ✅ |
| Regions with ≥1 sig tradition | 6 of 8 | 8 (uncorrected) under the script's scheme | ⚠️ scheme-dependent |
| Significant after BH-FDR q < .05 | — | **5/22**: adianoeta, śleṣa, īhām, xiéyīn, qene | 🆕 |
| Significant after Holm | — | 4/22 | 🆕 |
| Kenningar | n=3, +0.035, p=.213 (SI) | **n=6, +0.136, p=.053** | ❌ SI |
| Difrasismo | n=5, −0.002, 40%, p=.512 (SI) | **n=8, +0.034, 87.5%, p=.099** | ❌ SI |
| Algospeak, Mele kaona, Whakataukī, Songlines rows (SI S2.2) | listed | not among the 22 traditions with n ≥ 3 | ❌ SI |

Full table: `results/per_tradition.csv` (includes Wilcoxon p, BH q, Holm q).

## 4. P2 / P3 — S-type

**Pipeline bug found (2026-10-21):** the original archaeology script read only the `x_s` key of the Phase-2 YAML,
but 52 items store their coding as `x_type: [r, e, s]`. Their S value was silently set to 0 — including every
Difrasismo and Old Norse kenning item. `analysis/reproduce_stats.py` now reads `x_type`. Corrected numbers:

| Statistic | Submitted | Recomputed (corrected coding) | Status |
|---|---|---|---|
| Items whose x_s changed | — | 51 | 🆕 |
| R/E (x_s = 0) | +0.060 | n = 109, +0.048 | ⚠️ |
| Mixed (0 < x_s < 0.5) | — | n = 78, +0.062 | 🆕 |
| S-dominant (x_s ≥ 0.5) | −0.002 (n = 5) | **n = 26, +0.075, 85% positive, t = 4.33, p < .001** | ❌ opposite |
| S vs R/E | "S ≈ 0" | Welch p = .19; mixed-model slope of x_s b = +0.034, p = .42 | ⚠️ no difference |

**Verdict:** S-type items *do* pull toward their targets at the phrase level. P2 as written is falsified.
A defensible S-type test is superadditivity: sim(pair, target) > sim(each element, target). This needs the
`components` field from the LLM coding, then a re-embedding run.

### Superseded first pass (before the bug fix)

| Statistic | Submitted | Recomputed | Status |
|---|---|---|---|
| R/E mean pull | +0.060 | x_s = 0: n = 160, +0.061 | ✅ approx. |
| S-type mean pull | −0.002 (n=5) | x_s ≥ 0.4: **n = 5, +0.049**; x_s ≥ 0.5: n = 3, +0.079 | ❌ |
| S vs R/E test | "S ≈ 0" | Welch t = −0.37, p = .73; mixed-model slope of x_s b = −0.042, p = .49 | ⚠️ no difference |
| Difrasismo / kenningar coded S? | yes (in text) | both coded x_s = 0 for every item | ⚠️ coding inconsistency |
| P3 COMPOUND vs UNITARY variance | "COMPOUND greater" | Phase-1 labels are MIXED (n=20) vs UNITARY (n=83); var 0.018 vs 0.009, Levene p = .28 | ⚠️ not significant |

**Verdict:** P2 and P3 are not supported by the repository data as coded. Either recode S-type explicitly (and get more items), or reframe as an open question.

## 5. P4 — Channels

| Statistic | Submitted | Recomputed | Status |
|---|---|---|---|
| Phonetic effect (SI S4) | b = −0.061, p = .001 | mixed model **b = +0.066, p = .002** (cluster-robust OLS b = +0.044, p = .09) | ❌ sign reversed |
| Formal effect (SI S4) | b = −0.043, p = .031 | b = +0.030, p = .21 | ❌ |
| S-proportion (SI S4) | b = −0.058, p = .016 | x_s b = +0.024, p = .61 (corrected coding) | ❌ |
| ICC (SI S4) | 0.463 | 0.329 (residual ICC in that model) | ❌ |
| Pure phonetic items | "weakly recoverable" | n = 23, +0.079, 91% positive, p < .001 (10 of 23 are xiéyīn) | ⚠️ |
| Kakekotoba | +0.002, p = .470 | +0.0015, p = .470 | ✅ |
| Xiéyīn | p = .002 | p = .002 | ✅ |

Per-channel descriptives (any-channel coding): Conceptual +0.082 (n=85), Phonetic +0.101 (n=33), Formal +0.069 (n=33), Morphological +0.065 (n=21), Structural +0.060 (n=31), Chromatic +0.029 (n=10), Behavioural +0.026 (n=39), Metonymic +0.023 (n=62).

**Verdict:** "Phonetic channels are invisible to text embeddings" is false as a channel-level claim. The data show: kakekotoba null, but phonetic items in general pull strongly. The defensible story is about entrenchment/circulation, not channel.

## 6. Metonymic — Phrasal Differential

| Group | p (Mann–Whitney) | g | BH q | Submitted |
|---|---|---|---|---|
| Japanese sexual | .042 | 0.67 | .078 | ✅ |
| English drinking | .047 | 1.15 | .078 | ✅ |
| Ritual Type-E | .033 | 1.08 | .078 | ✅ |
| English sexual | .139 | 0.70 | .139 | ✅ |
| Japanese drinking | .120 | 0.33 | .139 | ✅ |

3/5 significant uncorrected; **0/5 after BH-FDR**. Template-leakage control (Netflix-and-X) exists in repo (p = .15, not significant leakage) — usable as a robustness check that was never reported.

## 7. P5 — Tradition as unit

| Statistic | Submitted | Recomputed | Status |
|---|---|---|---|
| pseudo-ICC | 0.285 | 0.270 (script formula, nonstandard) | ⚠️ |
| ICC, random-intercept model, 213 items | — | **0.318**; LRT for tradition effect χ² = 33.9, p < .001 | 🆕 |
| ICC, traditions n ≥ 3 | — | 0.373 | 🆕 |

**Verdict:** P5 is supported, and more strongly with a proper ICC. Replace pseudo-ICC.

## 8. Other submitted material

| Item | Status |
|---|---|
| Context-gating table (botan/momiji/sakura + nabe) | ❌ no data or script in repo; repo similarity for sakura→馬肉 is 0.164 vs 0.169 in paper |
| SI S1 v1/v2 control comparison | ❌ only one control cache exists |
| SI S3 similarity gradient values (algospeak 0.405, Songlines 0.114 …) | ❌ not reproducible; recomputed gradient for the 22 traditions is in `verified_stats.json` |
| SI S3.1 eggplant/banana example | no data; anecdote only |

## 9. Robustness — BAAI/bge-m3 (run 2026-10-21, `embeddings/scripts/nnpm_replicate.py`)

Same 213 items, same stored control words, re-embedded with bge-m3 (1024-d, multilingual, open weights).

| Condition | Mean pull | % positive | t | Mixed-model z | Traditions sig (p / BH) | ρ with OpenAI pull |
|---|---|---|---|---|---|---|
| A: host as in original pipeline (mostly romanized), k = 10 | +0.002 | 50.7% | 0.29 | 1.44 | 4 / 2 | 0.47 |
| **B: host in original script, k = 10** | **+0.040** | **70.9%** | **7.19** | **6.67** | 7 / 2 | 0.19 |
| B, k = 3 / 5 / 7 | +0.038 / +0.039 / +0.039 | 64–70% | 6.5–7.1 | 6.4–6.7 | 6 / 0–2 | ~0.2 |

**Script mismatch confound.** In 99 of 213 items the host was romanized while the LLM-generated controls were
in the original script (CJK, Arabic, Cyrillic, Devanagari, Ethiopic). In bge-m3 condition A, these items have
negative pull (−0.028) and Latin-control items positive pull (+0.033). The same split exists in the original
OpenAI pull (+0.074 vs +0.040, p = .017). Comparing like with like (condition B) restores the effect.

**Gloss confound.** 62 pipeline hosts carry an English gloss (e.g. `pingguo_apple`, `Hodam (glutton / fat stomach)`).
In the clean subset (Latin controls, no gloss, n = 69): OpenAI +0.074 (83% positive), bge-m3 A +0.025 (57%, p = .002).

**Control quality.** In 54 items the control list contains the host string itself, and in 15 more a control contains
the host as a substring (e.g. 苹果 → 红苹果). This biases pull toward zero (conservative), but should be fixed.

**Verdict:** the aggregate effect (P1) replicates in an independent multilingual model when host and controls
share a script. Tradition-level significance is model-dependent and should not carry weight on its own.

## 10. Summary for the rewrite

Keep and strengthen: P1 (all cluster-aware checks + bge-m3 replication in original script), P5 (proper ICC),
metonymic method (report BH honestly, add template-leakage test), per-tradition table with FDR.
Rewrite: P4 (entrenchment, not channel), P2/P3 (S-type pulls; test superadditivity instead).
Fix in the pipeline: original-script hosts as the primary condition, remove glosses (`clean_host`),
drop controls identical to or containing the host.
Remove or redo: context-gating table, SI S1, S3, S4 as submitted.

## 12. Recording status (spec v1.0, run 2026-10-21, `analysis/analyze_recording.py`)

Three LLM coders (gpt-5.6-sol, gemini-3.1-pro-preview, claude-opus-5-5) coded form_status and recording_status for all 217 items. Fleiss' κ: form_status 0.869, recording_status 0.773.

Analysed for H1/H2: 203 of 213 items with pulls (9 novel_form, 0 form UNRESOLVED, 1 recording UNRESOLVED excluded).

| hypothesis | main | sensitivity (unanimous labels) |
|---|---|---|
| H1 recording gradient | not supported (0/3) | not supported (0/3) |
| H2 unrecorded CPG > 0 | not supported (0/3 Holm) | mixed (1/3 Holm) |
| H3 neologisms excluded | procedural rule applied (no inferential test, per spec) | – |

% positive by recording_status (existing_form), main analysis:

| model | unrecorded | cultural_reference | general_dictionary |
|---|---|---|---|
| pull_index | 40.0% (n=10) | 80.2% (n=126) | 70.1% (n=67) |
| pull_bge_B | 80.0% (n=10) | 76.2% (n=126) | 59.7% (n=67) |
| pull_gem_B | 90.0% (n=10) | 82.5% (n=126) | 95.5% (n=67) |

Full tables: `results/recording_results.md`.

## 13. T1–T4: clean hosts, condition C, codebook v1.1 (run 2026-10-22, `analysis/analyze_t1.py`, `coding/scripts/aggregate_v1_1.py`, `analysis/analyze_final.py`)

Author decisions of 2026-10-22 applied here (full text: `docs/DECISIONS_2026-10-22.md`):
1. ask_author items: no host form (MAYA_005, QCH_006) and several different words listed (NET_003; also NEU_006) are excluded; Exeter Book riddles and whole verse lines are kept with host_unit = utterance; items matching none of these are kept and listed (`docs/T3_exclusions.md`).
2. use_first_form: variants of one item → first form becomes clean_host (embeddings recomputed with the three models); different words with different meanings → excluded; unclear → not adopted, listed.
3. k = 10 (n fixed in advance) stays the primary analysis; k = 3, k = 5 and kall are sensitivity analyses.
4. R2 kept as is; dropped controls per rule are at the top of `results/t1_results.md`.
5. CPG scope = form_type existing_form or transparent_combination; two_storey and outsider_inference are reported descriptively only (the first definition is kept as cpg_scope_two_storey).
6. outsider_inference UNRESOLVED stays a separate level in the descriptives; items whose form_type is UNRESOLVED are listed below.

Items: 217 → 213 with controls → 209 after T3 (SEA_012, MAYA_005, QCH_006, NET_003, NEU_006, QCH_003; those with controls) → 185 not low_control → k = 10 pull for 94 (items with 10 kept controls).

| hypothesis | OpenAI | bge-m3 | Gemini | verdict |
|---|---|---|---|---|
| H-C1 | n=94, z=5.3901, Holm p=0.0000 | n=94, z=2.2285, Holm p=0.0129 | n=94, z=5.1288, Holm p=0.0000 | supported |
| H-C2 | n=63, z=4.2347, Holm p=0.0000 | n=63, z=2.3060, Holm p=0.0106 | n=63, z=4.0527, Holm p=0.0001 | supported |
| H-C1 before the decisions | n=93, z=5.9285, Holm p=0.0000 | n=93, z=2.9401, Holm p=0.0016 | n=93, z=5.7019, Holm p=0.0000 | supported |
| H-C2 before the decisions | n=62, z=4.8550, Holm p=0.0000 | n=62, z=2.8990, Holm p=0.0019 | n=62, z=4.8577, Holm p=0.0000 | supported |
| old_criterion_existing_form (exploratory) | n=89, z=5.3731, Holm p=0.0000 | n=89, z=2.4610, Holm p=0.0069 | n=89, z=5.3645, Holm p=0.0000 | supported |

The H-C1 test restricted to cpg_scope = yes is identical to the test on all main-set items (all_main), because every item kept after T3 meets the form criterion; it is therefore not tabulated separately.
The exclusions of 2026-10-22 were decided after the first results had been seen; the rows 'before the decisions' are kept for that reason (SI).

Codebook v1.1 agreement (Fleiss' κ / % unanimous): form_type 0.722 / 84.8%; two_storey 0.269 / 82.5%; content_or_attitude 0.587 / 91.7%; outsider_inference 0.444 / 53.0%; host_unit 0.939 / 94.5%.
cpg_scope (form_type) after T3: yes 211; first definition cpg_scope_two_storey (descriptive): yes 107, no 89, UNRESOLVED 15.
H-C1 / H-C2 verdicts: H-C1 supported → supported; H-C2 supported → supported.

Full tables: `results/t1_results.md`, `results/final_results.md`.
