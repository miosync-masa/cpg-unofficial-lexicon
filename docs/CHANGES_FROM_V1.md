# Changes from the submitted version (v1)

The submitted version is kept in `legacy/v1_submission/` and under the tag `v1-language-sciences-2026-03`.
Five problems were found when the analysis was recomputed from the repository data. Each is fixed in the current
version. Numbers are not repeated here; they are in `results/verified_numbers.md`, cited by section (§).

## 1. S-type values read from the wrong YAML key

**Problem.** The v1 archaeology script read the S component only from the `x_s` key of the Phase-2 YAML. However,
52 items store their direction coding as `x_type: [r, e, s]`, and for those items S was silently set to 0. This
includes every Difrasismo and Old Norse kenning item. The S value changed for 51 items once the key was read.

**Fix.** `analysis/reproduce_stats.py` reads `x_type` when it is present.

**Effect.** With the corrected coding, S-dominant items pull toward their targets instead of showing no pull, and
S-type items do not differ from R/E items. The S-type prediction (P2) as written is therefore not supported (§4;
§4 also keeps the superseded first pass for comparison). The revision also recodes direction blind with three
LLMs (first coding run, `coding/results/first_run/`).

## 2. Romanized hosts compared with original-script controls

**Problem.** In 99 of the 213 items the host was embedded in romanized form, while the LLM-generated control
words were in the original script (CJK, Arabic, Cyrillic, Devanagari, Ethiopic). Similarity between a
romanized host and original-script controls is not comparable with similarity within one script. These items
had negative pull in an independent model (§9, "Script mismatch confound").

**Fix.** The host is embedded in its original script: condition B (§9) and condition C (§13, T1).

**Effect.** With host and controls in the same script, the overall effect replicates in BAAI/bge-m3 (§9). Under condition C the fixed hypotheses H-C1 and H-C2 are tested with three embedding
models (§13; full tables in `results/t1_results.md`).

## 3. English glosses in hosts

**Problem.** In 62 items the host string used for embedding carried an English gloss, such as `pingguo_apple` or
`Hodam (glutton / fat stomach)`. A gloss moves the host toward its literal meaning and away from the covert
target (§9, "Gloss confound").

**Fix.** Each item gets a clean host, the bare form, taken from the majority of the three LLM coders
(`embeddings/controls/clean_hosts.csv`, built by `embeddings/scripts/build_clean.py`). Hosts with a problem a
rule cannot settle were checked separately (T3, `docs/T3_exclusions.md`) and decided by the author
(`docs/DECISIONS_2026-10-22.md`).

**Effect.** Condition C uses the clean host (§13).

## 4. Controls identical to or containing the host

**Problem.** Some stored control lists contained the host itself or a compound of it (e.g. 苹果 → 红苹果). Such
controls are as close to the target as the host, so they bias the pull toward zero (§9, "Control quality").

**Fix.** Controls are cleaned by four rules, applied in order:
- R1: identical to the host.
- R2: contains the host or is contained in it.
- R3: written in a different script from the host.
- R4: a duplicate.

Items left with fewer than five controls are flagged `low_control`. Controls are never regenerated.

**Effect.** The number of controls dropped by each rule is the first table of `results/t1_results.md`. The cleaned
lists are in `embeddings/controls/controls_clean.csv`. Condition C uses only the kept controls (§13).

## 5. Supplementary statistics that could not be traced to code

**Problem.** Several values in the supplementary material could not be reproduced from any script or data file in
the v1 repository:
- the context-gating table;
- the SI S1 control comparison;
- the SI S3 similarity gradient;
- the SI S4 channel regression, whose recomputed signs are reversed;
- several per-tradition rows of SI S2.2.

These are listed in §3, §5 and §8.

**Fix.** Every number in the revision is written by a script. `results/verified_numbers.md` records where each
number comes from, and `rerun_offline.sh` regenerates all results from the stored codings and vectors without any
API call.

**Effect.** Material that cannot be traced is removed or redone (§10).
