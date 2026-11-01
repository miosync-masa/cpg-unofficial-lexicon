# Analysis plan: source audit of all 217 items

Fixed on 2026-10-24, before any judgement of stage B was made. The commit that adds this file is the record of that.

## Why

The Phase 1 items (103) have no per-item source. The Phase 2 items (114) were each given sources by the agents that
collected them, but nobody checked that the sources exist or that they support the item. The audit asks, for each
item, whether a document that was actually retrieved states that in this community this host carries this covert
meaning.

## Rules

1. No judgement from a document that was not retrieved. Knowledge or memory is never evidence.
2. Evidence collection (stage A) and judgement (stage B) are separate. Stage B sees only the quotes.
3. Nobody doing the audit looks at an item's pull. Items are processed in a random order fixed by seed 20261023
   (`numpy.random.default_rng(20261023).permutation` of the item IDs of `data/blind_items.csv` in file order). The
   audit reads items from `data/blind_items.csv`, which has no pull column.
4. Only URLs, bibliographic data, access dates, judgements and short quotes (at most 40 words) are published. Full
   texts are stored outside the repository (`private/`, ignored by git). Paywalls are not circumvented; a site that
   refuses access is not approached by another route (including web archives).

## Stage A: evidence (all 217 items)

- Phase 2: the recorded sources, in the recorded order. Phase 1, or Phase 2 when the recorded sources are not
  enough: a search on host, covert target and tradition; scholarly literature first, then dictionaries and
  encyclopaedias, then Wikipedia, blogs and media.
- At most 3 documents per item; collection may stop once a quote that supports the item is found.
- A quote must be a verbatim passage of the stored text of the retrieved document (checked by script after
  normalising whitespace) and at most 40 words. A book whose contents cannot be retrieved is `record_only`.
- One row per document in `results/source_audit_evidence.csv`: `item_id, doc_rank, source_type, citation, url,
  accessed, retrieved, quote, snapshot_sha256, listed_by_agent`.

## Stage B: judgement

- Coders: `gpt-5.6-sol` (no temperature), `claude-opus-5-5` (no temperature), `gemini-3.1-pro-preview`
  (temperature 0), as in the earlier coding runs. Batches of 10 items in the seed order above; no earlier answers
  are sent.
- Input per item: host (original and romanised), covert target, tradition, and the quotes, each with its
  `source_type` and `retrieved` value. Not sent: citation, URL, pull, item ID beyond a batch-local number.
- The prompt states that the judgement must rest only on the quoted text and not on any other knowledge.
- One label per item:
  - `supported`: a quote states that this host carries this covert meaning in this community;
  - `partly`: a quote states that the tradition or technique exists, but not the link between this host and this
    covert meaning; or it states a related but different meaning;
  - `contradicted`: a quote states something that conflicts with the item;
  - `no_evidence`: no relevant quote.
- An item with no quote at all (all documents `failed` or `record_only`, or nothing relevant found) is labelled
  `no_evidence` by rule and is not sent to the coders.
- Majority of the three coders; three different labels: `UNRESOLVED`. Fleiss' kappa over the items sent to the
  coders. A Claude refusal is resent as in the earlier runs (up to 3 times, then item by item); every refusal is
  logged and counted.

## Author's spot check

- After stage B, 20 items drawn with `numpy.random.default_rng(20261024)`: 10 from the items with majority
  `supported`, 10 from all other items (including `UNRESOLVED` and rule-based `no_evidence`). If a stratum has
  fewer than 10 items, all of them are taken.
- The author reads the documents and assigns the same four labels without seeing the LLM labels. Reported:
  agreement (percent) with the majority label, and the items that differ.

## Analysis

- Counts per label: all items, per phase, and per best `source_type` of the item.
- Sensitivity: the condition C main set (k = 10, n = 94) restricted to (a) items with majority `supported`, and
  (b) `supported` or `partly`. H-C1 and H-C2 recomputed exactly as in `analysis/analyze_t1.py` (mixed model
  `pull ~ 1 + (1 | tradition)`, raw labels, REML, one-sided z, Holm across the 3 models). Verdicts are reported
  whatever they are.
- `contradicted` items are listed and mentioned in the text; whether to remove them is the author's decision.

## Output

- `results/source_audit_evidence.csv`, `results/source_audit_judgements.csv` (per coder and majority),
  `results/source_audit_summary.{md,json}`.
- Macros through `analysis/make_numbers.py`: `NauditSupported`, `NauditPartly`, `NauditContradicted`,
  `NauditNoEvidence`, `NauditUnresolved`, `Kaudit`, `PauditAuthorAgree`, and the sensitivity z and p
  (`HConeZsup*`, `HConePsup*`, `HConeZsupPart*`, ..., with the model suffix).

## Amendments

### Decisions made during stage A (2026-10-24, before stage B)

- A page that shows only an abstract is `retrieved = abstract`; earlier rows were corrected. A library catalogue
  record (e.g. Open Library) is `record_only`.
- A document marked as AI-generated is kept in the record with `ai_generated = yes` and is not sent to stage B; a
  different document is looked for. A page that mixes source text with AI-generated paraphrases is treated the same
  way (T_029, document 1); the source text is retrieved again from another edition.
- When the only quotes are at the level of the tradition or technique, one search on host + covert meaning is
  required before collection stops.
- **Host criterion**: a quote counts as mentioning the host when it names the host itself or refers to what the host
  denotes in translation (e.g. "the mole on the cheek" for an item whose host is the word for mole).
- Columns added to `results/source_audit_evidence.csv`: `quote_2` (a second quote from the same document),
  `agent_source` (for sources recorded by the agents: `exists`, `inaccessible`, `not_found`) and `ai_generated`.

### Decisions made after stage B (2026-10-25)

- **Retrieval date.** All documents were retrieved on 2026-10-24, three days after the disclosure of two critical
  PHP vulnerabilities (CVE-2026-109618, CVE-2026-109619) on 2026-10-21, which led many sites (universities,
  governments, journal platforms running OJS) to go offline or restrict access. Some `failed` documents may
  therefore reflect a temporary outage.
- **Retrieval again.** Around 2026-11-01, every document that failed with a server error (HTTP 5xx) or a connection
  error (timeout, SSL, connection reset) is retrieved again from the same URL. New quotes are added, and the items
  whose quotes change are judged again in stage B by the three coders (same prompt and procedure; both the old and
  the new labels are reported).
- **Web archives.** Rule 4 is relaxed for documents that could not be retrieved because of a server outage or a
  dead link (HTTP 5xx, connection errors, HTTP 404). It is not relaxed for a site that refused access (HTTP 401,
  403, 429, bot challenges). For these documents the Wayback Machine is used, taking the capture closest to
  2026-03-13; such rows have `retrieved = archive`. The archived text may be quoted and used in stage B.
- **Types of `not_found`** (new column `not_found_type`, only for agent sources with `agent_source = not_found`):
  - (a) `unspecific`: the source does not name a specific document (e.g. "Slavic mythology scholarship");
  - (b) `specific_not_found`: a specific document (author/title or a specific page) that a search did not find;
  - (c) `url_unretrievable`: a URL was given but could not be retrieved, and no other copy was found.
  Only (b) is treated in the paper as a source whose existence is doubtful. Macros: `NnotFoundUnspecific`,
  `NnotFoundSpecific`, `NnotFoundURL`.
- **Sensitivity in condition D.** The two restrictions are applied also to condition D: H-D1 and H-D2 recomputed
  as in `analysis/analyze_target_blind.py` on the main-set items with a k = 10 condition D pull. Macros
  `HDoneZsup*`, `HDonePsup*`, `HDtwoZsupPart*`, ..., with the model suffix.
- **Corrections of item fields** are listed in `audit/corrections.csv` and applied to the stage B input. NEU_001:
  tradition "Old Norse kenning (skaldic poetry)" → "Old English kenning" (hronrád / hwælweg are Old English forms);
  the item was judged again by the three coders (`audit/runs/judge_<model>_rejudge_2026-10-25.*`). The correction is
  not applied to the tradition grouping of the embedding analyses.
- **Contradicted items** (EEU_008, T_041, MAYA_003) stay in the lexicon and in every analysis, with
  `audit_flag = contradicted` (`data/audit_flags.csv`, also in `results/items_final.csv`).

### Author's spot check and components (2026-10-25)

- **Spot check, items.** The 20 items are the draw made right after stage B (commit 7d31fce; strata from the
  stage B majority, seed 20261024 on the audit order of seed 20261023), and the author judged exactly these. When
  NEU_001 was judged again, `analysis/analyze_source_audit.py` was run again and drew the strata from the new
  majority (NEU_001 moved from `contradicted` to `supported`), which changed 5 of the 20 items and overwrote the
  local sheet. This was a fault of the script, not a new draw by design: the strata now use the stage B majority
  before any judging again (`majority_first`), which gives the original 20 items again. Items the author may judge
  outside these 20 are reported for reference only.
- **Spot check, measures.** Besides the percent agreement with the LLM majority (`PauditAuthorAgree`), Cohen's kappa
  between the author and the majority over the four labels (`KauditAuthor`); the items that differ are listed in
  `results/numbers_report.md`. The author's labels are in `data/audit_spot_check_author.csv`.
- **Reason codes.** The author's table also gives H, M and C (yes / partial / no), reason codes for `partly`
  (`missing_H`, `missing_M`, `partial_M`, `missing_C_device`, `other_form`, `different_work`) and notes. These are
  kept as an internal record outside the repository, unchanged.
- **Components (exploratory).** With the same input as stage B, the three coders answer for every item H (host
  named), M (covert meaning stated), C_community (placed in the community or tradition) and C_device (meaning carried
  through the named technique), each yes / partial / no (`audit/scripts/judge_evidence.py --components`). H = yes
  also when a translation names what the host refers to; H = partial only when the host appears solely as part of a
  longer, different expression. Run together with the judging again around 2026-11-01, on the input of that time.
  The four labels are not changed by this run. Items with no quote are not sent; they are left out of the
  reason-code tallies and counted separately as `no_evidence`.
- **Components → reason codes**, fixed before the run, on the majority (2 of 3) of each component (no majority:
  `UNRESOLVED`): H = no → `missing_H`; H = partial → `other_form`; M = no → `missing_M`; M = partial → `partial_M`;
  C_device ≠ yes → `missing_C_device`; C_community = no → `missing_C_community` (not one of the author's codes).
  `different_work` cannot be derived from the components and is not computed. An item can have several codes.
  Reported: the distribution of the codes over the items sent and within each majority label, in particular the
  number of `partial_M` items, with the items not sent (`no_evidence`) as a separate count.

### Retrieval again, first run (2026-10-30)

- Run on 2026-10-30, two days before the planned date, on the author's instruction.
- Scope: the document rows with `retrieved = failed` whose last request ended in HTTP 5xx or a connection error
  (7 rows: NAH_004/3, MET_006/2, MET_007/1, EEU_011/1, NET_003/2, MET_008/2, MET_009/3). Each was requested once more
  from the same URL (`audit_tool.py refetch`). Failed requests of searched documents that were never kept as rows are
  not documents of the audit and were not requested again.
- All 7 failed again with the same kind of error.
- Wayback Machine, for these 7 rows, for MET_001/1 (HTTP 404) and for SEA_008/2 (the address the agent gave, where
  archive.org had answered 429 on 2026-10-25): 5 had no capture; for the other 4 (NAH_004/3, MET_007/1, MET_009/3,
  SEA_008/2) archive.org answered HTTP 429 when the capture was requested, again after 90 seconds. Not tried further.
- No document changed, so no item was judged again. Every request is in the attempt logs.

### Retrieval again, second run, and components (2026-11-01)

- The same 7 rows were requested once more from the same URL: all failed again with the same kind of error. The
  4 Wayback captures that archive.org had refused (HTTP 429) were requested once more: 429 again. Not tried further.
  No document changed, so no item was judged again.
- The components were then judged on this input by the three coders (`audit/runs/judge_components_<model>.*`):
  214 items sent, 3 not sent (no quote), no refusals. Tallies by `analysis/analyze_components.py` →
  `results/audit_components.{csv,json,md}`. Three items sent have the majority label `no_evidence` (their quotes
  were judged not relevant); they are kept in the tallies under that label.

### Macros added later (2026-11-01)

The plan did not name macros for the components. Added on 2026-11-01, after the run, on the author's request
(`analysis/make_numbers.py`, section 15, from `results/audit_components.*`):
- `NcompSent`: items sent to the component judging.
- `NcompMissingDevice`: items with the reason code `missing_C_device`.
- `NcompMissingDeviceSup`: of these, the items whose majority label is `supported`.
- `NcompPartialM`: items with the reason code `partial_M`.

The documents that retrieval again and the Wayback Machine did not retrieve are listed in
`results/audit_unretrieved.{csv,md}` (`analysis/list_unretrieved.py`, from the attempt logs).

### Corrected labels (author's decision, 2026-11-01)

- The analyses keep the published labels; their numbers and macros do not change. The tradition grouping of the
  embedding analyses was fixed on the published labels before any analysis.
- `results/items_final.csv` gets `tradition_corrected`, `language_corrected` and `correction_reason`
  (`data/label_corrections.csv`); the original columns are not overwritten. The only item with a corrected label is
  NEU_001 (the only row of `audit/corrections.csv`): tradition → `Old English kenning`, language → `Old English`.
- Sensitivity (`analysis/make_numbers.py`, section 16): H-C1, H-C2, H-D1 and H-D2 with the tradition intercept
  grouped by the corrected labels, 3 models, Holm as in the main analysis. Macro `SensCorrSame` (built as
  `SensTradSame`). Also computed, not fixed in advance: NEU_001 joined to the existing label `Anglo-Saxon kenning
  (Beowulf)` instead of a label of its own. Table in `results/numbers_report.md`.
