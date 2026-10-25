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
