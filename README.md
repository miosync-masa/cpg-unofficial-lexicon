# CPG: Covert Path Generation

Covert Path Generation (CPG) is the case in which a word or phrase keeps its ordinary meaning and also carries a
second meaning that a community shares. Examples are a euphemism, a poetic pivot word or a food with an auspicious
meaning. CPG asks whether that second meaning is recorded in dictionaries, and whether text embeddings place the
form closer to its covert target than to comparable words.

This repository holds:
- 217 items from many traditions;
- blind codings of every item by three LLMs;
- embedding analyses with three models (the nearest-neighbour pull method, NNPM);
- a dictionary check of the Japanese items;
- the scripts that produce every reported number.

## Status

This is the revised version (October 2026). The analysis as submitted to Language Sciences (March 2026) is kept
in `legacy/v1_submission/` and under the tag `v1-language-sciences-2026-03`. It is superseded; the problems that
were fixed are described in `docs/CHANGES_FROM_V1.md`.

## Layout

```
├── README.md
├── LICENSE                    code: MIT
├── LICENSE-DATA               data and coding results: CC BY 4.0
├── CITATION.cff
├── rerun_offline.sh           reruns every step without any API call
├── requirements.txt
├── data/                      inputs (not changed by any script except the T3 lists)
│   ├── source/                Phase-2 YAML (7 files), Phase-1 table, coding table
│   ├── blind_items.csv        the 217 items as shown to the LLM coders
│   ├── control_cache.json     the stored control words (v1 pipeline)
│   ├── cpg_archaeology_results.csv   v1 pipeline item list, tradition labels and pull
│   ├── exclusions.csv         T3 exclusions with reasons; exclusion_candidates.csv, author_decisions.csv
├── coding/                    blind coding by three LLMs
│   ├── codebooks/             codebook.md, codebook_recording.md, codebook_v1_1.md
│   ├── scripts/               run_*.py, aggregate_*.py, build_blind_table.py, build_expert_sheets.py
│   └── results/               first_run/, recording/, v1_1/ (per-model CSV, consensus, agreement, JSONL logs)
├── embeddings/                NNPM
│   ├── scripts/               nnpm_replicate.py, analyze_replication.py, check_hosts.py, build_clean.py
│   ├── controls/              clean_hosts.csv, controls_clean.csv, controls_summary.csv
│   └── results/               pulls for conditions B and C (3 models), JSONL logs, vectors
├── analysis/                  reproduce_stats.py, analyze_t1.py, analyze_recording.py, analyze_final.py, …
├── results/                   the source of every number in the paper
│   ├── verified_numbers.md
│   ├── final_results.{md,json}, t1_results.{md,json}, recording_results.{md,json}, tables
│   └── before_decisions_2026-10-22/   outputs before the author's decisions (SI comparison table)
├── dictionary_check/          judgements against デジタル大辞泉 (dictionary text not included)
├── docs/                      specs, decisions, deviations, CHANGES_FROM_V1.md
└── legacy/v1_submission/      the submitted version (superseded)
```

## Reproducing the results

### Without API calls

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
./rerun_offline.sh            # or PYTHON=.venv/bin/python ./rerun_offline.sh
```

`rerun_offline.sh` recomputes everything from the stored LLM codings and embedding vectors:
- the v1 statistics;
- the agreement and consensus of each coding run;
- the T3 exclusions;
- the clean hosts and cleaned controls;
- condition C, from the saved vectors;
- the T1 and T4 analyses;
- `results/verified_numbers.md` §12–13.

It copies `results/` to `backups/` before overwriting it. Two consecutive runs give byte-identical output.

The Gemini vectors (`embeddings/results/nnpm_C_gemini_embedding_001_vectors.npz`) are larger than 50 MB. They
are a release asset rather than a file in git. Download them first:

```bash
gh release download data-2026-10-22 -p nnpm_C_gemini_embedding_001_vectors.npz -D embeddings/results
```

### Steps that call an API

API keys are read from `.env` at the repository root: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`.
Rerunning these steps replaces stored results. Copy `results/` (and the affected `coding/results/` or
`embeddings/results/` files) to `backups/` first.

**LLM coding.** Three runs, each of all 217 items, in batches of 10, with no previous answers sent:

| run | codebook | dates | output |
|---|---|---|---|
| first run | `codebook.md` | 2026-10-21 | `coding/results/first_run/` |
| recording status | `codebook_recording.md` | 2026-10-21 | `coding/results/recording/` |
| codebook v1.1 | `codebook_v1_1.md` | 2026-10-22 | `coding/results/v1_1/` |

| model ID | provider | temperature |
|---|---|---|
| `gpt-5.6-sol` | OpenAI | not set (the model rejects it) |
| `gemini-3.1-pro-preview` | Google | 0 |
| `claude-opus-5-5` | Anthropic | not set (the model rejects it) |

```bash
V=coding/results/v1_1
python3 coding/scripts/run_v1_1_coding.py --provider openai    --model gpt-5.6-sol            --out $V/v1_1_gpt.csv --no-temperature
python3 coding/scripts/run_v1_1_coding.py --provider google    --model gemini-3.1-pro-preview --out $V/v1_1_gemini.csv
python3 coding/scripts/run_v1_1_coding.py --provider anthropic --model claude-opus-5-5        --out $V/v1_1_claude.csv --no-temperature
# add --limit 10 for a pilot; run_llm_coding.py and run_recording_coding.py take the same options
```

Every request and response is logged next to the CSV as `<file>.csv.jsonl`.

**Embeddings.** The models are:
- `text-embedding-3-large` (OpenAI);
- `gemini-embedding-001` (Google, `task_type = SEMANTIC_SIMILARITY`);
- `BAAI/bge-m3` (run locally with sentence-transformers 6.2.0).

Conditions A and B were run on 2026-10-21 and condition C on 2026-10-22.

```bash
for b in openai gemini bge-m3; do
  python3 -I embeddings/scripts/nnpm_replicate.py data embeddings/results --backend $b --condition-c embeddings/controls
done
```

Every embedding call is logged to `embeddings/results/nnpm_C_<model>.jsonl`. Strings that could not be embedded
are listed in `nnpm_C_<model>_failed.csv`; there were none.

**Dictionary check.** See `dictionary_check/README.md`.

## Where the numbers come from

`results/verified_numbers.md` lists every reported number with its source file and script. The detailed tables
are in:
- `results/t1_results.md` (condition C, hypotheses H-C1 and H-C2);
- `results/final_results.md` (integrated analysis, exploratory). Its before/after table of H-C1 and H-C2 is
  part of the supplementary material.

How each analytic choice was made is recorded in `docs/DECISIONS_2026-10-22.md` and `docs/DEVIATIONS_T1_T3.md`.

## Licence and citation

The code is under the MIT License (`LICENSE`). The data, codings and results are under CC BY 4.0
(`LICENSE-DATA`). Please cite as given in `CITATION.cff`.

Author: Masamichi Iizumi (Miosync, Inc.)

## What is not published

- **Dictionary text.** The デジタル大辞泉 entries (raw HTML and extracted text) and any photographs or scans of
  printed dictionaries are copyrighted. Only the judgements, URLs and short quotations are published.
- **Review material.** Peer-review comments and submission planning are confidential.
- **API keys and local environment files.** These are never committed.
