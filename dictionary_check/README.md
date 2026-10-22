# Dictionary check

Does a standard Japanese dictionary record the covert meaning of each Japanese item? 41 headwords were checked
against デジタル大辞泉 (Shogakukan), as published on Kotobank.

## Files

- `daijisen_judgments.csv` — one row per headword:
  - `item_id`, `見出し語`
  - `URL` (the Kotobank page), `取得日時` (time retrieved)
  - `判定`: where the covert meaning appears (see Criteria)
  - `一致度`
  - `引用`: only the short passage that supports the judgement, verbatim
  - `判定メモ`
- `fetch_log.csv` — every request: query, URL, final URL, HTTP status, time.
- `scripts/fetch_kotobank.py` — fetches the pages and extracts the 大辞泉 entries.
- `scripts/apply_judgements.py` — adds the judgements; each quotation is checked to be a verbatim substring of the entry text.
- `scripts/export_public.py` — writes `daijisen_judgments.csv` without the entry text.

## Retrieval

The pages were retrieved on 2026-10-21. The lookup rules for each headword (alternatives, idiom pages, readings,
spelling variants) are in the docstring of `scripts/fetch_kotobank.py`. Nothing was filled in by hand. A headword
with no retrieved entry is left blank.

## Criteria

`判定` records where the covert meaning appears in the retrieved entry text:
- `番号つきの語義`: a numbered sense, or a sub-sense of one.
- `補説・用例`: not a sense, but stated in a usage note, an example or the explanatory part of a definition.
- `なし`: not stated.

`一致度` is `一致` when the text states the covert meaning itself and `近い` when it gives a closely related sense
(explained in `判定メモ`).

The judgements were made by an LLM (claude-opus-5-5) reading the extracted text. They are not the judgements of
dictionary editors.

## Not published

The dictionary text is copyrighted. The raw HTML and the extracted entry text (`本文`) are kept locally in
`private/` and are not in this repository; every entry can be checked at its URL. Photographs or scans of
printed dictionaries are not published either.

A check against 三省堂国語辞典 (from 2026-10-24) will be added in the same form: the judgement and short
quotations of the label, such as 〔俗〕, only.
