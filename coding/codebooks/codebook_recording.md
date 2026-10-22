# Codebook — recording status (CPG revision, v1.0, 2026-10-21)

You are given expressions (the *host*) that carry, or are claimed to carry, a hidden meaning (the *covert
target*) in a language or tradition. Judge only the fields below.

## `form_status` (exactly one)
- `existing_form` — the host is an ordinary word, phrase, number, emoji, or fixed expression that already had a
  surface meaning before the covert meaning attached (e.g. 草 "grass" → "lol"; *corn* → porn; "Netflix and chill";
  304; 海老).
- `novel_form` — the host was created for the covert meaning and has no independent surface meaning: a coinage,
  blend, respelling, or abbreviation (e.g. *seggs*, *unalive*, メダカブス).

## `recording_status` (exactly one; judge the covert meaning, not the surface meaning)
- `general_dictionary` — the covert meaning is listed as a sense, or a clearly marked usage (e.g. "euphemism",
  "slang", "vulgar"), in a standard general-language dictionary of that language (e.g. Kōjien / Daijirin,
  Le Robert, Oxford/Merriam-Webster, Monier-Williams, Lane, Xiàndài Hànyǔ Cídiǎn).
- `cultural_reference` — not in a general dictionary's sense list, but documented in specialist or cultural
  references: poetic-diction or rhetoric dictionaries, symbol and folklore dictionaries, scholarly commentaries
  and editions, ethnographies, encyclopaedias, or established guides to customs.
- `unrecorded` — not documented in either kind of source as far as you know; known mainly through use within the
  community (online communities, oral tradition, in-group speech).

## `recording_evidence`
The **type** of source that supports the label (e.g. "general dictionary sense", "poetic-diction dictionary",
"folklore guide", "scholarly edition", "online usage only"). Name a specific title only if you are certain it
exists and contains this item; otherwise give the type only. Never invent titles.

## `recording_confidence`
`1` guess, `2` fairly sure, `3` sure.

## Output
JSON array, one object per item, same order as input, keys exactly:
`item_id, form_status, recording_status, recording_evidence, recording_confidence`.
