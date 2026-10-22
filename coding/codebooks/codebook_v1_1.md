# Codebook v1.1 — form and boundary (CPG revision, 2026-10-22)

You are given expressions (the *host*) that carry, or are claimed to carry, a hidden meaning (the *covert
target*) in a language or tradition. A CPG item is two-storeyed: a surface layer that anyone can read from the
words, and a covert layer that a community keeps. Judge only the fields below.

## `form_type` (exactly one)
- `existing_form` — an ordinary word, phrase, number, emoji or fixed expression that already had its own surface meaning before the covert meaning attached (草 "grass", tea, 304, 海老, "Netflix and chill").
- `transparent_combination` — a new combination of existing elements whose **literal** meaning an outsider can still read, while the covert meaning needs community knowledge (whale-road, 草泥马 "grass-mud horse", 五月三十五日 "May 35th", メダカブス).
- `opaque_coinage` — a new form with no literal meaning an outsider can read (マジ卍, ズボラズギュン).
- `blend_or_abbreviation` — a new form whose meaning is just the sum of its recognisable parts, with no second layer (チョベリバ, サボる).
- `expressive_coinage` — a new form motivated by sound or feeling only, with no second layer (ぴえん).
- `respelling` — a deliberate misspelling or variant of an existing word that keeps the same meaning (seggs, unalive).

## `two_storey` (`yes` / `no`)
`yes` only if the item has BOTH (a) a literal surface layer an outsider can read and (b) a different covert layer shared by a community.

## `content_or_attitude` (exactly one)
- `content` — under the covert reading, the thing referred to changes (草: plant → laughter; 牡丹: flower → boar meat).
- `attitude` — the referent stays the same and only the speaker's evaluation or stance changes (いい度胸してるな; ご自由にどうぞ).
- `both`

## `outsider_inference` (exactly one)
Imagine a fluent speaker of the language who is not a member of the community, given the full context of use.
- `inferable` — they could work out the covert meaning from context and general reasoning alone.
- `inferable_with_effort` — they could work it out, but it is a conventional shortcut (e.g. "Can you pass the salt?").
- `not_inferable` — they would need community or cultural knowledge; once told, the link makes sense.
- `not_motivated` — even once told, there is no understandable link.

## `host_unit` (exactly one)
`word` / `phrase` / `utterance` (a whole sentence or saying, e.g. ぶぶ漬けでもどうどす)

## `confidence` and `rationale`
`confidence`: `1` guess, `2` fairly sure, `3` sure. `rationale`: one or two sentences.

## Output
JSON array, one object per item, same order as input, keys exactly:
`item_id, form_type, two_storey, content_or_attitude, outsider_inference, host_unit, confidence, rationale`.
