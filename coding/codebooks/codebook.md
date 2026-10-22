# CPG Coding Codebook (v1.0, 2026-10-21)

You are coding items for a study of **covert meaning**: cases where a word or phrase keeps its ordinary
(surface) meaning while also carrying a second meaning that a community shares but dictionaries do not record.

Each item gives: language, tradition, the host form (original script and romanization), the covert target
meaning, and sometimes a literature source. Judge each item **independently**, using only your own knowledge
of the language and tradition plus the information given. Do not assume the item is a valid case.

Answer every field for every item. When you are unsure, still choose the best label and lower `confidence`.

---

## Field 1 — `familiarity`  (how well you know this item)

- `known` — you know this specific form and its covert meaning from your own knowledge.
- `tradition_only` — you know the tradition but not this specific item.
- `unknown` — you are relying only on the information given.

## Field 2 — `cpg_valid`  (is this a case of covert meaning at all?)

`yes` only if **all three** hold:
1. **Surface preserved** — the host form has an ordinary meaning that remains available.
2. **Socially shared** — the covert meaning is shared within a community (not a private association, not a
   one-off inference made up on the spot).
3. **Motivated link** — there is an identifiable link (sound, shape, colour, concept, function, behaviour,
   contiguity, or form) between the surface form and the covert meaning.

Otherwise `no`, and put the failing condition(s) in `cpg_fail_reason`: `surface`, `shared`, `motivated`,
or `other` (for example: it is an ordinary dictionary sense, a brand-new coinage with no surface meaning,
or an institutional relabelling).

## Field 3 — `direction`  (how the second meaning is carried; exactly one label)

Apply the **S test first**. Answer the three checks:

- `s_juxtaposition` — Is the host made of **two or more** words/elements placed side by side
  (a pair, couplet, compound, or fixed collocation)? (`yes`/`no`)
- `s_noncompositional` — Is the covert meaning **not obtainable** from any single element on its own,
  arising only from the combination? (`yes`/`no`)
- `s_conventional` — Is this particular combination → meaning link **fixed by convention** in the tradition
  (not created freshly in one text)? (`yes`/`no`)

If all three are `yes` → `direction = S`.

Otherwise choose between R and E:

- `R` (**Refuge**) — the covert meaning existed independently and is **hidden inside** an ordinary form,
  typically to avoid saying it directly (taboo, euphemism, concealment, political coding, double entendre).
  Test: *the meaning picks a word to hide in.*
- `E` (**Enchantment**) — a new, additional significance is **attached to** an ordinary form, typically
  symbolic or auspicious, and is displayed rather than hidden (ritual food symbolism, emblematic meaning).
  Test: *the word is given a meaning.*

If `cpg_valid = no`, still give your best `direction`.

## Field 4 — `channels`  (which links connect surface and covert meaning; one or more)

Select **every** channel that is necessary for the link, then name the single most important one in
`primary_channel`.

| Channel | Definition | Example |
|---|---|---|
| `Phonetic` | sound similarity or identity (homophony, near-homophony, puns) | Chinese 鱼 yú "fish" → 余 yú "surplus" |
| `Morphological` | visual shape or form resemblance of the referent or the written form | shrimp's curved back → old age |
| `Chromatic` | colour association | red-and-white → celebration |
| `Conceptual` | abstract analogy or metaphor between the two meanings | *la petite mort* → orgasm |
| `Structural` | transfer of a functional property of the referent | lotus root's holes → seeing ahead |
| `Behavioural` | projection of an observed behaviour of the referent | crane pair-bonding → marital fidelity |
| `Metonymic` | contiguity in an action chain, frame, or scene (A leads to / co-occurs with B) | "Netflix and chill" → sex |
| `Formal` | the poetic or textual form itself creates the link (couplet structure, parallelism, metre) | pantun foreshadow–meaning parallelism |

## Field 5 — `activation`  (when is the covert meaning active?)

- `always_on` — the covert reading is available in essentially any context.
- `register_gated` — active in a particular register, speech style, or community of speakers.
- `context_gated` — needs a specific co-text or situation (a collocate, an occasion, a genre).
- `ritual_exclusive` — restricted to ritual, initiated, or ceremonial settings.

## Field 6 — `components`

If the host consists of several elements, list them as separate strings in their original form
(for example `["in xōchitl", "in cuīcatl"]`). Otherwise give an empty list.

## Field 7 — `clean_host`

The host form as a speaker would write it, in its original script, **without** any English gloss,
explanation, or alternative spellings. One form only.

## Field 8 — `confidence`

`1` = guess, `2` = fairly sure, `3` = sure. Rate your confidence in `direction` and `primary_channel`.

## Field 9 — `rationale`

One or two short sentences justifying `direction` and `primary_channel`.

---

## Output format

Return **only** a JSON array, one object per item, in the same order as the input, with exactly these keys:

```json
{
  "item_id": "...",
  "familiarity": "known | tradition_only | unknown",
  "cpg_valid": "yes | no",
  "cpg_fail_reason": "" ,
  "s_juxtaposition": "yes | no",
  "s_noncompositional": "yes | no",
  "s_conventional": "yes | no",
  "direction": "R | E | S",
  "channels": ["..."],
  "primary_channel": "...",
  "activation": "always_on | register_gated | context_gated | ritual_exclusive",
  "components": [],
  "clean_host": "...",
  "confidence": 1,
  "rationale": "..."
}
```
