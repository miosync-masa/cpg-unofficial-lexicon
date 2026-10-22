# Expert review — instructions for reviewers

Thank you for helping. Each sheet holds the items from one tradition group on which
automated coders disagreed with each other or with the original coding. You only need to judge
the items you feel competent to judge; leave the rest blank.

**Priority:** rows with `tier` 1 or 2 first. Tier 3 is optional.

For each row, read `host_original` (the expression), `covert_target` (the hidden meaning) and
`source_reference`, then fill:

| column | values | question |
|---|---|---|
| `expert_cpg_valid` | yes / no | Is this a genuine conventional covert meaning in the tradition (not an ad-hoc pun or a plain literal meaning)? |
| `expert_direction` | R / E / S | R = the hidden meaning is *reached* through the host; E = the host *encodes/embeds* it; S = both readings hold *simultaneously* and conventionally (see codebook §direction) |
| `expert_primary_channel` | Phonetic, Morphological, Chromatic, Conceptual, Structural, Behavioural, Metonymic, Formal | Which link between host and hidden meaning matters most? |
| `expert_confidence_1to3` | 1 = unsure, 2 = fairly sure, 3 = sure | |
| `expert_note` | free text | Anything the codes miss, or a reference |

The columns `gpt_*`, `gemini_*`, `claude_*` and `original_*` show what others decided; you may ignore
them or disagree freely. Full definitions: `../codebook.md`.
