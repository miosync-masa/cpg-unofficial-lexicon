# Codebook: taboo domain

Each item is an expression (the host) that carries a covert meaning shared by a community. You see the host, the
covert meaning, the tradition and the language. Judge the domain of the **covert meaning**, not of the surface
meaning of the host.

## `taboo_domain` (exactly one)

- `sex` — sexual acts, sexual organs, sexual desire, prostitution, promiscuity.
- `excretion` — urine, faeces, flatulence, vomit, menstrual blood and other bodily discharges, and the toilet.
- `death_illness` — death, dying, killing, suicide, corpses, disease, disability.
- `insult` — a disparaging label for a person or a group, a slur, or an obscene curse.
- `drugs_crime` — illegal drugs and their use, theft, fraud, and other criminal acts.
- `none` — none of the above.

If more than one applies, choose the one at the centre of the covert meaning.

## Output

JSON array, one object per item, same order as input, keys exactly: `item_id, taboo_domain`.
