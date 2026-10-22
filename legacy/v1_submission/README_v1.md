# CPG: Covert Path Generation

**The Unofficial Lexicon — A Cross-Cultural Framework for Community-Shared Meaning Between Semantics and Pragmatics**

> *"Dictionaries record tatemae. Embeddings preserve honne."*

## What is CPG?

Covert Path Generation (CPG) is the phenomenon in which a word or phrase retains its surface (dictionary) meaning while simultaneously carrying a second meaning that is socially shared within a community but absent from any lexicographic record.

The totality of such covert paths within a community constitutes the **Unofficial Lexicon** — a parallel meaning system that is learned, transmitted, and understood by community members, yet never formally documented.

## Repository Structure

```
cpg-unofficial-lexicon/
├── data/
│   ├── shogun/          # Cross-cultural survey data (114 items, 8 regions)
│   ├── torami/          # Linguistic coding data (103 items, 12 traditions)
│   └── merged/          # Unified dataset (217 items, 53 traditions)
├── embeddings/          # Pre-computed embeddings (text-embedding-3-large, 3072d)
├── scripts/
│   ├── archaeology/     # 69 Method + candidate vs control
│   ├── metonymic/       # Kurisu's Method 3 (phrasal differential)
│   └── regression/      # Supplementary HLM + Gate×Channel (appendix only)
├── results/             # Analysis outputs
├── paper/               # Manuscript drafts and figures
└── docs/                # Design documents, session reports, coding instructions
```

## Key Findings

- **217 items** across **53 poetic/rhetorical traditions** and **15+ languages**
- **74.6%** of CPG items show statistically significant covert path pull (sign test p ≈ 0)
- **10/22 traditions** individually significant (p < 0.05) via the 69 Method
- **3/5 metonymic groups** significant via Kurisu's Method 3 (phrasal differential)
- Two detection methods correspond to the **metaphor/metonymy** distinction in cognitive linguistics

## The 69 Method

Named after the observation that 68, 69, 70 are topological neighbors, but only 69 carries a covert sexual meaning. Controls are the candidate's nearest semantic neighbors — words that are "next to" the candidate in meaning space but lack the covert path.

## CPG Three-Axis Model

| Axis | Description | Not a scalar |
|------|-------------|--------------|
| **Directionality** (R/E/S) | How meaning multiplexes onto a word | Typological |
| **Mediating Channel** (8 vectors) | Cognitive operation connecting surface and covert | Categorical |
| **Social Circulation** (Z) | How the community transmits and maintains the path | Qualitative |

## Target Journal

**Language Sciences** (Elsevier)

## License

Research data and code: MIT License
---

*Miosync, Inc. | 2026*
