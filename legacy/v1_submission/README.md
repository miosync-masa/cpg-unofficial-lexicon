# v1 submission (superseded)

This folder contains the analysis as submitted to Language Sciences (March 2026). It is kept for
reproducibility only and is superseded by the current version at the repository root. The same state is
available under the tag `v1-language-sciences-2026-03`.

Known issues, each fixed in the current version:

1. S-type values were read from the wrong YAML key (51 items set to 0).
2. Romanized hosts were compared with original-script controls (99 items).
3. Hosts contained English glosses.
4. Some controls were identical to the host or contained it.
5. Several supplementary statistics could not be traced to code.

Details and the effect of each fix: `docs/CHANGES_FROM_V1.md`.

The files are as submitted. The original READMEs are `README_v1.md` (repository root) and `README_v1_inner.md`. Paths in
these READMEs and the scripts refer to the v1 layout. The source data (`data/shogun/ashigaru*.yaml`,
`data/torami/torami_103_with_tradition.csv`, `cpg_coding_table_v3.xlsx`, `cpg_all_items_for_embedding.csv`)
now live in `data/source/` at the repository root.
