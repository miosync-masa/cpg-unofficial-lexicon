"""Summarise NNPM replications and the script/gloss confound.

Usage: python3 -I analyze_replication.py <nnpm_csv> [<nnpm_csv> ...] --out summary.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--out", required=True)
args = ap.parse_args()
r4 = lambda x: None if x is None or np.isnan(x) else round(float(x), 4)


def summarise(df, col):
    x = df[col].dropna()
    d = df.dropna(subset=[col])
    t, p = stats.ttest_1samp(x, 0, alternative="greater")
    mm = smf.mixedlm(f"{col} ~ 1", d, groups=d["tradition"]).fit(reml=True)
    z = mm.params["Intercept"] / mm.bse["Intercept"]
    trad = []
    for tr, g in d.groupby("tradition"):
        if len(g) >= 3:
            trad.append((tr, stats.ttest_1samp(g[col], 0, alternative="greater").pvalue, g[col].mean()))
    q = multipletests([p for _, p, _ in trad], method="fdr_bh")[1]
    return {"n": len(x), "mean": r4(x.mean()), "pct_positive": round(100 * (x > 0).mean(), 1),
            "t": r4(t), "p": r4(p), "mixed_intercept": r4(mm.params["Intercept"]), "mixed_z": r4(z),
            "mixed_p_one_sided": r4(stats.norm.sf(z)),
            "traditions_sig_p05": int(sum(p < .05 for _, p, _ in trad)),
            "traditions_sig_bh": int((q < .05).sum()), "traditions_tested": len(trad),
            "spearman_with_openai": r4(stats.spearmanr(d[col], d["pull_openai"]).statistic)}


out = {}
for f in args.files:
    df = pd.read_csv(f, encoding="utf-8-sig")
    res = {"conditions": {}}
    for col in [c for c in df.columns if c.startswith("pull_") and c != "pull_openai"]:
        res["conditions"][col] = summarise(df, col)
    # confound checks on the original OpenAI pull and on this model's condition A, k=10
    conf = {}
    for target in ["pull_openai", "pull_A_k10", "pull_B_k10"]:
        if target not in df:
            continue
        c = {}
        for flag in ["controls_latin", "host_has_gloss", "host_is_romanization"]:
            g1, g0 = df.loc[df[flag] == 1, target].dropna(), df.loc[df[flag] == 0, target].dropna()
            c[flag] = {"n_1": len(g1), "mean_1": r4(g1.mean()), "n_0": len(g0), "mean_0": r4(g0.mean()),
                       "welch_p": r4(stats.ttest_ind(g1, g0, equal_var=False).pvalue)}
        same = df[(df.controls_latin == 1) & (df.host_has_gloss == 0)][target].dropna()
        c["clean_subset_latin_controls_no_gloss"] = {
            "n": len(same), "mean": r4(same.mean()), "pct_positive": round(100 * (same > 0).mean(), 1),
            "p": r4(stats.ttest_1samp(same, 0, alternative="greater").pvalue)}
        conf[target] = c
    res["confound_checks"] = conf
    out[Path(f).stem] = res
json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
