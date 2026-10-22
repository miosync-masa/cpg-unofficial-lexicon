"""
CPG revision — recompute every reportable statistic from the repository data.

Usage:
    python3 -I analysis/reproduce_stats.py        (from the repository root; writes to results/)

Inputs (read-only):
    data/cpg_archaeology_results.csv               per-item pull of the v1 pipeline (213 items, 10 LLM-generated
                                                   controls each)
    data/source/cpg_all_items_for_embedding.csv    Phase-2 ("shogun") item coding (114 items)
    data/source/torami_103_with_tradition.csv      Phase-1 ("torami") item coding (103 items)
    data/source/ashigaru*.yaml                     Phase-2 YAML (ternary x_type coding)
    legacy/v1_submission/results/metonymic/metonymic_results.json         phrasal differential analysis (5 groups)
    legacy/v1_submission/results/metonymic/template_leakage_results.json

Every number reported in the revised manuscript should come from the JSON this script writes.
No API calls are made; stored per-item similarities are used as-is.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


warnings.filterwarnings("ignore")
RNG = np.random.default_rng(20261021)
N_BOOT = 10000

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
V1 = ROOT / "legacy/v1_submission/results"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

CHANNELS = ["Phonetic", "Morphological", "Chromatic", "Conceptual",
            "Structural", "Behavioural", "Metonymic", "Formal"]

# Explicit region scheme (documented in output so the manuscript can state it)
REGION_SHOGUN = {"AFR": "Africa", "SEA": "Southeast Asia", "NAH": "Mesoamerica",
                 "MAYA": "Mesoamerica", "QCH": "South America", "NEU": "Northern Europe",
                 "MET": "Middle East", "EEU": "Eastern Europe", "OCE": "Oceania",
                 "NET": "Digital / Internet"}
REGION_TORAMI = {"JA": "East Asia", "ZH": "East Asia", "SA": "South Asia",
                 "ME": "Middle East", "FR": "Western Europe", "EN": "English (global)"}


def r(x, k=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def norm_channels(s):
    s = str(s or "").replace("?", "").replace("Behavioral", "Behavioural")
    return [c.strip() for c in s.split("+") if c.strip()]


def boot_ci(x, fn=np.mean, n=N_BOOT):
    x = np.asarray(x)
    bs = fn(x[RNG.integers(0, len(x), (n, len(x)))], axis=1)
    return [r(np.percentile(bs, 2.5)), r(np.percentile(bs, 97.5))]


def describe_one_sample(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = {"n": n, "mean": r(x.mean()), "sd": r(x.std(ddof=1)) if n > 1 else None,
           "pct_positive": r(100 * (x > 0).mean(), 1)}
    if n >= 3:
        t, p = stats.ttest_1samp(x, 0, alternative="greater")
        out.update(t=r(t, 3), p_t_one_sided=r(p, 6),
                   cohen_d=r(x.mean() / x.std(ddof=1), 3),
                   ci95_t=[r(v) for v in stats.t.interval(0.95, n - 1, x.mean(), stats.sem(x))],
                   p_wilcoxon_one_sided=r(stats.wilcoxon(x, alternative="greater").pvalue, 6),
                   p_sign_one_sided=r(stats.binomtest(int((x > 0).sum()), n, 0.5,
                                                      alternative="greater").pvalue, 6))
    return out


# ---------------------------------------------------------------- load
res = pd.read_csv(DATA / "cpg_archaeology_results.csv", encoding="utf-8-sig")
sho = pd.read_csv(DATA / "source/cpg_all_items_for_embedding.csv", encoding="utf-8-sig")
tor = pd.read_csv(DATA / "source/torami_103_with_tradition.csv")
tor["id"] = tor["id"].map(lambda i: "T_%03d" % int(i))

meta = pd.concat([
    pd.DataFrame({"id": sho.id, "language": sho.language, "channel_raw": sho.vector,
                  "region": sho.id.str.split("_").str[0].map(REGION_SHOGUN)}),
    pd.DataFrame({"id": tor.id, "language": tor.language, "channel_raw": tor.y_vector,
                  "region": tor.region.map(REGION_TORAMI),
                  "tradition_type": tor.tradition_type}),
])
df = res.merge(meta, on="id", how="left")

# BUG FIX: the original archaeology script read only the `x_s` key of the Phase-2 YAML,
# but 52 items store the ternary coding as `x_type: [r, e, s]`, so their S value was set to 0.
import yaml


yx = {}
for f in sorted((DATA / "source").glob("ashigaru*.yaml")):
    for it in yaml.safe_load(open(f, encoding="utf-8"))["items"]:
        xt = it.get("x_type")
        if isinstance(xt, list) and len(xt) == 3:
            yx[it["id"]] = float(xt[2])
        elif it.get("x_s") is not None:
            yx[it["id"]] = float(it["x_s"])
df["x_s_original_pipeline"] = df["x_s"]
df["x_s"] = df.apply(lambda row: yx.get(row["id"], row["x_s"]), axis=1)
df["channels"] = df.channel_raw.map(norm_channels)
df["primary_channel"] = df.channels.map(lambda c: c[0] if c else "Unknown")
for c in CHANNELS:
    df["has_" + c] = df.channels.map(lambda cs, c=c: int(c in cs))
df["n_channels"] = df.channels.map(len)
df["multi_channel"] = (df.n_channels > 1).astype(int)

R = {"_note": "All values recomputed from repository data by reproduce_stats.py"}

# ---------------------------------------------------------------- inventory
all_ids = set(sho.id) | set(tor.id)
R["inventory"] = {
    "items_coded": len(all_ids),
    "items_phase1_torami": len(tor), "items_phase2_shogun": len(sho),
    "items_with_pull": len(df),
    "excluded_ids": sorted(all_ids - set(df.id)),
    "distinct_tradition_labels": int(df.tradition.nunique()),
    "tradition_labels_n_ge_3": int((df.tradition.value_counts() >= 3).sum()),
    "items_in_traditions_n_ge_3": int(df.tradition.map(df.tradition.value_counts()).ge(3).sum()),
    "distinct_language_labels": int(df.language.nunique()),
    "regions_scheme": sorted(df.region.dropna().unique().tolist()),
    "items_per_region": df.region.value_counts().to_dict(),
    "controls_per_item": sorted(df.n_controls.unique().tolist()),
    "controls_source": "LLM-generated nearest-neighbour words (results/69method/control_cache.json)",
    "control_embeddings_stored": False,
}

# ---------------------------------------------------------------- P1 item level
R["P1_item_level"] = describe_one_sample(df.pull_index)
R["P1_item_level"]["ci95_bootstrap"] = boot_ci(df.pull_index)

# ---------------------------------------------------------------- P1 cluster-aware
m0 = smf.mixedlm("pull_index ~ 1", df, groups=df["tradition"]).fit(reml=True)
var_t, var_e = float(m0.cov_re.iloc[0, 0]), float(m0.scale)
z = m0.params["Intercept"] / m0.bse["Intercept"]
R["P1_mixed_model"] = {
    "model": "pull ~ 1 + (1 | tradition), REML, all 213 items, 72 tradition labels",
    "intercept": r(m0.params["Intercept"]), "se": r(m0.bse["Intercept"]),
    "z": r(z, 3), "p_one_sided": r(stats.norm.sf(z), 8),
    "ci95": [r(v) for v in m0.conf_int().loc["Intercept"]],
    "var_tradition": r(var_t, 6), "var_residual": r(var_e, 6),
    "icc_tradition": r(var_t / (var_t + var_e), 4),
}
tmeans = df.groupby("tradition").pull_index.mean()
R["P1_tradition_means"] = {"unit": "one mean per tradition label (72)",
                           **describe_one_sample(tmeans.values)}
groups = df.groupby("tradition").pull_index.apply(np.array).tolist()
cb = [np.concatenate([groups[i] for i in RNG.integers(0, len(groups), len(groups))]).mean()
      for _ in range(N_BOOT)]
R["P1_cluster_bootstrap_ci95"] = [r(np.percentile(cb, 2.5)), r(np.percentile(cb, 97.5))]

# ---------------------------------------------------------------- per tradition
rows = []
for trad, g in df.groupby("tradition"):
    if len(g) < 3:
        continue
    d = describe_one_sample(g.pull_index)
    rows.append({"tradition": trad, "region": g.region.mode().iat[0], "source": g.source.iat[0],
                 "n": d["n"], "mean_pull": d["mean"], "pct_positive": d["pct_positive"],
                 "t": d["t"], "p_t": d["p_t_one_sided"], "p_wilcoxon": d["p_wilcoxon_one_sided"],
                 "mean_candidate_sim": r(g.candidate_sim.mean())})
trad = pd.DataFrame(rows).sort_values("mean_pull", ascending=False)
trad["q_bh"] = multipletests(trad.p_t, method="fdr_bh")[1].round(4)
trad["q_holm"] = multipletests(trad.p_t, method="holm")[1].round(4)
trad.to_csv(OUT / "per_tradition.csv", index=False)
sig_p = trad[trad.p_t < .05]
sig_q = trad[trad.q_bh < .05]
R["per_tradition"] = {
    "n_tested": len(trad),
    "n_sig_uncorrected_p05": len(sig_p), "regions_sig_uncorrected": sorted(sig_p.region.unique()),
    "n_sig_bh_q05": len(sig_q), "regions_sig_bh": sorted(sig_q.region.unique()),
    "n_sig_holm_05": int((trad.q_holm < .05).sum()),
    "table": trad.to_dict("records"),
}

# ---------------------------------------------------------------- P5 ICC variants
ov = df.pull_index.var()
pseudo = max(0, ov - df.groupby("tradition").pull_index.var().mean()) / ov
d3 = df[df.tradition.map(df.tradition.value_counts()) >= 3]
m3 = smf.mixedlm("pull_index ~ 1", d3, groups=d3["tradition"]).fit(reml=True)
v3t, v3e = float(m3.cov_re.iloc[0, 0]), float(m3.scale)
R["P5_icc"] = {
    "pseudo_icc_script_formula": r(pseudo, 4),
    "pseudo_icc_note": "script formula (total var - mean within var)/total var; nonstandard",
    "icc_mixed_all_213": R["P1_mixed_model"]["icc_tradition"],
    "icc_mixed_traditions_n_ge_3": r(v3t / (v3t + v3e), 4),
    "n_items_n_ge_3": len(d3),
    "lrt_tradition_effect_p": None,
}
# LRT for tradition random effect (ML fits): mixed vs plain OLS
ml_mixed = smf.mixedlm("pull_index ~ 1", df, groups=df["tradition"]).fit(reml=False)
ml_ols = smf.ols("pull_index ~ 1", df).fit()
lr = 2 * (ml_mixed.llf - ml_ols.llf)
R["P5_icc"]["lrt_stat"] = r(lr, 3)
R["P5_icc"]["lrt_tradition_effect_p"] = r(0.5 * stats.chi2.sf(lr, 1), 8)  # boundary-corrected

# ---------------------------------------------------------------- P2 / P3 S-type
def grp(mask):
    x = df.loc[mask, "pull_index"]
    return {"n": int(mask.sum()), "mean_pull": r(x.mean()) if len(x) else None,
            "mean_candidate_sim": r(df.loc[mask, "candidate_sim"].mean()) if len(x) else None}

re_mask = df.x_s == 0
mix_mask = (df.x_s > 0) & (df.x_s < 0.5)
s_mask = df.x_s >= 0.5
R["P2_S_type"] = {
    "coding_fix": {"items_with_corrected_x_s": int((df.x_s != df.x_s_original_pipeline).sum()),
                   "note": "x_s read from YAML x_type[2]; original pipeline had set these to 0"},
    "R_E_xs_eq_0": grp(re_mask), "mixed_0_lt_xs_lt_0.5": grp(mix_mask),
    "S_dominant_xs_ge_0.5": grp(s_mask),
    "S_dominant_one_sample": describe_one_sample(df.loc[s_mask, "pull_index"]),
    "x_s_distribution": df.x_s.round(2).value_counts().sort_index().to_dict(),
}
if s_mask.sum() >= 2:
    w = stats.ttest_ind(df.loc[s_mask, "pull_index"], df.loc[re_mask, "pull_index"], equal_var=False)
    R["P2_S_type"]["welch_S_vs_RE"] = {"t": r(w.statistic, 3), "p_two_sided": r(w.pvalue, 4)}
mx = smf.mixedlm("pull_index ~ x_s", df, groups=df["tradition"]).fit(reml=True)
R["P2_S_type"]["mixed_slope_x_s"] = {"b": r(mx.params["x_s"]), "se": r(mx.bse["x_s"]),
                                     "p_two_sided": r(mx.pvalues["x_s"], 4)}
if "tradition_type" in df:
    tt = df.dropna(subset=["tradition_type"])
    R["P3_tradition_type_phase1"] = {
        k: {"n": len(g), "mean_pull": r(g.pull_index.mean()), "var_pull": r(g.pull_index.var(), 6)}
        for k, g in tt.groupby("tradition_type")}
    a, b = [g.pull_index.values for _, g in tt.groupby("tradition_type")]
    R["P3_tradition_type_phase1"]["levene_p"] = r(stats.levene(a, b).pvalue, 4)
for name in ["Nahuatl Difrasismo", "Old Norse kenning (skaldic poetry)"]:
    g = df[df.tradition == name]
    R["P2_S_type"]["tradition_" + name] = {**describe_one_sample(g.pull_index),
                                           "x_s_values": g.x_s.tolist()}

# ---------------------------------------------------------------- P4 channels
ch_rows = []
for c in CHANNELS:
    g = df[df["has_" + c] == 1].pull_index
    d = describe_one_sample(g) if len(g) else {"n": 0}
    ch_rows.append({"channel": c, "n_any": d["n"], "mean_pull": d.get("mean"),
                    "p_t": d.get("p_t_one_sided"),
                    "n_primary": int((df.primary_channel == c).sum()),
                    "n_single_channel": int(((df.primary_channel == c) & (df.n_channels == 1)).sum())})
R["P4_channels_descriptive"] = ch_rows
pure_phon = df[(df.has_Phonetic == 1) & (df.n_channels == 1)].pull_index
R["P4_pure_phonetic"] = describe_one_sample(pure_phon)
formula = "pull_index ~ has_Phonetic + has_Formal + has_Metonymic + x_s"
mc = smf.mixedlm(formula, df, groups=df["tradition"]).fit(reml=True)
R["P4_mixed_model"] = {
    "formula": formula + " + (1|tradition)",
    "coef": {k: {"b": r(mc.params[k]), "se": r(mc.bse[k]), "p": r(mc.pvalues[k], 4)}
             for k in mc.params.index if k != "Group Var"},
    "icc_residual": r(float(mc.cov_re.iloc[0, 0]) / (float(mc.cov_re.iloc[0, 0]) + mc.scale), 4),
}
ols_c = smf.ols(formula, df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.tradition)[0]})
R["P4_ols_cluster_robust"] = {k: {"b": r(ols_c.params[k]), "se": r(ols_c.bse[k]),
                                  "p": r(ols_c.pvalues[k], 4)} for k in ols_c.params.index}

# ---------------------------------------------------------------- metonymic
met = json.load(open(V1 / "metonymic/metonymic_results.json"))
mrows = []
for gname in met:
    dp = gname["differential_pull"]
    mrows.append({"group": gname["group_name"], "n_candidates": gname["n_candidates"],
                  "n_controls": gname["n_controls"], "cand_mean": r(dp["candidate_mean"]),
                  "ctrl_mean": r(dp["control_mean"]), "U": dp["mann_whitney_U"],
                  "p": r(dp["p_value"], 4), "hedges_g": r(dp["hedges_g"], 3)})
mdf = pd.DataFrame(mrows)
mdf["q_bh"] = multipletests(mdf.p, method="fdr_bh")[1].round(4)
R["metonymic_phrasal_differential"] = {
    "n_sig_uncorrected": int((mdf.p < .05).sum()), "n_sig_bh": int((mdf.q_bh < .05).sum()),
    "table": mdf.to_dict("records"),
    "available_metrics_per_group": list(met[0].keys()),
}
tl = json.load(open(V1 / "metonymic/template_leakage_results.json"))
R["metonymic_template_leakage"] = {k: {kk: (r(vv, 4) if isinstance(vv, float) else vv)
                                       for kk, vv in v.items()} for k, v in tl.items()
                                   if isinstance(v, dict)}

# ---------------------------------------------------------------- similarity gradient
R["similarity_gradient_by_tradition_n_ge_3"] = (
    trad[["tradition", "region", "n", "mean_candidate_sim"]]
    .sort_values("mean_candidate_sim", ascending=False).to_dict("records"))

# ---------------------------------------------------------------- not reproducible from repo
R["not_in_repository"] = [
    "Context-gating experiment (botan/momiji/sakura + nabe): no data or script",
    "Control-condition comparison v1/v2 (GPT-4o-mini, domain-matched): only one control cache",
    "Control word embeddings: not stored, so alternative-model robustness needs re-embedding",
]

df.to_csv(OUT / "items_with_metadata.csv", index=False, encoding="utf-8-sig")
json.dump(R, open(OUT / "verified_stats.json", "w"), ensure_ascii=False, indent=2, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str)[:20000])
