"""T1 analysis: NNPM condition C (clean host, cleaned controls) with three embedding models.

Usage: python3 analysis/analyze_t1.py

Inputs : embeddings/results/nnpm_C_<model>.csv (condition C), embeddings/results/nnpm_<model>.csv (condition B, earlier run),
         embeddings/controls/controls_summary.csv (low_control), data/exclusions.csv, coding/results/first_run/consensus_codes.csv (first run),
         coding/results/recording/recording_consensus.csv
Outputs: results/t1_results.json, results/t1_results.md. No random numbers are used (seed fixed anyway).

Hypotheses, fixed by the instruction sheet (v1.0, 2026-10-22) before condition C was computed:
  H-C1  T3 exclusions applied, low_control items removed: mean pull_C_k10 > 0. Per model, mixed model
        pull ~ 1 + (1 | tradition), one-sided test of the intercept (z, normal sf); Holm across the 3 models.
        3 models with Holm p < .05 -> supported, 1-2 -> partly supported, 0 -> not supported.
  H-C2  the same test on the items whose first-run consensus cpg_valid = yes.
k rule: an item with fewer than k kept controls has no pull for that k, so pull_C_k10 exists only for items with
10 kept controls. pull_C_kall (all kept controls, up to 10) is reported as an extra sensitivity analysis.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
np.random.seed(20261022)
ROOT = Path(__file__).resolve().parents[1]
MODELS = {"text_embedding_3_large": "OpenAI text-embedding-3-large",
          "BAAI_bge_m3": "BAAI/bge-m3",
          "gemini_embedding_001": "gemini-embedding-001"}
r4 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def describe(x):
    x = pd.Series(x, dtype=float).dropna()
    out = {"n": int(len(x)), "mean": r4(x.mean()) if len(x) else None,
           "sd": r4(x.std(ddof=1)) if len(x) > 1 else None,
           "pct_positive": round(100 * float((x > 0).mean()), 1) if len(x) else None}
    if len(x) >= 3:
        out["t"] = r4(stats.ttest_1samp(x, 0).statistic)
        out["p_t_one_sided"] = r4(stats.ttest_1samp(x, 0, alternative="greater").pvalue)
    return out


def mixed(d, col):
    d = d.dropna(subset=[col])
    try:
        m = smf.mixedlm(f"{col} ~ 1", d, groups=d["tradition"]).fit(reml=True)
    except Exception as e:  # singular fits etc. are reported, not hidden
        return {"n": int(len(d)), "error": str(e)}
    z = m.params["Intercept"] / m.bse["Intercept"]
    return {"n": int(len(d)), "n_traditions": int(d.tradition.nunique()), "intercept": r4(m.params["Intercept"]),
            "se": r4(m.bse["Intercept"]), "z": r4(z), "p_one_sided": r4(stats.norm.sf(z)), "_p_raw": float(stats.norm.sf(z))}


def full(d, col):
    m = mixed(d, col)
    m.pop("_p_raw", None)
    return {**describe(d[col]), "mixed": m}


def verdict(n_pass):
    return "supported" if n_pass == 3 else "partly supported" if n_pass >= 1 else "not supported"


def holm_block(subsets):
    """subsets: {model: DataFrame}; mixed test of pull_C_k10 in each, Holm across models."""
    res = {m: mixed(d, "pull_C_k10") for m, d in subsets.items()}
    adj = multipletests([res[m].pop("_p_raw", 1.0) for m in subsets], method="holm")[1]
    for m, a in zip(subsets, adj):
        res[m]["p_holm"] = r4(a)
    n_pass = int(sum(a < .05 for a in adj))
    return {"per_model": res, "n_models_holm_p05": n_pass, "verdict": verdict(n_pass)}


# ------------------------------------------------------------ data
summ = pd.read_csv(ROOT / "embeddings/controls/controls_summary.csv")
excl = set(pd.read_csv(ROOT / "data/exclusions.csv").item_id)
first = pd.read_csv(ROOT / "coding/results/first_run/consensus_codes.csv")[["item_id", "cpg_valid", "direction",
                                                                      "primary_channel"]]
rec = pd.read_csv(ROOT / "coding/results/recording/recording_consensus.csv")[["item_id", "recording_status"]]
n_blind = len(pd.read_csv(ROOT / "data/blind_items.csv"))
D = {}
for tag in MODELS:
    c = pd.read_csv(ROOT / f"embeddings/results/nnpm_C_{tag}.csv", encoding="utf-8-sig")
    b = pd.read_csv(ROOT / f"embeddings/results/nnpm_{tag}.csv", encoding="utf-8-sig")[["id", "pull_B_k10"]]
    d = (c.merge(b.rename(columns={"id": "item_id"}), on="item_id", how="left")
         .merge(summ[["item_id", "n_kept", "low_control"]], on="item_id", how="left")
         .merge(first, on="item_id", how="left").merge(rec, on="item_id", how="left"))
    assert len(d) == len(c) == 213 and (d.n_kept == d.n_controls_kept).all()
    d["excluded_T3"] = d.item_id.isin(excl).astype(int)
    D[tag] = d
base = D["text_embedding_3_large"]
R = {"models": MODELS, "hypotheses": {}, "counts": {}}

# ------------------------------------------------------------ counts
after_t3 = base[base.excluded_T3 == 0]
main = after_t3[after_t3.low_control == 0]
R["counts"] = {
    "blind_items": n_blind, "items_with_controls": int(len(base)),
    "no_controls_stored": n_blind - int(len(base)),
    "excluded_T3": int(base.excluded_T3.sum()), "after_T3": int(len(after_t3)),
    "low_control_after_T3": int(after_t3.low_control.sum()), "after_low_control": int(len(main)),
    "n_kept_distribution_after_T3": {int(k): int(v) for k, v in after_t3.n_kept.value_counts().sort_index().items()},
    "eligible_by_k_after_low_control": {f"k{k}": int((main.n_kept >= k).sum()) for k in [3, 5, 10]},
    "analysed_by_model": {}}
for tag, d in D.items():
    m = d[(d.excluded_T3 == 0) & (d.low_control == 0)]
    R["counts"]["analysed_by_model"][tag] = {f"pull_C_k{k}": int(m[f"pull_C_k{k}"].notna().sum()) for k in [3, 5, 10]}
    R["counts"]["analysed_by_model"][tag]["pull_C_kall"] = int(m["pull_C_kall"].notna().sum())
    R["counts"]["analysed_by_model"][tag]["missing_because_embedding_failed"] = int(
        (m.n_kept >= 10).sum() - m["pull_C_k10"].notna().sum())
rules = {"R1": "identical to the host", "R2": "contains the host or is contained in it",
         "R3": "different script from the host", "R4": "duplicate"}
s_t3 = summ[~summ.item_id.isin(excl)]
R["counts"]["control_cleaning"] = {
    lab: {"items": int(len(x)), "controls": int(x.n_original.sum()), "kept": int(x.n_kept.sum()),
          **{r: int(x[f"n_dropped_{r}"].sum()) for r in rules}}
    for lab, x in [("all_items_with_controls", summ), ("after_T3", s_t3)]}
c = R["counts"]
assert all(v["controls"] == v["kept"] + sum(v[r] for r in rules) for v in c["control_cleaning"].values())
assert c["after_T3"] == c["items_with_controls"] - c["excluded_T3"]
assert c["after_low_control"] == c["after_T3"] - c["low_control_after_T3"]

# ------------------------------------------------------------ hypotheses
sub = lambda d: d[(d.excluded_T3 == 0) & (d.low_control == 0)]
R["hypotheses"]["H-C1"] = holm_block({t: sub(d) for t, d in D.items()})
R["hypotheses"]["H-C2"] = holm_block({t: sub(d)[sub(d).cpg_valid == "yes"] for t, d in D.items()})

# ------------------------------------------------------------ analyses
A = {}
for tag, d in D.items():
    m = sub(d)
    both = m.dropna(subset=["pull_B_k10", "pull_C_k10"])
    a = {"B_vs_C": {"B_k10_all_after_T3": full(d[d.excluded_T3 == 0], "pull_B_k10"),
                    "C_k10_main": full(m, "pull_C_k10"),
                    "same_items": {"n": int(len(both)), "B_k10": full(both, "pull_B_k10"),
                                   "C_k10": full(both, "pull_C_k10"),
                                   "paired_t_C_minus_B": r4(stats.ttest_rel(both.pull_C_k10, both.pull_B_k10).statistic)
                                   if len(both) > 2 else None,
                                   "paired_p_two_sided": r4(stats.ttest_rel(both.pull_C_k10, both.pull_B_k10).pvalue)
                                   if len(both) > 2 else None}}}
    for g in ["recording_status", "direction", "primary_channel"]:
        a[f"by_{g}"] = {str(k): describe(v.pull_C_k10) for k, v in m.groupby(g)}
    trad = []
    for tr, g in m.dropna(subset=["pull_C_k10"]).groupby("tradition"):
        if len(g) >= 3:
            trad.append({"tradition": tr, "n": int(len(g)), "mean": r4(g.pull_C_k10.mean()),
                         "p_one_sided": r4(stats.ttest_1samp(g.pull_C_k10, 0, alternative="greater").pvalue)})
    if trad:
        q = multipletests([t["p_one_sided"] for t in trad], method="fdr_bh")[1]
        for t, qq in zip(trad, q):
            t["q_bh"] = r4(qq)
    a["by_tradition_n3"] = {"tested": len(trad), "p05": int(sum(t["p_one_sided"] < .05 for t in trad)),
                            "bh_q05": int(sum(t["q_bh"] < .05 for t in trad)), "rows": trad}
    with_low = d[d.excluded_T3 == 0]
    a["sensitivity"] = {"k3_main": full(m, "pull_C_k3"), "k5_main": full(m, "pull_C_k5"),
                        "k3_with_low_control": full(with_low, "pull_C_k3"),
                        "k10_with_low_control": full(with_low, "pull_C_k10"),
                        "kall_main": full(m, "pull_C_kall"), "kall_with_low_control": full(with_low, "pull_C_kall")}
    A[tag] = a
R["analyses"] = A
sp = {}
tags = list(MODELS)
for i in range(3):
    for j in range(i + 1, 3):
        x = sub(D[tags[i]])[["item_id", "pull_C_k10"]].merge(sub(D[tags[j]])[["item_id", "pull_C_k10"]],
                                                            on="item_id").dropna()
        sp[f"{tags[i]} ~ {tags[j]}"] = {"n": int(len(x)),
                                         "rho": r4(stats.spearmanr(x.iloc[:, 1], x.iloc[:, 2]).statistic)}
R["spearman_between_models_C_k10"] = sp
json.dump(R, open(ROOT / "results/t1_results.json", "w"), ensure_ascii=False, indent=1)

# ------------------------------------------------------------ markdown
f = lambda x: "–" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)
pf = lambda x: "–" if x is None else f"{x:.1f}"
L = ["# T1 results — NNPM condition C (generated by analysis/analyze_t1.py)", "",
     "## Controls dropped per cleaning rule (`embeddings/scripts/build_clean.py`; first rule that fires)", "",
     "| set | items | controls | " + " | ".join(f"{r} {v}" for r, v in rules.items()) + " | kept |",
     "|---|---|---|---|---|---|---|---|"]
L += [f"| {lab} | {v['items']} | {v['controls']} | " + " | ".join(str(v[r]) for r in rules) + f" | {v['kept']} |"
      for lab, v in c["control_cleaning"].items()]
L += ["", "Primary analysis: k = 10 (fixed in advance). k = 3, k = 5 and kall (all kept controls, up to 10) are "
      "sensitivity analyses (author decision 3, 2026-10-22).", "",
     "## Item counts", "",
     f"- blind items {c['blind_items']} → with stored controls {c['items_with_controls']} "
     f"({c['no_controls_stored']} have none) → after T3 exclusion ({c['excluded_T3']}) {c['after_T3']} "
     f"→ after removing low_control ({c['low_control_after_T3']}) {c['after_low_control']}.",
     f"- Of these {c['after_low_control']}, items with at least k kept controls: "
     + ", ".join(f"{k} = {v}" for k, v in c["eligible_by_k_after_low_control"].items()) + ".",
     "- Analysed per model (pull present): "
     + "; ".join(f"{MODELS[t]}: " + ", ".join(f"{k} {v}" for k, v in x.items()) for t, x in c["analysed_by_model"].items()),
     "", "## Hypotheses (fixed in advance)", "",
     "| | model | n | intercept | z | p (one-sided) | p Holm |", "|---|---|---|---|---|---|---|"]
for h in ["H-C1", "H-C2"]:
    for t, r in R["hypotheses"][h]["per_model"].items():
        L.append(f"| {h} | {MODELS[t]} | {r['n']} | {f(r.get('intercept'))} | {f(r.get('z'))} | "
                 f"{f(r.get('p_one_sided'))} | {f(r.get('p_holm'))} |")
L += [""] + [f"- **{h}: {R['hypotheses'][h]['verdict']}** ({R['hypotheses'][h]['n_models_holm_p05']} of 3 models "
             f"with Holm p < .05)" for h in ["H-C1", "H-C2"]]
L += ["", "## Condition B vs C (k = 10)", "",
      "| model | set | n | mean | % > 0 | t | mixed z |", "|---|---|---|---|---|---|---|"]
for t, a in A.items():
    bc = a["B_vs_C"]
    for lab, r in [("B, all after T3", bc["B_k10_all_after_T3"]), ("C, main", bc["C_k10_main"]),
                   ("B, same items", bc["same_items"]["B_k10"]), ("C, same items", bc["same_items"]["C_k10"])]:
        L.append(f"| {MODELS[t]} | {lab} | {r['n']} | {f(r['mean'])} | {pf(r['pct_positive'])} | {f(r.get('t'))} | "
                 f"{f(r['mixed'].get('z'))} |")
L += ["", "## Descriptive breakdowns of pull_C_k10 (main set)", ""]
for g in ["recording_status", "direction", "primary_channel"]:
    L += [f"### {g}", "", "| level | " + " | ".join(MODELS.values()) + " |", "|---|---|---|---|"]
    levels = sorted({k for a in A.values() for k in a[f"by_{g}"]})
    for lv in levels:
        cells = []
        for t in MODELS:
            r = A[t][f"by_{g}"].get(lv)
            cells.append("–" if not r else f"n={r['n']}, mean={f(r['mean'])}, {pf(r['pct_positive'])}% > 0")
        L.append(f"| {lv} | " + " | ".join(cells) + " |")
    L.append("")
L += ["## Traditions (n ≥ 3), one-sided t, BH", "", "| model | tested | p < .05 | BH q < .05 |", "|---|---|---|---|"]
for t, a in A.items():
    b = a["by_tradition_n3"]
    L.append(f"| {MODELS[t]} | {b['tested']} | {b['p05']} | {b['bh_q05']} |")
L += ["", "## Sensitivity", "", "| model | analysis | n | mean | % > 0 | mixed z | mixed p |", "|---|---|---|---|---|---|---|"]
for t, a in A.items():
    for lab, r in a["sensitivity"].items():
        L.append(f"| {MODELS[t]} | {lab} | {r['n']} | {f(r['mean'])} | {pf(r['pct_positive'])} | "
                 f"{f(r['mixed'].get('z'))} | {f(r['mixed'].get('p_one_sided'))} |")
L += ["", "k10_with_low_control equals the main k = 10 analysis by construction (a low_control item has < 5 kept "
      "controls, so no k = 10 pull). kall uses all kept controls (up to 10) and is not in the spec.", "",
      "## Spearman between models (pull_C_k10, items in both)", ""]
L += [f"- {k}: n = {v['n']}, rho = {f(v['rho'])}" for k, v in sp.items()]
(ROOT / "results/t1_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
