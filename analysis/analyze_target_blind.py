"""Conditions D and E: controls generated without the covert target (docs/analysis_plan_target_blind_controls.md).

Usage: python3 analysis/analyze_target_blind.py

Inputs : embeddings/results/nnpm_DE_<model>.csv, data/controls_D.json, data/controls_E_<model>.json,
         coding/results/first_run/consensus_codes.csv (cpg_valid)
Outputs: results/target_blind_controls.json, results/target_blind_controls.md

Hypotheses (fixed in the plan before the controls were generated):
  H-D1  condition D, main set: mean pull_D_k10 > 0. Mixed model pull ~ 1 + (1 | tradition) (raw labels), REML,
        one-sided z test of the intercept, Holm across the 3 models; 3 models with Holm p < .05 -> supported,
        1-2 -> partly supported, 0 -> not supported.
  H-D2  the same on the items whose first-run consensus cpg_valid = yes.
  H-E1, H-E2  the same tests in condition E (exploratory).
Diagnostic: mean sim(controls, target) in conditions C, D, E on the items that have a k = 10 pull in all compared
conditions.
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
ROOT = Path(__file__).resolve().parents[1]
MODELS = {"text_embedding_3_large": "OpenAI text-embedding-3-large", "BAAI_bge_m3": "BAAI/bge-m3",
          "gemini_embedding_001": "gemini-embedding-001"}
r4 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)  # noqa: E731


def mixed(d, col):
    d = d.dropna(subset=[col])
    m = smf.mixedlm(f"{col} ~ 1", d, groups=d["tradition"]).fit(reml=True)
    z = m.params["Intercept"] / m.bse["Intercept"]
    return {"n": int(len(d)), "n_groups": int(d.tradition.nunique()), "intercept": r4(m.params["Intercept"]),
            "se": r4(m.bse["Intercept"]), "z": r4(z), "p_one_sided": r4(stats.norm.sf(z)),
            "_p": float(stats.norm.sf(z))}


first = pd.read_csv(ROOT / "coding/results/first_run/consensus_codes.csv")[["item_id", "cpg_valid"]]
D = {t: pd.read_csv(ROOT / f"embeddings/results/nnpm_DE_{t}.csv", encoding="utf-8-sig").merge(first, on="item_id",
                                                                                                how="left")
     for t in MODELS}
assert all(len(d) == 94 for d in D.values())
R = {"plan": "docs/analysis_plan_target_blind_controls.md", "rows": [], "verdicts": {}, "diagnostic": [],
     "counts": {}}
for X in "DE":
    for h, sel in [("1", lambda d: d), ("2", lambda d: d[d.cpg_valid == "yes"])]:
        hyp = f"H-{X}{h}"
        res = {t: mixed(sel(d), f"pull_{X}_k10") for t, d in D.items()}
        adj = multipletests([res[t].pop("_p") for t in MODELS], method="holm")[1]
        n_pass = int(sum(a < .05 for a in adj))
        v = "supported" if n_pass == 3 else "partly supported" if n_pass else "not supported"
        R["verdicts"][hyp] = {"n_models_holm_p05": n_pass, "verdict": v}
        for t, a in zip(MODELS, adj):
            R["rows"].append({"condition": X, "hypothesis": hyp, "model": t, **res[t], "p_holm": r4(a),
                              "verdict": v})
# for comparison: condition C on the same items (H-C1, H-C2 as reported in t1_results.json)
t1 = json.load(open(ROOT / "results/t1_results.json"))
R["condition_C_reference"] = {h: {t: {k: t1["hypotheses"][h]["per_model"][t].get(k) for k in
                                      ["n", "n_traditions", "intercept", "z", "p_holm"]} for t in MODELS}
                              for h in ["H-C1", "H-C2"]}
for t, d in D.items():
    for lab, conds in [("C_D", "CD"), ("C_E", "CE"), ("C_D_E", "CDE")]:
        x = d.dropna(subset=[f"pull_{X}_k10" for X in conds])
        row = {"model": t, "items": lab, "n": int(len(x)), "sim_host_target": r4(x.sim_host_target.mean())}
        for X in conds:
            row[f"simctrl_{X}"] = r4(x[f"simctrl_{X}"].mean())
            row[f"pull_{X}"] = r4(x[f"pull_{X}_k10"].mean())
        for X in conds[1:]:
            diff = x[f"simctrl_{X}"] - x.simctrl_C
            row[f"simctrl_{X}_minus_C"] = r4(diff.mean())
            row[f"paired_t_{X}_minus_C"] = r4(stats.ttest_1samp(diff, 0).statistic) if len(x) > 2 else None
            row[f"paired_p_{X}_minus_C"] = r4(stats.ttest_1samp(diff, 0).pvalue) if len(x) > 2 else None
        R["diagnostic"].append(row)

cd = json.load(open(ROOT / "data/controls_D.json", encoding="utf-8"))
R["counts"]["D"] = {"items": len(cd), "with_10_kept": sum(len(v["kept"]) >= 10 for v in cd.values()),
                    "low_control": sorted(k for k, v in cd.items() if len(v["kept"]) < 10),
                    "no_usable_response": sorted(k for k, v in cd.items() if not v["generated"])}
for t in MODELS:
    ce = json.load(open(ROOT / f"data/controls_E_{t}.json", encoding="utf-8"))
    its = {k: v for k, v in ce.items() if not k.startswith("_")}
    R["counts"][f"E_{t}"] = {"items_in_mapped_languages": len(its),
                             "with_10_kept": sum(len(v["kept"]) >= 10 for v in its.values()),
                             "excluded_languages": ce["_excluded"],
                             "excluded_items": int(sum(ce["_excluded"].values())),
                             "items_per_language": pd.Series([v["wordfreq_lang"] for v in its.values()])
                             .value_counts().sort_index().to_dict()}
json.dump(R, open(ROOT / "results/target_blind_controls.json", "w"), ensure_ascii=False, indent=1)

f = lambda x: "–" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)  # noqa: E731
L = ["# Controls generated without the covert target — conditions D and E "
     "(generated by analysis/analyze_target_blind.py)", "",
     "Plan fixed before computing: `docs/analysis_plan_target_blind_controls.md`. D: v1 prompt without the "
     "EXCLUSION lines (gpt-5.2); E: nearest wordfreq words to the clean host in each embedding space (exploratory). "
     "Host and target vectors are those of condition C.", "",
     "## Hypotheses", "", "| condition | hypothesis | model | n | n_groups | intercept | z | p_holm | verdict |",
     "|---|---|---|---|---|---|---|---|---|"]
L += [f"| {r['condition']} | {r['hypothesis']} | {MODELS[r['model']]} | {r['n']} | {r['n_groups']} | "
      f"{f(r['intercept'])} | {f(r['z'])} | {f(r['p_holm'])} | {r['verdict']} |" for r in R["rows"]]
L += ["", "Condition C on its main set, for reference (results/t1_results.json):", "",
      "| hypothesis | model | n | intercept | z | p_holm |", "|---|---|---|---|---|---|"]
L += [f"| {h} | {MODELS[t]} | {v['n']} | {f(v['intercept'])} | {f(v['z'])} | {f(v['p_holm'])} |"
      for h, x in R["condition_C_reference"].items() for t, v in x.items()]
L += ["", "## Diagnostic: mean sim(controls, target)", "",
      "Items with a k = 10 pull in every listed condition. Paired t on the per-item difference from C.", "",
      "| model | items | n | sim(host, target) | C | D | E | D − C (t, p) | E − C (t, p) | pull C | pull D | pull E |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in R["diagnostic"]:
    g = lambda X: (f"{f(r[f'simctrl_{X}_minus_C'])} ({f(r[f'paired_t_{X}_minus_C'])}, {f(r[f'paired_p_{X}_minus_C'])})"  # noqa: E731
                   if f"simctrl_{X}_minus_C" in r else "")
    L.append(f"| {MODELS[r['model']]} | {r['items']} | {r['n']} | {f(r['sim_host_target'])} | {f(r['simctrl_C'])} | "
             f"{f(r.get('simctrl_D'))} | {f(r.get('simctrl_E'))} | {g('D')} | {g('E')} | {f(r['pull_C'])} | "
             f"{f(r.get('pull_D'))} | {f(r.get('pull_E'))} |")
c = R["counts"]
L += ["", "## Items", "", f"- D: {c['D']['items']} items, {c['D']['with_10_kept']} with 10 kept controls; "
      f"low_control: {', '.join(c['D']['low_control'])}. No usable response: {', '.join(c['D']['no_usable_response'])}."]
for t in MODELS:
    e = c[f"E_{t}"]
    L.append(f"- E ({MODELS[t]}): {e['items_in_mapped_languages']} items in mapped languages "
             f"({', '.join(f'{k} {v}' for k, v in e['items_per_language'].items())}), {e['with_10_kept']} with 10 "
             f"kept; {e['excluded_items']} items left out (language not mapped).")
L += ["", "Languages left out of E (items): " + "; ".join(f"{k} ({v})" for k, v in
                                                          c["E_text_embedding_3_large"]["excluded_languages"].items())]
(ROOT / "results/target_blind_controls.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
