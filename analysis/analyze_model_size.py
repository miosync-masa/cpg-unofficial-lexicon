"""Model size and pull: multilingual-e5 small, base, large (docs/analysis_plan_model_size.md).

Usage: python3 analysis/analyze_model_size.py

Inputs : embeddings/results/nnpm_e5_<size>.csv, embeddings/results/e5_meta.json (parameter counts),
         results/items_final.csv (outsider_inference), embeddings/results/nnpm_DE_<model>.csv (reference d of the
         three earlier models).
Outputs: results/model_size_e5.json, results/model_size_e5.md

Main sets: condition D = the 67 items of the condition C main set with 10 kept condition D controls; condition C =
the 94 items of the condition C main set. pull_std = pull / SD (ddof 1, per model, on the main set); d = mean / SD.
H-S1: pull_std ~ log10(params) + (1 | item) + (1 | tradition), stacked over the three sizes, REML, items nested in
traditions (groups = tradition, variance component for item). H-S2: the same with log10(params) * hard, hard = 1
for outsider_inference inferable_with_effort / not_inferable, 0 for inferable, UNRESOLVED left out. One-sided Wald
tests (coefficient > 0), 95% CI two-sided Wald. Fallback fixed in the plan: if a fit does not converge, the
tradition component is dropped ((1 | item) only) and this is recorded.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "embeddings/results"
SIZES = ["small", "base", "large"]
EARLIER = {"text_embedding_3_large": "OpenAI text-embedding-3-large", "BAAI_bge_m3": "BAAI/bge-m3",
           "gemini_embedding_001": "gemini-embedding-001"}
r4 = lambda x: round(float(x), 4)

meta = json.load(open(E / "e5_meta.json"))
params = {s: meta["models"][s]["params"] for s in SIZES}
oi = pd.read_csv(ROOT / "results/items_final.csv", encoding="utf-8").set_index("item_id").outsider_inference
P = {s: pd.read_csv(E / f"nnpm_e5_{s}.csv", encoding="utf-8-sig") for s in SIZES}
assert all(len(d) == 94 and list(d.item_id) == list(P["small"].item_id) for d in P.values())


def main_set(d, X):
    d = d.dropna(subset=[f"pull_{X}_k10"])
    assert len(d) == {"C": 94, "D": 67}[X], (X, len(d))
    return d


def ref_mixed(d, y):
    m = smf.mixedlm(f"{y} ~ 1", d, groups=d["tradition"]).fit(reml=True)
    b, se = m.params["Intercept"], m.bse["Intercept"]
    return {"intercept": r4(b), "ci95": [r4(b - 1.96 * se), r4(b + 1.96 * se)], "z": r4(b / se)}


def fit(formula, d, term):
    """Nested random effects; the fallback of the plan when the fit does not converge."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        m = smf.mixedlm(formula, d, groups=d["tradition"], re_formula="1",
                        vc_formula={"item": "0 + C(item_id)"}).fit(reml=True)
        bad = (not m.converged) or any(issubclass(x.category, ConvergenceWarning) for x in w)
    structure = "(1 | item) + (1 | tradition)"
    if bad:
        m = smf.mixedlm(formula, d, groups=d["item_id"]).fit(reml=True)
        structure = "(1 | item) only (fallback: the nested fit did not converge)"
    b, se = m.params[term], m.bse[term]
    z = b / se
    return {"term": term, "b": r4(b), "se": r4(se), "z": r4(z), "p_one_sided": r4(stats.norm.sf(z)),
            "ci95": [r4(b - 1.96 * se), r4(b + 1.96 * se)], "n_rows": int(len(d)),
            "n_items": int(d.item_id.nunique()), "n_traditions": int(d.tradition.nunique()),
            "random_effects": structure, "converged": bool(m.converged),
            "_b": float(b), "_se": float(se)}


out = {"plan": "docs/analysis_plan_model_size.md", "models": meta["models"], "versions": meta["versions"],
       "conditions": {}}
for X in ["D", "C"]:
    rows, stack = [], []
    for s in SIZES:
        d = main_set(P[s], X)
        y = d[f"pull_{X}_k10"]
        sd = float(y.std(ddof=1))
        rows.append({"size": s, "params": params[s], "n": int(len(d)), "mean_pull": r4(y.mean()), "sd": r4(sd),
                     "d": r4(y.mean() / sd), **ref_mixed(d, f"pull_{X}_k10")})
        stack.append(d.assign(size=s, log10_params=np.log10(params[s]), pull_std=y / sd)
                     [["item_id", "tradition", "size", "log10_params", "pull_std"]])
    S = pd.concat(stack, ignore_index=True)
    S["outsider_inference"] = S.item_id.map(oi)
    assert S.outsider_inference.notna().all()
    h1 = fit("pull_std ~ log10_params", S, "log10_params")
    S2 = S[S.outsider_inference != "UNRESOLVED"].assign(
        hard=lambda x: x.outsider_inference.isin(["inferable_with_effort", "not_inferable"]).astype(int))
    h2 = fit("pull_std ~ log10_params * hard", S2, "log10_params:hard")
    h2["n_hard_items"] = int(S2[S2.hard == 1].item_id.nunique())
    h2["n_inferable_items"] = int(S2[S2.hard == 0].item_id.nunique())
    ds = [r["d"] for r in rows]
    ref = []
    for tag, name in EARLIER.items():
        e = pd.read_csv(E / f"nnpm_DE_{tag}.csv", encoding="utf-8-sig")
        assert sorted(e.item_id) == sorted(P["small"].item_id)
        y = main_set(e, X)[f"pull_{X}_k10"]
        ref.append({"model": name, "n": int(len(y)), "mean_pull": r4(y.mean()), "sd": r4(y.std(ddof=1)),
                    "d": r4(y.mean() / y.std(ddof=1))})
    out["conditions"][X] = {"per_size": rows, "d_ordered_small_base_large": bool(ds[0] < ds[1] < ds[2]),
                            "H-S1": h1, "H-S2": h2, "reference_earlier_models": ref}
    sig = h1["p_one_sided"] < .05 and h1["b"] > 0
    out["conditions"][X]["interpretation_row"] = (
        "1: positive and significant slope, d in size order" if sig and ds[0] < ds[1] < ds[2] else
        "3: slope near 0 or negative" if h1["b"] <= 0 else
        "2: positive slope, not significant or d not in size order")

for X in out["conditions"]:  # private unrounded values are not stored
    for h in ("H-S1", "H-S2"):
        out["conditions"][X][h].pop("_b"), out["conditions"][X][h].pop("_se")
json.dump(out, open(ROOT / "results/model_size_e5.json", "w"), indent=1)

md = ["# Model size and pull: multilingual-e5 (generated by analysis/analyze_model_size.py)", "",
      "Plan: docs/analysis_plan_model_size.md. Prefix \"query: \" on every string; normalised embeddings, cosine.", "",
      "| size | Hugging Face | revision | parameters (counted) |", "|---|---|---|---|"]
md += [f"| {s} | {meta['models'][s]['name']} | {meta['models'][s]['revision']} | {params[s]:,} |" for s in SIZES]
md += ["", "Libraries: " + ", ".join(f"{k} {v}" for k, v in meta["versions"].items())]
for X, lab in [("D", "Condition D (main analysis)"), ("C", "Condition C (secondary analysis)")]:
    c = out["conditions"][X]
    md += ["", f"## {lab}", "", "| size | n | mean pull | SD | d | mixed-model intercept [95% CI] | z |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {r['size']} | {r['n']} | {r['mean_pull']:.4f} | {r['sd']:.4f} | {r['d']:.4f} | {r['intercept']:.4f} "
           f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}] | {r['z']:.4f} |" for r in c["per_size"]]
    md += ["", f"d ordered small < base < large: {c['d_ordered_small_base_large']}", ""]
    for h in ("H-S1", "H-S2"):
        v = c[h]
        md.append(f"- {h} ({v['term']}): b = {v['b']:.4f}, SE = {v['se']:.4f}, z = {v['z']:.4f}, one-sided p = "
                  f"{v['p_one_sided']:.4f}, 95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]; rows {v['n_rows']}, items "
                  f"{v['n_items']}, traditions {v['n_traditions']}; random effects {v['random_effects']}"
                  + (f"; hard items {v['n_hard_items']}, inferable items {v['n_inferable_items']}" if h == "H-S2" else ""))
    md += [f"- interpretation row (plan): {c['interpretation_row']}", "",
           "Reference (no test): d of the earlier models on the same main set", "", "| model | n | mean pull | SD | d |",
           "|---|---|---|---|---|"]
    md += [f"| {r['model']} | {r['n']} | {r['mean_pull']:.4f} | {r['sd']:.4f} | {r['d']:.4f} |"
           for r in c["reference_earlier_models"]]
(ROOT / "results/model_size_e5.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
