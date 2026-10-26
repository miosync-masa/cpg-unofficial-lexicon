"""Taboo domain and dictionary recording (exploratory; docs/analysis_plan_taboo_domain.md).

Usage: python3 analysis/analyze_taboo.py

Inputs : coding/results/taboo/taboo_{gpt,gemini,claude}.csv (+ .jsonl logs), results/items_final.csv (211 items,
         consensus recording_status, raw tradition), dictionary_check/daijisen_judgments.csv.
Outputs: coding/results/taboo/taboo_consensus.csv, results/taboo_results.json, results/taboo_results.md

Consensus = majority (>= 2 of 3), else "unresolved". taboo = consensus taboo_domain != "none". general = recording
status general_dictionary. Difference of shares (taboo minus no taboo) with Newcombe's hybrid Wilson 95% CI;
Fisher's exact test, one-sided (taboo lower) and two-sided. Auxiliary: general ~ taboo + (1 | tradition) with
BinomialBayesMixedGLM.fit_vb (statsmodels has no ML fit for it); posterior mean, SD, mean +- 1.96 SD.
Fleiss kappa on all 217 items. Refusals: requests that ended with an error, by coder; per consensus label, the items
that were in at least one refused request and the items resent one by one.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from statsmodels.stats.proportion import confint_proportions_2indep
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "coding/results/taboo"
CODERS = ["gpt", "gemini", "claude"]
DOMAINS = ["sex", "excretion", "death_illness", "insult", "drugs_crime", "none"]
REC = ["general_dictionary", "cultural_reference", "unrecorded"]
r4 = lambda x: round(float(x), 4)

codes = {}
for c in CODERS:
    d = pd.read_csv(T / f"taboo_{c}.csv", dtype=str)
    d = d[d.status == "ok"].drop_duplicates("item_id", keep="last").set_index("item_id")
    codes[c] = d.taboo_domain
blind = pd.read_csv(ROOT / "data/blind_items.csv", dtype=str).item_id
assert all(set(codes[c].index) == set(blind) and len(codes[c]) == 217 for c in CODERS), "coding incomplete"
ids = list(blind)
lab = pd.DataFrame({c: codes[c].loc[ids] for c in CODERS}, index=ids)


def majority(r):
    v = r.value_counts()
    return v.index[0] if v.iloc[0] >= 2 else "unresolved"


lab["taboo_domain"] = lab[CODERS].apply(majority, axis=1)
lab["unanimous"] = lab[CODERS].nunique(axis=1) == 1
lab.rename(columns={c: f"taboo_domain_{c}" for c in CODERS}).rename_axis("item_id").to_csv(
    T / "taboo_consensus.csv", encoding="utf-8")


def fleiss(cols):
    table, _ = aggregate_raters(np.array(cols).T)
    return float(fleiss_kappa(table, method="fleiss"))


kappa = {"taboo_domain": r4(fleiss([lab[c].values for c in CODERS])),
         "taboo_binary": r4(fleiss([(lab[c] != "none").astype(int).values for c in CODERS])),
         "n_items": len(lab), "pct_unanimous_domain": r4(100 * lab.unanimous.mean())}

# ---- refusals (every request in the logs)
ref = {}
item_ref = {}
for c in CODERS:
    log = [json.loads(l) for l in open(T / f"taboo_{c}.csv.jsonl", encoding="utf-8")]
    err = [r for r in log if r["error"]]
    is_ref = lambda r: "refusal" in (r["error"] or "")
    refused_items = sorted({i for r in log if is_ref(r) for i in r["item_ids"]})
    single = sorted({r["item_ids"][0] for r in log if r.get("single") and len(r["item_ids"]) == 1})
    ref[c] = {"requests": len(log), "requests_with_error": len(err), "requests_refused": sum(map(is_ref, log)),
              "batches_refused": len({tuple(r["item_ids"]) for r in log if is_ref(r) and len(r["item_ids"]) > 1}),
              "items_in_refused_request": len(refused_items), "items_resent_one_by_one": len(single),
              "items_refused_when_sent_alone": sorted({r["item_ids"][0] for r in log if is_ref(r)
                                                       and len(r["item_ids"]) == 1}),
              "other_errors": sorted({r["error"][:120] for r in err if not is_ref(r)})}
    item_ref[c] = (set(refused_items), set(single))
ref_by_label = []
for dmn in DOMAINS + ["unresolved"]:
    s = set(lab.index[lab.taboo_domain == dmn])
    row = {"taboo_domain": dmn, "n_items_217": len(s)}
    for c in CODERS:
        row[f"in_refused_request_{c}"] = len(s & item_ref[c][0])
        row[f"resent_one_by_one_{c}"] = len(s & item_ref[c][1])
    row["in_refused_request_any"] = len(s & set().union(*(item_ref[c][0] for c in CODERS)))
    ref_by_label.append(row)

# ---- analysis set
fin = pd.read_csv(ROOT / "results/items_final.csv", encoding="utf-8").set_index("item_id")
assert len(fin) == 211
A = fin[["tradition", "recording_status"]].join(lab[["taboo_domain"]])
n_rec_unres = int((A.recording_status == "UNRESOLVED").sum())
n_tab_unres = int((A.taboo_domain == "unresolved").sum())
A = A[(A.recording_status != "UNRESOLVED") & (A.taboo_domain != "unresolved")].copy()
A["taboo"] = (A.taboo_domain != "none").astype(int)
A["general"] = (A.recording_status == "general_dictionary").astype(int)
cross = pd.crosstab(A.taboo.map({1: "taboo", 0: "no taboo"}), A.recording_status).reindex(
    index=["taboo", "no taboo"], columns=REC, fill_value=0)
ky, ny = int(A.general[A.taboo == 1].sum()), int((A.taboo == 1).sum())
kn, nn = int(A.general[A.taboo == 0].sum()), int((A.taboo == 0).sum())
lo, hi = confint_proportions_2indep(ky, ny, kn, nn, method="newcomb", compare="diff")
tab = [[ky, ny - ky], [kn, nn - kn]]
p_less = stats.fisher_exact(tab, alternative="less")[1]
odds, p_two = stats.fisher_exact(tab, alternative="two-sided")
diff = ky / ny - kn / nn
main = {"n": int(len(A)), "n_left_out_recording_unresolved": n_rec_unres, "n_left_out_taboo_unresolved": n_tab_unres,
        "n_taboo": ny, "n_no_taboo": nn, "general_taboo": ky, "general_no_taboo": kn,
        "pct_general_taboo": r4(100 * ky / ny), "pct_general_no_taboo": r4(100 * kn / nn),
        "diff_pct_points": r4(100 * diff), "ci95_pct_points_newcombe": [r4(100 * lo), r4(100 * hi)],
        "fisher_p_one_sided_less": r4(p_less), "fisher_p_two_sided": r4(p_two), "odds_ratio": r4(odds)}
main["reading"] = ("prediction holds" if diff < 0 and p_less < .05 else
                   "predicted direction, not shown" if diff < 0 else "prediction does not hold")

# ---- auxiliary logistic mixed model (variational Bayes)
g = BinomialBayesMixedGLM.from_formula("general ~ taboo", {"tradition": "0 + C(tradition)"}, A.reset_index())
gf = g.fit_vb()
j = list(gf.model.exog_names).index("taboo")
gm, gs = float(gf.fe_mean[j]), float(gf.fe_sd[j])
conv = getattr(gf, "optim_retvals", None)
glmm = {"method": "BinomialBayesMixedGLM.fit_vb (variational Bayes, default priors)", "n": int(len(A)),
        "n_traditions": int(A.tradition.nunique()), "taboo_post_mean": r4(gm), "taboo_post_sd": r4(gs),
        "interval_mean_pm_1.96sd": [r4(gm - 1.96 * gs), r4(gm + 1.96 * gs)],
        "converged": bool(conv["success"]) if isinstance(conv, dict) and "success" in conv else None,
        "optimizer_message": str(conv.get("message")) if isinstance(conv, dict) else "not reported by statsmodels"}

per_domain = [{"taboo_domain": d, "n": int((A.taboo_domain == d).sum()),
               "n_general": int(A.general[A.taboo_domain == d].sum()),
               "pct_general": r4(100 * A.general[A.taboo_domain == d].mean()) if (A.taboo_domain == d).any() else None}
              for d in DOMAINS]

# ---- Japanese items: Daijisen
dj = pd.read_csv(ROOT / "dictionary_check/daijisen_judgments.csv", encoding="utf-8-sig").set_index("item_id")
J = dj[["判定"]].join(lab[["taboo_domain"]])
J = J[J.taboo_domain != "unresolved"]
J["taboo"] = np.where(J.taboo_domain != "none", "taboo", "no taboo")
jx = pd.crosstab(J.taboo, J["判定"]).reindex(index=["taboo", "no taboo"],
                                             columns=["番号つきの語義", "補説・用例", "なし"], fill_value=0)
jap = {"n": int(len(J)), "n_left_out_taboo_unresolved": int(len(dj) - len(J)),
       "crosstab": {i: {c: int(jx.loc[i, c]) for c in jx.columns} for i in jx.index},
       "items_taboo": {i: J.loc[i, "taboo_domain"] for i in J.index[J.taboo == "taboo"]}}

out = {"plan": "docs/analysis_plan_taboo_domain.md", "exploratory": True, "kappa": kappa,
       "label_counts_217": lab.taboo_domain.value_counts().to_dict(), "refusals": ref, "refusals_by_label": ref_by_label,
       "crosstab": {i: {c: int(cross.loc[i, c]) for c in REC} for i in cross.index}, "main": main,
       "glmm": glmm, "per_domain": per_domain, "japanese_daijisen": jap}
json.dump(out, open(ROOT / "results/taboo_results.json", "w"), indent=1, ensure_ascii=False)

md = ["# Taboo domain and dictionary recording (exploratory; generated by analysis/analyze_taboo.py)", "",
      "Plan: docs/analysis_plan_taboo_domain.md.", "",
      f"## Agreement (3 coders, {kappa['n_items']} items)", "",
      f"- Fleiss kappa, taboo_domain (6 labels): {kappa['taboo_domain']:.4f}; all three agree on "
      f"{kappa['pct_unanimous_domain']:.1f}% of items",
      f"- Fleiss kappa, taboo (binary): {kappa['taboo_binary']:.4f}",
      "- Consensus labels (217 items): " + ", ".join(f"{k} {v}" for k, v in out["label_counts_217"].items()), "",
      f"## Taboo × recording status (analysis set: {main['n']} items; left out: recording UNRESOLVED "
      f"{n_rec_unres}, taboo unresolved {n_tab_unres})", "",
      "| | " + " | ".join(REC) + " | total |", "|---|---|---|---|---|"]
md += [f"| {i} | " + " | ".join(str(int(cross.loc[i, c])) for c in REC) + f" | {int(cross.loc[i].sum())} |"
       for i in cross.index]
md += ["", f"- general_dictionary: taboo {ky}/{ny} = {main['pct_general_taboo']:.1f}%, no taboo {kn}/{nn} = "
       f"{main['pct_general_no_taboo']:.1f}%",
       f"- Difference (taboo − no taboo): {main['diff_pct_points']:.1f} percentage points, 95% CI (Newcombe) "
       f"[{main['ci95_pct_points_newcombe'][0]:.1f}, {main['ci95_pct_points_newcombe'][1]:.1f}]",
       f"- Fisher's exact test: one-sided (taboo lower) p = {main['fisher_p_one_sided_less']:.4f}; two-sided p = "
       f"{main['fisher_p_two_sided']:.4f}; odds ratio {main['odds_ratio']:.3f}",
       f"- Reading: {main['reading']}", "",
       f"Auxiliary: general ~ taboo + (1 | tradition), {glmm['method']}, {glmm['n_traditions']} traditions: taboo "
       f"posterior mean {glmm['taboo_post_mean']:.3f}, SD {glmm['taboo_post_sd']:.3f}, mean ± 1.96 SD "
       f"[{glmm['interval_mean_pm_1.96sd'][0]:.3f}, {glmm['interval_mean_pm_1.96sd'][1]:.3f}]; converged: "
       f"{glmm['converged']} ({glmm['optimizer_message']})", "",
       "## Per domain (analysis set)", "", "| taboo_domain | n | general_dictionary | % |", "|---|---|---|---|"]
md += [f"| {r['taboo_domain']} | {r['n']} | {r['n_general']} | "
       f"{'' if r['pct_general'] is None else format(r['pct_general'], '.1f')} |" for r in per_domain]
md += ["", "## Refusals", "", "| coder | requests | with error | refused | batches refused | items in a refused "
       "request | items resent one by one | refused when sent alone |", "|---|---|---|---|---|---|---|---|"]
md += [f"| {c} | {v['requests']} | {v['requests_with_error']} | {v['requests_refused']} | {v['batches_refused']} | "
       f"{v['items_in_refused_request']} | {v['items_resent_one_by_one']} | "
       f"{', '.join(v['items_refused_when_sent_alone']) or 'none'} |" for c, v in ref.items()]
oe = {c: v["other_errors"] for c, v in ref.items() if v["other_errors"]}
md += ["", "Other errors: " + ("; ".join(f"{c}: {' / '.join(e)}" for c, e in oe.items()) if oe else "none"), "",
       "By consensus label (217 items): items in at least one refused request", "",
       "| taboo_domain | n | " + " | ".join(f"refused {c}" for c in CODERS) + " | refused, any coder | "
       + " | ".join(f"resent {c}" for c in CODERS) + " |", "|---|---|" + "---|" * 7]
md += [f"| {r['taboo_domain']} | {r['n_items_217']} | " + " | ".join(str(r[f'in_refused_request_{c}']) for c in CODERS)
       + f" | {r['in_refused_request_any']} | " + " | ".join(str(r[f'resent_one_by_one_{c}']) for c in CODERS) + " |"
       for r in ref_by_label]
md += ["", f"## Japanese items: Daijisen ({jap['n']} headwords; taboo unresolved left out: "
       f"{jap['n_left_out_taboo_unresolved']})", "", "| | 番号つきの語義 | 補説・用例 | なし |", "|---|---|---|---|"]
md += [f"| {i} | " + " | ".join(str(v) for v in jap["crosstab"][i].values()) + " |" for i in jap["crosstab"]]
md += ["", "Taboo items: " + (", ".join(f"{i} ({d})" for i, d in jap["items_taboo"].items()) or "none")]
(ROOT / "results/taboo_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
