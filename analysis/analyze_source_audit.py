"""Source audit, stage B results and sensitivity analysis (docs/analysis_plan_source_audit.md).

Usage: python3 analysis/analyze_source_audit.py

Inputs : results/source_audit_evidence.csv, audit/runs/judge_<model>.csv (3 coders), audit/runs/judge_<model>.jsonl
         (refusals), data/blind_items.csv; for the sensitivity analysis the condition C inputs of analyze_t1.py.
Outputs: results/source_audit_judgements.csv (per coder and majority, all 217 items),
         results/source_audit_summary.{json,md}, private/spot_check_sheet.md and private/spot_check_key.json
         (the 20 items of the author's spot check; the sheet shows no LLM label and no stratum).

- Items with no quote (all documents failed / record_only / ai_generated, or nothing relevant) are no_evidence by
  rule and were not sent to the coders.
- Majority of the three coders; three different labels: UNRESOLVED. Fleiss' kappa over the items sent.
- Judging again (author's decisions after stage B): audit/runs/judge_<model>_<run>.csv replace the labels of their
  items (runs in name order); the first-run labels and majority are kept in label_<coder>_first and majority_first.
  Corrections of item fields (audit/corrections.csv) are applied. audit_flag comes from data/audit_flags.csv.
- Spot check: numpy.random.default_rng(20261024); 10 items from majority supported, 10 from all other items (in
  that order of drawing; candidates in the audit order, seed 20261023); all items of a stratum if it has fewer.
- Sensitivity: the condition C main set (k = 10) restricted to (a) majority supported, (b) supported or partly;
  H-C1 and H-C2 recomputed exactly as in analyze_t1.py (mixedlm pull_C_k10 ~ 1 + (1 | tradition), raw labels,
  REML, one-sided z, Holm across the 3 models). The same for condition D: H-D1 and H-D2 as in analyze_target_blind.py
  (main-set items with a pull_D_k10, embeddings/results/nnpm_DE_<model>.csv).
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "audit/scripts"))
from audit_tool import order  # noqa: E402

CODERS = ["gpt-5.6-sol", "claude-opus-5-5", "gemini-3.1-pro-preview"]
TAG = {"gpt-5.6-sol": "gpt", "claude-opus-5-5": "claude", "gemini-3.1-pro-preview": "gemini"}
LABELS = ["supported", "partly", "contradicted", "no_evidence"]
TYPES = ["scholarly", "reference", "wikipedia", "web"]
MODELS = {"text_embedding_3_large": "OpenAI text-embedding-3-large", "BAAI_bge_m3": "BAAI/bge-m3",
          "gemini_embedding_001": "gemini-embedding-001"}
r4 = lambda x: round(float(x), 4)

blind = pd.read_csv(ROOT / "data/blind_items.csv", dtype=str).fillna("")
corr = pd.read_csv(ROOT / "audit/corrections.csv", dtype=str)
for c in corr.itertuples():
    k = blind.index[blind.item_id == c.item_id][0]
    assert blind.at[k, c.field] == c.old, (c.item_id, c.field)
    blind.at[k, c.field] = c.new
ev = pd.read_csv(ROOT / "results/source_audit_evidence.csv", dtype=str).fillna("")
quoted = ev[(ev.ai_generated != "yes") & ((ev.quote != "") | (ev.quote_2 != ""))]
ids = order()
assert sorted(ids) == sorted(blind.item_id) and len(ids) == 217

# ------------------------------------------------------------ judgements
J = pd.DataFrame({"item_id": ids})
J["audit_position"] = range(1, len(ids) + 1)
J["phase"] = np.where(J.item_id.str.startswith("T_"), "1", "2")
best = quoted.groupby("item_id").source_type.agg(lambda s: next(t for t in TYPES if t in set(s)))
J["best_source_type"] = J.item_id.map(best).fillna("none")
J["sent"] = J.item_id.isin(set(quoted.item_id))
refusals = {}
for c in CODERS:
    d = pd.read_csv(ROOT / f"audit/runs/judge_{c}.csv", dtype=str).fillna("")
    d = d[d.status == "ok"].drop_duplicates("item_id", keep="last").set_index("item_id")
    assert set(d.index) == set(J.item_id[J.sent]), f"{c}: labelled items != items sent"
    J[f"label_{TAG[c]}_first"] = J.item_id.map(d.label).fillna("")
    J["rejudged"] = J.get("rejudged", "")
    for f in sorted((ROOT / "audit/runs").glob(f"judge_{c}_*.csv")):
        r = pd.read_csv(f, dtype=str).fillna("")
        r = r[r.status == "ok"].drop_duplicates("item_id", keep="last").set_index("item_id")
        assert set(r.index) <= set(d.index)
        d.loc[r.index, ["label", "reason"]] = r[["label", "reason"]]
        J.loc[J.item_id.isin(r.index), "rejudged"] = f.stem.removeprefix(f"judge_{c}_")
    J[f"label_{TAG[c]}"] = J.item_id.map(d.label).fillna("")
    J[f"reason_{TAG[c]}"] = J.item_id.map(d.reason).fillna("")
    log = [json.loads(x) for x in open(ROOT / f"audit/runs/judge_{c}.jsonl", encoding="utf-8")]
    refusals[c] = {"requests": len(log), "refused_requests": sum(1 for x in log if x.get("refusal")),
                   "items_in_refused_requests": sorted({i for x in log if x.get("refusal") for i in x["item_ids"]}),
                   "single_item_requests": sum(1 for x in log if x.get("mode") == "single")}


def majority(row, suf=""):
    if not row.sent:
        return "no_evidence"
    v = pd.Series([row[f"label_{TAG[c]}{suf}"] for c in CODERS]).value_counts()
    return v.index[0] if v.iloc[0] >= 2 else "UNRESOLVED"


J["majority"] = J.apply(majority, axis=1)
J["majority_first"] = J.apply(majority, axis=1, suf="_first")
flags = pd.read_csv(ROOT / "data/audit_flags.csv", dtype=str).set_index("item_id")
J["audit_flag"] = J.item_id.map(flags.audit_flag).fillna("")
J["rule_based"] = ~J.sent
S = J[J.sent]
table, _ = aggregate_raters(np.array([S[f"label_{TAG[c]}"].values for c in CODERS]).T)
kappa = float(fleiss_kappa(table, method="fleiss"))
J.to_csv(ROOT / "results/source_audit_judgements.csv", index=False)

cats = LABELS + ["UNRESOLVED"]
cnt = lambda x: {k: int((x == k).sum()) for k in cats}
summary = {
    "n_items": len(J), "n_sent": int(J.sent.sum()), "n_rule_based_no_evidence": int((~J.sent).sum()),
    "coders": CODERS, "fleiss_kappa": r4(kappa), "pct_unanimous": round(100 * float(
        (S[[f"label_{TAG[c]}" for c in CODERS]].nunique(axis=1) == 1).mean()), 1),
    "majority": cnt(J.majority),
    "majority_by_phase": {p: cnt(g.majority) for p, g in J.groupby("phase")},
    "majority_by_best_source_type": {t: cnt(g.majority) for t, g in J.groupby("best_source_type")},
    "per_coder": {c: {k: int((S[f"label_{TAG[c]}"] == k).sum()) for k in LABELS} for c in CODERS},
    "refusals": refusals,
    "contradicted_items": J.item_id[J.majority == "contradicted"].tolist(),
    "rejudged": {r.item_id: {"run": r.rejudged, "majority_first": r.majority_first, "majority": r.majority}
                 for r in J[J.rejudged != ""].itertuples()},
    "corrections": corr.to_dict(orient="records"),
    "audit_flags": J.set_index("item_id").audit_flag[J.audit_flag.values != ""].to_dict(),
    "unresolved_items": J.item_id[J.majority == "UNRESOLVED"].tolist(),
}
assert sum(summary["majority"].values()) == 217

# ------------------------------------------------------------ spot check (20 items)
rng = np.random.default_rng(20261024)
sup = [i for i in ids if J.set_index("item_id").at[i, "majority"] == "supported"]
oth = [i for i in ids if i not in set(sup)]
draw = lambda pool: [pool[k] for k in rng.choice(len(pool), size=min(10, len(pool)), replace=False)]
spot_sup, spot_oth = draw(sup), draw(oth)
spot = sorted(spot_sup + spot_oth)  # the sheet lists them sorted by item ID, so the order reveals no stratum
summary["spot_check_items"] = spot
PRIV = ROOT / "private"
PRIV.mkdir(exist_ok=True)
json.dump({"seed": 20261024, "supported": spot_sup, "other": spot_oth}, open(PRIV / "spot_check_key.json", "w"),
          indent=1)
B = blind.set_index("item_id")
sheet = ["# Spot check of the source audit (author)", "",
         "For each item: read the documents (open the URL) and give one label: supported / partly / contradicted /",
         "no_evidence (definitions in docs/analysis_plan_source_audit.md, Stage B). The LLM labels are not shown.", ""]
for i in spot:
    r = B.loc[i]
    sheet += [f"## {i}", f"- host: {r.host_original} ({r.host_romanized})", f"- covert target: {r.covert_target}",
              f"- tradition: {r.tradition}", "- documents:"]
    for d in ev[ev.item_id == i].assign(_r=lambda x: x.doc_rank.astype(int)).sort_values("_r").itertuples():
        sheet.append(f"  - D{d.doc_rank} [{d.source_type}, {d.retrieved}{', ai_generated' if d.ai_generated == 'yes' else ''}]"
                     f" {d.citation} — {d.url or '(no URL)'}")
        for q in (d.quote, d.quote_2):
            if q:
                sheet.append(f"    - quote: {q}")
    sheet += ["- author's label: ", ""]
(PRIV / "spot_check_sheet.md").write_text("\n".join(sheet) + "\n", encoding="utf-8")

# ------------------------------------------------------------ sensitivity (H-C1, H-C2 on restricted main sets)
summ = pd.read_csv(ROOT / "embeddings/controls/controls_summary.csv")
excl = set(pd.read_csv(ROOT / "data/exclusions.csv").item_id)
first = pd.read_csv(ROOT / "coding/results/first_run/consensus_codes.csv")[["item_id", "cpg_valid"]]
maj = J.set_index("item_id").majority


def mixed(d, y="pull_C_k10"):
    m = smf.mixedlm(f"{y} ~ 1", d, groups=d["tradition"]).fit(reml=True)
    z = m.params["Intercept"] / m.bse["Intercept"]
    return {"n": int(len(d)), "n_traditions": int(d.tradition.nunique()), "intercept": r4(m.params["Intercept"]),
            "se": r4(m.bse["Intercept"]), "z": r4(z), "_p": float(stats.norm.sf(z))}


main = {}
for tag in MODELS:
    c = pd.read_csv(ROOT / f"embeddings/results/nnpm_C_{tag}.csv", encoding="utf-8-sig")
    d = c.merge(summ[["item_id", "low_control"]], on="item_id", how="left").merge(first, on="item_id", how="left")
    d = d[~d.item_id.isin(excl) & (d.low_control == 0)].dropna(subset=["pull_C_k10"])
    main[tag] = d.assign(audit=d.item_id.map(maj))
n_main = {len(d) for d in main.values()}
assert n_main == {94}, n_main
t1 = json.load(open(ROOT / "results/t1_results.json"))
tb = json.load(open(ROOT / "results/target_blind_controls.json"))
mainD = {}
for tag in MODELS:
    d = pd.read_csv(ROOT / f"embeddings/results/nnpm_DE_{tag}.csv", encoding="utf-8-sig").merge(first, on="item_id",
                                                                                              how="left")
    assert sorted(d.item_id) == sorted(main[tag].item_id)
    mainD[tag] = d.dropna(subset=["pull_D_k10"]).assign(audit=lambda x: x.item_id.map(maj))
HYPS = [("H-C1", "pull_C_k10", main, lambda d: d), ("H-C2", "pull_C_k10", main, lambda d: d[d.cpg_valid == "yes"]),
        ("H-D1", "pull_D_k10", mainD, lambda d: d), ("H-D2", "pull_D_k10", mainD, lambda d: d[d.cpg_valid == "yes"])]
for hyp, y, base, sel in HYPS:
    for t, d in base.items():  # the unrestricted fit must reproduce analyze_t1.py / analyze_target_blind.py
        stored = (t1["hypotheses"][hyp]["per_model"][t]["z"] if hyp.startswith("H-C") else
                  next(r["z"] for r in tb["rows"] if r["hypothesis"] == hyp and r["model"] == t))
        assert mixed(sel(d), y)["z"] == stored, (hyp, t)
SETS = {"supported": {"supported"}, "supported_or_partly": {"supported", "partly"}}
sens = {"main_set_n": 94, "main_set_majority": cnt(main["text_embedding_3_large"].audit),
        "condition_D_n": len(mainD["text_embedding_3_large"]),
        "condition_D_majority": cnt(mainD["text_embedding_3_large"].audit), "rows": []}
for sname, keep in SETS.items():
    for hyp, y, base, sel in HYPS:
        res = {t: mixed(sel(d[d.audit.isin(keep)]), y) for t, d in base.items()}
        adj = multipletests([res[t].pop("_p") for t in MODELS], method="holm")[1]
        n_pass = int(sum(a < .05 for a in adj))
        for t, a in zip(MODELS, adj):
            sens["rows"].append({"set": sname, "hypothesis": hyp, "model": MODELS[t], **res[t], "p_holm": r4(a)})
        sens.setdefault("verdicts", {})[f"{sname} {hyp}"] = ("supported" if n_pass == 3 else
                                                            "partly supported" if n_pass else "not supported")
summary["sensitivity"] = sens
json.dump(summary, open(ROOT / "results/source_audit_summary.json", "w"), indent=1, ensure_ascii=False)

# ------------------------------------------------------------ markdown
md = ["# Source audit: summary (generated by analysis/analyze_source_audit.py)", "",
      f"Items: {len(J)}; sent to the coders: {summary['n_sent']}; no quote (no_evidence by rule): "
      f"{summary['n_rule_based_no_evidence']}. Coders: {', '.join(CODERS)}. Fleiss' kappa (items sent): "
      f"{kappa:.3f}; unanimous: {summary['pct_unanimous']}%.", "",
      "| | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|",
      "| all | " + " | ".join(str(summary["majority"][k]) for k in cats) + " |"]
for p, v in summary["majority_by_phase"].items():
    md.append(f"| Phase {p} | " + " | ".join(str(v[k]) for k in cats) + " |")
for t in TYPES + ["none"]:
    if t in summary["majority_by_best_source_type"]:
        v = summary["majority_by_best_source_type"][t]
        md.append(f"| best source {t} | " + " | ".join(str(v[k]) for k in cats) + " |")
md += ["", "Per coder (items sent): " + "; ".join(
    f"{c}: " + ", ".join(f"{k} {v}" for k, v in summary["per_coder"][c].items()) for c in CODERS), "",
       "Refusals: " + "; ".join(f"{c}: {r['refused_requests']} of {r['requests']} requests" for c, r in refusals.items()),
       "", "Contradicted (majority): " + (", ".join(summary["contradicted_items"]) or "none"),
       "", "UNRESOLVED: " + (", ".join(summary["unresolved_items"]) or "none"), "",
       "Judged again: " + (", ".join(f"{k} ({v['run']}: {v['majority_first']} -> {v['majority']})"
                                     for k, v in summary["rejudged"].items()) or "none"), "",
       "Corrections of item fields: " + ("; ".join(f"{c['item_id']} {c['field']}: {c['old']} -> {c['new']}"
                                                   for c in summary["corrections"]) or "none"), "",
       "audit_flag: " + (", ".join(f"{k} {v}" for k, v in summary["audit_flags"].items()) or "none"), "",
       "## Sensitivity: condition C main set (k = 10, n = 94) and condition D restricted by the audit", "",
       "Majority labels in the main set: " + ", ".join(f"{k} {v}" for k, v in sens["main_set_majority"].items()), "",
       f"Majority labels in condition D (main-set items with a k = 10 pull, n = {sens['condition_D_n']}): "
       + ", ".join(f"{k} {v}" for k, v in sens["condition_D_majority"].items()), "",
       "| set | hypothesis | model | n | z | Holm p (4 decimals) |", "|---|---|---|---|---|---|"]
md += [f"| {r['set']} | {r['hypothesis']} | {r['model']} | {r['n']} | {r['z']:.4f} | {r['p_holm']:.4f} |"
       for r in sens["rows"]]
md += ["", "Verdicts: " + "; ".join(f"{k}: {v}" for k, v in sens["verdicts"].items())]
(ROOT / "results/source_audit_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
