"""T4 (analysis plan, 2026-10-22): integrated table and analysis after T1, T2 and T3.

Usage: python3 analysis/analyze_final.py      (after embeddings/scripts/check_hosts.py, embeddings/scripts/build_clean.py, the condition-C embeddings,
                                      analysis/analyze_t1.py and coding/scripts/aggregate_v1_1.py)

Outputs: results/items_final.csv, results/final_results.json, results/final_results.md, section 13 of results/verified_numbers.md
(replaced on rerun). No random numbers are used (seed fixed anyway).

items_final.csv: the items left after the T3 exclusion (SEA_012 plus the author's exclusions of 2026-10-22), one
row each: blind-table fields, T1 clean host and control counts, condition-C pulls for the three embedding models,
T2 consensus labels, cpg_scope and cpg_scope_two_storey, first-run consensus codes, recording status, the T3
proposed_action and author decision, and audit_flag (data/audit_flags.csv: items kept although the source audit
majority is contradicted).

Analyses (all exploratory, labelled as such in the report), on pull_C_k10, main set = T3 exclusion applied and
low_control removed (the same set as H-C1):
  - the H-C1 test (mixed model intercept, one-sided, Holm over 3 models; 3 -> supported, 1-2 -> partly, 0 -> not)
    restricted to cpg_scope = yes (author decision 5, 2026-10-22: form_type existing_form or
    transparent_combination), and, for comparison, to the old criterion form_status = existing_form
  - descriptive statistics by outsider_inference (UNRESOLVED kept as its own level), content_or_attitude,
    two_storey and the first scope definition cpg_scope_two_storey (descriptive only, decision 5)
k = 10 is primary; k = 3, k = 5 and kall (all kept controls, up to 10) are sensitivity analyses (decision 3).
The verdicts of H-C1/H-C2 before the decisions are read from results/before_decisions_2026-10-22/t1_results.json
(the T1 output saved just before the decisions were applied) and compared with the current ones.
"""
import json
import re
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
MODELS = {"oai": ("text_embedding_3_large", "OpenAI text-embedding-3-large"),
          "bge": ("BAAI_bge_m3", "BAAI/bge-m3"),
          "gem": ("gemini_embedding_001", "gemini-embedding-001")}
T2F = ["form_type", "two_storey", "content_or_attitude", "outsider_inference", "host_unit", "cpg_scope",
       "cpg_scope_two_storey"]
KS = ["k10", "k3", "k5", "kall"]
DECISIONS = [
    "Author decisions of 2026-10-22 applied here:",
    "1. ask_author items: no host form (MAYA_005, QCH_006) and several different words listed (NET_003; also NEU_006) "
    "are excluded; Exeter Book riddles and whole verse lines are kept with host_unit = utterance; items matching "
    "none of these are kept and listed (`docs/T3_exclusions.md`).",
    "2. use_first_form: variants of one item → first form becomes clean_host (embeddings recomputed with the "
    "three models); different words with different meanings → excluded; unclear → not adopted, listed.",
    "3. k = 10 (n fixed in advance) stays the primary analysis; k = 3, k = 5 and kall are sensitivity analyses.",
    "4. R2 kept as is; dropped controls per rule are at the top of `results/t1_results.md`.",
    "5. CPG scope = form_type existing_form or transparent_combination; two_storey and outsider_inference are "
    "reported descriptively only (the first definition is kept as cpg_scope_two_storey).",
    "6. outsider_inference UNRESOLVED stays a separate level in the descriptives; items whose form_type is "
    "UNRESOLVED are listed below."]
r4 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)

blind = pd.read_csv(ROOT / "data/blind_items.csv")
excl = pd.read_csv(ROOT / "data/exclusions.csv")
cand = pd.read_csv(ROOT / "data/exclusion_candidates.csv")[["item_id", "problem_type", "proposed_action"]]
dec = pd.read_csv(ROOT / "data/author_decisions.csv")[["item_id", "decision"]]
ch = pd.read_csv(ROOT / "embeddings/controls/clean_hosts.csv")
cs = pd.read_csv(ROOT / "embeddings/controls/controls_summary.csv")[["item_id", "n_original", "n_kept", "low_control"]]
v11 = pd.read_csv(ROOT / "coding/results/v1_1/v1_1_consensus.csv")[["item_id"] + T2F + ["mean_confidence"]]
first = pd.read_csv(ROOT / "coding/results/first_run/consensus_codes.csv")[
    ["item_id", "cpg_valid", "direction", "primary_channel", "activation", "familiarity"]]
rec = pd.read_csv(ROOT / "coding/results/recording/recording_consensus.csv")[["item_id", "form_status", "recording_status"]]
df = (blind.drop(columns=["order", "source_reference"])
      .merge(ch, on="item_id", how="left").merge(cs, on="item_id", how="left")
      .merge(v11.rename(columns={"mean_confidence": "v1_1_mean_confidence"}), on="item_id", how="left")
      .merge(first.rename(columns={c: f"first_{c}" for c in first.columns if c != "item_id"}), on="item_id", how="left")
      .merge(rec, on="item_id", how="left").merge(cand, on="item_id", how="left")
      .merge(dec.rename(columns={"decision": "t3_author_decision"}), on="item_id", how="left"))
df["t3_proposed_action"] = df.pop("proposed_action").fillna("no_problem_found")
df["t3_problem_type"] = df.pop("problem_type").fillna("")
df["t3_author_decision"] = df.t3_author_decision.fillna("")
for short, (tag, _) in MODELS.items():
    c = pd.read_csv(ROOT / f"embeddings/results/nnpm_C_{tag}.csv", encoding="utf-8-sig")
    if short == "oai":
        df = df.merge(c[["item_id", "tradition"]].rename(columns={"tradition": "tradition_pipeline"}),
                      on="item_id", how="left")
    df = df.merge(c[["item_id", "sim_host_target", "pull_C_k3", "pull_C_k5", "pull_C_k10", "pull_C_kall"]]
                  .rename(columns=lambda x: x if x == "item_id" else f"{x}_{short}"), on="item_id", how="left")
assert len(df) == 217 and df.form_type.notna().all()
df = df[~df.item_id.isin(excl.item_id)].reset_index(drop=True)
assert len(df) == 217 - len(excl)
utt = df[df.t3_author_decision == "keep_utterance"]
# decision 1: riddles and verse lines are utterance hosts; the T2 consensus already says so for every one of them
assert (utt.host_unit == "utterance").all(), utt[["item_id", "host_unit"]]
# mixed models group on the tradition labels of the pipeline (as in every earlier analysis); items without
# stored controls have no pull and no pipeline label
df["tradition_model"] = df.tradition_pipeline
# author's decision of 2026-10-25: items with majority contradicted in the source audit stay, flagged
df["audit_flag"] = df.item_id.map(pd.read_csv(ROOT / "data/audit_flags.csv").set_index("item_id").audit_flag).fillna("")
df.to_csv(ROOT / "results/items_final.csv", index=False, encoding="utf-8")


def describe(x):
    x = pd.Series(x, dtype=float).dropna()
    out = {"n": int(len(x)), "mean": r4(x.mean()) if len(x) else None,
           "sd": r4(x.std(ddof=1)) if len(x) > 1 else None,
           "pct_positive": round(100 * float((x > 0).mean()), 1) if len(x) else None}
    if len(x) >= 3:
        out["p_t_one_sided"] = r4(stats.ttest_1samp(x, 0, alternative="greater").pvalue)
    return out


def mixed(d, col):
    d = d.dropna(subset=[col])
    try:
        m = smf.mixedlm(f"{col} ~ 1", d, groups=d["tradition_model"]).fit(reml=True)
    except Exception as e:
        return {"n": int(len(d)), "error": str(e)}
    z = m.params["Intercept"] / m.bse["Intercept"]
    return {"n": int(len(d)), "n_traditions": int(d.tradition_model.nunique()),
            "intercept": r4(m.params["Intercept"]), "z": r4(z), "p_one_sided": r4(stats.norm.sf(z)), "_p_raw": float(stats.norm.sf(z))}


def holm_block(d, col):
    res = {s: mixed(d, f"{col}_{s}") for s in MODELS}
    adj = multipletests([res[s].pop("_p_raw", 1.0) for s in MODELS], method="holm")[1]
    for s, a in zip(MODELS, adj):
        res[s]["p_holm"] = r4(a)
    n = int(sum(a < .05 for a in adj))
    return {"per_model": res, "n_models_holm_p05": n,
            "verdict": "supported" if n == 3 else "partly supported" if n else "not supported"}


main = df[(df.low_control == 0)]
C = {"items_after_T3": int(len(df)), "with_controls_after_T3": int(df.n_kept.notna().sum()),
     "main_set_not_low_control": int(len(main)),
     "main_with_k10": int(main.pull_C_k10_oai.notna().sum()),
     "cpg_scope_distribution_after_T3": df.cpg_scope.value_counts().to_dict(),
     "cpg_scope_yes_main_k10": int(((main.cpg_scope == "yes") & main.pull_C_k10_oai.notna()).sum()),
     "cpg_scope_unresolved_after_T3": int((df.cpg_scope == "UNRESOLVED").sum()),
     "cpg_scope_two_storey_distribution_after_T3": df.cpg_scope_two_storey.value_counts().to_dict(),
     "outsider_inference_unresolved_after_T3": sorted(df.item_id[df.outsider_inference == "UNRESOLVED"]),
     "form_type_unresolved_after_T3": sorted(df.item_id[df.form_type == "UNRESOLVED"]),
     "excluded_T3": excl.item_id.tolist(),
     "keep_utterance_items": utt.item_id.tolist(),
     "cpg_scope_yes_main_equals_all_main": bool((main.cpg_scope == "yes").all())}
R = {"exploratory": True, "counts": C, "tests": {}, "descriptives": {}}
for lab, d in [("cpg_scope_yes", main[main.cpg_scope == "yes"]),
               ("old_criterion_existing_form", main[main.form_status == "existing_form"]),
               ("all_main", main)]:
    R["tests"][lab] = {k: holm_block(d, f"pull_C_{k}") for k in KS}
# decision of 2026-10-22 (second round): when the scope test is identical to all_main, the report only notes this
R["cpg_scope_yes_test_identical_to_all_main"] = R["tests"]["cpg_scope_yes"] == R["tests"]["all_main"]
SHOWN = [lab for lab in R["tests"] if not (lab == "cpg_scope_yes" and R["cpg_scope_yes_test_identical_to_all_main"])]
SCOPE_NOTE = ("The H-C1 test restricted to cpg_scope = yes is identical to the test on all main-set items (all_main), "
              "because every item kept after T3 meets the form criterion; it is therefore not tabulated separately.")
GROUPS = ["outsider_inference", "content_or_attitude", "two_storey", "cpg_scope_two_storey", "cpg_scope", "form_type"]
for g in GROUPS:
    R["descriptives"][g] = {s: {str(k): {"k10": describe(v[f"pull_C_k10_{s}"]), "kall": describe(v[f"pull_C_kall_{s}"])}
                                for k, v in main.groupby(g)} for s in MODELS}
# verdicts of the fixed hypotheses before / after the decisions
t1 = json.load(open(ROOT / "results/t1_results.json"))
t1_before = json.load(open(ROOT / "results/before_decisions_2026-10-22/t1_results.json"))
R["hypotheses_before_after_decisions"] = {
    h: {lab: {"verdict": t["hypotheses"][h]["verdict"], "n_models_holm_p05": t["hypotheses"][h]["n_models_holm_p05"],
              "per_model": {m: {k: v[k] for k in ["n", "z", "p_holm"] if k in v}
                            for m, v in t["hypotheses"][h]["per_model"].items()}}
        for lab, t in [("before", t1_before), ("after", t1)]}
    for h in ["H-C1", "H-C2"]}
for h, v in R["hypotheses_before_after_decisions"].items():
    v["verdict_changed"] = v["before"]["verdict"] != v["after"]["verdict"]
json.dump(R, open(ROOT / "results/final_results.json", "w"), ensure_ascii=False, indent=1)

# ------------------------------------------------------------ report
f = lambda x: "–" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)
pf = lambda x: "–" if x is None else f"{x:.1f}"
L = ["# Final integrated analysis (T4) — exploratory (generated by analysis/analyze_final.py)", ""] + DECISIONS + ["",
     f"- Items after T3 exclusion: {C['items_after_T3']}; with stored controls: {C['with_controls_after_T3']}; "
     f"main set (not low_control): {C['main_set_not_low_control']}; of these with a k = 10 pull: {C['main_with_k10']}.",
     f"- Excluded in T3 ({len(C['excluded_T3'])}): {', '.join(C['excluded_T3'])}.",
     f"- cpg_scope (decision 5: form_type existing_form or transparent_combination), after T3: "
     + ", ".join(f"{k} {v}" for k, v in C["cpg_scope_distribution_after_T3"].items())
     + f". In the main set with a k = 10 pull: {C['cpg_scope_yes_main_k10']} cpg_scope = yes"
     + (" — every main-set item is in scope." if C["cpg_scope_yes_main_equals_all_main"] else "."),
     f"- form_type UNRESOLVED after T3: {', '.join(C['form_type_unresolved_after_T3']) or 'none'}.",
     f"- First definition cpg_scope_two_storey (descriptive only), after T3: "
     + ", ".join(f"{k} {v}" for k, v in C["cpg_scope_two_storey_distribution_after_T3"].items()) + ".",
     f"- outsider_inference UNRESOLVED after T3 ({len(C['outsider_inference_unresolved_after_T3'])}): "
     + ", ".join(C["outsider_inference_unresolved_after_T3"]) + ".",
     f"- Kept as utterance hosts (decision 1; T2 host_unit = utterance for all): "
     + ", ".join(C["keep_utterance_items"]) + ".", "",
     "## Fixed hypotheses H-C1 / H-C2 before and after the decisions (k = 10)", "",
     "SI table — keep. The exclusions of 2026-10-22 (e.g. QCH_006) were decided after the first results had been "
     "seen, so both the verdicts before and after the decisions are reported.", "",
     "| hypothesis | | " + " | ".join(m[1] for m in MODELS.values()) + " | verdict |", "|---|---|---|---|---|---|"]
for h, v in R["hypotheses_before_after_decisions"].items():
    for lab in ["before", "after"]:
        pm = v[lab]["per_model"]
        L.append(f"| {h} | {lab} | " + " | ".join(
            f"n={pm[m]['n']}, z={f(pm[m].get('z'))}, Holm p={f(pm[m].get('p_holm'))}" for m in pm)
                 + f" | {v[lab]['verdict']} ({v[lab]['n_models_holm_p05']}/3) |")
L += [""] + [f"- {h}: verdict {'changed' if v['verdict_changed'] else 'unchanged'}"
             for h, v in R["hypotheses_before_after_decisions"].items()]
L += ["", "## H-C1 test on subsets (exploratory; k10 primary, k3 / k5 / kall sensitivity)", "",
     "| subset | pull | model | n | intercept | z | p (one-sided) | p Holm |", "|---|---|---|---|---|---|---|---|"]
for lab in SHOWN:
    for k, blk in R["tests"][lab].items():
        for s, r in blk["per_model"].items():
            L.append(f"| {lab} | {k} | {MODELS[s][1]} | {r['n']} | {f(r.get('intercept'))} | {f(r.get('z'))} | "
                     f"{f(r.get('p_one_sided'))} | {f(r.get('p_holm'))} |")
L.append("")
if not set(R["tests"]) <= set(SHOWN):
    L += [SCOPE_NOTE, ""]
for lab in SHOWN:
    t = R["tests"][lab]
    L.append(f"- {lab}: k10 **{t['k10']['verdict']}** ({t['k10']['n_models_holm_p05']}/3); " + ", ".join(
        f"{k} {t[k]['verdict']} ({t[k]['n_models_holm_p05']}/3)" for k in KS[1:]))
for g in GROUPS:
    L += ["", f"## pull_C_k10 by {g} (main set; kall in brackets)", "",
          "| level | " + " | ".join(m[1] for m in MODELS.values()) + " |", "|---|---|---|---|"]
    for lv in sorted({k for s in MODELS for k in R["descriptives"][g][s]}):
        cells = []
        for s in MODELS:
            r = R["descriptives"][g][s].get(lv)
            cells.append("–" if not r else f"n={r['k10']['n']}, mean={f(r['k10']['mean'])}, "
                         f"{pf(r['k10']['pct_positive'])}% > 0 [n={r['kall']['n']}, mean={f(r['kall']['mean'])}]")
        L.append(f"| {lv} | " + " | ".join(cells) + " |")
(ROOT / "results/final_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ------------------------------------------------------------ section 13 of verified_numbers.md
agr = json.load(open(ROOT / "coding/results/v1_1/v1_1_agreement.json"))
c1 = t1["counts"]
S = ["## 13. T1–T4: clean hosts, condition C, codebook v1.1 (run 2026-10-22, `analysis/analyze_t1.py`, "
     "`coding/scripts/aggregate_v1_1.py`, `analysis/analyze_final.py`)", ""] + DECISIONS + ["",
     f"Items: {c1['blind_items']} → {c1['items_with_controls']} with controls → {c1['after_T3']} after T3 "
     f"({', '.join(C['excluded_T3'])}; those with controls) → {c1['after_low_control']} not low_control → k = 10 pull for "
     f"{c1['eligible_by_k_after_low_control']['k10']} (items with 10 kept controls).", "",
     "| hypothesis | OpenAI | bge-m3 | Gemini | verdict |", "|---|---|---|---|---|"]
for h in ["H-C1", "H-C2"]:
    pm = t1["hypotheses"][h]["per_model"]
    S.append(f"| {h} | " + " | ".join(f"n={pm[t]['n']}, z={f(pm[t].get('z'))}, Holm p={f(pm[t].get('p_holm'))}"
                                      for t in ["text_embedding_3_large", "BAAI_bge_m3", "gemini_embedding_001"])
             + f" | {t1['hypotheses'][h]['verdict']} |")
for h, v in R["hypotheses_before_after_decisions"].items():
    S.append(f"| {h} before the decisions | " + " | ".join(
        f"n={x['n']}, z={f(x.get('z'))}, Holm p={f(x.get('p_holm'))}" for x in v["before"]["per_model"].values())
             + f" | {v['before']['verdict']} |")
for lab in [x for x in ["cpg_scope_yes", "old_criterion_existing_form"] if x in SHOWN]:
    pm = R["tests"][lab]["k10"]["per_model"]
    S.append(f"| {lab} (exploratory) | " + " | ".join(
        f"n={pm[s]['n']}, z={f(pm[s].get('z'))}, Holm p={f(pm[s].get('p_holm'))}" for s in MODELS)
             + f" | {R['tests'][lab]['k10']['verdict']} |")
S += ([""] + ([SCOPE_NOTE] if "cpg_scope_yes" not in SHOWN else []) + [
      "The exclusions of 2026-10-22 were decided after the first results had been seen; the rows 'before the "
      "decisions' are kept for that reason (SI).",
      "", "Codebook v1.1 agreement (Fleiss' κ / % unanimous): " + "; ".join(
    f"{k} {v} / {agr['pct_unanimous'][k]}%" for k, v in agr["fleiss_kappa"].items()) + ".",
      "cpg_scope (form_type) after T3: " + ", ".join(f"{k} {v}" for k, v in C["cpg_scope_distribution_after_T3"].items())
      + "; first definition cpg_scope_two_storey (descriptive): "
      + ", ".join(f"{k} {v}" for k, v in C["cpg_scope_two_storey_distribution_after_T3"].items()) + ".",
      "H-C1 / H-C2 verdicts: " + "; ".join(f"{h} {v['before']['verdict']} → {v['after']['verdict']}"
                                         for h, v in R["hypotheses_before_after_decisions"].items()) + ".",
      "", "Full tables: `results/t1_results.md`, `results/final_results.md`.", ""])
vn = ROOT / "results/verified_numbers.md"
txt = vn.read_text(encoding="utf-8")
head, sep, rest = txt.partition("\n## 13. ")
nxt = re.search(r"\n## \d+\. ", rest) if sep else None
txt = head.rstrip("\n") + "\n\n" + "\n".join(S).rstrip("\n") + "\n" + (rest[nxt.start():] if nxt else "")
vn.write_text(txt, encoding="utf-8")
print("\n".join(L))
print("\n".join(S))
