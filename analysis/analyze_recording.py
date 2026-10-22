"""Recording-status analysis (spec v1.0, 2026-10-21): H1 recording gradient, H2 unrecorded CPG, H3 neologisms.

Usage: python3 analyze_recording.py

Inputs : results/items_with_consensus.csv (build_items_with_consensus.py),
         coding/results/recording/recording_consensus.csv and recording_agreement.json (aggregate_recording.py)
Outputs: results/recording_results.json, results/recording_results.md, and section 12 of results/verified_numbers.md
         (replaced on rerun, so the script is idempotent). No random numbers are used.

Decision rules (fixed in this script before the full coding results were inspected):
  H1, per embedding model: supported if (a) the ordered trend slope is positive with Holm-adjusted one-sided
      p < .05, (b) group means satisfy general_dictionary >= cultural_reference > unrecorded, and (c) each of the
      three groups has mean > 0 with one-sided t-test p < .05. Overall: supported if all three models pass,
      not supported if none does, mixed otherwise.
  H2: primary test = mixed-model intercept (one-sided) on unrecorded existing_form items, Holm across models.
      supported if >= 2 models have Holm p < .05; mixed if exactly 1, or if >= 2 pass only before correction;
      not supported otherwise.
  H3: the spec defines H3 as an exclusion rule with descriptive reporting only, so it receives no inferential
      verdict; the report states that the rule was applied and gives novel_form pulls descriptively.
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
ROOT = Path(__file__).resolve().parents[1]
PULLS = {"pull_index": "OpenAI text-embedding-3-large (original)",
         "pull_bge_B": "BAAI/bge-m3 (original script, k=10)",
         "pull_gem_B": "gemini-embedding-001 (original script, k=10)"}
LEVELS = ["unrecorded", "cultural_reference", "general_dictionary"]
r4 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)

items = pd.read_csv(ROOT / "results/items_with_consensus.csv")
rec = pd.read_csv(ROOT / "coding/results/recording/recording_consensus.csv")
agree = json.load(open(ROOT / "coding/results/recording/recording_agreement.json"))
df = items.merge(rec[["item_id", "form_status", "recording_status", "recording_unanimous", "form_unanimous",
                      "mean_recording_confidence"]], on="item_id", how="left")
assert len(df) == 213 and df.form_status.notna().all()


def describe(x):
    x = pd.Series(x).dropna()
    out = {"n": int(len(x)), "mean": r4(x.mean()) if len(x) else None,
           "sd": r4(x.std(ddof=1)) if len(x) > 1 else None,
           "pct_positive": round(100 * float((x > 0).mean()), 1) if len(x) else None}
    if len(x) >= 3:
        out["t"] = r4(stats.ttest_1samp(x, 0).statistic)
        out["p_t_one_sided"] = r4(stats.ttest_1samp(x, 0, alternative="greater").pvalue)
        out["p_wilcoxon_one_sided"] = r4(stats.wilcoxon(x, alternative="greater").pvalue)
    else:
        out["t"] = out["p_t_one_sided"] = out["p_wilcoxon_one_sided"] = None
    return out


def mixed(formula, d):
    try:
        return smf.mixedlm(formula, d, groups=d["tradition"]).fit(reml=True)
    except Exception as e:  # singular fits etc. are reported, not hidden
        return str(e)


def analyse(d, col):
    """Steps 1-4 of the spec for one pull column on one item subset."""
    res = {"by_form_status": {f: describe(g[col]) for f, g in d.groupby("form_status")}}
    ex = d[d.form_status == "existing_form"]
    res["existing_by_recording_status"] = {r: describe(g[col]) for r, g in ex.groupby("recording_status")}
    exr = ex[ex.recording_status.isin(LEVELS)].dropna(subset=[col]).copy()
    res["n_existing_resolved"] = int(len(exr))
    m = mixed(f"{col} ~ C(recording_status, Treatment('unrecorded'))", exr)
    if isinstance(m, str):
        res["mixed_levels"] = {"error": m}
    else:
        res["mixed_levels"] = {}
        for lv in LEVELS[1:]:
            k = f"C(recording_status, Treatment('unrecorded'))[T.{lv}]"
            if k in m.params:
                res["mixed_levels"][lv] = {"b": r4(m.params[k]), "se": r4(m.bse[k]), "p_two_sided": r4(m.pvalues[k])}
        res["mixed_levels"]["intercept_unrecorded"] = {"b": r4(m.params["Intercept"]), "se": r4(m.bse["Intercept"])}
    exr["level"] = exr.recording_status.map({lv: i for i, lv in enumerate(LEVELS)})
    m = mixed(f"{col} ~ level", exr)
    if isinstance(m, str):
        res["trend"] = {"error": m}
    else:
        z = m.params["level"] / m.bse["level"]
        res["trend"] = {"slope": r4(m.params["level"]), "se": r4(m.bse["level"]), "z": r4(z),
                        "p_one_sided": r4(stats.norm.sf(z))}
    un = ex[ex.recording_status == "unrecorded"].dropna(subset=[col])
    h2 = describe(un[col])
    if len(un) >= 3 and un.tradition.nunique() >= 2:
        m = mixed(f"{col} ~ 1", un)
        if isinstance(m, str):
            h2["mixed_error"] = m
        else:
            z = m.params["Intercept"] / m.bse["Intercept"]
            h2.update(mixed_intercept=r4(m.params["Intercept"]), mixed_se=r4(m.bse["Intercept"]),
                      mixed_z=r4(z), p_mixed_one_sided=r4(stats.norm.sf(z)))
    h2["n_traditions"] = int(un.tradition.nunique())
    res["h2_unrecorded_existing"] = h2
    return res


def holm(ps):
    ok = [i for i, p in enumerate(ps) if p is not None]
    adj = [None] * len(ps)
    if ok:
        for i, a in zip(ok, multipletests([ps[i] for i in ok], method="holm")[1]):
            adj[i] = r4(a)
    return adj


def verdicts(block):
    cols = list(PULLS)
    # H1
    trend_p = [block[c]["trend"].get("p_one_sided") for c in cols]
    trend_adj = holm(trend_p)
    h1 = {}
    for c, padj in zip(cols, trend_adj):
        g = block[c]["existing_by_recording_status"]
        means = {lv: g.get(lv, {}).get("mean") for lv in LEVELS}
        order = None not in means.values() and \
            means["general_dictionary"] >= means["cultural_reference"] > means["unrecorded"]
        pos = all((g.get(lv, {}).get("p_t_one_sided") or 1) < .05 for lv in LEVELS)
        trend_ok = padj is not None and padj < .05 and (block[c]["trend"].get("slope") or 0) > 0
        h1[c] = {"trend_p_holm": padj, "trend_ok": bool(trend_ok), "order_ok": bool(order),
                 "all_groups_positive": bool(pos), "means": means, "passes": bool(trend_ok and order and pos)}
    n1 = sum(v["passes"] for v in h1.values())
    h1_v = "supported" if n1 == 3 else "not supported" if n1 == 0 else "mixed"
    # H2
    p2 = [block[c]["h2_unrecorded_existing"].get("p_mixed_one_sided") for c in cols]
    p2adj = holm(p2)
    n_raw = sum(p is not None and p < .05 for p in p2)
    n_adj = sum(p is not None and p < .05 for p in p2adj)
    h2_v = "supported" if n_adj >= 2 else "mixed" if (n_adj == 1 or n_raw >= 2) else "not supported"
    h2 = {c: {"p_mixed_one_sided": p, "p_holm": a} for c, p, a in zip(cols, p2, p2adj)}
    return {"H1": {"verdict": h1_v, "models_passing": n1, "per_model": h1},
            "H2": {"verdict": h2_v, "models_sig_holm": n_adj, "models_sig_raw": n_raw, "per_model": h2},
            "H3": {"verdict": "procedural rule applied (no inferential test, per spec)",
                   "novel_form_descriptives": {c: block[c]["by_form_status"].get("novel_form") for c in cols}}}


R = {"spec": "Recording-status coding and analysis v1.0 (2026-10-21)", "pull_columns": PULLS,
     "agreement": agree}
cnt = {"items_with_pulls": int(len(df)),
       "form_status": df.form_status.value_counts().to_dict(),
       "existing_form_by_recording_status": df[df.form_status == "existing_form"].recording_status.value_counts().to_dict()}
cnt["analysed_existing_resolved"] = int(((df.form_status == "existing_form") & df.recording_status.isin(LEVELS)).sum())
cnt["excluded_form_unresolved"] = int((df.form_status == "UNRESOLVED").sum())
cnt["excluded_recording_unresolved_in_existing"] = int(((df.form_status == "existing_form") &
                                                        (df.recording_status == "UNRESOLVED")).sum())
cnt["novel_form"] = int((df.form_status == "novel_form").sum())
R["counts"] = cnt
uni = df[df.recording_unanimous == 1]
R["counts"]["sensitivity_unanimous_recording"] = int(len(uni))
R["counts"]["sensitivity_analysed_existing_resolved"] = int(((uni.form_status == "existing_form") &
                                                             uni.recording_status.isin(LEVELS)).sum())
R["main"] = {c: analyse(df, c) for c in PULLS}
R["sensitivity_unanimous"] = {c: analyse(uni, c) for c in PULLS}
R["verdicts_main"] = verdicts(R["main"])
R["verdicts_sensitivity"] = verdicts(R["sensitivity_unanimous"])
nov = df[df.form_status == "novel_form"][["item_id", "host", "target", "tradition", "recording_status",
                                          "pull_index", "pull_bge_B", "pull_gem_B"]]
R["novel_form_items"] = nov.round(4).to_dict("records")
json.dump(R, open(ROOT / "results/recording_results.json", "w"), ensure_ascii=False, indent=1, default=str)

# ------------------------------------------------------------------ markdown report
f = lambda x, d=3: "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
fp = lambda p: "–" if p is None else ("<.001" if p < .001 else f"{p:.3f}")
L = ["# Recording-status analysis (spec v1.0, 2026-10-21)", "",
     "All numbers in this file are generated by `analyze_recording.py`. Models differ in cosine scale: compare them by "
     "% positive and test statistics, not raw means. One-sided tests are against 0 (pull > 0).", ""]
L += ["## Coding agreement (3 LLM coders)", "",
      f"Coders: {', '.join(agree['coders'])}; items coded by all three: {agree['n_items']}.", "",
      "| field | Fleiss' κ | % unanimous |", "|---|---|---|"]
for k in ["form_status", "recording_status"]:
    L.append(f"| {k} | {f(agree['fleiss_kappa'][k])} | {agree['pct_unanimous'][k]} |")
L += ["", "Consensus label distribution (217 coded items):", ""]
for k, v in agree["consensus_distribution"].items():
    L.append(f"- `{k}`: " + ", ".join(f"{a} {b}" for a, b in v.items()))
ct = agree["crosstab_recording_vs_first_cpg_valid"]
cv = sorted({c for r in ct.values() for c in r})
L += ["", "Consensus `recording_status` × first-run consensus `cpg_valid`:", "",
      "| recording_status | " + " | ".join(f"cpg_valid={c}" for c in cv) + " |", "|---" * (len(cv) + 1) + "|"]
for r, row in ct.items():
    L.append(f"| {r} | " + " | ".join(str(row.get(c, 0)) for c in cv) + " |")
c = R["counts"]
L += ["", "## Counts", "",
      f"- Items with pulls: {c['items_with_pulls']}; form_status: " + ", ".join(f"{a} {b}" for a, b in c["form_status"].items()),
      f"- Existing-form items by recording_status: " + ", ".join(f"{a} {b}" for a, b in c["existing_form_by_recording_status"].items()),
      f"- **Analysed for H1/H2: {c['analysed_existing_resolved']}** = 213 − {c['novel_form']} novel_form − "
      f"{c['excluded_form_unresolved']} form UNRESOLVED − {c['excluded_recording_unresolved_in_existing']} recording UNRESOLVED",
      f"- Sensitivity subset (all three coders agree on recording_status): {c['sensitivity_unanimous_recording']} items, "
      f"{c['sensitivity_analysed_existing_resolved']} analysed", ""]


def tables(block, ver, title):
    out = [f"## {title}", ""]
    cols = list(PULLS)
    out += ["### Descriptives by recording_status (existing_form only)", "",
            "| model | group | n | mean | SD | % pos | t | p (t) | p (Wilcoxon) |", "|---|---|---|---|---|---|---|---|---|"]
    for col in cols:
        g = block[col]["existing_by_recording_status"]
        for lv in LEVELS + [k for k in g if k not in LEVELS]:
            if lv in g:
                s = g[lv]
                out.append(f"| {col} | {lv} | {s['n']} | {f(s['mean'])} | {f(s['sd'])} | {s['pct_positive']} | "
                           f"{f(s['t'], 2)} | {fp(s['p_t_one_sided'])} | {fp(s['p_wilcoxon_one_sided'])} |")
    out += ["", "### Descriptives by form_status (all items)", "",
            "| model | form_status | n | mean | % pos | p (t) |", "|---|---|---|---|---|---|"]
    for col in cols:
        for k, s in block[col]["by_form_status"].items():
            out.append(f"| {col} | {k} | {s['n']} | {f(s['mean'])} | {s['pct_positive']} | {fp(s['p_t_one_sided'])} |")
    v = ver["H1"]
    out += ["", f"### H1 — recording gradient: **{v['verdict']}** ({v['models_passing']}/3 models pass)", "",
            "| model | n | b cultural vs unrec. (SE, p) | b general vs unrec. (SE, p) | trend slope (SE) | p one-sided | p Holm | order ok | all groups > 0 | passes |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for col in cols:
        b, t, pm = block[col], block[col]["trend"], v["per_model"][col]
        ml = b["mixed_levels"]
        lv = lambda k: "–" if k not in ml else f"{f(ml[k]['b'])} ({f(ml[k]['se'])}, {fp(ml[k]['p_two_sided'])})"
        out.append(f"| {col} | {b['n_existing_resolved']} | {lv('cultural_reference')} | {lv('general_dictionary')} | "
                   f"{f(t.get('slope'))} ({f(t.get('se'))}) | {fp(t.get('p_one_sided'))} | {fp(pm['trend_p_holm'])} | "
                   f"{'yes' if pm['order_ok'] else 'no'} | {'yes' if pm['all_groups_positive'] else 'no'} | "
                   f"{'yes' if pm['passes'] else 'no'} |")
    v = ver["H2"]
    out += ["", f"### H2 — unrecorded CPG detectable: **{v['verdict']}** "
                f"({v['models_sig_holm']}/3 models p_Holm < .05; {v['models_sig_raw']}/3 before correction)", "",
            "| model | n | traditions | mean | % pos | p (t) | p (Wilcoxon) | mixed intercept (SE) | p mixed | p Holm |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for col in cols:
        h, pm = block[col]["h2_unrecorded_existing"], v["per_model"][col]
        out.append(f"| {col} | {h['n']} | {h['n_traditions']} | {f(h['mean'])} | {h['pct_positive']} | "
                   f"{fp(h['p_t_one_sided'])} | {fp(h['p_wilcoxon_one_sided'])} | "
                   f"{f(h.get('mixed_intercept'))} ({f(h.get('mixed_se'))}) | {fp(h.get('p_mixed_one_sided'))} | "
                   f"{fp(pm['p_holm'])} |")
    v = ver["H3"]
    out += ["", f"### H3 — neologisms: {v['verdict']}", "",
            "| model | n novel_form | mean | % pos | p (t), descriptive only |", "|---|---|---|---|---|"]
    for col in cols:
        s = v["novel_form_descriptives"][col] or {}
        out.append(f"| {col} | {s.get('n', 0)} | {f(s.get('mean'))} | {s.get('pct_positive', '–')} | "
                   f"{fp(s.get('p_t_one_sided'))} |")
    return out + [""]


L += tables(R["main"], R["verdicts_main"], "Main analysis (majority consensus)")
L += tables(R["sensitivity_unanimous"], R["verdicts_sensitivity"],
            "Sensitivity (only items where all three coders agree on recording_status)")
L += ["## novel_form items (slang / not-CPG boundary discussion)", "",
      "| item_id | host | target | tradition | recording | pull OpenAI | pull bge B | pull Gemini B |",
      "|---|---|---|---|---|---|---|---|"]
for r in R["novel_form_items"]:
    L.append(f"| {r['item_id']} | {r['host']} | {r['target']} | {r['tradition']} | {r['recording_status']} | "
             f"{f(r['pull_index'])} | {f(r['pull_bge_B'])} | {f(r['pull_gem_B'])} |")
(ROOT / "results/recording_results.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ section 12 of verified_numbers.md
vm, vs = R["verdicts_main"], R["verdicts_sensitivity"]
S = ["## 12. Recording status (spec v1.0, run 2026-10-21, `analysis/analyze_recording.py`)", "",
     f"Three LLM coders ({', '.join(agree['coders'])}) coded form_status and recording_status for all 217 items. "
     f"Fleiss' κ: form_status {f(agree['fleiss_kappa']['form_status'])}, recording_status "
     f"{f(agree['fleiss_kappa']['recording_status'])}.", "",
     f"Analysed for H1/H2: {c['analysed_existing_resolved']} of 213 items with pulls ({c['novel_form']} novel_form, "
     f"{c['excluded_form_unresolved']} form UNRESOLVED, {c['excluded_recording_unresolved_in_existing']} recording "
     f"UNRESOLVED excluded).", "",
     "| hypothesis | main | sensitivity (unanimous labels) |", "|---|---|---|",
     f"| H1 recording gradient | {vm['H1']['verdict']} ({vm['H1']['models_passing']}/3) | "
     f"{vs['H1']['verdict']} ({vs['H1']['models_passing']}/3) |",
     f"| H2 unrecorded CPG > 0 | {vm['H2']['verdict']} ({vm['H2']['models_sig_holm']}/3 Holm) | "
     f"{vs['H2']['verdict']} ({vs['H2']['models_sig_holm']}/3 Holm) |",
     f"| H3 neologisms excluded | {vm['H3']['verdict']} | – |", "",
     "% positive by recording_status (existing_form), main analysis:", "",
     "| model | " + " | ".join(LEVELS) + " |", "|---" * 4 + "|"]
for col in PULLS:
    g = R["main"][col]["existing_by_recording_status"]
    S.append(f"| {col} | " + " | ".join(f"{g[lv]['pct_positive']}% (n={g[lv]['n']})" if lv in g else "–"
                                         for lv in LEVELS) + " |")
S += ["", "Full tables: `results/recording_results.md`.", ""]
vn = ROOT / "results/verified_numbers.md"
txt = vn.read_text(encoding="utf-8")
head, sep, rest = txt.partition("\n## 12. Recording status")
nxt = re.search(r"\n## \d+\. ", rest) if sep else None  # keep later sections (e.g. 13) on rerun
tail = rest[nxt.start():] if nxt else ""
txt = head.rstrip("\n") + "\n\n" + "\n".join(S).rstrip("\n") + "\n" + tail
vn.write_text(txt, encoding="utf-8")
print("\n".join(S))
