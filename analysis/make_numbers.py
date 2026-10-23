"""Generate paper/numbers.tex: every number in the manuscript, read from result files or computed from the data.

Usage: python3 analysis/make_numbers.py          (no arguments; no API calls; same output on every run)

Outputs:
  paper/numbers.tex              the macros (header: generated, do not edit; time; git commit)
  results/numbers_manifest.csv   one row per macro: macro, value, source_file, source_key_or_query, rule, source_sha256
  results/numbers_report.md      macros left as \\missing, macros that differ from the interim file, new values,
                                 consistency checks
  paper/manuscript_checks.md     hard-coded numbers and macro use in the manuscript (untracked; only if the
                                 manuscript is present in paper/)
  data/tradition_harmonisation.csv       strong candidates merged (written only if the file does not exist)
  results/harmonisation_weak_for_author.csv   weak candidates, not merged (author's decision: final)
  results/harmonisation_all_weak_merged.csv   every weak pair merged too (sensitivity only)
  results/sensitivity_tradition_unit.{json,md}   H-C1/H-C2 under raw / strong-merged / all-merged tradition labels

Rules:
  - No value is typed by hand. A value that cannot be read or computed is written as \\missing{reason}.
  - Rounding is decimal ROUND_HALF_UP of the unrounded value. Values that exist only in rounded form in a result
    file (pre-decision run, verified_stats.json) are used only when that rounding cannot change the result.
  - Values computed here are checked against the stored result files where those hold the same quantity.
  - The interim macro values (data/numbers_interim_2026-10-23.tex, copied from reports, fixed) are compared
    with the generated values; differences are listed in the report.
"""
import csv
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import unicodedata
import warnings
from collections import Counter
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import yaml
from scipy import stats
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
INTERIM = ROOT / "data/numbers_interim_2026-10-23.tex"
MODELS = {"oai": "text_embedding_3_large", "bge": "BAAI_bge_m3", "gem": "gemini_embedding_001"}
MODEL_NAMES = {"oai": "OpenAI text-embedding-3-large", "bge": "BAAI/bge-m3", "gem": "gemini-embedding-001"}
CODERS = ["gpt", "gemini", "claude"]

M = {}            # macro -> row
NOTES = []        # (c) explanations of newly computed values
CHECKS = []       # consistency checks (text, ok)
EXTRA = []        # extra markdown blocks for the report
_sha = {}


# ------------------------------------------------------------------ helpers
def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def sha(*paths):
    out = []
    for p in paths:
        p = Path(p)
        if p not in _sha:
            _sha[p] = hashlib.sha256(p.read_bytes()).hexdigest()
        out.append(_sha[p])
    return ";".join(out)


def put(name, value, src, key, rule):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    assert name not in M, name
    srcs = src if isinstance(src, (list, tuple)) else [src]
    M[name] = {"macro": name, "value": value, "source_file": ";".join(rel(s) for s in srcs),
               "source_key_or_query": key, "rule": rule, "source_sha256": sha(*srcs)}


def missing(name, reason, src, key, rule):
    put(name, r"\missing{" + reason + "}", src, key, rule)


def check(text, ok):
    CHECKS.append((text, bool(ok)))
    return ok


def hu(x, places):
    """ROUND_HALF_UP of an unrounded float, as Decimal."""
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def safe_from_rounded(x, stored_places, places):
    """A value stored with `stored_places` decimals can be re-rounded to `places` only if the stored value is not
    exactly half-way at the target precision (then the unrounded value could fall on either side)."""
    d = Decimal(repr(float(x)))
    rem = (d.scaleb(places) - d.scaleb(places).to_integral_value(rounding="ROUND_FLOOR"))
    return not (stored_places > places and rem == Decimal("0.5"))


def f_int(n):
    n = int(n)
    s = f"{n:,}"
    return s.replace(",", "{,}") if n >= 1000 else s


def f_dec(x, places):
    d = hu(x, places)
    if d == 0:
        d = abs(d)
    s = f"{d:.{places}f}"
    return (r"\ensuremath{-}" + s[1:]) if s.startswith("-") else s


def f_p(p):
    if p < .001:
        return "< .001"
    s = f"{hu(p, 3):.3f}"
    return "= " + (s[1:] if s.startswith("0") else s)


RULE = {"int": "count, integer; {,} from 1,000",
        "k2": "kappa, 2 decimals, ROUND_HALF_UP",
        "icc3": "ICC, 3 decimals, ROUND_HALF_UP",
        "z2": "z, 2 decimals, ROUND_HALF_UP",
        "p": "p, APA: '< .001' if p < .001, else '= .xxx' (3 decimals, ROUND_HALF_UP, no leading 0)",
        "pct1": "percent, 1 decimal, ROUND_HALF_UP, no % sign",
        "m3": "mean pull / intercept, 3 decimals, ROUND_HALF_UP, minus sign only when negative",
        "rho": "Spearman rho range min--max, 2 decimals, ROUND_HALF_UP",
        "empty": "empty string if the check passes, else \\missing{...}"}


def read_csv(p, **kw):
    return pd.read_csv(ROOT / p, encoding="utf-8-sig", **kw)


def fleiss_unrounded(cols):
    table, _ = aggregate_raters(np.array(cols).T)
    return float(fleiss_kappa(table, method="fleiss"))


def cohen(a, b):
    a, b = np.asarray(a, dtype=str), np.asarray(b, dtype=str)
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in set(a) | set(b))
    return (po - pe) / (1 - pe)


def coders(path_fmt, require_ok):
    out = []
    for c in CODERS:
        d = read_csv(path_fmt.format(c=c), dtype=str)
        if require_ok:
            assert (d.status == "ok").all()
        else:
            d = d[d.status == "ok"]
        out.append(d.drop_duplicates("item_id", keep="last").set_index("item_id"))
    ids = sorted(set.intersection(*[set(d.index) for d in out]))
    return out, ids


def mixed(d, col, group="tradition"):
    m = smf.mixedlm(f"{col} ~ 1", d, groups=d[group]).fit(reml=True)
    var_u, var_e = float(m.cov_re.iloc[0, 0]), float(m.scale)
    return {"b": float(m.params["Intercept"]), "z": float(m.params["Intercept"] / m.bse["Intercept"]),
            "icc": var_u / (var_u + var_e)}


# ------------------------------------------------------------------ 1. data
F_BLIND = ROOT / "data/blind_items.csv"
F_TOR = ROOT / "data/source/torami_103_with_tradition.csv"
F_YAML = sorted((ROOT / "data/source").glob("ashigaru*.yaml"))
F_EXCL = ROOT / "data/exclusions.csv"
F_REC = ROOT / "coding/results/recording/recording_consensus.csv"
F_V11 = ROOT / "coding/results/v1_1/v1_1_consensus.csv"
F_T1 = ROOT / "results/t1_results.json"
F_T1PRE = ROOT / "results/before_decisions_2026-10-22/t1_results.json"

blind = read_csv("data/blind_items.csv")
tor = read_csv("data/source/torami_103_with_tradition.csv")
yaml_ids = [it["id"] for f in F_YAML for it in yaml.safe_load(open(f, encoding="utf-8"))["items"]]
check("Phase-1 + Phase-2 items = blind items (217), ids identical",
      len(tor) + len(yaml_ids) == len(blind)
      and set(yaml_ids) | {"T_%03d" % int(i) for i in tor["id"]} == set(blind.item_id))
put("Nitems", f_int(len(tor) + len(yaml_ids)), [F_TOR, *F_YAML], "rows of torami_103 + items of ashigaru*.yaml",
    RULE["int"])
put("NphaseOne", f_int(len(tor)), F_TOR, "rows", RULE["int"])
put("NphaseTwo", f_int(len(yaml_ids)), F_YAML, "len(items) summed over the 7 YAML files", RULE["int"])

# traditions / languages: strong candidates are merged (author's decision of 2026-10-23), weak ones are not
F_HARM = ROOT / "data/tradition_harmonisation.csv"
F_WEAK = ROOT / "results/harmonisation_weak_for_author.csv"
# ISO codes whose language name is unambiguous; a code is merged only with a label that is exactly that name
ISO = {"ja": "japanese", "en": "english", "zh": "chinese", "hi": "hindi", "fa": "persian", "fr": "french"}
STEPS = [("parenthetical", lambda s: re.sub(r"\s*\([^()]*\)", "", s).strip()),
         ("subtitle", lambda s: re.split(r"\s+—\s+", s)[0].strip()),
         ("diacritics", lambda s: "".join(c for c in unicodedata.normalize("NFKD", s)
                                          if not unicodedata.combining(c) and c not in "ʻ'’ʼ")),
         ("case", lambda s: re.sub(r"\s+", " ", s).strip().casefold()),
         ("iso_code", lambda s: ISO.get(s, s))]
STEPS_FOR = {"tradition": [k for k, _ in STEPS if k != "iso_code"], "language": [k for k, _ in STEPS if k != "subtitle"]}


def hkey(kind, lab):
    for k, f in STEPS:
        if k in STEPS_FOR[kind]:
            lab = f(lab)
    return lab


def hrule(kind, a, b):
    """steps, in order, that are needed before a and b become equal"""
    used = []
    for k, f in STEPS:
        if a == b:
            break
        if k in STEPS_FOR[kind]:
            a2, b2 = f(a), f(b)
            if (a2, b2) != (a, b):
                used.append(k)
            a, b = a2, b2
    return "+".join(used) if used else "unchanged"


def weak_relations(kind, a, b):
    """weak links between two labels that are not merged: never used for merging"""
    out = []
    segs = lambda s: {hkey(kind, x) for x in re.split(r"/", hkey(kind, s)) if x.strip()} | {hkey(kind, s)}
    sa, sb = segs(a), segs(b)
    if (sa - {hkey(kind, a)}) & (sb - {hkey(kind, b)}) or hkey(kind, a) in sb or hkey(kind, b) in sa:
        out.append("slash_segment")
    if any(re.search(rf"\b{re.escape(x)}\b", y) or re.search(rf"\b{re.escape(y)}\b", x) for x in sa for y in sb):
        out.append("contains")
    elif any(difflib.SequenceMatcher(None, x, y).ratio() >= 0.85 for x in sa for y in sb):
        out.append("near")
    return out


hrows, wrows, harm, weak_pairs = [], [], {}, []
for kind in ["tradition", "language"]:
    vc = blind[kind].value_counts()
    groups = {}
    for lab in vc.index:
        groups.setdefault(hkey(kind, lab), []).append(lab)
    target = {}
    for labs in groups.values():
        best = sorted(labs, key=lambda x: (-vc[x], len(STEPS[0][1](x)), len(x), x))[0]
        for lab in labs:
            target[lab] = best
    for lab in sorted(vc.index, key=lambda x: (target[x], x)):
        hrows.append({"kind": kind, "original_label": lab, "harmonised_label": target[lab],
                      "rule": hrule(kind, lab, target[lab])})
    pairs = 0
    for lab in vc.index:
        cands = []
        for o in vc.index:
            if o != lab and target[o] != target[lab]:
                rel_ = weak_relations(kind, lab, o)
                if rel_:
                    cands.append(f"{o} [{'+'.join(rel_)}]")
                    if lab < o:
                        weak_pairs.append((kind, lab, o, "+".join(rel_)))
                    pairs += 1
        if cands:
            wrows.append({"kind": kind, "label": lab, "n": int(vc[lab]), "harmonised_label": target[lab],
                          "weak_candidates": "; ".join(cands)})
    harm[kind] = {"before": len(vc), "after": len(set(target.values())),
                  "merged_labels": sum(1 for l in vc.index if len(groups[hkey(kind, l)]) > 1),
                  "weak_labels": sum(1 for r in wrows if r["kind"] == kind), "weak_pairs": pairs // 2}
rule_table = pd.DataFrame(hrows)
if F_HARM.exists():
    ht = read_csv("data/tradition_harmonisation.csv").fillna("")
    check("data/tradition_harmonisation.csv covers every tradition and language label of blind_items.csv exactly once",
          all(sorted(ht[ht.kind == k].original_label) == sorted(blind[k].unique()) for k in ["tradition", "language"]))
    edited = ht.merge(rule_table, on=["kind", "original_label"], suffixes=("", "_rule"))
    edited = edited[edited.harmonised_label != edited.harmonised_label_rule]
    NOTES.append(f"`data/tradition_harmonisation.csv` exists and is used as it is; rows that differ from the "
                 f"strong-candidate rule: {len(edited)}"
                 + ("" if edited.empty else " (" + "; ".join(f"{r.kind}: {r.original_label} → {r.harmonised_label}"
                                                           for r in edited.itertuples()) + ")") + ".")
else:
    ht = rule_table
    ht.to_csv(F_HARM, index=False, encoding="utf-8")
    NOTES.append("`data/tradition_harmonisation.csv` did not exist and was written from the strong-candidate rule.")
pd.DataFrame(wrows, columns=["kind", "label", "n", "harmonised_label", "weak_candidates"]).to_csv(
    F_WEAK, index=False, encoding="utf-8")
for macro, kind in [("NtradLabels", "tradition"), ("Nlanguages", "language")]:
    x = ht[ht.kind == kind]
    put(macro, f_int(x.harmonised_label.nunique()), [F_HARM, F_BLIND],
        f"nunique(harmonised_label) where kind == '{kind}' (labels of the 217 blind items; strong candidates "
        f"merged, weak ones not)", RULE["int"])

# sensitivity: every weak pair merged as well (connected components), except pairs known to be false matches.
# Not an input: written to results/ only (author's decision of 2026-10-23: weak candidates are not merged).
F_ALLW = ROOT / "results/harmonisation_all_weak_merged.csv"
FALSE_WEAK = {("language", "ja", "Javanese"): "ISO code ja is Japanese, not Javanese",
              ("language", "Old English", "en"): "Old English is not (modern) English"}
strong = {(r.kind, r.original_label): r.harmonised_label for r in ht.itertuples()}
parent = {}


def find(x):
    while parent.setdefault(x, x) != x:
        x = parent[x]
    return x


dropped, used_pairs = [], 0
for kind, a, b, why in weak_pairs:
    ta, tb = strong[(kind, a)], strong[(kind, b)]
    fp = FALSE_WEAK.get((kind, ta, tb)) or FALSE_WEAK.get((kind, tb, ta))
    if fp:
        dropped.append(f"{kind}: {a} — {b} [{why}]: {fp}")
        continue
    used_pairs += 1
    parent[find((kind, ta))] = find((kind, tb))
check(f"weak pairs used for the all-merged table ({used_pairs}) + dropped false matches ({len(dropped)}) = weak pairs "
      f"in results/harmonisation_weak_for_author.csv ({sum(v['weak_pairs'] for v in harm.values())})",
      used_pairs + len(dropped) == sum(v["weak_pairs"] for v in harm.values()) == len(weak_pairs))
arows, allweak = [], {}
for kind in ["tradition", "language"]:
    n_by = blind[kind].map(lambda l: strong[(kind, l)]).value_counts()
    comp = {}
    for t in n_by.index:
        comp.setdefault(find((kind, t)), []).append(t)
    for members in comp.values():
        best = sorted(members, key=lambda x: (-n_by[x], len(STEPS[0][1](x)), len(x), x))[0]
        for t in members:
            for (k2, o), t2 in strong.items():
                if k2 == kind and t2 == t:
                    allweak[(kind, o)] = best
                    arows.append({"kind": kind, "original_label": o, "strong_label": t, "all_weak_label": best})
aw = pd.DataFrame(arows).sort_values(["kind", "all_weak_label", "original_label"])
aw.to_csv(F_ALLW, index=False, encoding="utf-8")
for macro, kind in [("NtradLabelsMin", "tradition"), ("NlanguagesMin", "language")]:
    put(macro, f_int(aw[aw.kind == kind].all_weak_label.nunique()), [F_HARM, F_BLIND],
        f"nunique(all_weak_label) where kind == '{kind}': strong candidates and every weak pair merged (connected "
        f"components), known false matches not merged; table in results/harmonisation_all_weak_merged.csv",
        RULE["int"])
EXTRA.append(("Weak pairs not merged even in the all-merged count (known false matches)",
              [f"- {x}" for x in dropped] or ["none"]))
EXTRA.append(("All-merged groups (sensitivity only)", [
    f"- {k}: {t} ← " + "; ".join(sorted(set(g.strong_label) - {t}))
    for (k, t), g in aw.groupby(["kind", "all_weak_label"]) if g.strong_label.nunique() > 1]))

rec = read_csv("coding/results/recording/recording_consensus.csv")
check("recording_consensus has 217 items", len(rec) == 217)
for macro, lab in [("NrecCultural", "cultural_reference"), ("NrecGeneral", "general_dictionary"),
                   ("NrecUnrecorded", "unrecorded"), ("NrecUnresolved", "UNRESOLVED")]:
    put(macro, f_int((rec.recording_status == lab).sum()), F_REC, f"count(recording_status == '{lab}'), 217 items",
        RULE["int"])
check("recording labels sum to 217", sum(int(M[m]["value"]) for m in
                                         ["NrecCultural", "NrecGeneral", "NrecUnrecorded", "NrecUnresolved"]) == 217)

excl = read_csv("data/exclusions.csv")
check("every excluded item is a blind item", set(excl.item_id) <= set(blind.item_id))
put("Nexcluded", f_int(len(excl)), F_EXCL, "rows", RULE["int"])
EXCL_KIND = [("NexclNoHost", "no host form", lambda r: "no host form" in r),
             ("NexclMultiword", "several words listed", lambda r: "several different words listed" in r)]
left = excl.copy()
for macro, what, f in EXCL_KIND:
    hit = left.reason.map(f)
    put(macro, f_int(int(hit.sum())), F_EXCL, f"count(reason contains '{what if what != 'several words listed' else 'several different words listed'}')",
        RULE["int"])
    left = left[~hit]
put("NexclOther", f_int(len(left)), F_EXCL, "rows not counted in NexclNoHost or NexclMultiword", RULE["int"])
check(f"NexclNoHost + NexclMultiword + NexclOther = Nexcluded ({len(excl)})",
      sum(int(M[m]["value"]) for m in ["NexclNoHost", "NexclMultiword", "NexclOther"]) == len(excl))
EXTRA.append(("T3 exclusions by reason (data/exclusions.csv)", ["| item | counted in | reason |", "|---|---|---|"] + [
    f"| {r.item_id} | " + next((m for m, _, f in EXCL_KIND if f(r.reason)), "NexclOther") + f" | {r.reason} |"
    for r in excl.itertuples()]))
put("NafterExcl", f_int(len(blind) - len(excl)), [F_BLIND, F_EXCL], "rows(blind_items) - rows(exclusions)", RULE["int"])
check(f"NafterExcl = 217 - 6 = 211 (got {len(blind)} - {len(excl)} = {len(blind) - len(excl)})",
      len(blind) - len(excl) == 211)

t1 = json.load(open(F_T1))
c = t1["counts"]
hc1, hc2 = t1["hypotheses"]["H-C1"]["per_model"], t1["hypotheses"]["H-C2"]["per_model"]
put("NwithControls", f_int(c["items_with_controls"]), F_T1, "counts.items_with_controls", RULE["int"])
put("NcontrolsAfterExcl", f_int(c["after_T3"]), F_T1, "counts.after_T3 (items, not controls)", RULE["int"])
put("NnotLowControl", f_int(c["after_low_control"]), F_T1, "counts.after_low_control", RULE["int"])
n10 = {hc1[MODELS[m]]["n"] for m in MODELS}
nv = {hc2[MODELS[m]]["n"] for m in MODELS}
check("H-C1 n is the same in the three models", len(n10) == 1)
check("H-C2 n is the same in the three models", len(nv) == 1)
put("NkTen", f_int(n10.pop()), F_T1, "hypotheses.H-C1.per_model.*.n", RULE["int"])
put("NkTenValid", f_int(nv.pop()), F_T1, "hypotheses.H-C2.per_model.*.n", RULE["int"])

v11 = read_csv("coding/results/v1_1/v1_1_consensus.csv")
v11x = v11[~v11.item_id.isin(excl.item_id)]
nfe, nft = int((v11x.form_type == "existing_form").sum()), int((v11x.form_type == "transparent_combination").sum())
put("NformExisting", f_int(nfe), [F_V11, F_EXCL], "count(form_type == 'existing_form'), items not in exclusions",
    RULE["int"])
put("NformTransparent", f_int(nft), [F_V11, F_EXCL],
    "count(form_type == 'transparent_combination'), items not in exclusions", RULE["int"])
check(f"form_type after T3: {nfe} + {nft} = {len(v11x)} items (211)", nfe + nft == len(v11x) == 211)
NOTES.append(f"`NformExisting`, `NformTransparent`: v1.1 consensus `form_type` of the {len(v11x)} items left after the "
             f"T3 exclusions: existing_form {nfe}, transparent_combination {nft}; other levels: "
             + (", ".join(f"{k} {v}" for k, v in v11x.form_type.value_counts().items()
                          if k not in ("existing_form", "transparent_combination")) or "none") + ".")

# ------------------------------------------------------------------ 2. controls
cc = c["control_cleaning"]["all_items_with_controls"]
put("NcontrolsTotal", f_int(cc["controls"]), F_T1, "counts.control_cleaning.all_items_with_controls.controls",
    RULE["int"])
for macro, r in [("NdropRone", "R1"), ("NdropRtwo", "R2"), ("NdropRthree", "R3"), ("NdropRfour", "R4")]:
    put(macro, f_int(cc[r]), F_T1, f"counts.control_cleaning.all_items_with_controls.{r}", RULE["int"])
put("NcontrolsKept", f_int(cc["kept"]), F_T1, "counts.control_cleaning.all_items_with_controls.kept", RULE["int"])
check(f"controls {cc['controls']} = kept {cc['kept']} + R1-R4 {cc['R1'] + cc['R2'] + cc['R3'] + cc['R4']}",
      cc["controls"] == cc["kept"] + cc["R1"] + cc["R2"] + cc["R3"] + cc["R4"])
summ = read_csv("embeddings/controls/controls_summary.csv")
check("control counts in t1_results.json = controls_summary.csv",
      int(summ.n_original.sum()) == cc["controls"] and int(summ.n_kept.sum()) == cc["kept"])
ca = c["control_cleaning"]["after_T3"]
NOTES.append(f"`NcontrolsKept`: controls kept after cleaning, all {cc['items']} items with controls (the same base as "
             f"`NcontrolsTotal` and `NdropR*`): {cc['kept']}. For the {ca['items']} items left after T3 the figures are "
             f"{ca['controls']} controls, {ca['kept']} kept (R1 {ca['R1']}, R2 {ca['R2']}, R3 {ca['R3']}, R4 {ca['R4']}).")

# ------------------------------------------------------------------ 3. coding agreement (recomputed unrounded)
F_AG1 = ROOT / "coding/results/first_run/agreement.json"
F_AGR = ROOT / "coding/results/recording/recording_agreement.json"
F_AG11 = ROOT / "coding/results/v1_1/v1_1_agreement.json"
F_CONS1 = ROOT / "coding/results/first_run/consensus_codes.csv"
c1, ids1 = coders("coding/results/first_run/codes_{c}.csv", require_ok=False)
ag1 = json.load(open(F_AG1))
src1 = [ROOT / f"coding/results/first_run/codes_{x}.csv" for x in CODERS]


def kappa_macro(macro, value, stored, src, key):
    check(f"{macro}: recomputed kappa {value:.4f} rounds to stored {stored}", round(value, 3) == stored)
    put(macro, f_dec(value, 2), src, key, RULE["k2"])


for macro, field in [("KdirRunOne", "direction"), ("KchanRunOne", "primary_channel"), ("KvalidRunOne", "cpg_valid"),
                     ("KactivRunOne", "activation"), ("KsJux", "s_juxtaposition"), ("KsNoncomp", "s_noncompositional"),
                     ("KsConv", "s_conventional")]:
    k = fleiss_unrounded([d.loc[ids1, field].astype(str).values for d in c1])
    kappa_macro(macro, k, ag1["fleiss_kappa"][field], src1,
                f"Fleiss kappa of '{field}', 3 coders (stored: agreement.json fleiss_kappa.{field})")
for macro, ch in [("KphonChan", "Phonetic"), ("KstructChan", "Structural")]:
    cols = [d.loc[ids1, "channels"].map(json.loads).map(lambda s, ch=ch: int(ch in s)).values for d in c1]
    kappa_macro(macro, fleiss_unrounded(cols), ag1["fleiss_kappa_per_channel"][ch], src1,
                f"Fleiss kappa of '{ch} in channels', 3 coders (stored: agreement.json fleiss_kappa_per_channel.{ch})")

cr, idsr = coders("coding/results/recording/recording_{c}.csv", require_ok=True)
agr = json.load(open(F_AGR))
for macro, field in [("KformStatus", "form_status"), ("KrecStatus", "recording_status")]:
    kappa_macro(macro, fleiss_unrounded([d.loc[idsr, field].astype(str).values for d in cr]),
                agr["fleiss_kappa"][field], [ROOT / f"coding/results/recording/recording_{x}.csv" for x in CODERS],
                f"Fleiss kappa of '{field}' (stored: recording_agreement.json fleiss_kappa.{field})")

cv, idsv = coders("coding/results/v1_1/v1_1_{c}.csv", require_ok=True)
ag11 = json.load(open(F_AG11))
for macro, field in [("KformType", "form_type"), ("KtwoStorey", "two_storey"), ("KcontentAttitude", "content_or_attitude"),
                     ("KoutsiderInf", "outsider_inference"), ("KhostUnit", "host_unit")]:
    kappa_macro(macro, fleiss_unrounded([d.loc[idsv, field].astype(str).values for d in cv]),
                ag11["fleiss_kappa"][field], [ROOT / f"coding/results/v1_1/v1_1_{x}.csv" for x in CODERS],
                f"Fleiss kappa of '{field}' (stored: v1_1_agreement.json fleiss_kappa.{field})")

cons1 = read_csv("coding/results/first_run/consensus_codes.csv")
for kmacro, pmacro, field in [("KdirVsOrig", "PdirVsOrig", "direction"), ("KchanVsOrig", "PchanVsOrig", "primary_channel")]:
    x = cons1[(cons1[field] != "UNRESOLVED") & cons1[f"original_{field}"].notna()]
    k, pa = cohen(x[field], x[f"original_{field}"]), 100 * float(np.mean(x[field].values == x[f"original_{field}"].values))
    st = ag1["consensus_vs_original_cohen_kappa"][field]
    check(f"{kmacro}: Cohen kappa {k:.4f} (n = {len(x)}) matches stored {st['kappa']} (n = {st['n']})",
          round(k, 3) == st["kappa"] and len(x) == st["n"])
    check(f"{pmacro}: % agreement {pa:.2f} matches stored {st['pct_agree']}", round(pa, 1) == st["pct_agree"])
    q = f"consensus '{field}' vs 'original_{field}', items with consensus != UNRESOLVED (n = {len(x)})"
    put(kmacro, f_dec(k, 2), F_CONS1, "Cohen kappa, " + q, RULE["k2"])
    put(pmacro, f_dec(pa, 1), F_CONS1, "% agreement, " + q, RULE["pct1"])

# ------------------------------------------------------------------ 4. dictionary check
F_DICT = ROOT / "dictionary_check/daijisen_judgments.csv"
dj = read_csv("dictionary_check/daijisen_judgments.csv")
dx = dj.merge(rec, on="item_id", how="left")
check("every dictionary item has a recording consensus", dx.recording_status.notna().all())
llm_gd = dx.recording_status.eq("general_dictionary")
len_ = dx["判定"].eq("番号つきの語義")
strict = len_ & dx["一致度"].eq("一致")
put("NdictItems", f_int(len(dj)), F_DICT, "rows", RULE["int"])
for pm, km, dic, txt in [("PdictAgreeLenient", "KdictLenient", len_, "判定 == 番号つきの語義 (一致 or 近い)"),
                         ("PdictAgreeStrict", "KdictStrict", strict, "判定 == 番号つきの語義 and 一致度 == 一致")]:
    q = f"dictionary recorded := {txt}; LLM recorded := consensus recording_status == general_dictionary"
    put(pm, f_dec(100 * float((dic == llm_gd).mean()), 1), [F_DICT, F_REC], "% agreement; " + q, RULE["pct1"])
    put(km, f_dec(cohen(dic, llm_gd), 2), [F_DICT, F_REC], "Cohen kappa; " + q, RULE["k2"])
put("NagentAgree", f_int(dx.recording_unanimous.sum()), [F_DICT, F_REC],
    "count(recording_unanimous == 1) among the dictionary items (all three coders gave the same recording_status)",
    RULE["int"])
NOTES.append("Dictionary check: no stored script computed these figures. The definitions in `source_key_or_query` "
             "reproduce the interim values (85, 0.69, 78, 0.52, 33). `NagentAgree` is read as 'the three coders agree "
             "on recording_status'. Another reading gives the same number: the dictionary and the coders agree on "
             f"{int((dx['判定'].isin(['番号つきの語義', '補説・用例']) == llm_gd).sum())} items when 補説・用例 also counts "
             "as recorded. Please check which meaning the text uses.")

# ------------------------------------------------------------------ 5. condition C, main set (recomputed)
summ_k = summ[["item_id", "n_kept", "low_control"]]
first = cons1[["item_id", "cpg_valid", "direction", "primary_channel"]]
D = {}
for m, tag in MODELS.items():
    cdf = read_csv(f"embeddings/results/nnpm_C_{tag}.csv")
    bdf = read_csv(f"embeddings/results/nnpm_{tag}.csv")[["id", "pull_B_k10"]].rename(columns={"id": "item_id"})
    d = (cdf.merge(bdf, on="item_id", how="left").merge(summ_k, on="item_id", how="left")
         .merge(first, on="item_id", how="left").merge(v11[["item_id", "outsider_inference"]], on="item_id", how="left"))
    d = d[~d.item_id.isin(excl.item_id) & (d.low_control == 0)]
    D[m] = d.dropna(subset=["pull_C_k10"]).sort_values("item_id").reset_index(drop=True)
F_C = [ROOT / f"embeddings/results/nnpm_C_{t}.csv" for t in MODELS.values()]
F_B = [ROOT / f"embeddings/results/nnpm_{t}.csv" for t in MODELS.values()]
F_MAIN = F_C + [ROOT / "embeddings/controls/controls_summary.csv", F_EXCL]
ids = [tuple(D[m].item_id) for m in MODELS]
check("main set (condition C, k = 10, T3 and low_control removed) has the same items in the three models",
      len(set(ids)) == 1)
check(f"main set n = NkTen ({len(D['oai'])})", str(len(D["oai"])) == M["NkTen"]["value"])
MAIN_Q = "main set: not in exclusions, low_control == 0, pull_C_k10 present (n = %d)" % len(D["oai"])

fit = {}
for hyp, sel in [("H-C1", lambda d: d), ("H-C2", lambda d: d[d.cpg_valid == "yes"])]:
    res = {m: mixed(sel(D[m]), "pull_C_k10") for m in MODELS}
    pr = [stats.norm.sf(res[m]["z"]) for m in MODELS]
    for m, a in zip(MODELS, multipletests(pr, method="holm")[1]):
        res[m]["p_holm"] = float(a)
        st = t1["hypotheses"][hyp]["per_model"][MODELS[m]]
        check(f"{hyp} {m}: recomputed z {res[m]['z']:.4f}, intercept {res[m]['b']:.4f}, Holm p {a:.4f} match "
              f"t1_results.json", round(res[m]["z"], 4) == st["z"] and round(res[m]["b"], 4) == st["intercept"]
              and round(a, 4) == st["p_holm"] and len(sel(D[m])) == st["n"])
    fit[hyp] = res
for hyp, pre in [("H-C1", "HCone"), ("H-C2", "HCtwo")]:
    sub = MAIN_Q + ("" if hyp == "H-C1" else ", first-run consensus cpg_valid == yes")
    for m in MODELS:
        r = fit[hyp][m]
        put(f"{pre}Z{m}", f_dec(r["z"], 2), F_MAIN + [F_CONS1],
            f"mixedlm pull_C_k10 ~ 1 + (1|tradition), REML; z of intercept; {sub} (= t1_results.json "
            f"hypotheses.{hyp}.per_model.{MODELS[m]}.z)", RULE["z2"])
        put(f"{pre}P{m}", f_p(r["p_holm"]), F_MAIN + [F_CONS1],
            f"one-sided normal p of that z, Holm across the 3 models (= hypotheses.{hyp}.per_model.{MODELS[m]}.p_holm)",
            RULE["p"])
    if hyp == "H-C1":
        for m in MODELS:
            put(f"HConeB{m}", f_dec(fit[hyp][m]["b"], 3), F_MAIN,
                f"intercept of the H-C1 mixed model (= hypotheses.H-C1.per_model.{MODELS[m]}.intercept)", RULE["m3"])
# macro order as in the interim file: HConeZ*, HConeP*, HConeB*, HCtwoZ*, HCtwoP* (sorting happens at write time)

# sensitivity: unit of the tradition random intercept. The pipeline's tradition labels (used in fit above and in
# t1_results.json) correspond one-to-one to the raw blind_items labels, so the main analysis groups by raw labels.
traw = blind.set_index("item_id").tradition
for m in MODELS:
    D[m]["trad_raw"] = D[m].item_id.map(traw)
    D[m]["trad_strong"] = D[m].trad_raw.map(lambda l: strong[("tradition", l)])
    D[m]["trad_all"] = D[m].trad_raw.map(lambda l: allweak[("tradition", l)])
    check(f"main set {m}: pipeline tradition labels correspond one-to-one to raw blind labels",
          D[m].groupby("tradition").trad_raw.nunique().max() == 1 and D[m].groupby("trad_raw").tradition.nunique().max() == 1)
GROUPINGS = [("raw labels", "trad_raw"), ("strong candidates merged", "trad_strong"), ("all weak merged", "trad_all")]
sens = []
for gname, gcol in GROUPINGS:
    for hyp, sel in [("H-C1", lambda d: d), ("H-C2", lambda d: d[d.cpg_valid == "yes"])]:
        res = {m: mixed(sel(D[m]), "pull_C_k10", gcol) for m in MODELS}
        adj = multipletests([stats.norm.sf(res[m]["z"]) for m in MODELS], method="holm")[1]
        for m, a in zip(MODELS, adj):
            sens.append({"grouping": gname, "hypothesis": hyp, "model": MODEL_NAMES[m], "n": int(len(sel(D[m]))),
                         "n_groups": int(sel(D[m])[gcol].nunique()), "intercept": round(res[m]["b"], 4),
                         "z": round(res[m]["z"], 4), "p_holm": float(a), "verdict": "supported" if a < .05 else
                         "not supported"})
ref = {(r["hypothesis"], r["model"]): r["verdict"] for r in sens if r["grouping"] == "raw labels"}
for r in sens:
    st = t1["hypotheses"][r["hypothesis"]]["per_model"][MODELS[[k for k, v in MODEL_NAMES.items() if v == r["model"]][0]]]
    if r["grouping"] == "raw labels":
        check(f"sensitivity, raw labels, {r['hypothesis']} {r['model']}: z {r['z']} = t1_results.json, n_groups "
              f"{r['n_groups']} = n_traditions", r["z"] == st["z"] and r["n_groups"] == st["n_traditions"])
changed = [r for r in sens if r["verdict"] != ref[(r["hypothesis"], r["model"])]]
F_SENS = ROOT / "results/sensitivity_tradition_unit"
json.dump({"description": "H-C1 and H-C2 on the main set (condition C, k = 10, T3 and low_control removed), "
                          "random intercept for tradition under three groupings; Holm across the 3 models within each "
                          "grouping and hypothesis; verdict = supported if p_holm < .05. The main analysis "
                          "(t1_results.json) groups by the raw labels.",
           "rows": sens, "verdict_changes": changed}, open(F_SENS.with_suffix(".json"), "w"), indent=1,
          ensure_ascii=False)
cols = ["grouping", "hypothesis", "model", "n", "n_groups", "intercept", "z", "p_holm", "verdict"]
md = ["# Sensitivity: unit of the tradition random intercept (generated by analysis/make_numbers.py)", "",
      "Main set: condition C, k = 10, T3 exclusions and low_control removed. Mixed model pull_C_k10 ~ 1 + "
      "(1|tradition), REML; one-sided Wald z; Holm across the 3 models; supported if p_holm < .05.", "",
      "Groupings: raw labels = the labels in data/blind_items.csv (the main analysis in t1_results.json uses these); "
      "strong candidates merged = data/tradition_harmonisation.csv; all weak merged = "
      "results/harmonisation_all_weak_merged.csv.", "",
      "**Verdict changes against the raw-label grouping: " + (
          "none.**" if not changed else str(len(changed)) + "** — " + "; ".join(
              f"{r['grouping']}, {r['hypothesis']}, {r['model']}: {r['verdict']}" for r in changed)), "",
      "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
md += ["| " + " | ".join(f"{r[c]:.4f}" if c in ("intercept", "z", "p_holm") else str(r[c]) for c in cols) + " |"
       for r in sens]
F_SENS.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
q = "verdicts (p_holm < .05) of H-C1 and H-C2, 3 models, identical under raw / strong-merged / all-merged tradition " \
    "labels; results/sensitivity_tradition_unit.json"
if changed:
    missing("SensTradSame", "verdict changes: " + "; ".join(f"{r['grouping']}, {r['hypothesis']}, {r['model']}"
                                                          for r in changed), F_MAIN + [F_HARM, F_ALLW], q, RULE["empty"])
else:
    put("SensTradSame", "", F_MAIN + [F_HARM, F_ALLW], q, RULE["empty"])


def rho_range(col):
    v = []
    for a, b in [("oai", "bge"), ("oai", "gem"), ("bge", "gem")]:
        x = D[a][["item_id", col]].merge(D[b][["item_id", col]], on="item_id").dropna()
        v.append((a, b, len(x), float(stats.spearmanr(x.iloc[:, 1], x.iloc[:, 2]).statistic)))
    return v


rc_ = rho_range("pull_C_k10")
for a, b, n, r in rc_:
    st = t1["spearman_between_models_C_k10"][f"{MODELS[a]} ~ {MODELS[b]}"]
    check(f"Spearman C {a}~{b}: {r:.4f} (n = {n}) matches t1_results.json", round(r, 4) == st["rho"] and n == st["n"])
rb_ = rho_range("pull_B_k10")
check("condition-B pulls present for every main-set item", all(D[m].pull_B_k10.notna().all() for m in MODELS))
for macro, v, col, srcs in [("RhoRange", rc_, "pull_C_k10", F_MAIN), ("RhoRangeB", rb_, "pull_B_k10", F_MAIN + F_B)]:
    lo, hi = min(r for *_, r in v), max(r for *_, r in v)
    put(macro, f"{f_dec(lo, 2)}--{f_dec(hi, 2)}", srcs,
        f"Spearman rho of {col} between the 3 model pairs, {MAIN_Q}; min--max", RULE["rho"])
    NOTES.append(f"`{macro}`: " + ", ".join(f"{a}~{b} {r:.4f} (n = {n})" for a, b, n, r in v) + ".")

for m in MODELS:
    d = D[m]
    put(f"MeanBsame{m.capitalize()}", f_dec(d.pull_B_k10.mean(), 3), F_MAIN + F_B,
        f"mean pull_B_k10 ({MODEL_NAMES[m]}), {MAIN_Q} (= t1_results.json analyses.{MODELS[m]}.B_vs_C.same_items.B_k10.mean)",
        RULE["m3"])
    put(f"MeanCsame{m.capitalize()}", f_dec(d.pull_C_k10.mean(), 3), F_MAIN,
        f"mean pull_C_k10 ({MODEL_NAMES[m]}), {MAIN_Q} (= analyses.{MODELS[m]}.B_vs_C.same_items.C_k10.mean)",
        RULE["m3"])
    st = t1["analyses"][MODELS[m]]["B_vs_C"]["same_items"]
    check(f"same items {m}: n {len(d)}, mean B {d.pull_B_k10.mean():.4f}, mean C {d.pull_C_k10.mean():.4f} match "
          f"t1_results.json", st["n"] == len(d) and round(d.pull_B_k10.mean(), 4) == st["B_k10"]["mean"]
          and round(d.pull_C_k10.mean(), 4) == st["C_k10"]["mean"])

# outsider_inference
lv = D["oai"].outsider_inference.value_counts()
oi_rows = ["| outsider_inference | n | " + " | ".join(f"mean pull {m}" for m in MODELS) + " |",
           "|---|---|" + "---|" * len(MODELS)]
for lev in lv.index:
    oi_rows.append(f"| {lev} | {lv[lev]} | " + " | ".join(
        f"{D[m].loc[D[m].outsider_inference == lev, 'pull_C_k10'].mean():.4f}" for m in MODELS) + " |")
EXTRA.append(("Outsider inference levels in the main set (v1.1 consensus; condition C, k = 10)", oi_rows))
for macro_n, prefix, lev in [("NoiInf", "PullInf", "inferable"), ("NoiNot", "PullNot", "not_inferable")]:
    q = f"v1.1 consensus outsider_inference == '{lev}', {MAIN_Q}"
    if lev in lv.index:
        put(macro_n, f_int(lv[lev]), F_MAIN + [F_V11], "count; " + q, RULE["int"])
        for m in MODELS:
            put(f"{prefix}{m.capitalize()}", f_dec(D[m].loc[D[m].outsider_inference == lev, "pull_C_k10"].mean(), 3),
                F_MAIN + [F_V11], f"mean pull_C_k10 ({MODEL_NAMES[m]}); " + q, RULE["m3"])
    else:
        missing(macro_n, f"level {lev} not in data", F_V11, q, RULE["int"])
lev = "inferable_with_effort"
q = f"v1.1 consensus outsider_inference == '{lev}', {MAIN_Q}"
put("NoiEff", f_int(lv.get(lev, 0)), F_MAIN + [F_V11], "count; " + q, RULE["int"])
for m in MODELS:
    put(f"PullEff{m.capitalize()}", f_dec(D[m].loc[D[m].outsider_inference == lev, "pull_C_k10"].mean(), 3)
        if lev in lv.index else r"\missing{no item}", F_MAIN + [F_V11], f"mean pull_C_k10 ({MODEL_NAMES[m]}); " + q,
        RULE["m3"])
check("outsider_inference levels in the main set sum to NkTen",
      int(lv.sum()) == len(D["oai"]) == int(M["NkTen"]["value"]))
NOTES.append("`NoiInf`/`PullInf*` use the level `inferable` and `NoiNot`/`PullNot*` the level `not_inferable`. The "
             "consensus also has `inferable_with_effort` and `UNRESOLVED`; all levels are in the table below.")

# ICC
for m in MODELS:
    put(f"ICCtradC{m}", f_dec(mixed(D[m], "pull_C_k10")["icc"], 3), F_MAIN,
        f"ICC = var(tradition intercept) / (var(tradition intercept) + residual var) of mixedlm pull_C_k10 ~ 1 + "
        f"(1|tradition), REML, {MODEL_NAMES[m]}; {MAIN_Q}", RULE["icc3"])

# direction / channel check
bad, tab = [], ["| variable | level | n | " + " | ".join(f"mean {m}" for m in MODELS) + " |",
                "|---|---|---|" + "---|" * len(MODELS)]
for var in ["direction", "primary_channel"]:
    for lev, g in D["oai"].groupby(var):
        means = {m: D[m].loc[D[m][var] == lev, "pull_C_k10"].mean() for m in MODELS}
        tab.append(f"| {var} | {lev} | {len(g)} | " + " | ".join(f"{means[m]:.4f}" for m in MODELS) + " |")
        bad += [f"{var}={lev} ({m}, n={len(g)}, mean {means[m]:.4f})" for m in MODELS if not means[m] > 0]
EXTRA.append(("Direction and primary channel (first-run consensus) in the main set", tab))
q = f"mean pull_C_k10 per level of first-run consensus direction and primary_channel (all levels present, incl. " \
    f"UNRESOLVED), 3 models; {MAIN_Q}"
put("NchanNegBge", f_int(int((D["bge"].primary_channel == "Behavioural").sum())), F_MAIN + [F_CONS1],
    f"count(first-run consensus primary_channel == 'Behavioural'), {MAIN_Q}", RULE["int"])
# CheckDirChan is no longer a macro (not used in the text since 2026-10-23); the check itself stays in the report
DIRCHAN = ("every level of direction and primary_channel has a positive mean pull_C_k10 in all 3 models: "
           + ("yes" if not bad else "NO — " + "; ".join(bad)))
EXTRA[-1][1].extend(["", "Check: " + DIRCHAN])

# ------------------------------------------------------------------ 6. recording status (condition B)
F_RR = ROOT / "results/recording_results.json"
F_IWC = ROOT / "results/items_with_consensus.csv"
rr = json.load(open(F_RR))
iwc = read_csv("results/items_with_consensus.csv").merge(rec[["item_id", "form_status", "recording_status"]],
                                                          on="item_id", how="left")
ex = iwc[iwc.form_status == "existing_form"]
exr = ex[ex.recording_status != "UNRESOLVED"]
check(f"recording analysis: {len(exr)} existing_form items with resolved recording_status = stored "
      f"{rr['counts']['analysed_existing_resolved']}", len(exr) == rr["counts"]["analysed_existing_resolved"])
put("NrecAnalysed", f_int(len(exr)), [F_IWC, F_REC],
    "items with pulls (213), form_status == existing_form, recording_status != UNRESOLVED "
    "(= recording_results.json counts.analysed_existing_resolved)", RULE["int"])
for macro, lab in [("NrecUnrecAnalysed", "unrecorded"), ("NrecCultAnalysed", "cultural_reference"),
                   ("NrecGenAnalysed", "general_dictionary")]:
    n = int((exr.recording_status == lab).sum())
    check(f"{macro}: {n} = stored", n == rr["counts"]["existing_form_by_recording_status"][lab])
    put(macro, f_int(n), [F_IWC, F_REC], f"as NrecAnalysed, recording_status == '{lab}'", RULE["int"])
for lab, short in [("unrecorded", "Unrec"), ("cultural_reference", "Cultural"), ("general_dictionary", "General")]:
    for col, m in [("pull_index", "Oai"), ("pull_bge_B", "Bge"), ("pull_gem_B", "Gem")]:
        x = exr.loc[exr.recording_status == lab, col].dropna()
        pct = 100 * float((x > 0).mean())
        st = rr["main"][col]["existing_by_recording_status"][lab]
        check(f"Ppos{short}{m}: {pct:.3f}% (n = {len(x)}) matches stored {st['pct_positive']}",
              round(pct, 1) == st["pct_positive"] and len(x) == st["n"])
        put(f"Ppos{short}{m}", f_dec(pct, 1), [F_IWC, F_REC],
            f"% of {col} > 0, existing_form items with recording_status == '{lab}' (condition B for bge/gem; original "
            f"OpenAI pull) (= recording_results.json main.{col}.existing_by_recording_status.{lab}.pct_positive)",
            RULE["pct1"])

# ------------------------------------------------------------------ 7. positive control (first-run cpg_valid)
vc = iwc.cpg_valid.value_counts()
put("NvalidConsensus", f_int(vc.get("yes", 0)), F_IWC, "count(cpg_valid == 'yes'), 213 items with pulls "
    "(first-run consensus); there is no consensus_results.json", RULE["int"])
put("NinvalidConsensus", f_int(vc.get("no", 0)), F_IWC, "count(cpg_valid == 'no'), 213 items with pulls", RULE["int"])
check(f"cpg_valid: yes {vc.get('yes', 0)} + no {vc.get('no', 0)} + other "
      f"{len(iwc) - vc.get('yes', 0) - vc.get('no', 0)} = {len(iwc)}", len(iwc) == 213)

# ------------------------------------------------------------------ 8. earlier pipeline
F_VS = ROOT / "results/verified_stats.json"
vs = json.load(open(F_VS))
icc = vs["P5_icc"]["icc_mixed_all_213"]
if safe_from_rounded(icc, 4, 3):
    put("ICCtrad", f_dec(icc, 3), F_VS, "P5_icc.icc_mixed_all_213 (v1 pipeline, condition A, OpenAI; stored with 4 "
        "decimals)", RULE["icc3"])
else:
    missing("ICCtrad", "stored value is half-way at 3 decimals", F_VS, "P5_icc.icc_mixed_all_213", RULE["icc3"])
F_REP = ROOT / "embeddings/results/replication_summary.json"
F_NB = ROOT / "embeddings/results/nnpm_text_embedding_3_large.csv"
nb = read_csv("embeddings/results/nnpm_text_embedding_3_large.csv")
# Nscriptmismatch (definition chosen by the author on 2026-10-23): items whose host was embedded in a different script
# from its controls, i.e. a romanised host with controls in the original script. Base: the 213 items with a
# v1-pipeline pull.
def letters(s):
    # letters only; modifier letters (Lm: ʻokina, ʿayn sign) and marks carry no script
    return [ch for ch in str(s) if unicodedata.category(ch) in ("Lu", "Ll", "Lt", "Lo")]


def scripts(s):
    out = set()
    for ch in letters(s):
        w = unicodedata.name(ch, "UNKNOWN").split()[0]
        out.add("CJK/KANA" if w in ("CJK", "HIRAGANA", "KATAKANA") else w)
    return out


nonlatin = lambda s: bool(scripts(s) - {"LATIN"})
latin_only = lambda s: scripts(s) == {"LATIN"}
ctrl = {k.split("|||")[0]: v for k, v in json.load(open(ROOT / "data/control_cache.json")).items()}
check("every v1-pipeline host has exactly one control list in control_cache.json",
      all(sum(k.split("|||")[0] == h for k in json.load(open(ROOT / "data/control_cache.json"))) == 1
          for h in nb.host_pipeline))
check("v1 pipeline: 213 items, all with pull", len(nb) == 213 and nb.pull_openai.notna().all())
nb["n_ctrl"] = [len(ctrl[h]) for h in nb.host_pipeline]
nb["nl_ctrl"] = [sum(map(nonlatin, ctrl[h])) for h in nb.host_pipeline]
nb["orig_nl"], nb["pipe_lat"] = nb.host_original.map(nonlatin), nb.host_pipeline.map(latin_only)
base = nb[nb.orig_nl & nb.pipe_lat]
SM_DEF = ("host_original contains letters of a non-Latin script, host_pipeline contains only Latin letters, and every "
          "control contains letters of a non-Latin script")
sm = base[base.nl_ctrl == base.n_ctrl]
F_CC = ROOT / "data/control_cache.json"
put("Nscriptmismatch", f_int(len(sm)), [F_NB, F_CC], SM_DEF + " (v1 pipeline, 213 items with pull; letters = "
    "Unicode categories Lu/Ll/Lt/Lo, script = first word of the Unicode name)", RULE["int"])
part = base[(base.nl_ctrl > 0) & (base.nl_ctrl < base.n_ctrl)]
other_script = [r.id for r in sm.itertuples()
                if not set().union(*map(scripts, ctrl[r.host_pipeline])) - {"LATIN"} <= scripts(r.host_original) - {"LATIN"}]
not_lat = nb[nb.orig_nl & ~nb.pipe_lat]


def old_nonlatin(s):   # the test used in the first report (counted modifier letters such as ʻokina as non-Latin)
    return any(ch.isalpha() and not unicodedata.name(ch, "").startswith("LATIN") for ch in str(s))


ob = nb[nb.host_original.map(old_nonlatin) & ~nb.host_pipeline.map(old_nonlatin)]
onl = [sum(map(old_nonlatin, ctrl[h])) for h in ob.host_pipeline]
cand = [("**adopted**: " + SM_DEF, len(sm)),
        ("same, at least one control in a non-Latin script", int((base.nl_ctrl > 0).sum())),
        ("same, most controls in a non-Latin script", int((base.nl_ctrl > base.n_ctrl / 2).sum())),
        ("host_pipeline only Latin letters and every control non-Latin (host_original not checked)",
         int((nb.pipe_lat & (nb.nl_ctrl == nb.n_ctrl)).sum())),
        ("host_original non-Latin and host_pipeline only Latin (controls not checked)", len(base)),
        ("stored flags: host_is_romanization == 1 (host_pipeline != host_original) and controls_latin == 0",
         int(((nb.host_is_romanization == 1) & (nb.controls_latin == 0)).sum())),
        ("first report, old letter test: host non-Latin → Latin, >= 1 control non-Latin", sum(x > 0 for x in onl)),
        ("first report, old letter test: same, every control non-Latin",
         sum(x == len(ctrl[h]) for x, h in zip(onl, ob.host_pipeline)))]
EXTRA.append(("`Nscriptmismatch`: candidate definitions (base: 213 v1-pipeline items with pull)",
              ["| definition | n |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cand] + [
                  "",
                  "Adopted because it is the reading of the sentence in section 6.2: the host was romanised (the "
                  "original form is in a non-Latin script, the pipeline form is in Latin letters) and the controls "
                  "are in the original script (non-Latin). The old value 99 is not used.",
                  "",
                  f"- Items where only some controls are non-Latin: {len(part)}"
                  + (" (" + ", ".join(part.id) + ")" if len(part) else "") + ". So 'all', 'most' and 'at least one' "
                  "give the same count.",
                  f"- Adopted items whose controls use a non-Latin script that does not occur in host_original: "
                  f"{len(other_script)}" + (" (" + ", ".join(other_script) + ")" if other_script else "") + ".",
                  f"- Items with a non-Latin host_original and controls in Latin letters (romanised host, Latin "
                  f"controls; not counted): {len(base) - len(sm)} ("
                  + ", ".join(base[base.nl_ctrl == 0].id) + "). Most are Malay/Javanese items whose host_original "
                  "adds a Jawi or Javanese-script gloss.",
                  "- Items whose host_pipeline still contains non-Latin letters (not counted): "
                  + ", ".join(f"{r.id} (`{r.host_pipeline}`)" for r in not_lat.itertuples()) + ".",
                  "- The first report's letter test counted modifier letters (Hawaiian ʻokina, the Arabic ʿayn sign) "
                  "as non-Latin; this is corrected here, which is why the earlier 97/98 differ."]))
gl = int(nb.host_has_gloss.sum())
check(f"Ngloss: host_has_gloss in nnpm csv ({gl}) = replication_summary.json",
      gl == json.load(open(F_REP))["nnpm_text_embedding_3_large"]["confound_checks"]["pull_openai"]["host_has_gloss"]["n_1"])
put("Ngloss", f_int(gl), F_NB, "count(host_has_gloss == 1): pipeline host matches [(（] or _[a-z] "
    "(nnpm_replicate.py)", RULE["int"])
put("NxsBug", f_int(vs["P2_S_type"]["coding_fix"]["items_with_corrected_x_s"]), F_VS,
    "P2_S_type.coding_fix.items_with_corrected_x_s", RULE["int"])

# ------------------------------------------------------------------ 9. before the decisions (fixed file)
pre = json.load(open(F_T1PRE))["hypotheses"]["H-C1"]["per_model"]
npre = {pre[MODELS[m]]["n"] for m in MODELS}
check("pre-decision H-C1 n is the same in the three models", len(npre) == 1)
put("NkTenPre", f_int(npre.pop()), F_T1PRE, "hypotheses.H-C1.per_model.*.n", RULE["int"])
for m in MODELS:
    z = pre[MODELS[m]]["z"]
    name = f"HConeZ{m}Pre"
    if safe_from_rounded(z, 4, 2):
        put(name, f_dec(z, 2), F_T1PRE, f"hypotheses.H-C1.per_model.{MODELS[m]}.z (stored with 4 decimals)", RULE["z2"])
    else:
        missing(name, "stored value is half-way at 2 decimals", F_T1PRE, f"hypotheses.H-C1.per_model.{MODELS[m]}.z",
                RULE["z2"])

# ------------------------------------------------------------------ count chain
chain = [("blind items", c["blind_items"]), ("with controls", c["items_with_controls"]),
         ("after T3", c["after_T3"]), ("not low_control", c["after_low_control"]), ("k = 10", len(D["oai"]))]
check(f"chain {' → '.join(str(v) for _, v in chain)}, valid {len(D['oai'][D['oai'].cpg_valid == 'yes'])}: "
      f"{c['blind_items']} - {c['no_controls_stored']} no controls = {c['items_with_controls']}; "
      f"- {c['excluded_T3']} excluded with controls = {c['after_T3']}; - {c['low_control_after_T3']} low_control = "
      f"{c['after_low_control']}; of these {c['eligible_by_k_after_low_control']['k10']} have 10 kept controls",
      c["blind_items"] - c["no_controls_stored"] == c["items_with_controls"]
      and c["items_with_controls"] - c["excluded_T3"] == c["after_T3"]
      and c["after_T3"] - c["low_control_after_T3"] == c["after_low_control"]
      and c["eligible_by_k_after_low_control"]["k10"] == len(D["oai"])
      and [v for _, v in chain] == [217, 213, 209, 185, 94]
      and len(D["oai"][D["oai"].cpg_valid == "yes"]) == 63)
no_ctrl = sorted(set(blind.item_id) - set(summ.item_id))
excl_no_ctrl = sorted(set(excl.item_id) - set(summ.item_id))
check(f"211 after T3 ({len(blind)} - {len(excl)}) minus the items without controls that are not excluded "
      f"({len(no_ctrl) - len(excl_no_ctrl)} of {', '.join(no_ctrl)}) = {c['after_T3']} with controls after T3",
      len(blind) - len(excl) - (len(no_ctrl) - len(excl_no_ctrl)) == c["after_T3"])

# ------------------------------------------------------------------ write numbers.tex
interim = {}
for line in INTERIM.read_text(encoding="utf-8").splitlines():
    mm = re.match(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$", line)
    if mm and mm.group(1) != "missing":
        interim[mm.group(1)] = mm.group(2)
order = list(interim) + [k for k in M if k not in interim]
RETIRED = {"CheckDirChan": "not used in the text since 2026-10-23; the check stays in this report"}
order = [k for k in order if k in M]
check("every interim macro is generated (except retired ones: " + ", ".join(RETIRED) + ")",
      all(k in M for k in interim if k not in RETIRED))
try:
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    # this script's own outputs do not count as uncommitted changes
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", ".", ":!paper/numbers.tex",
                            ":!results/numbers_manifest.csv", ":!results/numbers_report.md",
                            ":!results/harmonisation_weak_for_author.csv",
                            ":!results/harmonisation_all_weak_merged.csv", ":!results/sensitivity_tradition_unit.json",
                            ":!results/sensitivity_tradition_unit.md"], capture_output=True, text=True).stdout.strip()
    commit += " (with uncommitted changes)" if dirty else ""
except Exception:
    commit = "unknown"
L = ["%% numbers.tex — AUTOMATICALLY GENERATED by analysis/make_numbers.py. DO NOT EDIT BY HAND.",
     f"%% generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
     f"%% git commit: {commit}",
     "%% Sources and rules for every macro: results/numbers_manifest.csv. Report: results/numbers_report.md.",
     r"\providecommand{\missing}[1]{\textcolor{red}{[\textit{#1}]}}", ""]
L += [f"\\newcommand{{\\{k}}}{{{M[k]['value']}}}" for k in order]
(PAPER / "numbers.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
with open(ROOT / "results/numbers_manifest.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["macro", "value", "source_file", "source_key_or_query", "rule", "source_sha256"])
    w.writeheader()
    for k in order:
        w.writerow(M[k])

# ------------------------------------------------------------------ manuscript checks (§6)
tex_files = sorted((PAPER / "sections").glob("*.tex")) + [p for p in [PAPER / "cpg_v2.tex", PAPER / "numbers_pending.tex"]
                                                          if p.exists()]
ms = ["## (d) Manuscript checks", ""]
if not tex_files:
    ms += ["`paper/sections/*.tex` and `paper/cpg_v2.tex` are not in the repository, so the search for hard-coded numbers "
           "and the macro-use check were not run. `pdflatex` was not run either."]
else:
    # running text only: comments, column specs, lengths, \texttt{}, labels, refs, citations and years are ignored
    blank = re.compile(r"\\(?:texttt|label|ref|eqref|cite[pt]?|citealp|input|include|url|href|begin|end)\*?"
                       r"(?:\[[^\]]*\])?\{[^{}]*\}|\{[^{}]*\\linewidth\}|[A-Za-z]?\{\d*\.?\d+\s*(?:cm|mm|pt|em|ex)\}"
                       r"|\d*\.?\d+\s*(?:cm|mm|pt|em|ex)\b|\\\\\[[^\]]*\]|\\multicolumn\{\d+\}|\$[^$]*\$"
                       r"|\b(?:Fig\.|Figure|Table|Section|Test|Conditions?|condition|Sankoku|Supplementary)[~\s]*S?\d+"
                       r"(?:\s*(?:and|/)\s*\d+)?|\bC\d\b")
    num = re.compile(r"(?<![\w.\\-])(\d+(?:\{,\}\d+|,\d{3})*(?:\.\d+)?)(?![\w])")
    year = re.compile(r"^(1[5-9]\d\d|20\d\d)$")
    words = re.compile(r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
                       r"sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|hundred|half|dozen)\b", re.I)
    hard, wordhits, used = [], [], set()
    for f in tex_files:
        if f.name == "numbers_pending.tex":
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            code = re.split(r"(?<!\\)%", line)[0]
            if not re.match(r"\s*\\(?:new|renew|provide)command", code):
                used |= set(re.findall(r"\\([A-Za-z]+)", code))
            if f.name == "cpg_v2.tex" or not code.strip():
                continue
            text = blank.sub(" ", code)
            for mt in num.finditer(text):
                if year.match(mt.group(1)) or re.search(r"\b[kK]\s*=\s*$", text[:mt.start()]):
                    continue
                ctx = text[max(0, mt.start() - 30):mt.end() + 30].strip()
                hard.append(f"- `{rel(f)}:{i}`: `{mt.group(1)}` — …{ctx}…")
            for mt in (words.finditer(text) if re.match(r"s[4-7]_", f.name) else []):
                ctx = text[max(0, mt.start() - 30):mt.end() + 30].strip()
                wordhits.append(f"- `{rel(f)}:{i}`: `{mt.group(1)}` — …{ctx}…")
    defined = set(M)
    newcmds = set()
    for f in tex_files:
        newcmds |= set(re.findall(r"\\(?:new|renew|provide)command\{?\\([A-Za-z]+)", f.read_text(encoding="utf-8")))
    cand_macros = {u for u in used if re.match(r"^(N[a-z]{3,}|[NKP][A-Za-z]*[A-Z]|HC|Mean|Pull|ICC|Rho|Check)", u)
                   and u not in {"NeedsTeXFormat"}}
    ms += ["Files: " + ", ".join(f"`{rel(f)}`" for f in tex_files) + ". Nothing in the manuscript was changed.", "",
           "### 1a. Hard-coded number candidates: digits in running text (years, k = …, lengths, labels, \\texttt "
           "and citations ignored)", ""] + (hard or ["none"])
    ms += ["", "### 1b. Hard-coded number candidates: cardinal number words in sections 4–7 (many are not counts of "
           "data; listed for checking)", ""] \
        + (wordhits or ["none"])
    ms += ["", "### Macros used but not defined in numbers.tex", ""]
    ms += [f"- `\\{u}`" + (" (defined in a manuscript file)" if u in newcmds else "")
           for u in sorted(cand_macros - defined)] or ["none"]
    hard_def, soft_def = set(), set()
    for f in tex_files:
        t = f.read_text(encoding="utf-8")
        hard_def |= set(re.findall(r"\\(?:new|renew)command\{?\\([A-Za-z]+)", t))
        soft_def |= set(re.findall(r"\\providecommand\{?\\([A-Za-z]+)", t))
    ms += ["", "### Macros defined with \\newcommand both in numbers.tex and in a manuscript file (LaTeX error)", ""]
    ms += [f"- `\\{u}`" for u in sorted(defined & hard_def)] or ["none"]
    ms += ["", "### Macros that numbers.tex now defines and a manuscript file still provides (\\providecommand; "
           "harmless, can be deleted there)", ""]
    ms += [f"- `\\{u}`" for u in sorted(defined & (soft_def - {"missing"}))] or ["none"]
    ms += ["", "### Macros defined but not used", ""] + ([f"- `\\{u}`" for u in sorted(defined - used)] or ["none"])
    if shutil.which("pdflatex"):
        main = PAPER / "cpg_v2.tex"
        logs = []
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", main.name], cwd=main.parent,
                               capture_output=True, text=True)
            logs.append(r.stdout)
        und = sorted(set(re.findall(r"Undefined control sequence.*?\n.*?(\\[A-Za-z]+)", logs[-1])))
        ms += ["", "### pdflatex (2 runs)", "", "undefined control sequences: " + (", ".join(und) or "none")]
    else:
        ms += ["", "`pdflatex` is not installed here; the compilation check was not run."]

# ------------------------------------------------------------------ report
miss = [k for k in order if M[k]["value"].startswith(r"\missing")]
diff = [(k, interim[k], M[k]["value"]) for k in order if k in interim and interim[k] != M[k]["value"]]
R = ["# numbers_report.md — generated by analysis/make_numbers.py", "",
     f"Macros: {len(M)} (manifest rows: {len(order)}). Interim values compared: `{rel(INTERIM)}`.", "",
     "## (a) Macros left as \\missing", ""]
R += [f"- `\\{k}`: {M[k]['value'][9:-1]}" for k in miss] or ["none"]
R += ["", "## (b) Generated value differs from the interim value", "", "| macro | interim | generated |", "|---|---|---|"]
R += [f"| `\\{k}` | `{a}` | `{b}` |" for k, a, b in diff] or ["| none | | |"]
R += ["", "## (c) Newly computed values and notes", ""] + [f"- {n}" for n in NOTES]
for title, rows in EXTRA:
    R += ["", f"### {title}", ""] + rows
R += ["", "### Tradition and language labels", "",
      "Strong candidates are merged (author's decision of 2026-10-23): labels that become the same name once a "
      "parenthetical, a ' — ' subtitle (traditions only), diacritics and modifier letters, case and an unambiguous "
      "ISO code (languages only) are removed. The merged name is the label with the most items (tie: the shorter one "
      "without its parenthetical, then the shorter full label, then alphabetical). Weak candidates (shared segment of a '/' list, one name "
      "contained in the other, very similar spelling) are not merged; they are listed in "
      "`results/harmonisation_weak_for_author.csv`. The author decided on 2026-10-23 not to merge weak candidates "
      "(final). `NtradLabelsMin` and `NlanguagesMin` count labels with every weak pair merged too (connected "
      "components, known false matches excluded; `results/harmonisation_all_weak_merged.csv`, sensitivity only).", "",
      "| kind | labels before | labels after | labels in a merged group | labels with a weak candidate | weak pairs |",
      "|---|---|---|---|---|---|"]
R += [f"| {k} | {v['before']} | {v['after']} | {v['merged_labels']} | {v['weak_labels']} | {v['weak_pairs']} |"
      for k, v in harm.items()]
R += ["", "Merged groups:", ""] + [
    f"- {k}: {t} ← " + "; ".join(f"{r.original_label} ({r.rule})" for r in g.itertuples() if r.original_label != t)
    for (k, t), g in ht.groupby(["kind", "harmonised_label"], sort=False) if len(g) > 1]
R += ["", "### Retired macros", ""] + [f"- `\\{k}`: {v}" for k, v in RETIRED.items()]
R += ["", "## Consistency checks", ""] + [f"- {'OK' if ok else 'FAILED'}: {t}" for t, ok in CHECKS]
# the manuscript is not in the repository: its checks go to an untracked file, so that this report is the same with
# or without a local copy of the manuscript
(PAPER / "manuscript_checks.md").write_text("# Manuscript checks — generated by analysis/make_numbers.py\n\n"
                                            + "\n".join(ms[2:]) + "\n", encoding="utf-8")
R += ["", "## (d) Manuscript checks", "",
      "Written to `paper/manuscript_checks.md`, which is not tracked (the manuscript is not in the repository)."]
R += ["", "## Rules", ""] + [f"- `{k}`: {v}" for k, v in RULE.items()]
(ROOT / "results/numbers_report.md").write_text("\n".join(R) + "\n", encoding="utf-8")
print(f"{len(M)} macros, {len(miss)} missing, {len(diff)} differ from interim, "
      f"{sum(not ok for _, ok in CHECKS)} failed checks")
