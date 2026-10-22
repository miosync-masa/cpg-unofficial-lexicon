"""Combine the three codebook-v1.1 codings: agreement, consensus labels, cross-tables against earlier codings.

Usage:
    python3 aggregate_v1_1.py results/v1_1_gpt.csv results/v1_1_gemini.csv results/v1_1_claude.csv --out results

Outputs:
    v1_1_consensus.csv   majority label per field (UNRESOLVED when no label has >= 2 of 3 votes), per-coder votes,
                         unanimity flags, mean confidence, the three rationales, cpg_scope (decision 5 of 2026-10-22:
                         yes iff form_type is existing_form or transparent_combination; UNRESOLVED if form_type is),
                         cpg_scope_two_storey (the sheet's first definition, two_storey = yes and outsider_inference =
                         not_inferable; descriptive only),
                         old form_status (recording run) and first-run cpg_valid
    v1_1_agreement.json  Fleiss' kappa and % unanimous per field, label distributions, cross-tables
                         (old form_status x new form_type, two_storey x first-run cpg_valid), and where the
                         focus items (the six kennings coded novel_form, 草泥马, 五月三十五日) moved
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

RES = Path(__file__).resolve().parents[1] / "results"
ap = argparse.ArgumentParser()
ap.add_argument("files", nargs=3)
ap.add_argument("--out", default="results")
args = ap.parse_args()
OUT = Path(args.out)
FIELDS = ["form_type", "two_storey", "content_or_attitude", "outsider_inference", "host_unit"]
TAGS = ["gpt", "gemini", "claude"]
FOCUS = {"NEU_001": "kenning", "NEU_002": "kenning", "NEU_003": "kenning", "NEU_005": "kenning",
         "NEU_011": "kenning", "NEU_012": "kenning", "NET_004": "草泥马", "NET_005": "五月三十五日"}

coders = []
for f in args.files:
    d = pd.read_csv(f, dtype=str)
    assert (d.status == "ok").all(), f"{f} has non-ok rows"
    coders.append(d.drop_duplicates("item_id", keep="last").set_index("item_id"))
names = [c.model.iat[0] for c in coders]
ids = sorted(set.intersection(*[set(c.index) for c in coders]))
assert len(ids) == 217, len(ids)


def fk(cols):
    table, _ = aggregate_raters(np.array(cols).T)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = float(fleiss_kappa(table, method="fleiss"))
    return None if np.isnan(k) else round(k, 3)


A = {"coders": names, "n_items": len(ids), "fleiss_kappa": {}, "pct_unanimous": {}, "consensus_distribution": {},
     "per_coder_distribution": {}}
cons = pd.DataFrame(index=pd.Index(ids, name="item_id"))
for field in FIELDS:
    cols = [c.loc[ids, field].astype(str).values for c in coders]
    stacked = np.array(cols).T
    A["fleiss_kappa"][field] = fk(cols)
    A["pct_unanimous"][field] = round(100 * float(np.mean([len(set(r)) == 1 for r in stacked])), 1)
    cons[field] = [Counter(r).most_common(1)[0][0] if Counter(r).most_common(1)[0][1] >= 2 else "UNRESOLVED"
                   for r in stacked]
    cons[f"{field}_unanimous"] = [int(len(set(r)) == 1) for r in stacked]
    for t, col in zip(TAGS, cols):
        cons[f"{field}_{t}"] = col
    A["consensus_distribution"][field] = cons[field].value_counts().to_dict()
    A["per_coder_distribution"][field] = {t: pd.Series(col).value_counts().to_dict() for t, col in zip(TAGS, cols)}
cons["mean_confidence"] = np.mean([c.loc[ids, "confidence"].astype(float).values for c in coders], axis=0).round(3)
for t, c in zip(TAGS, coders):
    cons[f"rationale_{t}"] = c.loc[ids, "rationale"].values


def scope_two_storey(r):
    """First definition (instruction sheet); kept for description only since the decision of 2026-10-22."""
    if "UNRESOLVED" in (r.two_storey, r.outsider_inference):
        return "UNRESOLVED"
    return "yes" if r.two_storey == "yes" and r.outsider_inference == "not_inferable" else "no"


def scope(r):
    """Decision 5 of 2026-10-22: in scope iff form_type is existing_form or transparent_combination."""
    if r.form_type == "UNRESOLVED":
        return "UNRESOLVED"
    return "yes" if r.form_type in ("existing_form", "transparent_combination") else "no"


cons["cpg_scope"] = cons.apply(scope, axis=1)
cons["cpg_scope_two_storey"] = cons.apply(scope_two_storey, axis=1)
A["consensus_distribution"]["cpg_scope"] = cons.cpg_scope.value_counts().to_dict()
A["consensus_distribution"]["cpg_scope_two_storey"] = cons.cpg_scope_two_storey.value_counts().to_dict()
A["form_type_unresolved_items"] = sorted(cons.index[cons.form_type == "UNRESOLVED"])
old = pd.read_csv(RES / "recording/recording_consensus.csv", dtype=str).set_index("item_id")
first = pd.read_csv(RES / "first_run/consensus_codes.csv", dtype=str).set_index("item_id")
cons["old_form_status"] = old.reindex(ids)["form_status"].values
cons["first_cpg_valid"] = first.reindex(ids)["cpg_valid"].values

ct = lambda a, b: {str(r): {str(k): int(v) for k, v in row.items()}
                   for r, row in pd.crosstab(cons[a], cons[b]).iterrows()}
A["crosstab_old_form_status_x_form_type"] = ct("old_form_status", "form_type")
A["crosstab_two_storey_x_first_cpg_valid"] = ct("two_storey", "first_cpg_valid")
A["crosstab_cpg_scope_x_first_cpg_valid"] = ct("cpg_scope", "first_cpg_valid")
A["crosstab_cpg_scope_two_storey_x_first_cpg_valid"] = ct("cpg_scope_two_storey", "first_cpg_valid")
A["focus_items"] = {i: {"label": lab, "old_form_status": cons.loc[i, "old_form_status"],
                        **{f: cons.loc[i, f] for f in FIELDS + ["cpg_scope", "cpg_scope_two_storey"]},
                        "form_type_votes": [cons.loc[i, f"form_type_{t}"] for t in TAGS]}
                    for i, lab in FOCUS.items()}
cons.to_csv(OUT / "v1_1_consensus.csv", encoding="utf-8")
json.dump(A, open(OUT / "v1_1_agreement.json", "w"), ensure_ascii=False, indent=1, default=int)
print(json.dumps({k: A[k] for k in ["fleiss_kappa", "pct_unanimous", "consensus_distribution"]}, ensure_ascii=False,
                 indent=1))
print(pd.crosstab(cons.old_form_status, cons.form_type))
print(pd.crosstab(cons.two_storey, cons.first_cpg_valid))
print(json.dumps(A["focus_items"], ensure_ascii=False, indent=1))
