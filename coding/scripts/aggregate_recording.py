"""Combine the three recording-status codings: agreement and consensus labels.

Usage:
    python3 aggregate_recording.py results/recording_gpt.csv results/recording_gemini.csv \
        results/recording_claude.csv --first agreement/consensus_codes.csv --out results

Outputs:
    recording_consensus.csv   majority label per field (UNRESOLVED when all three differ), per-coder votes,
                              and `recording_unanimous` (all three coders agree on recording_status)
    recording_agreement.json  Fleiss' kappa and % unanimous per field, label distributions,
                              crosstab of consensus recording_status against the first run's consensus cpg_valid
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs=3)
ap.add_argument("--first", default="agreement/consensus_codes.csv")
ap.add_argument("--out", default="results")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
FIELDS = ["form_status", "recording_status"]
TAGS = ["gpt", "gemini", "claude"]

coders = []
for f in args.files:
    d = pd.read_csv(f, dtype=str)
    assert (d.status == "ok").all(), f"{f} has non-ok rows"
    coders.append(d.drop_duplicates("item_id", keep="last").set_index("item_id"))
names = [c.model.iat[0] for c in coders]
ids = sorted(set.intersection(*[set(c.index) for c in coders]))
print("items coded by all three:", len(ids), "| coders:", names)


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
    for t, col in zip(TAGS, cols):
        cons[f"{field}_{t}"] = col
    A["consensus_distribution"][field] = cons[field].value_counts().to_dict()
    A["per_coder_distribution"][field] = {t: pd.Series(col).value_counts().to_dict() for t, col in zip(TAGS, cols)}
cons["recording_unanimous"] = [int(len({c.loc[i, "recording_status"] for c in coders}) == 1) for i in ids]
cons["form_unanimous"] = [int(len({c.loc[i, "form_status"] for c in coders}) == 1) for i in ids]
cons["mean_recording_confidence"] = np.mean(
    [c.loc[ids, "recording_confidence"].astype(float).values for c in coders], axis=0).round(3)
for t, c in zip(TAGS, coders):
    cons[f"recording_evidence_{t}"] = c.loc[ids, "recording_evidence"].values

first = pd.read_csv(args.first, dtype=str).set_index("item_id")
cons["first_cpg_valid"] = first.reindex(ids)["cpg_valid"].values
ct = pd.crosstab(cons["recording_status"], cons["first_cpg_valid"])
A["crosstab_recording_vs_first_cpg_valid"] = {r: ct.loc[r].to_dict() for r in ct.index}

cons.to_csv(OUT / "recording_consensus.csv", encoding="utf-8")
json.dump(A, open(OUT / "recording_agreement.json", "w"), ensure_ascii=False, indent=1, default=int)
print(json.dumps({k: A[k] for k in ["fleiss_kappa", "pct_unanimous", "consensus_distribution"]}, indent=1))
print(ct)
