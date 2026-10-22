"""Combine the three LLM codings: agreement, consensus labels, and the human audit sheet.

Usage:
    python3 aggregate_codes.py codes_gpt.csv codes_gemini.csv codes_claude.csv \
        --out <out-dir>        (original coding read from data/source/)

Outputs:
    consensus_codes.csv   majority label per field (UNRESOLVED when all three differ)
    agreement.json        Fleiss' kappa per field, per-channel kappa, % unanimous,
                          and Cohen's kappa of the consensus against the authors' original coding
    audit_sheet.csv       every item with any disagreement on direction or primary channel,
                          plus a seeded random 20% of the rest, with blank columns for the human auditor
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa


ap = argparse.ArgumentParser()
ap.add_argument("files", nargs=3)
ap.add_argument("--out", required=True)
args = ap.parse_args()
DATA = Path(__file__).resolve().parents[2] / "data"
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
CHANNELS = ["Phonetic", "Morphological", "Chromatic", "Conceptual", "Structural",
            "Behavioural", "Metonymic", "Formal"]
CAT = ["cpg_valid", "direction", "primary_channel", "activation", "familiarity",
       "s_juxtaposition", "s_noncompositional", "s_conventional"]

coders = []
for f in args.files:
    d = pd.read_csv(f, dtype=str)
    d = d[d.status == "ok"].drop_duplicates("item_id", keep="last").set_index("item_id")
    coders.append(d)
names = [c.model.iat[0] for c in coders]
ids = sorted(set.intersection(*[set(c.index) for c in coders]))
print("items coded by all three:", len(ids), "| coders:", names)


def fk(cols):
    table, _ = aggregate_raters(np.array(cols).T)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = float(fleiss_kappa(table, method="fleiss"))
    return None if np.isnan(k) else round(k, 3)  # None = no variation (all coders, all items identical)


A = {"coders": names, "n_items": len(ids), "fleiss_kappa": {}, "pct_unanimous": {}}
cons = pd.DataFrame(index=ids)
for field in CAT:
    cols = [c.loc[ids, field].astype(str).values for c in coders]
    A["fleiss_kappa"][field] = fk(cols)
    stacked = np.array(cols).T
    A["pct_unanimous"][field] = round(100 * float(np.mean([len(set(r)) == 1 for r in stacked])), 1)
    cons[field] = [Counter(r).most_common(1)[0][0] if Counter(r).most_common(1)[0][1] >= 2
                   else "UNRESOLVED" for r in stacked]

chan = {c: [] for c in CHANNELS}
for c in coders:
    sets = c.loc[ids, "channels"].map(json.loads)
    for ch in CHANNELS:
        chan[ch].append(sets.map(lambda s, ch=ch: int(ch in s)).values)
A["fleiss_kappa_per_channel"] = {ch: fk(v) for ch, v in chan.items()}
for ch, v in chan.items():
    cons["has_" + ch] = (np.sum(v, axis=0) >= 2).astype(int)
for field in ["confidence"]:
    cons["mean_confidence"] = np.mean([c.loc[ids, field].astype(float).values for c in coders], axis=0)
cons["clean_host_" + "votes"] = [json.dumps([c.loc[i, "clean_host"] for c in coders], ensure_ascii=False)
                                 for i in ids]

# --- compare with the authors' original coding
orig = {}
for f in sorted((DATA / "source").glob("ashigaru*.yaml")):
    for it in yaml.safe_load(open(f, encoding="utf-8"))["items"]:
        xt = it.get("x_type")
        rse = xt if isinstance(xt, list) else [it.get("x_r", 0) or 0, it.get("x_e", 0) or 0, it.get("x_s", 0) or 0]
        orig[it["id"]] = {"direction": "RES"[int(np.argmax([float(v) for v in rse]))],
                          "primary_channel": str(it.get("y_vector", "")).split("+")[0].replace("?", "")
                          .replace("Behavioral", "Behavioural").strip()}
tor = pd.read_csv(DATA / "source/torami_103_with_tradition.csv")
for _, r in tor.iterrows():
    orig["T_%03d" % int(r["id"])] = {
        "direction": "RES"[int(np.argmax([r.x_r, r.x_e, r.x_s]))],
        "primary_channel": str(r.y_vector).split("+")[0].replace("Behavioral", "Behavioural")}
A["consensus_vs_original_cohen_kappa"] = {}
for field in ["direction", "primary_channel"]:
    pair = [(cons.loc[i, field], orig[i][field]) for i in ids
            if i in orig and cons.loc[i, field] != "UNRESOLVED"]
    a, b = zip(*pair)
    A["consensus_vs_original_cohen_kappa"][field] = {
        "kappa": round(float(cohen_kappa_score(a, b)), 3), "n": len(pair),
        "pct_agree": round(100 * float(np.mean([x == y for x, y in pair])), 1),
        "confusion": pd.crosstab(pd.Series(b, name="original"), pd.Series(a, name="consensus")).to_dict()}
    cons["original_" + field] = [orig.get(i, {}).get(field, "") for i in ids]

cons.index.name = "item_id"
cons.to_csv(OUT / "consensus_codes.csv", encoding="utf-8")
json.dump(A, open(OUT / "agreement.json", "w"), ensure_ascii=False, indent=1, default=str)

# --- audit sheet
dis = [i for i in ids if len({c.loc[i, "direction"] for c in coders}) > 1
       or len({c.loc[i, "primary_channel"] for c in coders}) > 1]
rest = [i for i in ids if i not in dis]
rng = np.random.default_rng(20261021)
rand = list(rng.choice(rest, size=max(1, round(0.2 * len(rest))), replace=False)) if rest else []
blind = pd.read_csv(DATA / "blind_items.csv", dtype=str).set_index("item_id")
audit = []
for i in dis + rand:
    row = {"item_id": i, "audit_reason": "disagreement" if i in dis else "random_20pct",
           **blind.loc[i, ["language", "tradition", "host_original", "host_romanized", "covert_target"]].to_dict()}
    for n, c in zip(["A", "B", "C"], coders):
        row[f"{n}_direction"] = c.loc[i, "direction"]
        row[f"{n}_primary_channel"] = c.loc[i, "primary_channel"]
        row[f"{n}_rationale"] = c.loc[i, "rationale"]
    row.update(human_cpg_valid="", human_direction="", human_primary_channel="", human_note="")
    audit.append(row)
pd.DataFrame(audit).to_csv(OUT / "audit_sheet.csv", index=False, encoding="utf-8-sig")
print(json.dumps({k: A[k] for k in ["fleiss_kappa", "pct_unanimous", "fleiss_kappa_per_channel"]}, indent=1))
print("audit items:", len(audit), f"({len(dis)} disagreements + {len(rand)} random)")
