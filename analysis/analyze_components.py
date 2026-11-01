"""Source audit, components (exploratory; docs/analysis_plan_source_audit.md, "Components" and "Components -> reason
codes", fixed 2026-10-25 before the run).

Input: audit/runs/judge_components_<model>.csv (three coders), results/source_audit_judgements.csv (majority label,
sent).
Per item and component: the majority (2 of 3) of yes / partial / no; no majority -> UNRESOLVED.
Reason codes: H = no -> missing_H; H = partial -> other_form; M = no -> missing_M; M = partial -> partial_M;
C_device != yes -> missing_C_device; C_community = no -> missing_C_community. An UNRESOLVED component gives no code.
Items not sent (no quote) are counted as no_evidence and left out of the tallies.
Output: results/audit_components.csv, results/audit_components.json, results/audit_components.md.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"gpt": "gpt-5.6-sol", "claude": "claude-opus-5-5", "gemini": "gemini-3.1-pro-preview"}
COMPONENTS = ["H", "M", "C_community", "C_device"]
LEVELS = ["yes", "partial", "no"]
CODES = ["missing_H", "other_form", "missing_M", "partial_M", "missing_C_device", "missing_C_community"]
LABELS = ["supported", "partly", "contradicted", "no_evidence"]

J = pd.read_csv(ROOT / "results/source_audit_judgements.csv", dtype=str).fillna("")
sent = J[J.sent == "True"].set_index("item_id")
runs = {}
for k, m in MODELS.items():
    r = pd.read_csv(ROOT / f"audit/runs/judge_components_{m}.csv", dtype=str).fillna("").set_index("item_id")
    assert set(r.index) == set(sent.index), (k, len(r), len(sent))
    runs[k] = r


def maj(vals):
    for v in LEVELS:
        if sum(x == v for x in vals) >= 2:
            return v
    return "UNRESOLVED"


rows = []
for it in sent.index:
    row = {"item_id": it, "majority_label": sent.at[it, "majority"]}
    for c in COMPONENTS:
        for k in MODELS:
            row[f"{c}_{k}"] = runs[k].at[it, c]
        row[c] = maj([runs[k].at[it, c] for k in MODELS])
    codes = []
    if row["H"] == "no":
        codes.append("missing_H")
    if row["H"] == "partial":
        codes.append("other_form")
    if row["M"] == "no":
        codes.append("missing_M")
    if row["M"] == "partial":
        codes.append("partial_M")
    if row["C_device"] in ("partial", "no"):
        codes.append("missing_C_device")
    if row["C_community"] == "no":
        codes.append("missing_C_community")
    row["codes"] = ";".join(codes)
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(ROOT / "results/audit_components.csv", index=False)

n_sent, n_noev = len(T), int((J.sent != "True").sum())
out = {"n_sent": n_sent, "n_no_evidence": n_noev, "components": {}, "codes": {}, "codes_by_label": {},
       "n_without_code": int((T.codes == "").sum())}
for c in COMPONENTS:
    out["components"][c] = {v: int((T[c] == v).sum()) for v in LEVELS + ["UNRESOLVED"]}
for code in CODES:
    has = T.codes.str.split(";").apply(lambda xs: code in xs)
    out["codes"][code] = int(has.sum())
    out["codes_by_label"][code] = {lab: int((has & (T.majority_label == lab)).sum()) for lab in LABELS}
out["n_by_label"] = {lab: int((T.majority_label == lab).sum()) for lab in LABELS}
out["n_label_other"] = int((~T.majority_label.isin(LABELS)).sum())
json.dump(out, open(ROOT / "results/audit_components.json", "w"), indent=1, ensure_ascii=False)

md = ["# Source audit: components (exploratory)", "",
      f"Items sent: {n_sent}; not sent (no quote, `no_evidence`): {n_noev}. Majority of 3 coders per component.", "",
      "| component | yes | partial | no | UNRESOLVED |", "|---|---|---|---|---|"]
md += [f"| {c} | " + " | ".join(str(out['components'][c][v]) for v in LEVELS + ['UNRESOLVED']) + " |"
       for c in COMPONENTS]
md += ["", "| reason code | all sent | " + " | ".join(LABELS) + " |", "|---|---|" + "---|" * len(LABELS)]
md += [f"| {code} | {out['codes'][code]} | " + " | ".join(str(out['codes_by_label'][code][lab]) for lab in LABELS)
       + " |" for code in CODES]
md += ["", f"Items per majority label: " + ", ".join(f"{lab} {out['n_by_label'][lab]}" for lab in LABELS)
       + f", other {out['n_label_other']}. Items with no code: {out['n_without_code']}.",
       "An item can have several codes. `different_work` cannot be derived from the components and is not computed."]
(ROOT / "results/audit_components.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
