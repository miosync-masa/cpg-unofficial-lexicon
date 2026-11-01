"""Source audit: the documents that retrieval again (2026-10-30, 2026-11-01) and the Wayback Machine did not retrieve
(docs/analysis_plan_source_audit.md, "Retrieval again"). For supplement S1.

Input: the attempt logs private/audit_attempts/<item>.jsonl (not in the repository; only item, rank, date and the
kind of failure are written out).
Output: results/audit_unretrieved.csv, results/audit_unretrieved.md.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS = [("NAH_004", "3"), ("MET_006", "2"), ("MET_007", "1"), ("EEU_011", "1"), ("NET_003", "2"), ("MET_009", "3"),
        ("MET_008", "2"), ("MET_001", "1"), ("SEA_008", "2")]


def kind(s):
    return s.replace("error ", "") if s.startswith("error ") else ("HTTP " + s if s[:1].isdigit() else s)


def day(a):
    return a["accessed"][5:10]


rows = []
for it, rank in DOCS:
    log = [json.loads(x) for x in open(ROOT / f"private/audit_attempts/{it}.jsonl", encoding="utf-8")]
    log = [a for a in log if a.get("doc_rank") == rank and "status" in a]
    first = [a for a in log if "wayback_for" not in a and not a.get("refetch") and a["status"] not in ("200", "record")]
    ref = {d: [kind(a["status"]) for a in log if a.get("refetch") and a["accessed"].startswith(d)]
           for d in ("2026-10-30", "2026-11-01")}
    wb = {}
    for a in log:
        if "wayback_for" in a:
            wb.setdefault(day(a), []).append(kind(a["status"]))
    rows.append({
        "item_id": it, "doc_rank": rank,
        "first_failure": f"{kind(first[-1]['status'])} ({day(first[-1])})" if first else "no request (row recorded)",
        "refetch_10-30": "; ".join(ref["2026-10-30"]) or "—",
        "refetch_11-01": "; ".join(ref["2026-11-01"]) or "—",
        "wayback": "; ".join(f"{d}: " + ", ".join(k + (f" ×{v.count(k)}" if v.count(k) > 1 else "")
                                                  for k in dict.fromkeys(v)) for d, v in wb.items()),
    })
T = pd.DataFrame(rows)
T.to_csv(ROOT / "results/audit_unretrieved.csv", index=False)
md = ["# Source audit: documents not retrieved by retrieval again or the Wayback Machine", "",
      "First failure: the last failed request of 2026-10-24. Retrieval again: once more from the same URL on 2026-10-30 "
      "and 2026-11-01 (only rows that had failed with HTTP 5xx or a connection error). Wayback Machine: capture "
      "closest to 2026-03-13; `HTTP 429` = archive.org refused the capture; dates are MM-DD of 2026.", "",
      "| item | doc | first failure | again 10-30 | again 11-01 | Wayback |", "|---|---|---|---|---|---|"]
md += [f"| {r.item_id} | {r.doc_rank} | {r.first_failure} | {r['refetch_10-30']} | {r['refetch_11-01']} | {r.wayback} |"
       for _, r in T.iterrows()]
(ROOT / "results/audit_unretrieved.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
