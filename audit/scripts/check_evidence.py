"""Source audit, stage A: progress and integrity of the evidence (docs/analysis_plan_source_audit.md).

Usage: python3 audit/scripts/check_evidence.py [N | START END]
       (items N first in audit order, or positions START..END, 1-based inclusive; default all)

Checks every document row of private/audit_work/: the snapshot exists and its SHA-256 matches, every quote is a
verbatim passage of the stored text and within the length limit, at most 2 quotes, listed sources have an
`agent_source` value, at most 3 documents with text per item (archive copies of listed sources do not count),
every `not_found` source has a `not_found_type`, every `archive` row has an `archive_url`. Prints counts, including the share of sources listed
by the agents that could not be found (not_found).
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_tool import NFTYPES, SNAP, WDIR, norm, order, too_long, work  # noqa: E402

ids = order()
if len(sys.argv) > 2:
    sel = ids[int(sys.argv[1]) - 1:int(sys.argv[2])]
else:
    sel = ids[:int(sys.argv[1]) if len(sys.argv) > 1 else len(ids)]
rows = pd.concat([work(i) for i in sel if (WDIR / f"{i}.csv").exists()], ignore_index=True)
problems = []
for r in rows.itertuples():
    if r.snapshot_sha256:
        raw = [p for p in SNAP.glob(f"{r.snapshot_sha256}.*") if p.suffix != ".txt"]
        if not raw or hashlib.sha256(raw[0].read_bytes()).hexdigest() != r.snapshot_sha256:
            problems.append(f"{r.item_id} {r.doc_rank}: snapshot missing or hash mismatch")
            continue
        t = norm((SNAP / f"{r.snapshot_sha256}.txt").read_text(encoding="utf-8"))
    else:
        t = ""
    for q in [r.quote, r.quote_2]:
        if q and (q not in t or too_long(q)):
            problems.append(f"{r.item_id} {r.doc_rank}: quote not verbatim or too long")
    if r.quote_2 and not r.quote:
        problems.append(f"{r.item_id} {r.doc_rank}: quote_2 without quote")
    if r.listed_by_agent == "yes" and r.agent_source not in ("exists", "not_found", "inaccessible"):
        problems.append(f"{r.item_id} {r.doc_rank}: listed source without agent_source")
    if r.retrieved in ("record_only", "failed") and (r.quote or r.quote_2):
        problems.append(f"{r.item_id} {r.doc_rank}: quote on a {r.retrieved} row")
    if (r.agent_source == "not_found") != (r.not_found_type in NFTYPES) or (r.not_found_type and r.not_found_type
                                                                            not in NFTYPES):
        problems.append(f"{r.item_id} {r.doc_rank}: not_found_type {r.not_found_type!r} with agent_source "
                        f"{r.agent_source!r}")
    if (r.retrieved == "archive") != bool(r.archive_url):
        problems.append(f"{r.item_id} {r.doc_rank}: archive row without archive_url, or archive_url on another row")
text_docs = rows[rows.retrieved.isin(["full_text", "abstract", "snippet"]) & (rows.ai_generated != "yes")] \
    .groupby("item_id").size()
problems += [f"{i}: {k} documents with text (> 3)" for i, k in text_docs.items() if k > 3]

done = [i for i in sel if (WDIR / f"{i}.csv").exists() and len(work(i))]
lst = rows[rows.listed_by_agent == "yes"]
has_q = rows[(rows.quote != "") & (rows.ai_generated != "yes")].item_id.nunique()
print(f"items {len(sel)} | with any document {len(done)} | with a quote {has_q} | no document "
      f"{len(sel) - len(done)}")
print("documents:", len(rows), rows.retrieved.value_counts().to_dict(), "| types:", rows.source_type.value_counts().to_dict())
if len(lst):
    vc = lst.agent_source.value_counts()
    print(f"listed sources: {len(lst)} {vc.to_dict()} | not_found share {vc.get('not_found', 0) / len(lst):.1%}")
    nf = lst[lst.agent_source == "not_found"].not_found_type.value_counts()
    print("not_found types:", nf.to_dict())
print("ai_generated:", int((rows.ai_generated == "yes").sum()))
print(f"problems: {len(problems)}")
for p in problems:
    print("  " + p)
