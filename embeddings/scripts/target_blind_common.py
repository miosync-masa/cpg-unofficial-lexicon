"""Items and hosts shared by the condition D and E scripts (docs/analysis_plan_target_blind_controls.md).

main_items() returns the 94 items of the condition C main set (T3 exclusions applied, 10 kept controls in
condition C) with: item_id, tradition, target, language (v1 cache key), host_v1 (the host string sent in v1),
clean_host, host_keys (keys of clean host, host_original, host_romanized, as in build_clean.py).
"""
import json
from pathlib import Path

import pandas as pd

from clean_rules import key

ROOT = Path(__file__).resolve().parents[2]


def main_items():
    items = pd.read_csv(ROOT / "data/blind_items.csv", dtype=str).fillna("").set_index("item_id")
    ch = pd.read_csv(ROOT / "embeddings/controls/clean_hosts.csv", dtype=str).fillna("").set_index("item_id")
    summ = pd.read_csv(ROOT / "embeddings/controls/controls_summary.csv").set_index("item_id")
    excl = set(pd.read_csv(ROOT / "data/exclusions.csv").item_id)
    res = pd.read_csv(ROOT / "data/cpg_archaeology_results.csv", encoding="utf-8-sig")
    cache = json.load(open(ROOT / "data/control_cache.json", encoding="utf-8"))
    by_ht = {}
    for k in cache:
        h, t, lang, trad = k.split("|||")
        by_ht[(h.strip(), t.strip(), trad.strip())] = (h, t, lang, trad)
    out = []
    for _, r in res.iterrows():
        iid = r["id"]
        kk = by_ht.get((str(r["host"]).strip(), str(r["target"]).strip(), str(r["tradition"]).strip()))
        if kk is None or iid in excl or summ.loc[iid, "n_kept"] < 10:
            continue
        h = ch.loc[iid, "clean_host"]
        keys = {key(x) for x in [h, items.loc[iid, "host_original"], items.loc[iid, "host_romanized"]] if str(x).strip()}
        out.append({"item_id": iid, "tradition": r["tradition"], "target": r["target"], "language": kk[2],
                    "host_v1": kk[0], "clean_host": h, "host_keys": keys, "cache_key": "|||".join(kk)})
    assert len(out) == 94, len(out)
    return out
