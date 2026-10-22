"""Build the blind item table for LLM coding.

Usage: python3 -I build_blind_table.py <out-csv>        (reads data/source/)

Keeps only what a coder needs to judge an item: language, tradition, the host form
(original script and romanized), the covert target, and the literature source when recorded.
Drops every existing code (x/y/z, notes, rationales) and every embedding result.
Rows are shuffled with a fixed seed so that traditions are not presented in blocks.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

SRC, out = Path(__file__).resolve().parents[2] / "data/source", Path(sys.argv[1])
rows = []
for f in sorted(SRC.glob("ashigaru*.yaml")):
    for it in yaml.safe_load(open(f, encoding="utf-8"))["items"]:
        rows.append({
            "item_id": it["id"],
            "language": it.get("language", ""),
            "tradition": it.get("tradition", ""),
            "host_original": str(it.get("host_word", "")).strip(),
            "host_romanized": str(it.get("host_romanized", "")).strip(),
            "covert_target": str(it.get("covert_target", "")).strip(),
            "source_reference": str(it.get("source", "") or "").strip(),
        })
tor = pd.read_csv(SRC / "torami_103_with_tradition.csv")
NAMES = {"kakekotoba": "Japanese kakekotoba (waka pivot words)", "osechi": "Japanese osechi New Year food symbolism",
         "food_euphemism_ja": "Japanese game-meat food euphemisms", "slesa": "Sanskrit śleṣa",
         "iham": "Arabic/Persian īhām", "adianoeta": "Adianoeta / double entendre (European)",
         "xieyin": "Chinese xiéyīn (homophonic symbolism)", "metonymic_sex_ja": "Japanese sexual euphemism",
         "metonymic_sex_en": "English sexual euphemism", "metonymic_drink_ja": "Japanese drinking euphemism",
         "metonymic_drink_en": "English drinking euphemism", "metonymic_ritual": "Cross-cultural ritual gesture/phrase"}
for _, r in tor.iterrows():
    rows.append({
        "item_id": "T_%03d" % int(r["id"]),
        "language": r["language"],
        "tradition": NAMES.get(r["tradition"], r["tradition"]),
        "host_original": str(r["item_orig"]).strip(),
        "host_romanized": str(r["item_roman"]).strip(),
        "covert_target": str(r["covert_target"]).strip(),
        "source_reference": "",
    })
df = pd.DataFrame(rows).sample(frac=1, random_state=20261021).reset_index(drop=True)
df.insert(0, "order", range(1, len(df) + 1))
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, encoding="utf-8")
print(len(df), "items written to", out)
