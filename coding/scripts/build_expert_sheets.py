"""Split the items that need a human look into small per-expertise review sheets.

Usage:
    python3 build_expert_sheets.py codes_gpt.csv codes_gemini.csv codes_claude.csv \
        --agreement agreement --out expert_review

An item is sent for expert review only if, on direction or primary channel,
    tier 1  the three models all disagree (consensus UNRESOLVED), or
    tier 2  all three models agree with each other but contradict the authors' original coding, or
    tier 3  a 2-of-3 consensus contradicts the original coding.
Tier 1-2 are the core ask; tier 3 is optional if the expert has time.
Items are grouped by the expertise needed to judge them (one sheet per group), not by raw language label.

Outputs:
    expert_review/<group>.csv   one row per item, model codes and rationales, blank expert_* columns
    expert_review/summary.csv   item counts per group and tier
"""
import argparse
from pathlib import Path

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs=3)
ap.add_argument("--agreement", default="agreement")
ap.add_argument("--out", default="expert_review")
args = ap.parse_args()
DATA = Path(__file__).resolve().parents[2] / "data"
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# tradition keyword -> expert group (first match wins; order matters)
GROUPS = [
    ("japanese", ["Japanese", "kakekotoba", "osechi"]),
    ("arabic_persian_turkish", ["īhām", "tawriya", "tevriye", "Arabic", "Persian", "Ottoman"]),
    ("sanskrit_south_asian", ["Sanskrit", "śleṣa", "Hindi"]),
    ("mesoamerican_andean", ["Nahuatl", "Maya", "Quechua", "Andean"]),
    ("african", ["Oríkì", "Izibongo", "Akan", "Ɛbɛ", "Qene", "Yoruba", "Zulu", "Xhosa", "Amharic"]),
    ("slavic", ["Russian", "zagadka", "bylina", "Polish", "Czech", "Belarus", "Ukrain"]),
    ("old_northern_european", ["Norse", "kenning", "Anglo-Saxon", "Exeter", "Old English", "Irish",
                               "Kalevala", "Finnish"]),
    ("southeast_asian_pacific", ["pantun", "Malay", "Javanese", "macapat", "Vietnamese", "Nôm", "Thai",
                                 "Māori", "Whakataukī", "Hawai", "Pitjantjatjara", "Aboriginal"]),
    ("english_euphemism_internet", ["English", "euphemism", "Internet", "Cross-cultural", "gesture",
                                    "Adianoeta", "double entendre"]),
]


def group_of(tradition, language):
    text = f"{tradition} {language}"
    for g, keys in GROUPS:
        if any(k.lower() in text.lower() for k in keys):
            return g
    return "other"


blind = pd.read_csv(DATA / "blind_items.csv", dtype=str).fillna("").set_index("item_id")
cons = pd.read_csv(Path(args.agreement) / "consensus_codes.csv", dtype=str).fillna("").set_index("item_id")
coders = []
for f in args.files:
    d = pd.read_csv(f, dtype=str).fillna("")
    coders.append(d[d.status == "ok"].drop_duplicates("item_id", keep="last").set_index("item_id"))

rows = []
for i in cons.index:
    tier, why = None, []
    for field in ["direction", "primary_channel"]:
        votes = [c.loc[i, field] for c in coders]
        orig, con = cons.loc[i, "original_" + field], cons.loc[i, field]
        if con == "UNRESOLVED":
            t = 1
        elif orig and con != orig:
            t = 2 if len(set(votes)) == 1 else 3
        else:
            continue
        why.append(f"{field}: models {'/'.join(votes)} vs original {orig or '-'}")
        tier = t if tier is None else min(tier, t)
    if tier is None:
        continue
    b = blind.loc[i]
    row = {"item_id": i, "tier": tier, "reason": "; ".join(why),
           "language": b.language, "tradition": b.tradition, "host_original": b.host_original,
           "host_romanized": b.host_romanized, "covert_target": b.covert_target,
           "source_reference": b.source_reference,
           "original_direction": cons.loc[i, "original_direction"],
           "original_primary_channel": cons.loc[i, "original_primary_channel"]}
    for name, c in zip(["gpt", "gemini", "claude"], coders):
        for field in ["direction", "primary_channel", "cpg_valid", "rationale"]:
            row[f"{name}_{field}"] = c.loc[i, field]
    row.update(expert_direction="", expert_primary_channel="", expert_cpg_valid="",
               expert_confidence_1to3="", expert_note="")
    rows.append(row)

df = pd.DataFrame(rows)
df["group"] = [group_of(t, l) for t, l in zip(df.tradition, df.language)]
for g, sub in df.groupby("group"):
    sub.sort_values(["tier", "item_id"]).drop(columns="group").to_csv(
        OUT / f"{g}.csv", index=False, encoding="utf-8-sig")  # utf-8-sig so Excel shows scripts correctly
summ = df.pivot_table(index="group", columns="tier", values="item_id", aggfunc="count", fill_value=0)
summ.columns = [f"tier{c}" for c in summ.columns]
summ["core_tier1_2"] = summ.get("tier1", 0) + summ.get("tier2", 0)
summ["total"] = summ.filter(like="tier").sum(axis=1)
summ.to_csv(OUT / "summary.csv")
print(summ.to_string())
print("total items:", len(df), "| core (tier 1-2):", int((df.tier <= 2).sum()))
