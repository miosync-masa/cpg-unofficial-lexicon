"""T1 steps 1-2: clean host per item, and the control-word cleaning (rules R1-R4).

Usage: python3 embeddings/scripts/build_clean.py

Inputs : data/blind_items.csv, coding/results/first_run/codes_{gpt,gemini,claude}.csv (first coding run, `clean_host`),
         data/control_cache.json and data/cpg_archaeology_results.csv (213 items with controls)
Outputs: embeddings/controls/clean_hosts.csv     item_id, clean_host, clean_host_source, script
         embeddings/controls/controls_clean.csv  item_id, control, kept, rule_dropped   (one row per stored control, stored order)
         embeddings/controls/controls_summary.csv item_id, n_original, n_kept, n_dropped_R1..R4, low_control

Clean host: the three coders' `clean_host` values are compared after normalisation (norm(): NFKC, case-fold,
whitespace collapsed, Arabic harakat removed); a value given by >= 2 coders is taken, in the spelling of the first
coder (gpt, gemini, claude order) that gave it. Otherwise host_original, source = fallback_original.
Items whose use_first_form proposal the author adopted (data/author_decisions.csv, decision adopt_first_form) take
the first form instead, source = first_form_T3, when it differs from the majority value.

Script: the Unicode block of the majority of letters: Latin / CJK (Han, kana, CJK punctuation) / Arabic / Cyrillic /
Devanagari / Ethiopic / Javanese / Other (anything else, or no letters).

Controls, applied in order, first rule that fires is recorded:
  R1 identical    key(control) equals key() of clean_host, host_original or host_romanized
                  (key = norm() plus combining marks, i.e. tone and other diacritics, removed, and spaces removed)
  R2 substring    key(control) contains key(clean_host) or vice versa, both keys >= 2 characters
  R3 script       script(control) != script(clean_host); not applied when clean_host has no letters at all
                  (digits or emoji, e.g. 304, 🍆 🍑), since such a host has no script to match
  R4 duplicate    key(control) already kept for this item
low_control = fewer than 5 kept. Controls are never regenerated.
"""
import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "embeddings/controls"
ap = argparse.ArgumentParser()
args = ap.parse_args()
DATA = ROOT / "data"
HARAKAT = re.compile("[ً-ٰٟ]")


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold().replace("’", "'").replace("ʼ", "'")
    return HARAKAT.sub("", re.sub(r"\s+", " ", s)).strip()


def key(s):
    d = unicodedata.normalize("NFD", norm(s))
    return unicodedata.normalize("NFC", "".join(c for c in d if not unicodedata.combining(c))).replace(" ", "")


def char_script(c):
    o = ord(c)
    if 0x0400 <= o <= 0x052F or 0x1C80 <= o <= 0x1C8F or 0x2DE0 <= o <= 0x2DFF or 0xA640 <= o <= 0xA69F:
        return "Cyrillic"
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF or 0xFB50 <= o <= 0xFDFF \
            or 0xFE70 <= o <= 0xFEFF:
        return "Arabic"
    if 0x0900 <= o <= 0x097F or 0xA8E0 <= o <= 0xA8FF:
        return "Devanagari"
    if 0x1200 <= o <= 0x139F or 0x2D80 <= o <= 0x2DDF or 0xAB00 <= o <= 0xAB2F:
        return "Ethiopic"
    if 0xA980 <= o <= 0xA9DF:
        return "Javanese"
    if 0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF \
            or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F or 0x20000 <= o <= 0x2FA1F:
        return "CJK"
    if c.isascii() or 0x00C0 <= o <= 0x024F or 0x1E00 <= o <= 0x1EFF or 0x0250 <= o <= 0x02AF \
            or 0xA720 <= o <= 0xA7FF:
        return "Latin"
    return "Other"


def script(s):
    cs = [char_script(c) for c in str(s) if c.isalpha()]
    return Counter(cs).most_common(1)[0][0] if cs else "Other"


items = pd.read_csv(ROOT / "data/blind_items.csv", dtype=str).fillna("")
coders = [pd.read_csv(ROOT / f"coding/results/first_run/codes_{t}.csv", dtype=str).fillna("").set_index("item_id")
          for t in ["gpt", "gemini", "claude"]]
rows = []
for _, r in items.iterrows():
    vals = [c.loc[r.item_id, "clean_host"].strip() if r.item_id in c.index else "" for c in coders]
    counts = Counter(norm(v) for v in vals if v)
    best = [n for n, k in counts.items() if k >= 2]
    if best:
        host, src = next(v for v in vals if v and norm(v) == best[0]), "majority"
    else:
        host, src = r.host_original, "fallback_original"
    rows.append({"item_id": r.item_id, "clean_host": host, "clean_host_source": src, "script": script(host)})
ch = pd.DataFrame(rows)
# author decision 2 (2026-10-22): adopted use_first_form items take the first form as clean host
dec = pd.read_csv(ROOT / "data/author_decisions.csv", dtype=str).fillna("")
for r in dec[dec.decision == "adopt_first_form"].itertuples():
    i = ch.index[ch.item_id == r.item_id][0]
    if ch.at[i, "clean_host"] != r.first_form:
        ch.loc[i, ["clean_host", "clean_host_source", "script"]] = [r.first_form, "first_form_T3", script(r.first_form)]
ch.to_csv(HERE / "clean_hosts.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------ controls
res = pd.read_csv(DATA / "cpg_archaeology_results.csv", encoding="utf-8-sig")
cache = json.load(open(DATA / "control_cache.json", encoding="utf-8"))
by_ht = {}
for k, v in cache.items():
    h, t, lang, trad = k.split("|||")
    by_ht[(h.strip(), t.strip(), trad.strip())] = v
info = items.set_index("item_id")
chi = ch.set_index("item_id")
crow, srow = [], []
for _, r in res.iterrows():
    ctrls = by_ht.get((str(r["host"]).strip(), str(r["target"]).strip(), str(r["tradition"]).strip()), [])
    h = chi.loc[r["id"]]
    host_keys = {key(x) for x in [h.clean_host, info.loc[r["id"], "host_original"],
                                  info.loc[r["id"], "host_romanized"]] if str(x).strip()}
    hk, hs = key(h.clean_host), h.script
    has_letters = any(ch_.isalpha() for ch_ in str(h.clean_host))
    seen, n = set(), Counter()
    for c in ctrls:
        ck = key(c)
        if ck in host_keys:
            rule = "R1"
        elif len(ck) >= 2 and len(hk) >= 2 and (hk in ck or ck in hk):
            rule = "R2"
        elif has_letters and script(c) != hs:
            rule = "R3"
        elif ck in seen:
            rule = "R4"
        else:
            rule = ""
            seen.add(ck)
        n[rule] += 1
        crow.append({"item_id": r["id"], "control": c, "kept": int(rule == ""), "rule_dropped": rule})
    srow.append({"item_id": r["id"], "n_original": len(ctrls), "n_kept": n[""],
                 **{f"n_dropped_R{i}": n[f"R{i}"] for i in range(1, 5)}, "low_control": int(n[""] < 5)})
pd.DataFrame(crow).to_csv(HERE / "controls_clean.csv", index=False, encoding="utf-8")
sm = pd.DataFrame(srow)
sm.to_csv(HERE / "controls_summary.csv", index=False, encoding="utf-8")
print("clean hosts:", ch.clean_host_source.value_counts().to_dict(), "| scripts:", ch.script.value_counts().to_dict())
print("items with controls:", len(sm), "| controls:", int(sm.n_original.sum()), "kept:", int(sm.n_kept.sum()),
      {f"R{i}": int(sm[f"n_dropped_R{i}"].sum()) for i in range(1, 5)})
print("low_control:", int(sm.low_control.sum()), "| n_kept distribution:", sm.n_kept.value_counts().sort_index().to_dict())
