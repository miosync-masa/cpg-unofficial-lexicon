"""Write the public table of the author's check against 三省堂国語辞典 (8th edition), without the dictionary text.

Usage: python3 dictionary_check/scripts/export_sankoku.py

Reads dictionary_check/private/T4_dictionary_check_2026-10-29.xlsx (sheet 辞書照合, filled in by the author on
2026-10-29; not published) and writes dictionary_check/sankoku_judgments.csv with only: item_id, 見出し語, 引く見出し,
三国_掲載 (column F). Column G (三国：ラベル・注記) holds the author's working notes and paraphrases of the entries, so
it stays in private/. The example row and the 参考 rows (outside the dataset) are left out.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
LEVELS = {"語義として載る", "注記・用例のみ", "載らない", "見出しなし"}
x = pd.read_excel(HERE / "private/T4_dictionary_check_2026-10-29.xlsx", sheet_name="辞書照合", header=7, dtype=str)
x = x[x["item_id"].fillna("").str.match(r"^[A-Z]+_\d+$")]
out = x[["item_id", "見出し語（原文）", "引く見出し（目安）", "三国：掲載"]].fillna("")
out.columns = ["item_id", "見出し語", "引く見出し", "三国_掲載"]
assert set(out["三国_掲載"]) - {""} <= LEVELS, set(out["三国_掲載"])
out.to_csv(HERE / "sankoku_judgments.csv", index=False, encoding="utf-8-sig")
print(f"wrote sankoku_judgments.csv: {len(out)} rows, {(out['三国_掲載'] != '').sum()} with a judgement")
