"""Write the public dictionary-check table without the dictionary text.

Usage: python3 dictionary_check/scripts/export_public.py   (after apply_judgements.py)

Reads dictionary_check/private/daijisen_text.csv (contains the copyrighted 本文 column; not published) and writes
dictionary_check/daijisen_judgments.csv with only: item_id, 見出し語, URL, 取得日時, 判定, 一致度, 引用, 判定メモ.
引用 holds only the short verbatim passage that supports the judgement (apply_judgements.py checks it against 本文).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
COLS = ["item_id", "見出し語", "URL", "取得日時", "判定", "一致度", "引用", "判定メモ"]
df = pd.read_csv(HERE / "private/daijisen_text.csv", encoding="utf-8-sig", dtype=str).fillna("")
df[COLS].to_csv(HERE / "daijisen_judgments.csv", index=False, encoding="utf-8-sig")
print(f"wrote daijisen_judgments.csv: {len(df)} rows, longest 引用 {df['引用'].str.len().max()} characters")
