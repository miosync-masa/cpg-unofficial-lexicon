"""Build results/items_with_consensus.csv: the 213 items with pulls, joined with the first 3-model consensus codes.

Usage: python3 build_items_with_consensus.py

Pull columns:
    pull_index   original OpenAI pull (results/items_with_metadata.csv, as reproduced by reproduce_stats.py)
    pull_bge_B   BAAI/bge-m3, host in original script, k = 10 controls   (embeddings/results/nnpm_BAAI_bge_m3.csv)
    pull_gem_B   gemini-embedding-001, same condition                     (embeddings/results/nnpm_gemini_embedding_001.csv)
Code columns come from coding/results/first_run/consensus_codes.csv (first coding run), prefixed with nothing.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
meta = pd.read_csv(ROOT / "results/items_with_metadata.csv", encoding="utf-8-sig")
bge = pd.read_csv(ROOT / "embeddings/results/nnpm_BAAI_bge_m3.csv", encoding="utf-8-sig")
gem = pd.read_csv(ROOT / "embeddings/results/nnpm_gemini_embedding_001.csv", encoding="utf-8-sig")
cons = pd.read_csv(ROOT / "coding/results/first_run/consensus_codes.csv", dtype=str)

df = meta[["id", "host", "target", "tradition", "language", "region", "pull_index"]].rename(columns={"id": "item_id"})
df = df.merge(bge[["id", "pull_B_k10"]].rename(columns={"id": "item_id", "pull_B_k10": "pull_bge_B"}), on="item_id")
df = df.merge(gem[["id", "pull_B_k10"]].rename(columns={"id": "item_id", "pull_B_k10": "pull_gem_B"}), on="item_id")
df = df.merge(cons[["item_id", "cpg_valid", "direction", "primary_channel", "activation", "familiarity"]],
              on="item_id", how="left")
assert len(df) == 213 and df.cpg_valid.notna().all(), len(df)
df.to_csv(ROOT / "results/items_with_consensus.csv", index=False, encoding="utf-8")
print("wrote results/items_with_consensus.csv", df.shape)
