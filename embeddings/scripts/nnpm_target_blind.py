"""Pulls for conditions D and E (controls generated without the covert target), per embedding model.

Usage: python3 embeddings/scripts/nnpm_target_blind.py --backend openai|gemini|bge-m3 [--from-vectors]

Plan: docs/analysis_plan_target_blind_controls.md. The clean host and the target keep their condition C vectors
(embeddings/results/nnpm_C_<model>_vectors.npz), so only the controls differ from condition C. The controls
(data/controls_D.json, data/controls_E_<model>.json; the first 10 kept) are embedded with the same model.
pull_X_k10 = sim(host, target) - mean sim(control, target) over 10 controls; empty with fewer than 10.
simctrl_X = mean sim(control, target) over the same 10 controls (diagnostic), X = C, D, E.

Outputs: embeddings/results/nnpm_DE_<model>.csv, nnpm_DE_<model>_vectors.npz (control vectors),
         embeddings/results/logs/nnpm_DE_<model>.jsonl (every embedding call: time, texts, error)
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from target_blind_common import ROOT, main_items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["openai", "gemini", "bge-m3"], required=True)
ap.add_argument("--from-vectors", action="store_true")
args = ap.parse_args()
NAME = {"openai": "text-embedding-3-large", "gemini": "gemini-embedding-001", "bge-m3": "BAAI/bge-m3"}[args.backend]
TAG = re.sub(r"[^A-Za-z0-9]+", "_", NAME).strip("_")
E = ROOT / "embeddings/results"

z = np.load(E / f"nnpm_C_{TAG}_vectors.npz", allow_pickle=True)
cvec = dict(zip(z["texts"].tolist(), z["vectors"]))  # as stored, so that condition C is reproduced exactly
cc = pd.read_csv(ROOT / "embeddings/controls/controls_clean.csv", dtype=str).fillna("")
ctrl = {"C": cc[cc.kept == "1"].groupby("item_id", sort=False).control.apply(lambda s: list(s)[:10]).to_dict(),
        "D": {k: v["kept"][:10] for k, v in json.load(open(ROOT / "data/controls_D.json", encoding="utf-8")).items()},
        "E": {k: v["kept"][:10] for k, v in json.load(open(ROOT / f"data/controls_E_{TAG}.json", encoding="utf-8")).items()
              if not k.startswith("_")}}
items = main_items()
need = sorted({c for X in "DE" for v in ctrl[X].values() for c in v})
vf = E / f"nnpm_DE_{TAG}_vectors.npz"
if args.from_vectors:
    zz = np.load(vf, allow_pickle=True)
    vec = dict(zip(zz["texts"].tolist(), zz["vectors"]))
else:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if args.backend == "bge-m3":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(NAME, device="cpu")
        emb = lambda t: m.encode(t, batch_size=16, normalize_embeddings=True, show_progress_bar=False)  # noqa: E731
    elif args.backend == "gemini":
        import os
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        emb = lambda t: [e.values for e in client.models.embed_content(  # noqa: E731
            model=NAME, contents=t, config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")).embeddings]
    else:
        from openai import OpenAI
        client = OpenAI()
        emb = lambda t: [d.embedding for d in client.embeddings.create(model=NAME, input=t).data]  # noqa: E731
    vec = {}
    log = E / f"logs/nnpm_DE_{TAG}.jsonl"
    for i in range(0, len(need), 100):
        chunk, err = need[i:i + 100], None
        for attempt in range(6):
            try:
                v = np.asarray(emb(chunk), dtype=float)
                err = None
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:300]
                import time
                time.sleep(2 ** attempt)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": dt.datetime.now().astimezone().isoformat(), "model": NAME, "n": len(chunk),
                                "texts": chunk, "error": err}, ensure_ascii=False) + "\n")
        if err:
            raise RuntimeError(err)
        for s, x in zip(chunk, v):
            vec[s] = x / np.linalg.norm(x)
    np.savez_compressed(vf, texts=np.array(need, dtype=object), vectors=np.array([vec[t] for t in need]))
rows = []
for it in items:
    h, t = cvec[it["clean_host"]], cvec[it["target"]]
    r = {"item_id": it["item_id"], "tradition": it["tradition"], "sim_host_target": float(h @ t)}
    for X in "CDE":
        cs = ctrl[X].get(it["item_id"])
        r[f"n_kept_{X}"] = None if cs is None else len(cs)
        if cs is not None and len(cs) >= 10:
            src = cvec if X == "C" else vec
            s = float(np.mean([src[c] @ t for c in cs]))
            r[f"simctrl_{X}"], r[f"pull_{X}_k10"] = s, r["sim_host_target"] - s
        else:
            r[f"simctrl_{X}"] = r[f"pull_{X}_k10"] = np.nan
    rows.append(r)
d = pd.DataFrame(rows)
# check: condition C reproduces nnpm_C_<model>.csv
c = pd.read_csv(E / f"nnpm_C_{TAG}.csv", encoding="utf-8-sig").set_index("item_id")
assert np.allclose(d.set_index("item_id").pull_C_k10, c.loc[d.item_id, "pull_C_k10"], atol=1e-9)
d.to_csv(E / f"nnpm_DE_{TAG}.csv", index=False, encoding="utf-8-sig")
print(NAME, {X: int(d[f"pull_{X}_k10"].notna().sum()) for X in "CDE"})
