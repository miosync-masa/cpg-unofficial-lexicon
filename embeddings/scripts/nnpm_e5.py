"""Pulls in conditions C and D with multilingual-e5 small, base and large (docs/analysis_plan_model_size.md).

Usage: python3 embeddings/scripts/nnpm_e5.py [--from-vectors]

Every string (clean host, covert target, controls) gets the prefix "query: " and is embedded with each size
(sentence-transformers, CPU, batch 16, normalised). The model files are fetched with huggingface_hub
snapshot_download and loaded from the snapshot, so the revision used is the snapshot's commit hash.
pull_X_k10 = sim(host, target) - mean sim(control, target) over the first 10 kept controls of condition X (C:
embeddings/controls/controls_clean.csv, D: data/controls_D.json); empty with fewer than 10.

Outputs: embeddings/results/nnpm_e5_<size>.csv, embeddings/results/e5_meta.json (revisions, parameter counts,
library versions); vectors in embeddings/results/e5/<size>.npz (ignored by git).
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from target_blind_common import ROOT, main_items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--from-vectors", action="store_true")
args = ap.parse_args()
SIZES = ["small", "base", "large"]
PREFIX = "query: "
E = ROOT / "embeddings/results"
(E / "e5").mkdir(parents=True, exist_ok=True)

cc = pd.read_csv(ROOT / "embeddings/controls/controls_clean.csv", dtype=str).fillna("")
ctrl = {"C": cc[cc.kept == "1"].groupby("item_id", sort=False).control.apply(lambda s: list(s)[:10]).to_dict(),
        "D": {k: v["kept"][:10] for k, v in json.load(open(ROOT / "data/controls_D.json", encoding="utf-8")).items()}}
items = main_items()
ids = {it["item_id"] for it in items}
need = sorted({it["clean_host"] for it in items} | {it["target"] for it in items}
              | {c for X in "CD" for k, v in ctrl[X].items() if k in ids for c in v})

meta_f = E / "e5_meta.json"
meta = json.load(open(meta_f)) if args.from_vectors else {"prefix": PREFIX, "n_strings": len(need), "models": {}}
if not args.from_vectors:
    import huggingface_hub
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer
    meta["versions"] = {"sentence_transformers": sentence_transformers.__version__,
                        "transformers": transformers.__version__, "torch": torch.__version__,
                        "huggingface_hub": huggingface_hub.__version__}
for size in SIZES:
    name = f"intfloat/multilingual-e5-{size}"
    vf = E / "e5" / f"{size}.npz"
    if args.from_vectors:
        z = np.load(vf, allow_pickle=True)
        vec = dict(zip(z["texts"].tolist(), z["vectors"]))
    else:
        path = huggingface_hub.snapshot_download(name)
        m = SentenceTransformer(path, device="cpu")
        v = m.encode([PREFIX + t for t in need], batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        vec = dict(zip(need, v))
        np.savez_compressed(vf, texts=np.array(need, dtype=object), vectors=np.asarray(v))
        meta["models"][size] = {"name": name, "revision": Path(path).name,
                                "params": int(sum(p.numel() for p in m.parameters())),
                                "dim": int(v.shape[1]), "max_seq_length": int(m.max_seq_length),
                                "time": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
        del m
    rows = []
    for it in items:
        h, t = vec[it["clean_host"]], vec[it["target"]]
        r = {"item_id": it["item_id"], "tradition": it["tradition"], "sim_host_target": float(h @ t)}
        for X in "CD":
            cs = ctrl[X].get(it["item_id"])
            r[f"n_kept_{X}"] = None if cs is None else len(cs)
            if cs is not None and len(cs) >= 10:
                s = float(np.mean([vec[c] @ t for c in cs]))
                r[f"simctrl_{X}"], r[f"pull_{X}_k10"] = s, r["sim_host_target"] - s
            else:
                r[f"simctrl_{X}"] = r[f"pull_{X}_k10"] = np.nan
        rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(E / f"nnpm_e5_{size}.csv", index=False, encoding="utf-8-sig")
    print(name, {X: int(d[f"pull_{X}_k10"].notna().sum()) for X in "CD"})
if not args.from_vectors:
    json.dump(meta, open(meta_f, "w"), indent=1)
