"""NNPM replication with an alternative embedding model.

Usage:
    python3 -I nnpm_replicate.py <data-dir> <out-dir> --backend bge-m3
    python3 -I nnpm_replicate.py <data-dir> <out-dir> --backend gemini --model gemini-embedding-001
    (<data-dir> = data/ of this repository: cpg_archaeology_results.csv, control_cache.json, source/)
        (gemini needs GEMINI_API_KEY in the environment and `pip install google-genai`)

For every item it re-embeds the host, the covert target and the stored control words
(data/control_cache.json — the same controls as the original analysis) and
recomputes the pull index under two host conditions:

    A  "pipeline" : the host string exactly as used in the original analysis (mostly romanized,
                    sometimes with an English gloss)
    B  "original" : the host in its original script (host_word / item_orig)

and for k = 3, 5, 7, 10 controls (the first k in the stored order).

Condition C (T1, 2026-10-22), run separately with --condition-c embeddings/controls; A and B are then not recomputed:

    C  host = clean_hosts.csv clean_host, controls = the kept controls of controls_clean.csv
       (at most 10, stored order), k = 3, 5, 10; an item with fewer than k kept controls has no pull for that k.

    Output <out-dir>/nnpm_C_<model>.csv (item_id, tradition, clean_host, n_controls_kept, sim_host_target,
    pull_C_k3, pull_C_k5, pull_C_k10, plus pull_C_kall = all kept controls up to 10, a sensitivity column not in
    the spec), the vectors in nnpm_C_<model>_vectors.npz, every embedding call logged to nnpm_C_<model>.jsonl (texts, error), and
    strings that could not be embedded listed in nnpm_C_<model>_failed.csv (their pulls are left empty).
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # keys live in .env at the repository root

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--backend", choices=["bge-m3", "gemini", "openai"], required=True)
ap.add_argument("--model", default=None)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--condition-c", default=None, help="t1 directory: run condition C only")
ap.add_argument("--from-vectors", action="store_true",
                help="condition C: recompute from the saved nnpm_C_<model>_vectors.npz, no API or model call")
args = ap.parse_args()
DATA, OUT = Path(args.data), Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
KS = [3, 5, 7, 10]


# ------------------------------------------------------------ embedding backends
def make_embedder():
    if args.backend == "bge-m3":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(args.model or "BAAI/bge-m3", device="cpu")
        name = args.model or "BAAI/bge-m3"
        return name, lambda texts: m.encode(texts, batch_size=16, normalize_embeddings=True,
                                            show_progress_bar=False)
    if args.backend == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        name = args.model or "gemini-embedding-001"

        def emb(texts):
            out = []
            for i in range(0, len(texts), 100):
                for attempt in range(6):
                    try:
                        r = client.models.embed_content(
                            model=name, contents=texts[i:i + 100],
                            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"))
                        out += [e.values for e in r.embeddings]
                        break
                    except Exception as e:  # rate limits
                        time.sleep(2 ** attempt)
                        if attempt == 5:
                            raise e
            v = np.array(out, dtype=float)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        return name, emb
    if args.backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        name = args.model or "text-embedding-3-large"

        def emb(texts):
            out = []
            for i in range(0, len(texts), 256):
                r = client.embeddings.create(model=name, input=texts[i:i + 256])
                out += [d.embedding for d in r.data]
            v = np.array(out, dtype=float)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        return name, emb


# ------------------------------------------------------------ items
res = pd.read_csv(DATA / "cpg_archaeology_results.csv", encoding="utf-8-sig")
orig = {}
for f in sorted((DATA / "source").glob("ashigaru*.yaml")):
    for it in yaml.safe_load(open(f, encoding="utf-8"))["items"]:
        orig[it["id"]] = str(it.get("host_word", "")).strip()
tor = pd.read_csv(DATA / "source/torami_103_with_tradition.csv")
for _, r in tor.iterrows():
    orig["T_%03d" % int(r["id"])] = str(r["item_orig"]).strip()

cache = json.load(open(DATA / "control_cache.json", encoding="utf-8"))
by_ht = {}
for k, v in cache.items():
    h, t, lang, trad = k.split("|||")
    by_ht[(h.strip(), t.strip(), trad.strip())] = v

items = []
for _, r in res.iterrows():
    ctrls = by_ht.get((str(r["host"]).strip(), str(r["target"]).strip(), str(r["tradition"]).strip()))
    if not ctrls:
        continue
    items.append({"id": r["id"], "tradition": r["tradition"], "target": r["target"],
                  "host_pipeline": str(r["host"]).strip(), "host_original": orig.get(r["id"], ""),
                  "controls": ctrls, "pull_openai": r["pull_index"]})
print(len(items), "items with controls")


def run_condition_c(t1):
    import datetime as dt
    ch = pd.read_csv(t1 / "clean_hosts.csv", dtype=str).fillna("").set_index("item_id")
    cc = pd.read_csv(t1 / "controls_clean.csv", dtype=str).fillna("")
    kept = cc[cc.kept == "1"].groupby("item_id", sort=False).control.apply(list).to_dict()
    rows_c = [{"item_id": it["id"], "tradition": it["tradition"], "target": it["target"],
               "clean_host": ch.loc[it["id"], "clean_host"], "controls": kept.get(it["id"], [])[:10]}
              for it in items]
    default = {"bge-m3": "BAAI/bge-m3", "gemini": "gemini-embedding-001", "openai": "text-embedding-3-large"}
    name = args.model or default[args.backend]
    tag = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    log = OUT / f"nnpm_C_{tag}.jsonl"
    texts = sorted({s for r in rows_c for s in [r["target"], r["clean_host"], *r["controls"]] if s})
    vecs, failed, t0 = {}, [], time.time()
    if args.from_vectors:
        z = np.load(OUT / f"nnpm_C_{tag}_vectors.npz", allow_pickle=True)
        saved = dict(zip(z["texts"].tolist(), z["vectors"]))
        vecs = {s1: saved[s1] for s1 in texts if s1 in saved}
        failed = [s1 for s1 in texts if s1 not in saved]
        texts = []  # nothing to embed
    else:
        name, emb = make_embedder()

    def call(chunk):
        err = None
        try:
            v = emb(chunk)
        except Exception as e:
            v, err = None, f"{type(e).__name__}: {e}"
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": dt.datetime.now().isoformat(), "model": name, "n": len(chunk),
                                "texts": chunk, "error": err}, ensure_ascii=False) + "\n")
        return v

    for i in range(0, len(texts), args.batch):
        chunk = texts[i:i + args.batch]
        v = call(chunk)
        if v is None:  # fall back to one string at a time so a single bad string is isolated
            for s1 in chunk:
                v1 = call([s1])
                if v1 is None:
                    failed.append(s1)
                else:
                    vecs[s1] = np.asarray(v1[0], dtype=float)
        else:
            for s1, v1 in zip(chunk, v):
                vecs[s1] = np.asarray(v1, dtype=float)
        print(f"  embedded {min(i + args.batch, len(texts))}/{len(texts)}  ({time.time() - t0:.0f}s)", flush=True)
    c = lambda a, b: float(np.dot(vecs[a], vecs[b]))
    out = []
    for r in rows_c:
        ok_ctrl = [x for x in r["controls"] if x in vecs]
        row = {"item_id": r["item_id"], "tradition": r["tradition"], "clean_host": r["clean_host"],
               "n_controls_kept": len(r["controls"]), "sim_host_target": np.nan}
        if r["clean_host"] in vecs and r["target"] in vecs:
            row["sim_host_target"] = c(r["clean_host"], r["target"])
        for k in [3, 5, 10]:
            ok = not np.isnan(row["sim_host_target"]) and len(r["controls"]) >= k \
                and all(x in vecs for x in r["controls"][:k])
            row[f"pull_C_k{k}"] = (row["sim_host_target"] - np.mean([c(x, r["target"]) for x in r["controls"][:k]])
                                   if ok else np.nan)
        # sensitivity (not in the spec): all kept controls (up to 10), whatever their number
        ok = not np.isnan(row["sim_host_target"]) and r["controls"] and all(x in vecs for x in r["controls"])
        row["pull_C_kall"] = (row["sim_host_target"] - np.mean([c(x, r["target"]) for x in r["controls"]])
                              if ok else np.nan)
        out.append(row)
    pd.DataFrame(out).to_csv(OUT / f"nnpm_C_{tag}.csv", index=False, encoding="utf-8-sig")
    if args.from_vectors:  # recompute only: vectors, meta and failure list stay as written by the API run
        print("recomputed", OUT / f"nnpm_C_{tag}.csv", "from saved vectors")
        return
    keys = sorted(vecs)  # vectors kept so the pulls can be recomputed without calling the API again
    np.savez_compressed(OUT / f"nnpm_C_{tag}_vectors.npz", texts=np.array(keys, dtype=object),
                        vectors=np.array([vecs[k] for k in keys]))
    pd.DataFrame({"text": failed}).to_csv(OUT / f"nnpm_C_{tag}_failed.csv", index=False, encoding="utf-8-sig")
    json.dump({"model": name, "n_items": len(out), "n_texts": len(texts), "n_failed": len(failed),
               "seconds": round(time.time() - t0)}, open(OUT / f"nnpm_C_{tag}_meta.json", "w"), indent=1)
    print("saved", OUT / f"nnpm_C_{tag}.csv", "| failed strings:", len(failed))


if args.condition_c:
    run_condition_c(Path(args.condition_c))
    raise SystemExit

LATIN = re.compile(r"^[\x00-\x7FÀ-ɏḀ-ỿ\s\W]*$")

# ------------------------------------------------------------ embed all unique strings once
name, emb = make_embedder()
texts = sorted({s for it in items for s in [it["target"], it["host_pipeline"], it["host_original"],
                                            *it["controls"]] if s})
t0 = time.time()
vecs = {}
for i in range(0, len(texts), args.batch):
    chunk = texts[i:i + args.batch]
    for s, v in zip(chunk, emb(chunk)):
        vecs[s] = np.asarray(v, dtype=float)
    print(f"  embedded {min(i + args.batch, len(texts))}/{len(texts)}  ({time.time() - t0:.0f}s)", flush=True)

cos = lambda a, b: float(np.dot(vecs[a], vecs[b]))

rows = []
for it in items:
    row = {"id": it["id"], "tradition": it["tradition"], "pull_openai": it["pull_openai"],
           "host_pipeline": it["host_pipeline"], "host_original": it["host_original"],
           "host_is_romanization": int(it["host_pipeline"] != it["host_original"]),
           "host_has_gloss": int(bool(re.search(r"[(（]|_[a-z]", it["host_pipeline"]))),
           "controls_latin": int(all(LATIN.match(c) for c in it["controls"]))}
    for cond, host in [("A", it["host_pipeline"]), ("B", it["host_original"])]:
        if not host:
            continue
        hs = cos(host, it["target"])
        row[f"sim_{cond}"] = hs
        for k in KS:
            row[f"pull_{cond}_k{k}"] = hs - np.mean([cos(c, it["target"]) for c in it["controls"][:k]])
    rows.append(row)
df = pd.DataFrame(rows)
tag = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
df.to_csv(OUT / f"nnpm_{tag}.csv", index=False, encoding="utf-8-sig")
json.dump({"model": name, "n_items": len(df), "n_texts": len(texts),
           "seconds": round(time.time() - t0)}, open(OUT / f"nnpm_{tag}_meta.json", "w"), indent=1)
print("saved", OUT / f"nnpm_{tag}.csv")
