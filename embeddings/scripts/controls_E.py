"""Condition E: control words = nearest vocabulary words to the clean host in one embedding space (no LLM, no target).

Usage: python3 embeddings/scripts/controls_E.py --backend openai|gemini|bge-m3

Plan: docs/analysis_plan_target_blind_controls.md. Items: the 94 of the condition C main set whose language maps to
a wordfreq language (LANG below; every other language is left out). Vocabulary: wordfreq 3.1.1 top_n_list(lang,
50000) (shorter where the list is shorter), words without a letter dropped. Every vocabulary word is embedded;
the cosine similarity to the clean host (the condition C vector of the clean host) ranks the words; R1-R4
(clean_rules.py) are applied in that order and the first 10 kept words are the controls.

Outputs: data/controls_E_<model>.json   {item_id: {language, wordfreq_lang, clean_host, vocab_size,
                                                    ranked: [[word, sim, rule], ...] up to the 10th kept word,
                                                    kept: [...]}}, plus "_excluded": {language label: n items}
         embeddings/results/logs/controls_E_<model>.jsonl   one line per embedding call (time, lang, offset, n, error)
         embeddings/vocab_cache/<model>/<lang>.npz   similarities of each item's host to the vocabulary (not tracked)
"""
import argparse
import datetime as dt
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_rules import clean  # noqa: E402
from target_blind_common import ROOT, main_items  # noqa: E402

LANG = {"en": "en", "English (primarily American English; global adoption)": "en", "ja": "ja", "hi": "hi",
        "fa": "fa", "fr": "fr", "zh": "zh", "Mandarin Chinese (普通话)": "zh", "Russian": "ru", "Arabic": "ar",
        "Vietnamese": "vi", "Polish": "pl", "Finnish": "fi"}
ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["openai", "gemini", "bge-m3"], required=True)
ap.add_argument("--workers", type=int, default=4)
args = ap.parse_args()
NAME = {"openai": "text-embedding-3-large", "gemini": "gemini-embedding-001", "bge-m3": "BAAI/bge-m3"}[args.backend]
TAG = re.sub(r"[^A-Za-z0-9]+", "_", NAME).strip("_")
LOG = ROOT / f"embeddings/results/logs/controls_E_{TAG}.jsonl"
CACHE = ROOT / "embeddings/vocab_cache" / TAG
CACHE.mkdir(parents=True, exist_ok=True)


def make_embedder():
    if args.backend == "bge-m3":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(NAME, device="cpu")
        return lambda t: m.encode(t, batch_size=64, normalize_embeddings=True, show_progress_bar=False), 512, 1
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if args.backend == "gemini":
        import os
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        def emb(t):
            r = client.models.embed_content(model=NAME, contents=t,
                                            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"))
            return np.array([e.values for e in r.embeddings], dtype=float)
        return emb, 100, args.workers
    from openai import OpenAI
    client = OpenAI()
    return lambda t: np.array([d.embedding for d in client.embeddings.create(model=NAME, input=t).data]), 1000, \
        args.workers


def vocab(lang):
    from wordfreq import top_n_list
    return [w for w in top_n_list(lang, 50000) if any(c.isalpha() for c in w)]


items = main_items()
z = np.load(ROOT / f"embeddings/results/nnpm_C_{TAG}_vectors.npz", allow_pickle=True)
cvec = dict(zip(z["texts"].tolist(), z["vectors"]))
by_lang, excluded = {}, {}
for it in items:
    wl = LANG.get(it["language"])
    if wl is None:
        excluded[it["language"]] = excluded.get(it["language"], 0) + 1
    else:
        by_lang.setdefault(wl, []).append(it)

emb, bs, workers = None, None, None
for wl, its in sorted(by_lang.items()):
    f = CACHE / f"{wl}.npz"
    ids = [it["item_id"] for it in its]
    if f.exists() and np.load(f, allow_pickle=True)["item_ids"].tolist() == ids:
        continue
    if emb is None:
        emb, bs, workers = make_embedder()
    V = vocab(wl)
    H = np.array([cvec[it["clean_host"]] for it in its])
    H = H / np.linalg.norm(H, axis=1, keepdims=True)
    chunks = [(o, V[o:o + bs]) for o in range(0, len(V), bs)]
    t0 = time.time()

    def run(c):
        o, words = c
        for attempt in range(8):
            err = None
            try:
                v = np.asarray(emb(words), dtype=float)
                v = v / np.linalg.norm(v, axis=1, keepdims=True)
            except Exception as e:  # rate limits
                v, err = None, f"{type(e).__name__}: {e}"[:300]
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"time": dt.datetime.now().astimezone().isoformat(), "model": NAME, "lang": wl,
                                     "offset": o, "n": len(words), "attempt": attempt + 1, "error": err}) + "\n")
            if v is not None:
                return o, (H @ v.T).astype(np.float32)
            time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"embedding failed: {wl} offset {o}")

    S = np.zeros((len(its), len(V)), dtype=np.float32)
    with ThreadPoolExecutor(workers) as ex:
        for k, (o, s) in enumerate(ex.map(run, chunks)):
            S[:, o:o + s.shape[1]] = s
            if k % 20 == 0:
                print(f"  {wl}: {o + s.shape[1]}/{len(V)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(f, item_ids=np.array(ids, dtype=object), vocab=np.array(V, dtype=object), sims=S)
    print(f"{wl}: {len(V)} words, {len(its)} items, {time.time() - t0:.0f}s", flush=True)

out = {"_excluded": excluded, "_wordfreq": "3.1.1", "_model": NAME}
for wl, its in sorted(by_lang.items()):
    d = np.load(CACHE / f"{wl}.npz", allow_pickle=True)
    V, S = d["vocab"].tolist(), d["sims"]
    for it, s in zip(its, S):
        order = np.argsort(-s, kind="stable")
        cl = clean([V[j] for j in order[:2000]], it["clean_host"], it["host_keys"], limit=10)
        sims = {V[j]: float(s[j]) for j in order[:len(cl)]}
        out[it["item_id"]] = {"language": it["language"], "wordfreq_lang": wl, "clean_host": it["clean_host"],
                              "vocab_size": len(V), "ranked": [[w, round(sims[w], 6), r] for w, r in cl],
                              "kept": [w for w, r in cl if r == ""]}
json.dump(out, open(ROOT / f"data/controls_E_{TAG}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n = [k for k in out if not k.startswith("_")]
print(NAME, "| items:", len(n), "| with 10 kept:", sum(len(out[k]["kept"]) >= 10 for k in n), "| excluded:", excluded)
