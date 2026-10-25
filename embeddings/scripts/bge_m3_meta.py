"""Base model and parameter count of BAAI/bge-m3, beside multilingual-e5 (docs/analysis_plan_model_size.md).

Usage: python3 embeddings/scripts/bge_m3_meta.py

Reads the local Hugging Face snapshot of BAAI/bge-m3 (refs/main of the cache; the earlier runs did not record a
revision): the architecture from config.json, the lineage from the model card (README.md, "Specs" table), and the
parameter count of the loaded SentenceTransformer (sum of the sizes of all parameter tensors, as for e5). The same
config fields of the multilingual-e5-large snapshot recorded in embeddings/results/e5_meta.json are stored beside
them.

Output: embeddings/results/bge_m3_meta.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from target_blind_common import ROOT  # noqa: E402

import huggingface_hub  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

NAME = "BAAI/bge-m3"
path = Path(huggingface_hub.snapshot_download(NAME, local_files_only=True))
cfg = json.load(open(path / "config.json"))
e5 = json.load(open(ROOT / "embeddings/results/e5_meta.json"))["models"]["large"]
lcfg = json.load(open(huggingface_hub.snapshot_download(e5["name"], revision=e5["revision"], local_files_only=True)
                      + "/config.json"))
KEYS = ("architectures", "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size",
        "max_position_embeddings", "vocab_size")
card = (path / "README.md").read_text(encoding="utf-8")
rows = {m.group(1): m.group(2).strip() for m in
        re.finditer(r"^\| \[(BAAI/bge-m3[\w-]*)\]\([^)]*\) \|[^|]*\|[^|]*\|(.*)\|\s*$", card, re.M)}
m = SentenceTransformer(str(path), device="cpu")
out = {"name": NAME, "revision": path.name,
       "config": {k: cfg[k] for k in KEYS},
       "model_card_specs": rows,
       "params": int(sum(p.numel() for p in m.parameters())),
       "e5_large_config": {"revision": e5["revision"], **{k: lcfg[k] for k in KEYS}}}
json.dump(out, open(ROOT / "embeddings/results/bge_m3_meta.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
