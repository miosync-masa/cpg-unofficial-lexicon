"""Condition D: control words from the v1 prompt without the EXCLUSION lines (no covert target).

Usage: python3 embeddings/scripts/generate_controls_D.py            calls the API (gpt-5.2), then cleans
       python3 embeddings/scripts/generate_controls_D.py --offline  cleans again from the stored responses

Plan: docs/analysis_plan_target_blind_controls.md. Items: the 94 of the condition C main set.
Prompt: legacy/v1_submission/scripts/archaeology/cpg_archaeology.py generate_controls() with the two EXCLUSION
lines removed; host and language as sent in v1 (keys of data/control_cache.json). gpt-5.2, temperature 0.3,
max_completion_tokens 300. API errors retried up to 3 times; empty or short answers are not regenerated.
Post-processing as in v1, then R1-R4 against the condition C clean host (clean_rules.py).

Outputs: embeddings/results/logs/controls_D.jsonl  every request: time, model, parameters, prompt, response, error
         data/controls_D.json  {item_id: {host_v1, clean_host, language, generated: [...], controls: [(word, rule)],
                                           kept: [...]}}
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_rules import clean  # noqa: E402
from target_blind_common import ROOT, main_items  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--offline", action="store_true")
ap.add_argument("--model", default="gpt-5.2")
args = ap.parse_args()
LOG = ROOT / "embeddings/results/logs/controls_D.jsonl"
N = 10


def prompt_for(host_word, language="", n=N):
    lang_str = language if language else "same language as the candidate"
    return (
        "You are helping with a linguistic experiment.\n\n"
        f"Give me exactly {n} words or phrases that are SEMANTICALLY CLOSEST to:\n"
        f'  "{host_word}"\n\n'
        f"LANGUAGE: {lang_str}\n"
        "ALL words MUST be in the same language as the candidate. This is mandatory.\n\n"
        "Think of it like numbers: if the candidate is 69, I want 68, 70, 67, 71...\n"
        f'The NEAREST NEIGHBORS in meaning. Words a native speaker would say are '
        f'most similar to "{host_word}".\n\n'
        "Return ONLY the words, one per line. No numbers, no explanation, no translations."
    )


def postprocess(text_r, host_word, n=N):  # v1 post-processing, unchanged
    controls = [line.strip().strip('.-•*0123456789) ') for line in text_r.strip().split('\n')
                if line.strip() and len(line.strip()) < 80]
    controls = [c for c in controls if c.lower() != host_word.lower() and len(c) > 0]
    return controls[:n]


items = main_items()
# check: clean_rules reproduces condition C on the stored controls
import pandas as pd  # noqa: E402
cache = json.load(open(ROOT / "data/control_cache.json", encoding="utf-8"))
cc = pd.read_csv(ROOT / "embeddings/controls/controls_clean.csv", dtype=str).fillna("")
for it in items:
    stored = cache[it["cache_key"]]
    mine = [r for _, r in clean(stored, it["clean_host"], it["host_keys"])]
    assert mine == cc[cc.item_id == it["item_id"]].rule_dropped.tolist(), it["item_id"]

responses = {}
if args.offline:
    for line in open(LOG, encoding="utf-8"):
        e = json.loads(line)
        if e["error"] is None:
            responses[e["item_id"]] = e["response"]
else:
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(ROOT / ".env")
    client = OpenAI()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for i, it in enumerate(items):
        p = prompt_for(it["host_v1"], it["language"])
        for attempt in range(4):
            err, text, meta = None, None, {}
            try:
                r = client.chat.completions.create(model=args.model, messages=[{"role": "user", "content": p}],
                                                   temperature=0.3, max_completion_tokens=300)
                text = r.choices[0].message.content or ""
                meta = {"response_model": r.model, "finish_reason": r.choices[0].finish_reason,
                        "usage": r.usage.model_dump() if r.usage else None}
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"time": dt.datetime.now().astimezone().isoformat(), "item_id": it["item_id"],
                                    "model": args.model, "temperature": 0.3, "max_completion_tokens": 300,
                                    "attempt": attempt + 1, "prompt": p, "response": text, "error": err, **meta},
                                   ensure_ascii=False) + "\n")
            if err is None:
                responses[it["item_id"]] = text
                break
            time.sleep(2 ** attempt)
        print(f"{i + 1}/{len(items)} {it['item_id']} {'ok' if it['item_id'] in responses else 'FAILED'}", flush=True)

out = {}
for it in items:
    gen = postprocess(responses.get(it["item_id"], ""), it["host_v1"])
    cl = clean(gen, it["clean_host"], it["host_keys"])
    out[it["item_id"]] = {"host_v1": it["host_v1"], "clean_host": it["clean_host"], "language": it["language"],
                          "generated": gen, "controls": cl, "kept": [c for c, r in cl if r == ""]}
json.dump(out, open(ROOT / "data/controls_D.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n10 = sum(len(v["kept"]) >= 10 for v in out.values())
print("items:", len(out), "| responses:", len(responses), "| with 10 kept:", n10)
