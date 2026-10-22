"""Code form and boundary fields (codebook v1.1) for every item in blind_items.csv, following codebook_v1_1.md.

Usage (run once per model, each in a fresh process):
    python3 run_v1_1_coding.py --provider openai    --model gpt-5.6-sol            --out results/v1_1_gpt.csv --no-temperature
    python3 run_v1_1_coding.py --provider google    --model gemini-3.1-pro-preview --out results/v1_1_gemini.csv
    python3 run_v1_1_coding.py --provider anthropic --model claude-opus-5-5        --out results/v1_1_claude.csv --no-temperature

Same structure as run_llm_coding.py (batching, validation, retries, JSONL log, resume); only the codebook,
the fields and validate() differ.

Keys come from the environment: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY.
Requires: pip install openai google-genai anthropic pandas

Every request and raw response is appended to <out>.jsonl, so the run is fully auditable.
Items are sent in batches (default 10). A batch whose output fails validation is retried up to 3 times;
items that still fail are written with status=failed so nothing is silently dropped.
Temperature is 0 where the model accepts it; the run log records whether it was applied.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env")  # keys live in .env at the repository root

ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["openai", "google", "anthropic"], required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--items", default=str(ROOT / "data/blind_items.csv"))
ap.add_argument("--codebook", default=str(ROOT / "coding/codebooks/codebook_v1_1.md"))
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--limit", type=int, default=0, help="only code the first N items (pilot)")
ap.add_argument("--no-temperature", action="store_true", help="for models that reject temperature")
args = ap.parse_args()

CODEBOOK = Path(args.codebook).read_text(encoding="utf-8")
SYSTEM = ("You are an expert in comparative linguistics, rhetoric and folklore acting as a blind coder "
          "for a research study. Follow the codebook exactly. Output only valid JSON.\n\n" + CODEBOOK)
ENUM = {
    "form_type": {"existing_form", "transparent_combination", "opaque_coinage", "blend_or_abbreviation",
                  "expressive_coinage", "respelling"},
    "two_storey": {"yes", "no"},
    "content_or_attitude": {"content", "attitude", "both"},
    "outsider_inference": {"inferable", "inferable_with_effort", "not_inferable", "not_motivated"},
    "host_unit": {"word", "phrase", "utterance"},
}
FIELDS = ["item_id", "form_type", "two_storey", "content_or_attitude", "outsider_inference", "host_unit",
          "confidence", "rationale"]
use_temp = not args.no_temperature


def call(user_msg):
    if args.provider == "openai":
        from openai import OpenAI
        kw = {"temperature": 0} if use_temp else {}
        r = OpenAI().chat.completions.create(
            model=args.model, messages=[{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": user_msg}], **kw)
        return r.choices[0].message.content
    if args.provider == "google":
        from google import genai
        from google.genai import types
        cfg = types.GenerateContentConfig(system_instruction=SYSTEM,
                                          response_mime_type="application/json",
                                          **({"temperature": 0} if use_temp else {}))
        global _gclient  # keep one client alive; a temporary one is closed before the response is read
        if "_gclient" not in globals():
            _gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        r = _gclient.models.generate_content(model=args.model, contents=user_msg, config=cfg)
        return r.text
    if args.provider == "anthropic":
        import anthropic
        kw = {"temperature": 0} if use_temp else {}
        r = anthropic.Anthropic().messages.create(
            model=args.model, max_tokens=16000, system=SYSTEM,
            messages=[{"role": "user", "content": user_msg}], **kw)
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        if not text.strip():  # surface why, instead of a bare JSONDecodeError
            raise RuntimeError(f"no text: stop_reason={r.stop_reason}, blocks={[b.type for b in r.content]}")
        return text


def parse(text):
    text = text.strip()
    m = re.search(r"\[.*\]", text, re.S)
    return json.loads(m.group(0) if m else text)


def validate(obj, expected_ids):
    if not isinstance(obj, list) or [o.get("item_id") for o in obj] != expected_ids:
        return "item_id list does not match input order"
    for o in obj:
        if set(o) != set(FIELDS):
            return f"{o.get('item_id')}: keys {sorted(o)} != {sorted(FIELDS)}"
        for k, allowed in ENUM.items():
            if str(o.get(k)) not in allowed:
                return f"{o.get('item_id')}: bad {k}={o.get(k)!r}"
        if not str(o.get("rationale", "")).strip():
            return f"{o.get('item_id')}: empty rationale"
        if o.get("confidence") not in (1, 2, 3):
            return f"{o.get('item_id')}: bad confidence"
    return None


items = pd.read_csv(args.items, dtype=str).fillna("")
if args.limit:
    items = items.head(args.limit)
out_csv, log = Path(args.out), Path(args.out + ".jsonl")
out_csv.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out_csv.exists():
    prev = pd.read_csv(out_csv, dtype=str)
    ok = prev[prev.status == "ok"].drop_duplicates("item_id", keep="last")
    if ok.empty:  # nothing usable yet: start the file over so the header covers all fields
        out_csv.unlink()
    elif len(ok) < len(prev):  # drop failed rows so those items are retried, not skipped
        ok.to_csv(out_csv, index=False, encoding="utf-8")
    done = set(ok.item_id)
cols = ["item_id", "language", "tradition", "host_original", "host_romanized",
        "covert_target", "source_reference"]
todo = items[~items.item_id.isin(done)]
print(f"{len(todo)} items to code with {args.provider}:{args.model}")

for start in range(0, len(todo), args.batch):
    batch = todo.iloc[start:start + args.batch]
    ids = batch.item_id.tolist()
    user = ("Code the following items. Return a JSON array with one object per item, same order.\n\n"
            + json.dumps(batch[cols].to_dict("records"), ensure_ascii=False, indent=1))
    result, err = None, None
    for attempt in range(1, 4):
        t0 = time.time()
        try:
            raw = call(user)
            obj = parse(raw)
            err = validate(obj, ids)
        except Exception as e:  # network or JSON errors
            raw, obj, err = "", None, f"{type(e).__name__}: {e}"
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": dt.datetime.now().isoformat(), "provider": args.provider,
                                "model": args.model, "temperature_applied": use_temp,
                                "attempt": attempt, "item_ids": ids, "error": err,
                                "seconds": round(time.time() - t0, 1), "raw": raw},
                               ensure_ascii=False) + "\n")
        if err is None:
            result = obj
            break
        print(f"  batch {ids[0]}.. attempt {attempt} failed: {err}", file=sys.stderr)
        time.sleep(5 * attempt)
    rows = []
    if result:
        for o in result:
            rows.append({**{k: (json.dumps(o[k], ensure_ascii=False) if isinstance(o.get(k), list)
                                else o.get(k, "")) for k in FIELDS},
                         "status": "ok", "model": args.model})
    else:
        rows = [{"item_id": i, "status": "failed", "model": args.model} for i in ids]
    pd.DataFrame(rows).reindex(columns=FIELDS + ["status", "model"]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False,
                              encoding="utf-8")
    print(f"  coded {min(start + args.batch, len(todo))}/{len(todo)}")
print("done:", out_csv)
