"""Source audit, stage B (docs/analysis_plan_source_audit.md): one coder judges every item from its quotes only.

Usage (run once per coder, each in a fresh process):
    python3 audit/scripts/judge_evidence.py --provider openai    --model gpt-5.6-sol            --no-temperature
    python3 audit/scripts/judge_evidence.py --provider anthropic --model claude-opus-5-5        --no-temperature
    python3 audit/scripts/judge_evidence.py --provider google    --model gemini-3.1-pro-preview
Output: audit/runs/judge_<model>.csv (one row per item sent) and audit/runs/judge_<model>.jsonl (every request and
raw response). Resumes: items already labelled ok are skipped.

Input per item: host (original and romanised), covert target, tradition, and the quotes of results/
source_audit_evidence.csv, each with its source_type and retrieved value (documents with ai_generated = yes are left
out). Not sent: citation, URL, item ID (the coder sees a batch-local number), pull. Items with no quote are not sent
(no_evidence by rule; see aggregate_judgements.py). Batches of 10 in the audit order (seed 20261023); no earlier
answers are sent. A batch whose output fails validation, or that a coder refuses, is resent up to 3 times; after
that its items are sent one at a time (up to 3 times each). Every refusal is logged (field "refusal").
Keys come from the environment: OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_tool import ROOT, order  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["openai", "google", "anthropic"], required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--limit", type=int, default=0, help="only the first N items sent (pilot)")
ap.add_argument("--no-temperature", action="store_true", help="for models that reject temperature")
args = ap.parse_args()
use_temp = not args.no_temperature
LABELS = ["supported", "partly", "contradicted", "no_evidence"]

SYSTEM = """You are a coder in a source audit for a research study. Each item claims that, in a given community or
tradition, a given expression (the host) carries a given covert meaning. For each item you receive short passages
quoted verbatim from documents that were retrieved for it, each with the kind of document (scholarly, reference =
dictionary or encyclopaedia, wikipedia, web) and whether the passage comes from the full text or only an abstract.

Judge each item ONLY on the quoted text. Do not use any other knowledge, including what you know about the
expression, the tradition or the documents. If the quotes do not say it, it is not there.

Give exactly one label per item:
- supported: a quote states that this host carries this covert meaning in this community;
- partly: a quote states that the tradition or technique exists, but not the link between this host and this
  covert meaning; or it states a related but different meaning;
- contradicted: a quote states something that conflicts with the item;
- no_evidence: no relevant quote.

Return only a JSON array with one object per item, in the input order:
[{"n": <the item's number>, "label": "<one of the four labels>", "reason": "<one short sentence>"}]"""


def call(user_msg):
    """Returns (text, refusal) where refusal is a short description or None."""
    if args.provider == "openai":
        from openai import OpenAI
        kw = {"temperature": 0} if use_temp else {}
        r = OpenAI().chat.completions.create(
            model=args.model, messages=[{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": user_msg}], **kw)
        m = r.choices[0].message
        if getattr(m, "refusal", None):
            return "", f"refusal: {m.refusal}"
        return m.content or "", None
    if args.provider == "google":
        from google import genai
        from google.genai import types
        cfg = types.GenerateContentConfig(system_instruction=SYSTEM, response_mime_type="application/json",
                                          **({"temperature": 0} if use_temp else {}))
        global _gclient  # keep one client alive; a temporary one is closed before the response is read
        if "_gclient" not in globals():
            _gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        r = _gclient.models.generate_content(model=args.model, contents=user_msg, config=cfg)
        if not r.text:
            fr = r.candidates[0].finish_reason if r.candidates else None
            return "", f"no text: finish_reason={fr}, feedback={r.prompt_feedback}"
        return r.text, None
    import anthropic
    kw = {"temperature": 0} if use_temp else {}
    r = anthropic.Anthropic().messages.create(model=args.model, max_tokens=16000, system=SYSTEM,
                                              messages=[{"role": "user", "content": user_msg}], **kw)
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    if r.stop_reason == "refusal" or not text.strip():
        return text, f"stop_reason={r.stop_reason}"
    return text, None


def parse(text):
    m = re.search(r"\[.*\]", text.strip(), re.S)
    return json.loads(m.group(0) if m else text)


def validate(obj, n):
    if not isinstance(obj, list) or [o.get("n") for o in obj] != list(range(1, n + 1)):
        return "numbers do not match the input order"
    for o in obj:
        if o.get("label") not in LABELS:
            return f"{o.get('n')}: bad label {o.get('label')!r}"
    return None


def payload(group):
    out = []
    for k, it in enumerate(group, 1):
        r = I.loc[it]
        qs = []
        for d in ev[ev.item_id == it].itertuples():
            for q in (d.quote, d.quote_2):
                if q:
                    qs.append({"document": f"D{d.doc_rank}", "source_type": d.source_type,
                               "retrieved": d.retrieved, "quote": q})
        out.append({"n": k, "host": r.host_original, "host_romanized": r.host_romanized,
                    "covert_target": r.covert_target, "tradition": r.tradition, "quotes": qs})
    return ("Judge the following items. Return a JSON array with one object per item, same order.\n\n"
            + json.dumps(out, ensure_ascii=False, indent=1))


def send(group, mode):
    """Up to 3 attempts; returns the parsed list or None. Every attempt is logged."""
    for attempt in range(1, 4):
        t0, refusal = time.time(), None
        try:
            raw, refusal = call(payload(group))
            obj = parse(raw) if refusal is None else None
            err = refusal or validate(obj, len(group))
        except Exception as e:  # network or JSON errors
            raw, obj, err = "", None, f"{type(e).__name__}: {e}"
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                                "provider": args.provider, "model": args.model, "temperature_applied": use_temp,
                                "mode": mode, "attempt": attempt, "item_ids": group, "refusal": refusal,
                                "error": err, "seconds": round(time.time() - t0, 1), "raw": raw},
                               ensure_ascii=False) + "\n")
        if err is None:
            return obj
        print(f"  {mode} {group[0]}.. attempt {attempt} failed: {err}", file=sys.stderr)
        time.sleep(5 * attempt)
    return None


I = pd.read_csv(ROOT / "data/blind_items.csv", dtype=str).fillna("").set_index("item_id")
ev = pd.read_csv(ROOT / "results/source_audit_evidence.csv", dtype=str).fillna("")
ev = ev[(ev.ai_generated != "yes") & ((ev.quote != "") | (ev.quote_2 != ""))]
ev = ev.assign(_r=ev.doc_rank.astype(int)).sort_values(["item_id", "_r"])
sent = [x for x in order() if x in set(ev.item_id)]
if args.limit:
    sent = sent[:args.limit]
RUNS = ROOT / "audit/runs"
RUNS.mkdir(parents=True, exist_ok=True)
OUT, LOG = RUNS / f"judge_{args.model}.csv", RUNS / f"judge_{args.model}.jsonl"
done = set()
if OUT.exists():
    prev = pd.read_csv(OUT, dtype=str)
    ok = prev[prev.status == "ok"].drop_duplicates("item_id", keep="last")
    ok.to_csv(OUT, index=False)
    done = set(ok.item_id)
todo = [x for x in sent if x not in done]
print(f"{len(todo)} of {len(sent)} items to judge with {args.provider}:{args.model}")
for s in range(0, len(todo), args.batch):
    group = todo[s:s + args.batch]
    rows = []
    res = send(group, "batch")
    if res is not None:
        rows = [(it, o) for it, o in zip(group, res)]
    else:
        for it in group:
            r1 = send([it], "single")
            rows.append((it, r1[0] if r1 else None))
    pd.DataFrame([{"item_id": it, "label": o["label"] if o else "", "reason": o.get("reason", "") if o else "",
                   "status": "ok" if o else "failed", "model": args.model} for it, o in rows]) \
        .to_csv(OUT, mode="a", header=not OUT.exists(), index=False)
    print(f"  judged {min(s + args.batch, len(todo))}/{len(todo)}")
print("done:", OUT)
