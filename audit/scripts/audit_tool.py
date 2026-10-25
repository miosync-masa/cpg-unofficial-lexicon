"""Source audit, stage A (docs/analysis_plan_source_audit.md): tools for collecting evidence.

  python3 audit/scripts/audit_tool.py order [N]       first N items in the audit order (seed 20261023)
  python3 audit/scripts/audit_tool.py item ITEM       the item, its recorded sources (Phase 2) and its documents so far
  python3 audit/scripts/audit_tool.py fetch ITEM RANK URL TYPE LISTED "CITATION" [--retrieved R] [--keep-failed]
        retrieves URL once and stores the raw page and its text under private/audit_snapshots/; TYPE =
        scholarly|reference|wikipedia|web, LISTED = yes|no. Every request is logged. A failed request is kept as a
        document row only with --keep-failed (for a source listed by the agent); otherwise it is only logged.
  python3 audit/scripts/audit_tool.py record ITEM RANK TYPE LISTED "CITATION" URL RETRIEVED
        a document row without a request (e.g. a listed source that could not be found: RETRIEVED = failed)
  python3 audit/scripts/audit_tool.py exists ITEM RANK exists|not_found|inaccessible
        for a source listed by the agent: could its existence be established?
  python3 audit/scripts/audit_tool.py grep ITEM RANK "regex"   passages of the stored text matching the regex
  python3 audit/scripts/audit_tool.py quote ITEM RANK "QUOTE" [--second]
        sets quote (or quote_2): a verbatim passage of the stored text (whitespace normalised), at most 40 words
        (Japanese/Chinese: at most 80 characters)
  python3 audit/scripts/audit_tool.py cite ITEM RANK "CITATION"  corrects the citation of a document row (no new request)
  python3 audit/scripts/audit_tool.py setret ITEM RANK full_text|abstract|snippet|record_only
        corrects the retrieved value (abstract = a page that gives only the abstract of a work)
  python3 audit/scripts/audit_tool.py flag ITEM RANK yes|no      ai_generated: the page says it is AI-generated (not used
        in stage B; does not count towards the 3 documents)
  python3 audit/scripts/audit_tool.py nftype ITEM RANK unspecific|specific_not_found|url_unretrievable
        not_found_type of an agent source with agent_source = not_found (author, 2026-10-25): (a) names no specific
        document, (b) a specific document that a search did not find, (c) a URL that could not be retrieved
  python3 audit/scripts/audit_tool.py wayback ITEM RANK [URL]
        for a document that failed because of a server outage or a dead link (never for a refusal: 401, 403, 429):
        asks the Wayback Machine for the capture closest to 2026-03-13 and stores it; the row gets retrieved =
        archive and archive_url. URL: the address given by the agent when the row has none.
  python3 audit/scripts/audit_tool.py drop ITEM RANK            removes a document row (the attempt stays logged)
  python3 audit/scripts/audit_tool.py export                    writes results/source_audit_evidence.csv

Rules (author, 2026-10-24): Phase 2 items: the first 3 sources listed by the agent are always checked, each with an
`agent_source` value. Further documents are searched only while the item has no supporting quote; at most 3
documents with text (full_text, abstract or snippet; not ai_generated) per item; record_only rows and failed
listed sources do not count. Author, 2026-10-24 (second set): abstract-only pages are `abstract`, not `full_text`;
pages marked as AI-generated get ai_generated = yes.
Working files: private/audit_work/<item>.csv; attempts: private/audit_attempts/<item>.jsonl (one file per item, so
that several collectors can work at the same time). Item data come from data/blind_items.csv (no pull column).
"""
import datetime as dt
import hashlib
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
PRIV = ROOT / "private"
SNAP = PRIV / "audit_snapshots"
WDIR = PRIV / "audit_work"
ADIR = PRIV / "audit_attempts"
COLS = ["item_id", "doc_rank", "source_type", "citation", "url", "accessed", "retrieved", "quote", "quote_2",
        "snapshot_sha256", "listed_by_agent", "agent_source", "ai_generated", "not_found_type", "archive_url"]
NFTYPES = ("unspecific", "specific_not_found", "url_unretrievable")
WAYBACK_DATE = "20260313"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15"


def items():
    return pd.read_csv(ROOT / "data/blind_items.csv", dtype=str).fillna("")


def order():
    ids = items().item_id.tolist()
    return [ids[i] for i in np.random.default_rng(20261023).permutation(len(ids))]


def work(it):
    f = WDIR / f"{it}.csv"
    if f.exists():
        w = pd.read_csv(f, dtype=str).fillna("")
        for c in COLS:
            if c not in w:
                w[c] = ""
        return w[COLS]
    return pd.DataFrame(columns=COLS)


def save(it, w):
    WDIR.mkdir(parents=True, exist_ok=True)
    w = w.assign(_r=w.doc_rank.astype(int)).sort_values("_r").drop(columns="_r")
    w[COLS].to_csv(WDIR / f"{it}.csv", index=False)


def upsert(row):
    it = row["item_id"]
    w = work(it)
    w = w[w.doc_rank != str(row["doc_rank"])]
    save(it, pd.concat([w, pd.DataFrame([{c: str(row.get(c, "")) for c in COLS}])], ignore_index=True))


def log(it, rec):
    ADIR.mkdir(parents=True, exist_ok=True)
    with open(ADIR / f"{it}.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()


def text_of(content, ctype):
    if b"%PDF" in content[:1024] or "pdf" in ctype:
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(content)).pages), "pdf"
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return soup.get_text("\n"), "html"


def row(it, rank):
    w = work(it)
    r = w[w.doc_rank == rank]
    assert len(r), f"no document {it} {rank}"
    return w, r.index[0]


def cjk(s):
    return sum(1 for c in s if "぀" <= c <= "ヿ" or "㐀" <= c <= "鿿" or "豈" <= c <= "﫿")


def too_long(q):
    return len(q) > 80 if cjk(q) > len(q) / 3 else len(q.split()) > 40


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "order":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 217
        for i, x in enumerate(order()[:n], 1):
            print(i, x)
    elif cmd == "item":
        it = sys.argv[2]
        r = items().set_index("item_id").loc[it]
        print(f"{it} | {r.language} | {r.tradition}\n host: {r.host_original} / {r.host_romanized}\n"
              f" target: {r.covert_target}")
        for f in sorted((ROOT / "data/source").glob("ashigaru*.yaml")):
            for x in yaml.safe_load(open(f, encoding="utf-8"))["items"]:
                if x["id"] == it:
                    print(" recorded source (Phase 2):", x.get("source"))
        w = work(it)
        if len(w):
            w = w.assign(q=(w.quote != "").astype(int) + (w.quote_2 != "").astype(int))
            print(w[["doc_rank", "source_type", "retrieved", "listed_by_agent", "agent_source", "ai_generated", "q", "url"]]
                  .to_string(index=False))
    elif cmd == "fetch":
        it, rank, url, typ, listed, cit = sys.argv[2:8]
        retrieved = sys.argv[sys.argv.index("--retrieved") + 1] if "--retrieved" in sys.argv else None
        acc = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            r = requests.get(url, headers={"User-Agent": UA, "Accept-Language": "en,ja;q=0.8"}, timeout=40)
            status, content, ctype = r.status_code, r.content, r.headers.get("content-type", "")
        except Exception as e:
            status, content, ctype = f"error {type(e).__name__}", b"", ""
        sha, ok = "", False
        if status == 200 and content:
            sha = hashlib.sha256(content).hexdigest()
            try:
                txt, kind = text_of(content, ctype)
            except Exception as e:
                txt, kind = "", "bin"
                print("text extraction failed:", type(e).__name__)
            SNAP.mkdir(parents=True, exist_ok=True)
            (SNAP / f"{sha}.{kind}").write_bytes(content)
            (SNAP / f"{sha}.txt").write_text(txt, encoding="utf-8")
            retrieved = retrieved or ("full_text" if len(norm(txt)) > 500 else "failed")
            ok = retrieved != "failed"
            print(f"HTTP 200, {len(content)} bytes, text {len(norm(txt))} chars -> {retrieved}")
        else:
            retrieved = "failed"
            print(f"not retrieved: {status}")
        keep = ok or "--keep-failed" in sys.argv
        log(it, {"item_id": it, "doc_rank": rank, "url": url, "accessed": acc, "status": str(status), "sha256": sha,
                 "retrieved": retrieved, "kept": keep})
        if keep:
            upsert({"item_id": it, "doc_rank": rank, "source_type": typ, "citation": cit, "url": url, "accessed": acc,
                    "retrieved": retrieved, "snapshot_sha256": sha, "listed_by_agent": listed})
        else:
            print("(not kept as a document; attempt logged)")
    elif cmd == "record":
        it, rank, typ, listed, cit, url, ret = sys.argv[2:9]
        acc = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        log(it, {"item_id": it, "doc_rank": rank, "url": url, "accessed": acc, "status": "record", "retrieved": ret,
                 "citation": cit})
        upsert({"item_id": it, "doc_rank": rank, "source_type": typ, "citation": cit, "url": url, "accessed": acc,
                "retrieved": ret, "listed_by_agent": listed})
    elif cmd == "exists":
        it, rank, v = sys.argv[2:5]
        assert v in ("exists", "not_found", "inaccessible")
        w, i = row(it, rank)
        assert w.at[i, "listed_by_agent"] == "yes", "only for sources listed by the agent"
        w.at[i, "agent_source"] = v
        save(it, w)
        print(it, rank, v)
    elif cmd == "grep":
        it, rank, pat = sys.argv[2:5]
        w, i = row(it, rank)
        t = norm((SNAP / f"{w.at[i, 'snapshot_sha256']}.txt").read_text(encoding="utf-8"))
        last, k = -10 ** 9, 0
        for m in re.finditer(pat, t, flags=re.I):
            if m.start() < last + 400 or k >= 10:
                continue
            last, k = m.start(), k + 1
            print("…" + t[max(0, m.start() - 250):m.end() + 250] + "…\n")
        if k == 0:
            print("(no match)")
    elif cmd == "quote":
        it, rank, q = sys.argv[2:5]
        nq = norm(q)
        assert not too_long(nq), "too long (40 words; Japanese/Chinese 80 characters)"
        w, i = row(it, rank)
        t = norm((SNAP / f"{w.at[i, 'snapshot_sha256']}.txt").read_text(encoding="utf-8"))
        assert nq in t, "quote is not a verbatim passage of the stored text"
        col = "quote_2" if "--second" in sys.argv else "quote"
        log(it, {"item_id": it, "doc_rank": rank, col + "_changed": [w.at[i, col], nq],
                 "time": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
        w.at[i, col] = nq
        save(it, w)
        print("quote set:", nq)
    elif cmd == "cite":
        it, rank, cit = sys.argv[2:5]
        w, i = row(it, rank)
        log(it, {"item_id": it, "doc_rank": rank, "citation_changed": [w.at[i, "citation"], cit]})
        w.at[i, "citation"] = cit
        save(it, w)
        print("citation set:", cit)
    elif cmd == "setret":
        it, rank, v = sys.argv[2:5]
        assert v in ("full_text", "abstract", "snippet", "record_only", "archive")
        w, i = row(it, rank)
        log(it, {"item_id": it, "doc_rank": rank, "retrieved_changed": [w.at[i, "retrieved"], v],
                 "time": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
        w.at[i, "retrieved"] = v
        save(it, w)
        print(it, rank, "retrieved =", v)
    elif cmd == "flag":
        it, rank, v = sys.argv[2:5]
        assert v in ("yes", "no")
        w, i = row(it, rank)
        log(it, {"item_id": it, "doc_rank": rank, "ai_generated": v,
                 "time": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
        w.at[i, "ai_generated"] = v
        save(it, w)
        print(it, rank, "ai_generated =", v)
    elif cmd == "nftype":
        it, rank, v = sys.argv[2:5]
        assert v in NFTYPES + ("",)
        w, i = row(it, rank)
        assert w.at[i, "agent_source"] == "not_found" or v == "", "only for agent sources with agent_source = not_found"
        log(it, {"item_id": it, "doc_rank": rank, "not_found_type": v,
                 "time": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
        w.at[i, "not_found_type"] = v
        save(it, w)
        print(it, rank, "not_found_type =", v)
    elif cmd == "wayback":
        it, rank = sys.argv[2:4]
        w, i = row(it, rank)
        url = sys.argv[4] if len(sys.argv) > 4 else w.at[i, "url"]
        assert url, "no URL"
        acc = dt.datetime.now().astimezone().isoformat(timespec="seconds")
        snap_url, status, sha, content, ctype = "", "", "", b"", ""
        try:
            a = requests.get("https://archive.org/wayback/available", params={"url": url, "timestamp": WAYBACK_DATE},
                             headers={"User-Agent": UA}, timeout=40).json()
            c = a.get("archived_snapshots", {}).get("closest") or {}
            if c.get("available") and str(c.get("status", "200")) == "200":
                snap_url = c["url"].replace("http://", "https://", 1)
                raw = re.sub(r"/web/(\d+)/", r"/web/\1id_/", snap_url, count=1)  # the archived page without the banner
                r = requests.get(raw, headers={"User-Agent": UA}, timeout=60)
                status, content, ctype = r.status_code, r.content, r.headers.get("content-type", "")
            else:
                status = "no capture"
        except Exception as e:
            status = f"error {type(e).__name__}"
        ok = False
        if status == 200 and content:
            sha = hashlib.sha256(content).hexdigest()
            try:
                txt, kind = text_of(content, ctype)
            except Exception as e:
                txt, kind = "", "bin"
            SNAP.mkdir(parents=True, exist_ok=True)
            (SNAP / f"{sha}.{kind}").write_bytes(content)
            (SNAP / f"{sha}.txt").write_text(txt, encoding="utf-8")
            ok = len(norm(txt)) > 500
            print(f"capture {snap_url}: {len(content)} bytes, text {len(norm(txt))} chars -> "
                  + ("archive" if ok else "too short, not kept"))
        else:
            print("not retrieved:", status, snap_url)
        log(it, {"item_id": it, "doc_rank": rank, "wayback_for": url, "archive_url": snap_url, "accessed": acc,
                 "status": str(status), "sha256": sha, "retrieved": "archive" if ok else "failed", "kept": ok})
        if ok:
            prev = w.loc[i, ["retrieved", "url", "accessed", "snapshot_sha256"]].to_dict()
            log(it, {"item_id": it, "doc_rank": rank, "replaced_by_archive": prev, "time": acc})
            if not w.at[i, "url"]:
                w.at[i, "url"] = url
            w.at[i, "retrieved"], w.at[i, "archive_url"] = "archive", snap_url
            w.at[i, "accessed"], w.at[i, "snapshot_sha256"] = acc, sha
            save(it, w)
    elif cmd == "drop":
        it, rank = sys.argv[2:4]
        w = work(it)
        save(it, w[w.doc_rank != rank])
        log(it, {"item_id": it, "doc_rank": rank, "dropped": True,
                 "time": dt.datetime.now().astimezone().isoformat(timespec="seconds")})
        print("dropped", it, rank)
    elif cmd == "export":
        fs = sorted(WDIR.glob("*.csv"))
        w = pd.concat([work(f.stem) for f in fs], ignore_index=True) if fs else pd.DataFrame(columns=COLS)
        pos = {x: i for i, x in enumerate(order())}
        w = (w.assign(_o=w.item_id.map(pos), _r=w.doc_rank.astype(int)).sort_values(["_o", "_r"])
             .drop(columns=["_o", "_r"]))
        w[COLS].to_csv(ROOT / "results/source_audit_evidence.csv", index=False)
        print(len(w), "rows,", w.item_id.nunique(), "items")
