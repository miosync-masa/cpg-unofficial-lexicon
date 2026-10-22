"""Fetch the Kotobank page for each of the 41 T4 headwords and extract the デジタル大辞泉 text.

Usage: python3 dictionary_check/scripts/fetch_kotobank.py

The dictionary text is copyrighted, so the worksheet, the raw HTML and the extracted text stay in
dictionary_check/private/ (not published). Set CONTACT_EMAIL in .env to put a contact address in the User-Agent.

Lookup rule, from the 「引く見出し（目安）」 column of T4_dictionary_check.xlsx:
    牡丹               -> 牡丹
    紅白／かまぼこ      -> 紅白 and かまぼこ (every alternative is fetched)
    手（手を出す）      -> 手を出す first (Kotobank has idiom pages); if it has no 大辞泉 entry, the parent 手
    床（とこ）          -> 床, keeping only 大辞泉 entries whose reading is とこ (kana in parentheses = reading)

Outputs:
    private/kotobank_raw/<item_id>__<query>.html   response bytes exactly as received (one file per request)
    fetch_log.csv                                  every request: item, query, URL, final URL, HTTP status, time
    private/daijisen_text.csv                      見出し語, 引いた語, URL, 取得日時, 本文 (blank when nothing
                                                   was retrieved)

Spelling variants: when a query yields no 大辞泉 entry, the 大辞泉 spellings in VARIANTS are tried. A variant
entry is kept only if its headline (h3, with ‐ ・ （ ） ＝ ▽ × removed) contains the original query, or its kana
reading equals the query's reading in READING; every other entry on that page is dropped. Each attempt is in fetch_log.csv.

Body text: the h3 headline plus section.description of every 大辞泉 entry on the page, figures/ads removed,
ruby readings dropped (base text kept), <br> as line break. Nothing is filled in by hand.
"""
import datetime as dt
import os
import re
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parents[1]
PRIVATE = HERE / "private"
load_dotenv(HERE.parent / ".env")
RAW = PRIVATE / "kotobank_raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0 (academic dictionary check; CPG revision" + (
    f"; contact {os.environ['CONTACT_EMAIL']})" if os.environ.get("CONTACT_EMAIL") else ")")
VARIANTS = {"逢う": ["会う", "遭う"], "伊達巻き": ["伊達巻"], "かまぼこ": ["蒲鉾"], "ごぼう": ["牛蒡"],
            "煮しめ": ["煮染め", "煮染"], "お泊まり": ["御泊まり"], "泊まり": ["泊り", "泊"], "付き合う": ["付合う", "付き合い"]}
# reading of each variant-looked-up word, so a same-spelling different word (煮染 にぞめ) is not taken
READING = {"逢う": "あう", "伊達巻き": "だてまき", "かまぼこ": "かまぼこ", "ごぼう": "ごぼう", "煮しめ": "にしめ",
           "お泊まり": "おとまり", "泊まり": "とまり", "付き合う": "つきあう"}
KANA = re.compile(r"^[ぁ-ゖァ-ヺー]+$")

sheet = pd.read_excel(PRIVATE / "T4_dictionary_check.xlsx", sheet_name="辞書照合", header=None)
hdr = sheet.index[sheet[0] == "item_id"][0]
rows = sheet.iloc[hdr + 1:, :5]
rows.columns = ["item_id", "headword", "lookup", "covert", "tradition"]
rows = rows[rows.item_id.astype(str).str.match(r"^T_\d+$")].reset_index(drop=True)
assert len(rows) == 41, len(rows)


def plan(lookup):
    """-> list of (query, reading_filter, fallback_query)."""
    out = []
    for alt in str(lookup).split("／"):
        m = re.match(r"^(.+?)（(.+)）$", alt.strip())
        if not m:
            out.append((alt.strip(), None, None))
        elif KANA.match(m.group(2)):
            out.append((m.group(1), m.group(2), None))
        else:
            out.append((m.group(2), None, m.group(1)))  # idiom first, parent word as fallback
    return out


def norm(s):
    return re.sub(r"[‐・（）()×▽＝=]", "", s)


def entries(html, reading, must=None):
    soup = BeautifulSoup(html, "html.parser")
    art = soup.select_one("article.daijisen")
    if art is None:
        return []
    found = []
    for ex in art.select("div.ex"):
        h3 = ex.find("h3")
        desc = ex.select_one("section.description")
        if h3 is None or desc is None:
            continue
        for junk in desc.select("div.media, figure, .pc-word-ad, script, style"):
            junk.decompose()
        for tag in ex.find_all("rt"):
            tag.decompose()
        for br in desc.find_all("br"):
            br.replace_with("\n")
        head = h3.get_text("", strip=True)
        if reading and not head.replace("‐", "").startswith(reading):
            continue
        if must and must not in norm(head) and norm(head).split("【")[0] not in (must, READING.get(must)):
            continue
        body = "\n".join(l.strip() for l in desc.get_text("").splitlines() if l.strip())
        found.append(f"【見出し】{head}\n{body}")
    return found


log, out = [], []
session = requests.Session()
session.headers["User-Agent"] = UA
for _, r in rows.iterrows():
    texts, urls, times = [], [], []
    for query, reading, fallback in plan(r.lookup):
        tries = [(query, None)] + [(v, query) for v in VARIANTS.get(query, [])]
        if fallback:
            tries += [(fallback, None)] + [(v, fallback) for v in VARIANTS.get(fallback, [])]
        for q, must in tries:
            url = "https://kotobank.jp/word/" + quote(q)
            t = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            try:
                resp = session.get(url, timeout=30)
                status, final, content = resp.status_code, resp.url, resp.content
            except requests.RequestException as e:
                status, final, content = f"error: {type(e).__name__}", "", b""
            fn = RAW / f"{r.item_id}__{q}.html"
            if content:
                fn.write_bytes(content)
            found = entries(content.decode("utf-8", "replace"), reading, must) if status == 200 else []
            log.append({"item_id": r.item_id, "query": q, "reading_filter": reading or "", "variant_of": must or "", "url": url,
                        "final_url": final, "http_status": status, "fetched_at": t,
                        "daijisen_entries": len(found), "raw_file": fn.name if content else ""})
            time.sleep(3)
            if found:
                texts += found
                urls.append(final)
                times.append(t)
                break  # idiom page found: parent word not needed
    out.append({"item_id": r.item_id, "見出し語": r.headword, "引く見出し": r.lookup, "裏の意味": r.covert,
                "URL": " | ".join(urls), "取得日時": " | ".join(times), "本文": "\n\n".join(texts)})
    print(r.item_id, r.lookup, "->", len(texts), "entries", flush=True)

pd.DataFrame(log).to_csv(HERE / "fetch_log.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(out).to_csv(PRIVATE / "daijisen_text.csv", index=False, encoding="utf-8-sig")
print("retrieved:", sum(bool(o["本文"]) for o in out), "/ 41")
