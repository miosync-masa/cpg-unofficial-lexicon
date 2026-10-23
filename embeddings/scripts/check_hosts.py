"""T3: mechanical check of every host string; candidate list + confirmed exclusion list.

Usage: python3 embeddings/scripts/check_hosts.py
       (writes data/exclusion_candidates.csv, data/exclusions.csv and docs/T3_exclusions.md)

Reads data/blind_items.csv (217 items). For each item, host_original and host_romanized are checked against the
rules below. Matching items are only *listed* (data/exclusion_candidates.csv); the confirmed exclusion list
(data/exclusions.csv) contains SEA_012, as fixed in the analysis plan, plus the items the author decided to
exclude on 2026-10-22 (data/author_decisions.csv, hand-entered). Rules (applied to both strings):

  placeholder_brackets  a [ ... ] segment
  no_form               nothing is left once (...) / （...） segments are removed, e.g. "(speech act genre)"
  english_description   outside parentheses, >= 2 distinct English descriptive words from DESC_WORDS
                        (not applied to English-language items)
  dash_appendix         a "—" outside parentheses that appends a title or description to the form: flagged when
                        the string is not long_text, or when it also contains a DESC_WORDS word (so a dash inside
                        a sentence of the saying itself is not flagged)
  multiple_forms        "/" or ";" outside parentheses separating segments of <= 20 characters each
  long_text             outside parentheses, a segment longer than 20 characters, or a newline;
                        long_text_lines when it also has a newline or several "/"- or ";"-separated segments
  combined_forms        " and " or " + " outside parentheses (two forms that make one host, e.g. two emoji)
  parenthetical_gloss   a (...) / （...） segment with content

proposed_action, first rule that applies:
  placeholder_brackets                      -> exclude
  no_form / english_description / dash_appendix / combined_forms -> ask_author
  multiple_forms with >= 3 alternatives     -> ask_author
  multiple_forms                            -> use_first_form
  long_text_lines                           -> ask_author   (verse: which line is the host is an author decision)
  long_text (one sentence or saying)        -> keep         (the saying itself is the host; T2 codes host_unit)
  parenthetical_gloss only                  -> keep         (T1's clean_host drops the gloss)
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "data"
DESC_WORDS = {"the", "of", "to", "and", "for", "with", "by", "from", "used", "riddle", "book", "motif", "address",
              "object", "song", "composed", "technique", "genre", "form", "emoji", "number", "entities",
              "performance", "frame", "narrative", "persona", "paired", "ritual", "spoken", "specific", "word",
              "examples", "require", "access"}
PAREN = re.compile(r"\([^()]*\)|（[^（）]*）")


def outside(s):
    prev = None
    while prev != s:  # nested parentheses
        prev, s = s, PAREN.sub("", s)
    return re.sub(r"\[[^\]]*\]", "", s).strip()


def problems(s, english):
    s = str(s)
    out, n_alt = [], 1
    o = outside(s)
    if re.search(r"\[[^\]]*\]", s):
        out.append("placeholder_brackets")
    if s.strip() and not o:
        out.append("no_form")
    words = {w.lower() for w in re.findall(r"[A-Za-z]+", o)}
    if not english and len(words & DESC_WORDS) >= 2:
        out.append("english_description")
    segs = [x.strip() for x in re.split(r"[/;]", o) if x.strip()]
    long_text = "\n" in s or max([len(x) for x in segs] + [0]) > 20
    if ("—" in o) and (not long_text or len(words & DESC_WORDS) >= 1):
        out.append("dash_appendix")
    if re.search(r"\s(and|\+)\s", o):
        out.append("combined_forms")
    if long_text:
        out.append("long_text_lines" if ("\n" in s or len(segs) > 1) else "long_text")
    elif len(segs) > 1:
        out.append("multiple_forms")
        n_alt = len(segs)
    if PAREN.search(s):
        out.append("parenthetical_gloss")
    return out, n_alt


def action(p, n_alt):
    if "placeholder_brackets" in p:
        return "exclude"
    if {"no_form", "english_description", "dash_appendix", "combined_forms"} & set(p):
        return "ask_author"
    if "multiple_forms" in p:
        return "ask_author" if n_alt >= 3 else "use_first_form"
    if "long_text_lines" in p:
        return "ask_author"
    return "keep"


items = pd.read_csv(ROOT / "data/blind_items.csv", dtype=str).fillna("")
rows = []
for _, r in items.iterrows():
    english = bool(re.search(r"english|^en$", r.language, re.I))
    p1, a1 = problems(r.host_original, english)
    p2, a2 = problems(r.host_romanized, english)
    p = [x for x in dict.fromkeys(p1 + p2)]
    if not p:
        continue
    rows.append({"item_id": r.item_id, "host_original": r.host_original, "host_romanized": r.host_romanized,
                 "problem_type": ";".join(p), "proposed_action": action(p, max(a1, a2))})
cand = pd.DataFrame(rows).sort_values("item_id").reset_index(drop=True)
assert (cand.loc[cand.item_id == "SEA_012", "proposed_action"] == "exclude").all()
cand.to_csv(HERE / "exclusion_candidates.csv", index=False, encoding="utf-8")
# author decisions of 2026-10-22 entered by hand in data/author_decisions.csv
dec = pd.read_csv(HERE / "author_decisions.csv", dtype=str).fillna("")
pending = cand[cand.proposed_action.isin(["ask_author", "use_first_form"])]
assert sorted(dec.item_id) == sorted(pending.item_id), "author_decisions.csv must cover every ask_author/use_first_form"
excl = pd.concat([pd.DataFrame([{"item_id": "SEA_012",
                                 "reason": "host is a placeholder string ([technique-level item ...]), not a form"}]),
                  dec[dec.decision == "exclude"][["item_id", "reason"]]], ignore_index=True)
excl.to_csv(HERE / "exclusions.csv", index=False, encoding="utf-8")
print("items checked:", len(items), "| candidates:", len(cand))
print(cand.proposed_action.value_counts().to_string())
pt = cand.problem_type.str.split(";").explode().value_counts()
print(pt.to_string())

# ------------------------------------------------------------ README.md (counts generated here, never typed)
ac = cand.proposed_action.value_counts()
ask = cand[cand.proposed_action == "ask_author"]
ufs = cand[cand.proposed_action == "use_first_form"]
cell = lambda x: str(x).replace("|", "/").replace("\n", " / ")
L = ["# T3 — exclusion list (2026-10-22)", "",
     "Generated by `embeddings/scripts/check_hosts.py` (rerunning it regenerates this file and both CSV files).", "",
     "## Files", "",
     f"- `data/exclusions.csv` — the confirmed exclusion list ({len(excl)} items): SEA_012 (host is the placeholder",
     "  `[technique-level item …]`) and the items excluded by the author's decisions of 2026-10-22.",
     "- `data/author_decisions.csv` — the author's decision on every ask_author / use_first_form item (entered by hand",
     "  from the decisions of 2026-10-22); `first_form` is the clean host that T1 uses for adopted use_first_form items.",
     "- `data/exclusion_candidates.csv` — every item whose `host_original` or `host_romanized` matches a rule below,",
     "  with `problem_type` (all rules that matched, `;`-joined) and `proposed_action`.", "",
     "## Check rules (mechanical; both host strings of all 217 items)", "",
     "| problem_type | rule |", "|---|---|",
     "| placeholder_brackets | a `[ … ]` segment |",
     "| no_form | nothing left once `(…)`/`（…）` are removed, e.g. `(speech act genre)` |",
     "| english_description | ≥ 2 distinct English descriptive words (fixed list in the script) outside parentheses; "
     "not applied to English-language items |",
     "| dash_appendix | a `—` outside parentheses appending a title or description (inside a long saying only "
     "when a descriptive word is also present) |",
     "| combined_forms | ` and ` or ` + ` outside parentheses (two forms making one host) |",
     "| multiple_forms | `/` or `;` outside parentheses between segments of ≤ 20 characters |",
     "| long_text | a segment of > 20 characters outside parentheses |",
     "| long_text_lines | long_text with a newline or several `/`/`;`-separated segments (verse lines) |",
     "| parenthetical_gloss | a `(…)`/`（…）` segment |", "",
     "`proposed_action` (first that applies): placeholder_brackets → `exclude`; no_form / english_description / "
     "dash_appendix / combined_forms → `ask_author`; multiple_forms with ≥ 3 alternatives → `ask_author`; "
     "multiple_forms → `use_first_form`; long_text_lines → `ask_author`; long_text (one saying) → `keep`; "
     "parenthetical_gloss only → `keep` (T1's clean host already drops the gloss).", "",
     "## Counts", "",
     f"- Items checked: {len(items)}; candidates: {len(cand)}.",
     "- proposed_action: " + ", ".join(f"{k} {v}" for k, v in ac.items()) + ".",
     "- problem_type (an item can have several): " + ", ".join(f"{k} {v}" for k, v in pt.items()) + ".", "",
     f"## Waiting for the author ({len(ask)} × ask_author)", "",
     "| item_id | host_original | problem_type |", "|---|---|---|"]
L += [f"| {r.item_id} | {cell(r.host_original)} | {r.problem_type} |" for r in ask.itertuples()]
L += ["", f"## use_first_form proposals ({len(ufs)})", "", "| item_id | host_original |", "|---|---|"]
L += [f"| {r.item_id} | {cell(r.host_original)} |" for r in ufs.itertuples()]
cnt = dec.groupby(["proposed_action", "decision"]).size()
L += ["", "## Author decisions (2026-10-22)", "",
      "- " + "; ".join(f"{a} → {d}: {n}" for (a, d), n in cnt.items()) + ".",
      "- `keep_utterance`: kept, host treated as an utterance (host_unit = utterance).",
      "- `keep_no_rule_matched`: none of the decision-1 cases applies; kept and listed for the author.",
      "- `not_adopted_unclear`: variant or different words is unclear; first form not adopted, item kept with the",
      "  T1 majority clean host, listed for the author.", "",
      "| item_id | proposed_action | decision | first_form | reason |", "|---|---|---|---|---|"]
L += [f"| {r.item_id} | {r.proposed_action} | {r.decision} | {r.first_form} | {cell(r.reason)} |"
      for r in dec.sort_values(["decision", "item_id"]).itertuples()]
(ROOT / "docs/T3_exclusions.md").write_text("\n".join(L) + "\n", encoding="utf-8")
