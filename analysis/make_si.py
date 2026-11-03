"""Supplementary information: every table (si/tables/*.tex) and every text included verbatim (si/text/*.txt) is
generated here from the result files, so that no number is typed by hand.

Usage: python3 analysis/make_si.py      (after analysis/make_numbers.py; reads results/numbers_manifest.csv)

Rules
- Numbers come from the result files; rounding as in paper/numbers.tex (the rules of analysis/make_numbers.py).
  Where a value also exists as a macro, the table uses or is checked against the macro value; the script fails
  on any mismatch.
- Tables: booktabs; table or longtable; \\caption{...} ends with the result file(s); \\label{tab:S<n>-...}.
- LaTeX special characters are escaped; Japanese is wrapped in \\ja{...}, Chinese in \\zh{...} (Han-only text
  follows the table's language; Hangul is wrapped in \\ja{...}, as no Korean command is defined).
- Not included: dictionary text, API keys, local paths, internal file names, the name of any venue.
- Texts (si/text): the prompts and codebooks as sent, re-wrapped at 100 characters per line (only whitespace
  changes, checked);
  the evidence-collection instructions with local paths replaced (marked in the file).
Output: si/tables/*.tex, si/text/*.txt, si/si_manifest.csv (file, sources, sha256, rules).
"""
import ast
import csv
import hashlib
import json
import re
import textwrap
import unicodedata
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "si"
TAB, TXT = SI / "tables", SI / "text"
MAN = []                                  # manifest rows
HALF = []                                 # stored values that cannot be re-rounded safely
CODERS = {"gpt": "gpt-5.6-sol", "gemini": "gemini-3.1-pro-preview", "claude": "claude-opus-5-5"}
EMB = {"oai": ("text_embedding_3_large", "OpenAI text-embedding-3-large"), "bge": ("BAAI_bge_m3", "BAAI/bge-m3"),
       "gem": ("gemini_embedding_001", "gemini-embedding-001")}
EMB_BY_TAG = {t: n for t, n in EMB.values()}
EMB_BY_NAME = {n: n for _, n in EMB.values()} | EMB_BY_TAG


# ------------------------------------------------------------------ rounding (as analysis/make_numbers.py)
def hu(x, places):
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def f_int(n):
    n = int(n)
    return f"{n:,}".replace(",", "{,}") if n >= 1000 else str(n)


def f_dec(x, places, stored=None):
    """stored: decimals the value was stored with; refuse a re-rounding that could go either way."""
    if stored and stored > places:
        if Decimal(repr(float(x))).scaleb(places) % 1 == Decimal("0.5"):
            HALF.append(f"{x} (stored with {stored} decimals) is half-way at {places} decimals")
    d = hu(x, places)
    if d == 0:
        d = abs(d)
    s = f"{d:.{places}f}"
    return (r"\ensuremath{-}" + s[1:]) if s.startswith("-") else s


def f_p(p, stored_places=None):
    """APA p; `stored_places`: the value was stored rounded, refuse a re-rounding that could go either way."""
    if stored_places and stored_places > 3:
        rem = Decimal(repr(float(p))).scaleb(3) % 1
        if rem == Decimal("0.5"):
            HALF.append(f"p {p} (stored with {stored_places} decimals) is half-way at 3 decimals")
    if p < .001:
        return "< .001"
    s = f"{hu(p, 3):.3f}"
    return "= " + (s[1:] if s.startswith("0") else s)


def p_cell(p, stored_places=None):
    return f_p(p, stored_places).replace("= ", "").replace("< ", "$<$ ")


def p_mac(name):
    """p cell from a macro (computed from the unrounded p by make_numbers.py)."""
    return mac(name).replace("= ", "").replace("< ", "$<$ ")


def f_ci(lo, hi, places=3):
    return f"[{f_dec(lo, places)}, {f_dec(hi, places)}]"


def f4(x, places):
    """a value stored with 4 decimals in a result file"""
    return f_dec(x, places, stored=4)


def f4ci(lo, hi, places=3):
    return f"[{f4(lo, places)}, {f4(hi, places)}]"


def pct(a, n, places=1):
    return f_dec(100 * a / n, places) if n else "--"


# ------------------------------------------------------------------ macros (paper/numbers.tex) for checks
MAC = {r["macro"]: r for r in csv.DictReader(open(ROOT / "results/numbers_manifest.csv", encoding="utf-8"))}


def mac(name):
    return MAC[name]["value"]


def same(value, macro):
    assert value == mac(macro), f"{macro}: table value {value!r} != macro {mac(macro)!r}"
    return value


# ------------------------------------------------------------------ LaTeX text
KANA = re.compile(r"[぀-ヿㇰ-ㇿｦ-ﾟ]")
CJK = re.compile(r"[　-〿぀-ヿㇰ-ㇿ㐀-䶿一-鿿豈-﫿＀-￯"
                 r"가-힯ᄀ-ᇿ㄰-㆏]+")
HANGUL = re.compile(r"[가-힯ᄀ-ᇿ㄰-㆏]")
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(s, lang="zh"):
    """Escape and wrap CJK runs. lang: language of Han-only runs ('ja' or 'zh'); kana in the cell makes it 'ja'."""
    s = "" if s is None or (isinstance(s, float) and np.isnan(s)) else str(s)
    cell_lang = "ja" if KANA.search(s) else lang
    out, pos = [], 0
    for m in CJK.finditer(s):
        out.append("".join(SPECIAL.get(c, c) for c in s[pos:m.start()]))
        run = m.group(0)
        cmd = "ja" if HANGUL.search(run) else cell_lang
        out.append(f"\\{cmd}{{" + "".join(SPECIAL.get(c, c) for c in run) + "}")
        pos = m.end()
    out.append("".join(SPECIAL.get(c, c) for c in s[pos:]))
    return "".join(out)


def src_caption(text, sources):
    return text.rstrip(".") + ". Source: " + "; ".join(tex(str(s)) for s in sources) + "."


def table(name, label, caption, sources, header, rows, colspec=None, notes=(), long=False, size=r"\small",
          midrules=()):
    """Write si/tables/<name>.tex. rows: lists of already formatted LaTeX cells (strings)."""
    colspec = colspec or "l" * len(header)
    cap = src_caption(caption, sources)
    head = " & ".join(header) + r" \\"
    body = []
    for i, r in enumerate(rows):
        if i in midrules:
            body.append(r"\midrule")
        body.append(" & ".join(r) + r" \\")
    lines = [f"% generated by analysis/make_si.py; do not edit by hand"]
    if long:
        lines += ["{" + size, rf"\begin{{longtable}}{{{colspec}}}", rf"\caption{{{cap}}}\label{{{label}}}\\",
                  r"\toprule", head, r"\midrule", r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead",
                  r"\bottomrule", r"\endfoot"] + body
        lines += [r"\end{longtable}"]
        lines += [rf"\par\noindent\footnotesize {n}\par" for n in notes]
        lines += ["}"]
    else:
        lines += [r"\begin{table}[htbp]", r"\centering", size, rf"\caption{{{cap}}}", rf"\label{{{label}}}",
                  rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", head, r"\midrule"] + body + [
                  r"\bottomrule", r"\end{tabular}"]
        if notes:
            lines += [r"\par\smallskip\parbox{\linewidth}{\footnotesize " + r" \\ ".join(notes) + "}"]
        lines += [r"\end{table}"]
    return "\n".join(lines) + "\n"


def write(path, text, sources, rule):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    srcs = [ROOT / s.split("=")[-1] for s in sources]
    MAN.append({"file": str(path.relative_to(ROOT)), "sources": ";".join(s.split("=")[0] for s in sources),
                "source_sha256": ";".join(hashlib.sha256(p.read_bytes()).hexdigest() for p in srcs), "rule": rule})


def write_tables(name, parts, sources, rule):
    write(TAB / f"{name}.tex", "\n".join(parts), sources, rule)


LIST = re.compile(r"(\s*)([-*] |\d+\. |\[\{|\{)?")


def wrap100(text):
    """Re-wrap paragraphs and list items at 100 characters. Headings, table rows, fenced code and indented code
    blocks (4+ spaces after a blank line) are kept as they are (a long one is broken at 100, continued with 4 more
    spaces). A line is joined to the previous one only if that one was hard-wrapped (85 characters or more), so
    intended short lines stay. Only whitespace changes: checked."""
    units, cur, raw_block, fence, prev_blank, prev_len = [], None, False, False, True, 0
    for line in text.split("\n"):
        st = line.strip()
        if st.startswith("```"):
            fence = not fence
        if fence or st.startswith("```") or not st or st.startswith(("#", "|", "===")) or (
                line.startswith("    ") and (prev_blank or raw_block) and not re.match(r"\s*([-*] |\d+\. )", line)):
            raw_block = bool(st) and line.startswith("    ") and not fence and (prev_blank or raw_block)
            units.append(["raw", line]); cur = None
        elif cur is not None and not re.match(r"\s*([-*] |\d+\. )", line) and not raw_block and prev_len >= 85:
            cur[1] += " " + st
        else:
            raw_block = False
            m = re.match(r"(\s*)([-*] |\d+\. )?", line)
            cur = ["para", st, m.group(1), m.group(1) + " " * len(m.group(2) or "")]
            units.append(cur)
        prev_blank, prev_len = not st, len(line)
    out = []
    for u in units:
        if u[0] == "raw":
            ind = " " * (len(u[1]) - len(u[1].lstrip()) + 4)
            out += [u[1]] if len(u[1]) <= 100 else textwrap.wrap(u[1], 100, subsequent_indent=ind,
                                                                  break_long_words=True, break_on_hyphens=False)
        else:
            out += textwrap.wrap(u[1], 100, initial_indent=u[2], subsequent_indent=u[3], break_long_words=True,
                                 break_on_hyphens=False)
    res = "\n".join(out) + "\n"
    assert all(len(x) <= 100 for x in out)
    assert re.sub(r"\s+", "", res) == re.sub(r"\s+", "", text), "wrapping changed more than whitespace"
    return res


def read_csv(p, **kw):
    return pd.read_csv(ROOT / p, encoding="utf-8-sig", **kw)


def load(p):
    return json.load(open(ROOT / p, encoding="utf-8"))


def same4(unr, rnd, path=""):
    """the unrounded result agrees with the stored (4-decimal) one everywhere"""
    if isinstance(unr, dict):
        for k in rnd:
            same4(unr[k], rnd[k], f"{path}.{k}")
    elif isinstance(unr, list):
        assert len(unr) == len(rnd), path
        for i, (a, b) in enumerate(zip(unr, rnd)):
            same4(a, b, f"{path}[{i}]")
    elif isinstance(unr, float) and isinstance(rnd, (int, float)) and not isinstance(rnd, bool):
        assert round(unr, 4) == rnd, (path, unr, rnd)
    else:
        assert unr == rnd, (path, unr, rnd)


def load_unrounded(rounded, unrounded):
    a, b = load(unrounded), load(rounded)
    same4(a, b)
    return a


def fleiss(cols):
    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
    t, _ = aggregate_raters(np.array(cols).T)
    return float(fleiss_kappa(t, method="fleiss"))


def cohen(a, b):
    a, b = np.asarray(a, dtype=str), np.asarray(b, dtype=str)
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in set(a) | set(b))
    return (po - pe) / (1 - pe)


def xtab_rows(rows, cols, rlev, clev, lang="ja"):
    t = pd.crosstab(pd.Series(list(rows)), pd.Series(list(cols))).reindex(index=rlev, columns=clev, fill_value=0)
    out = [[tex(i, lang)] + [str(int(t.loc[i, c])) for c in clev] + [str(int(t.loc[i].sum()))] for i in rlev]
    out.append([r"\textit{Total}"] + [str(int(t[c].sum())) for c in clev] + [str(int(t.values.sum()))])
    return out


# ================================================================== S1 lexicon and sources
blind = read_csv("data/blind_items.csv", dtype=str).fillna("")
B = blind.set_index("item_id")

# S1_label_merges
ht = read_csv("data/tradition_harmonisation.csv", dtype=str).fillna("")
mg = ht[ht.original_label != ht.harmonised_label].sort_values(["kind", "harmonised_label", "original_label"])
wk = read_csv("results/harmonisation_weak_for_author.csv", dtype=str).fillna("").sort_values(["kind", "label"])
n_after = {k: ht[ht.kind == k].harmonised_label.nunique() for k in ["tradition", "language"]}
assert f_int(n_after["tradition"]) == mac("NtradLabels") and f_int(n_after["language"]) == mac("Nlanguages")
t1 = table("S1_label_merges", "tab:S1-label-merges",
           "Tradition and language labels merged because they differ only in spelling (strong candidates; author's "
           f"decision of 2026-10-23). After merging: {f_int(n_after['tradition'])} tradition labels and "
           f"{f_int(n_after['language'])} language labels",
           ["data/tradition_harmonisation.csv"], ["Kind", "Label as published", "Merged into", "Rule"],
           [[r.kind, tex(r.original_label), tex(r.harmonised_label), tex(r.rule)] for r in mg.itertuples()],
           colspec="l p{0.33\\linewidth} p{0.33\\linewidth} l", long=True)
t2 = table("S1_label_merges", "tab:S1-label-weak",
           "Weak candidates that were not merged (author's decision of 2026-10-23). Each label is listed with its "
           "number of items and the labels it is weakly related to (slash\\_segment: shares a part separated by a "
           "slash; contains: one contains the other; near: string similarity at least 0.85)",
           ["results/harmonisation_weak_for_author.csv"], ["Kind", "Label", "Items", "Weak candidates"],
           [[r.kind, tex(r.label), r.n, tex(r.weak_candidates)] for r in wk.itertuples()],
           colspec="l p{0.3\\linewidth} r p{0.42\\linewidth}", long=True)
write_tables("S1_label_merges", [t1, t2], ["data/tradition_harmonisation.csv",
                                           "results/harmonisation_weak_for_author.csv"],
             "rows of data/tradition_harmonisation.csv with original_label != harmonised_label; all rows of "
             "results/harmonisation_weak_for_author.csv; counts checked against NtradLabels, Nlanguages")

# S1_label_corrections
lc = read_csv("data/label_corrections.csv", dtype=str).fillna("").set_index("item_id")
corr = read_csv("audit/corrections.csv", dtype=str).fillna("")
rows, star = [], False
for i, r in lc.iterrows():
    for fld in ["tradition", "language"]:
        old, new = B.at[i, fld], r[f"{fld}_corrected"]
        if old == new:
            continue
        d = corr[(corr.item_id == i) & (corr.field == fld)].decided
        dec = d.iloc[0] if len(d) else "2026-11-01*"
        star |= not len(d)
        rows.append([tex(i), fld, tex(old), tex(new), dec])
write_tables("S1_label_corrections", [table(
    "S1_label_corrections", "tab:S1-label-corrections",
    "Labels corrected after the analyses were fixed. The published columns are not overwritten; the corrected "
    "values are in the added columns tradition\\_corrected and language\\_corrected, with the reason (host forms "
    "are Old English: hronrád, Beowulf l.\\,10). Every analysis uses the published labels; the corrected labels are "
    "used only in the sensitivity analysis of Table~\\ref{tab:S4-corrected-labels}",
    ["data/label_corrections.csv", "audit/corrections.csv", "results/items_final.csv"],
    ["Item", "Field", "Published value", "Corrected value", "Decided"], rows,
    colspec="l l p{0.27\\linewidth} p{0.2\\linewidth} l",
    notes=["* The language correction was added on 2026-11-01. When the item was judged again in the source audit "
           "(2026-10-25), only the tradition correction was applied: the language was not corrected to Old English "
           "and was sent as Old Norse."] if star else [])],
    ["data/label_corrections.csv", "audit/corrections.csv", "data/blind_items.csv"],
    "published value from data/blind_items.csv; corrected value from data/label_corrections.csv; date from "
    "audit/corrections.csv, else 2026-11-01 (marked *)")

# S1_exclusions
ex = read_csv("data/exclusions.csv", dtype=str).fillna("")
assert f_int(len(ex)) == mac("Nexcluded")
write_tables("S1_exclusions", [table(
    "S1_exclusions", "tab:S1-exclusions", f"The {len(ex)} items excluded from the analyses, with the reason",
    ["data/exclusions.csv"], ["Item", "Language", "Tradition", "Reason"],
    [[tex(r.item_id), tex(B.at[r.item_id, "language"]), tex(B.at[r.item_id, "tradition"]), tex(r.reason)]
     for r in ex.itertuples()], colspec="l p{0.13\\linewidth} p{0.25\\linewidth} p{0.38\\linewidth}")],
    ["data/exclusions.csv", "data/blind_items.csv"], "all rows; count checked against Nexcluded")

# S1_audit_overall
aud = load("results/source_audit_summary.json")
LAB4 = ["supported", "partly", "contradicted", "no_evidence"]
aj = read_csv("results/source_audit_judgements.csv", dtype=str, keep_default_na=False)
sent = aj[aj.sent == "True"]
ka = fleiss([sent[f"label_{t}"].values for t in ["gpt", "claude", "gemini"]])
assert round(ka, 4) == aud["fleiss_kappa"]
rows = [[tex(l), str(aud["majority"][l])] + [str(aud["per_coder"][CODERS[c]][l]) for c in ["gpt", "claude", "gemini"]]
        for l in LAB4]
for l in LAB4:
    assert f_int(aud["majority"][l]) == mac({"supported": "NauditSupported", "partly": "NauditPartly",
                                             "contradicted": "NauditContradicted",
                                             "no_evidence": "NauditNoEvidence"}[l])
rows.append([r"\textit{Total}", str(sum(aud["majority"][l] for l in LAB4))]
            + [str(sum(aud["per_coder"][CODERS[c]].values())) for c in ["gpt", "claude", "gemini"]])
write_tables("S1_audit_overall", [table(
    "S1_audit_overall", "tab:S1-audit-overall",
    f"Source audit, stage B: labels of the {aud['n_items']} items. Majority = at least 2 of the 3 coders; the "
    f"{aud['n_rule_based_no_evidence']} items without any quote were not sent and are no\\_evidence by rule "
    f"(majority column only). Agreement over the {aud['n_sent']} items sent: Fleiss' $\\kappa$ = "
    f"{same(f_dec(ka, 2), 'Kaudit')}; all three coders gave the same label for {f_dec(aud['pct_unanimous'], 1)}\\% "
    f"of them. Unresolved (no majority): {aud['majority']['UNRESOLVED']}",
    ["results/source_audit_summary.json", "results/source_audit_judgements.csv"],
    ["Label", "Majority", "gpt-5.6-sol", "claude-opus-5-5", "gemini-3.1-pro-preview"], rows, colspec="lrrrr",
    midrules=(len(LAB4),))],
    ["results/source_audit_summary.json", "results/source_audit_judgements.csv"],
    "majority and per_coder counts; Fleiss kappa recomputed from the labels (checked against the stored value and "
    "Kaudit); pct_unanimous, 1 decimal")

# S1_author_spotcheck
asp = read_csv("data/audit_spot_check_author.csv", dtype=str, keep_default_na=False)
asp = asp[asp.in_sample == "yes"]
assert sorted(asp.item_id) == sorted(aud["spot_check_items"])
mj = aj.set_index("item_id").majority
a_lab, m_lab = asp.author_label.tolist(), [mj[i] for i in asp.item_id]
agree = same(f_dec(100 * np.mean([a == b for a, b in zip(a_lab, m_lab)]), 1), "PauditAuthorAgree")
kk = same(f_dec(cohen(a_lab, m_lab), 2), "KauditAuthor")
write_tables("S1_author_spotcheck", [table(
    "S1_author_spotcheck", "tab:S1-author-spotcheck",
    f"Author's spot check of {len(asp)} items drawn at random (stratified by the stage B majority): the author's "
    f"label (rows) against the majority of the three coders (columns). Agreement {agree}\\%, Cohen's $\\kappa$ = "
    f"{kk}", ["data/audit_spot_check_author.csv", "results/source_audit_judgements.csv"],
    ["Author \\textbackslash{} majority"] + [tex(l) for l in LAB4] + ["Total"],
    xtab_rows(a_lab, m_lab, LAB4, LAB4, "zh"), colspec="lrrrrr", midrules=(len(LAB4),))],
    ["data/audit_spot_check_author.csv", "results/source_audit_judgements.csv", "results/source_audit_summary.json"],
    "in_sample == yes (the drawn items); agreement and kappa recomputed and checked against PauditAuthorAgree, "
    "KauditAuthor; the author's notes are not included")

# S1_components, S1_reason_codes
cj = load("results/audit_components.json")
COMPS = [("H", "host named"), ("M", "covert meaning stated"), ("C_community", "placed in the community or "
         "tradition"), ("C_device", "meaning carried by the named technique")]
LEV = ["yes", "partial", "no", "UNRESOLVED"]
assert f_int(cj["n_sent"]) == mac("NcompSent")
write_tables("S1_components", [table(
    "S1_components", "tab:S1-components",
    f"Source audit, components (exploratory): majority (2 of 3 coders) for each component over the "
    f"{cj['n_sent']} items sent; {cj['n_no_evidence']} items without a quote were not sent. UNRESOLVED: no "
    "majority", ["results/audit_components.json", "results/audit_components.csv"],
    ["Component", "", "yes", "partial", "no", "UNRESOLVED"],
    [[tex(c), d] + [str(cj["components"][c][v]) for v in LEV] for c, d in COMPS], colspec="l l rrrr")],
    ["results/audit_components.json", "results/audit_components.csv"], "components counts; n_sent checked "
    "against NcompSent")
CODES = [("missing_H", "H = no"), ("other_form", "H = partial"), ("missing_M", "M = no"), ("partial_M", "M = partial"),
         ("missing_C_device", "C\\_device $\\neq$ yes"), ("missing_C_community", "C\\_community = no")]
LAB4m = ["supported", "partly", "contradicted", "no_evidence"]
assert f_int(cj["codes"]["missing_C_device"]) == mac("NcompMissingDevice")
assert f_int(cj["codes_by_label"]["missing_C_device"]["supported"]) == mac("NcompMissingDeviceSup")
assert f_int(cj["codes"]["partial_M"]) == mac("NcompPartialM")
rows = [[tex(c), d, str(cj["codes"][c])] + [str(cj["codes_by_label"][c][l]) for l in LAB4m] for c, d in CODES]
rows.append([r"\textit{Items}", "", str(cj["n_sent"])] + [str(cj["n_by_label"][l]) for l in LAB4m])
write_tables("S1_reason_codes", [table(
    "S1_reason_codes", "tab:S1-reason-codes",
    "Source audit, components (exploratory): reason codes derived from the component majorities (rules fixed on "
    "2026-10-25, before the run), by the stage B majority label. An item can have several codes; "
    f"{cj['n_without_code']} items have none. different\\_work cannot be derived from the components",
    ["results/audit_components.json"], ["Code", "Rule", "All sent"] + [tex(l) for l in LAB4m],
    rows, colspec="l l rrrrr", midrules=(len(CODES),))],
    ["results/audit_components.json"], "codes and codes_by_label; checked against NcompMissingDevice, "
    "NcompMissingDeviceSup, NcompPartialM")

# S1_unretrieved
un = read_csv("results/audit_unretrieved.csv", dtype=str).fillna("")
write_tables("S1_unretrieved", [table(
    "S1_unretrieved", "tab:S1-unretrieved",
    "Documents that could not be retrieved, neither again from the same address (2026-10-30 and 2026-11-01; "
    "only documents that had failed with a server or connection error) nor from the Wayback Machine (capture "
    "closest to 2026-03-13). HTTP 429 in the Wayback column: the archive refused the capture. Dates are MM-DD of "
    "2026; --: not requested again (HTTP 404, or no first request). Addresses are not given",
    ["results/audit_unretrieved.csv"], ["Item", "Doc.", "First failure", "Again 10-30", "Again 11-01", "Wayback"],
    [[tex(r.item_id), r.doc_rank, tex(r.first_failure), tex(r["refetch_10-30"]).replace("—", "--"),
      tex(r["refetch_11-01"]).replace("—", "--"), tex(r.wayback).replace("×", r"$\times$")]
     for _, r in un.iterrows()],
    colspec="l r p{0.14\\linewidth} p{0.12\\linewidth} p{0.12\\linewidth} p{0.3\\linewidth}", size=r"\footnotesize")],
    ["results/audit_unretrieved.csv"], "all rows; no URL")


# ================================================================== S2 codebooks, S5 prompts (texts as sent)
def module_strings(path):
    """Module-level string assignments and the constant prefix of SYSTEM = ("..." + CODEBOOK)."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            v = n.value
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out[n.targets[0].id] = v.value
            elif isinstance(v, ast.BinOp) and isinstance(v.left, ast.Constant):
                out[n.targets[0].id + "_PREFIX"] = v.left.value
            elif isinstance(v, ast.List) and all(isinstance(e, ast.Constant) for e in v.elts):
                out[n.targets[0].id] = [e.value for e in v.elts]
    return out


def user_template(path):
    s = (ROOT / path).read_text(encoding="utf-8")
    m = re.search(r'user = \("([^"]+)"', s) or re.search(r'return \("([^"]+)"', s)
    return m.group(1).encode().decode("unicode_escape")


CODING_RUNS = [("first run", "coding/scripts/run_llm_coding.py", "coding/codebooks/codebook.md",
                "a failed batch is retried up to 3 times"),
               ("form and boundary (v1.1)", "coding/scripts/run_v1_1_coding.py", "coding/codebooks/codebook_v1_1.md",
                "a failed batch is retried up to 3 times"),
               ("recording status", "coding/scripts/run_recording_coding.py", "coding/codebooks/codebook_recording.md",
                "a failed batch is retried up to 3 times; the items of the " + MAC["NrefBatchSingleRec"]["value"]
                + " batches that claude-opus-5-5 refused 3 times were then sent one item per request (batch size 1)"),
               ("taboo domain", "coding/scripts/run_taboo_coding.py", "coding/codebooks/codebook_taboo.md",
                "a failed batch is retried up to 3 times, then its items are sent one at a time")]
S2_FILES = {"coding/codebooks/codebook.md": "S2_codebook", "coding/codebooks/codebook_v1_1.md": "S2_codebook_v1_1",
            "coding/codebooks/codebook_recording.md": "S2_recording_status",
            "coding/codebooks/codebook_taboo.md": "S2_taboo_domain"}
coding_prompt = []
for run, script, cb, retry in CODING_RUNS:
    ms = module_strings(script)
    system = ms["SYSTEM_PREFIX"] + (ROOT / cb).read_text(encoding="utf-8")
    write(TXT / f"{S2_FILES[cb]}.txt", wrap100(system), [script, cb],
          "system prompt as sent: the fixed opening of SYSTEM in the script + the codebook file; wrapped at 100")
    coding_prompt += [f"=== {run} ===", "", "System prompt: the text of the codebook file for this run (supplement "
                      "S2), preceded by:", "", ms["SYSTEM_PREFIX"].rstrip("\n"), "",
                      f"User message (one per batch of 10 items; {retry}):", "", user_template(script).rstrip("\n"),
                      "<JSON array of the items, with the fields: " + ", ".join(ms["cols"]) + ">", ""]
write(TXT / "S5_prompt_coding.txt", wrap100("\n".join(coding_prompt)), [r[1] for r in CODING_RUNS],
      "SYSTEM opening, user message and item fields as in each coding script; wrapped at 100")

jm = module_strings("audit/scripts/judge_evidence.py")
judge_user = user_template("audit/scripts/judge_evidence.py")
JUDGE_FIELDS = "n (batch-local number), host, host_romanized, covert_target, tradition, quotes (document, " \
               "source_type, retrieved, quote)"
write(TXT / "S2_components.txt", wrap100(jm["SYSTEM_COMPONENTS"]), ["audit/scripts/judge_evidence.py"],
      "SYSTEM_COMPONENTS of judge_evidence.py as sent; wrapped at 100")

# evidence collection: the instructions given to the collectors, with local paths replaced
guide = (ROOT / "private/audit_collector_guide.md").read_text(encoding="utf-8")
guide = re.sub(r"Repository: \S+", "Repository: <the repository>", guide)
guide = re.sub(r"T=\S+", "T=<the evidence tool, audit/scripts/audit_tool.py>", guide)
assert "/Users" not in guide and "/private" not in guide and "scratchpad" not in guide
audit_txt = ["=== Stage A: evidence collection ===", "",
             "The instructions given to each evidence collector (an instance of claude-opus-5-5 with web search and "
             "the evidence tool), in their final form (last changed on 2026-10-24). They were revised during the "
             "collection, which began on 2026-10-24; earlier versions were not kept. The only change here: the "
             "local paths of the repository and of the tool are replaced by <...>. Later rule (2026-10-25): web "
             "archives may be used for documents lost to a server outage or a dead link (not for refusals).", "",
             guide.rstrip("\n"), "",
             "=== Stage B: judging (three coders) ===", "", "System prompt:", "", jm["SYSTEM"], "",
             "User message (one per batch of 10 items):", "", judge_user.rstrip("\n"),
             f"<JSON array of the items: {JUDGE_FIELDS}>", "",
             "=== Components (exploratory) ===", "",
             "The same user message; system prompt as in supplement S2 (components)."]
write(TXT / "S5_prompt_audit.txt", wrap100("\n".join(audit_txt)),
      ["collector instructions (kept outside the repository)=private/audit_collector_guide.md",
       "audit/scripts/judge_evidence.py"],
      "collector instructions (local paths replaced) + SYSTEM and user message of judge_evidence.py; wrapped at 100")

# control-word prompts
leg = (ROOT / "legacy/v1_submission/scripts/archaeology/cpg_archaeology.py").read_text(encoding="utf-8")
m = re.search(r"    prompt = \(\n(.*?)\n    \)\n", leg, re.S)
c_prompt = ast.literal_eval("(" + re.sub(r"(?<![A-Za-z])f([\"'])", r"\1", m.group(1)) + ")")
write(TXT / "S5_prompt_controls_C.txt", wrap100(
    "Control words, conditions A-C (v1 study; stored responses in data/control_cache.json). Prompt template as "
    "stored in the v1 script; {n} = 10, {host_word}, {lang_str} (the item's language) and {covert_target} are "
    "filled in per item. Model gpt-5.2, temperature 0.3, max_completion_tokens 300.\n\n" + c_prompt),
    ["legacy/v1_submission/scripts/archaeology/cpg_archaeology.py"], "prompt template of generate_controls(), "
    "f-string placeholders kept; wrapped at 100")
d_log = [json.loads(l) for l in open(ROOT / "embeddings/results/logs/controls_D.jsonl", encoding="utf-8")]
d_ex = d_log[0]
dm = module_strings("embeddings/scripts/generate_controls_D.py")
write(TXT / "S5_prompt_controls_D.txt", wrap100(
    "Control words, condition D: the prompt of conditions A-C without the two EXCLUSION lines (no covert target). "
    f"Model {d_ex['model']}, temperature {d_ex['temperature']}, max_completion_tokens {d_ex['max_completion_tokens']}. "
    "The prompt as sent for the first item of the log (the host word and the language change per item):\n\n"
    + d_ex["prompt"]), ["embeddings/results/logs/controls_D.jsonl"], "first logged prompt, verbatim; wrapped at 100")
assert "EXCLUSION" not in d_ex["prompt"] and "EXCLUSION" in c_prompt


# ================================================================== S3 coding
KROWS = [("First run (3 coders, 217 items)", [("direction", "KdirRunOne"), ("primary\\_channel", "KchanRunOne"),
                                              ("cpg\\_valid", "KvalidRunOne"), ("activation", "KactivRunOne"),
                                              ("s\\_juxtaposition", "KsJux"), ("s\\_noncompositional", "KsNoncomp"),
                                              ("s\\_conventional", "KsConv"), ("channel Phonetic", "KphonChan"),
                                              ("channel Structural", "KstructChan")]),
         ("Form and boundary, v1.1 (217 items)", [("form\\_type", "KformType"), ("two\\_storey", "KtwoStorey"),
                                                  ("content\\_or\\_attitude", "KcontentAttitude"),
                                                  ("outsider\\_inference", "KoutsiderInf"), ("host\\_unit", "KhostUnit")]),
         ("Recording status (217 items)", [("form\\_status", "KformStatus"), ("recording\\_status", "KrecStatus")]),
         (f"Recording status without the {mac('NrefItemsSingleRec')} items resent one by one to claude-opus-5-5 "
          f"({mac('NexSingle')} items)", [("form\\_status", "KformStatusExSingle"),
                                          ("recording\\_status", "KrecStatusExSingle")]),
         ("Taboo domain (217 items)", [("taboo\\_domain (6 labels)", "KtabooDomain"),
                                       ("taboo (binary)", "KtabooBin")]),
         (f"Source audit, stage B ({aud['n_sent']} items sent)", [("label (4 labels)", "Kaudit")])]
rows, mids = [], []
for grp, vs in KROWS:
    mids.append(len(rows))
    rows.append([rf"\multicolumn{{2}}{{l}}{{\textit{{{grp}}}}}"])
    rows += [[v, mac(k)] for v, k in vs]
write_tables("S3_agreement", [table(
    "S3_agreement", "tab:S3-agreement", "Agreement of the three LLM coders: Fleiss' $\\kappa$ of every coded "
    "variable reported in the main text", ["results/numbers_manifest.csv", "coding/results/first_run/agreement.json",
                                            "coding/results/v1_1/v1_1_agreement.json",
                                            "coding/results/recording/recording_agreement.json",
                                            "results/taboo_results.json", "results/source_audit_summary.json"],
    ["Variable", "Fleiss' $\\kappa$"], rows, colspec="lr", midrules=tuple(mids[1:]))],
    ["results/numbers_manifest.csv"], "values of the kappa macros (2 decimals, as in the main text)")


def refusals(path):
    L = [json.loads(l) for l in open(ROOT / path, encoding="utf-8")]
    ref = [a for a in L if a.get("refusal") or "stop_reason=refusal" in str(a.get("error"))]
    batches = {tuple(a["item_ids"]) for a in ref if len(a["item_ids"]) > 1}
    single = [a for a in L if len(a["item_ids"]) == 1]
    return len(L), len(ref), len(batches), len({i for a in single for i in a["item_ids"]})


RUNS = [("First run", "coding/results/first_run/codes_{c}.csv.jsonl"),
        ("Form and boundary (v1.1)", "coding/results/v1_1/v1_1_{c}.csv.jsonl"),
        ("Recording status", "coding/results/recording/recording_{c}.csv.jsonl"),
        ("Taboo domain", "coding/results/taboo/taboo_{c}.csv.jsonl"),
        ("Source audit, stage B", "audit/runs/judge_{m}.jsonl"),
        ("Source audit, components", "audit/runs/judge_components_{m}.jsonl")]
rows, srcs = [], []
for run, fmt in RUNS:
    for c, m in CODERS.items():
        p = fmt.format(c=c, m=m)
        srcs.append(p)
        n, r, b, s = refusals(p)
        rows.append([run if c == "gpt" else "", m, str(n), str(r), str(b), str(s)])
rr = refusals("coding/results/recording/recording_claude.csv.jsonl")
assert (f_int(rr[1]), f_int(rr[2]), f_int(rr[3])) == (mac("NrefReqRec"), mac("NrefBatchRec"), mac("NrefItemsSingleRec"))
tt = refusals("coding/results/taboo/taboo_claude.csv.jsonl")
assert (f_int(tt[1]), f_int(tt[2]), f_int(tt[3])) == (mac("NrefReqTaboo"), mac("NrefBatchTaboo"),
                                                       mac("NrefItemsSingleTaboo"))
write_tables("S3_refusals", [table(
    "S3_refusals", "tab:S3-refusals",
    "Refusals in each coding run, per coder. Requests: all requests in the log (including retries). Refused: "
    "requests the model refused (stop\\_reason = refusal, or a refusal field). Batches: distinct batches with at "
    "least one refused request. Resent singly: items sent one at a time after a batch failed three times. The "
    "audit's judging again of one item (2026-10-25) is not included",
    ["coding/results/*/*.jsonl", "audit/runs/*.jsonl"],
    ["Run", "Coder", "Requests", "Refused", "Batches", "Resent singly"], rows, colspec="llrrrr",
    midrules=tuple(range(3, len(rows), 3)))],
    srcs, "counted from the request logs; recording and taboo claude rows checked against the NrefRec / NrefTaboo "
    "macros")

# dictionaries
dj = read_csv("dictionary_check/daijisen_judgments.csv", dtype=str).fillna("")
rc = read_csv("coding/results/recording/recording_consensus.csv", dtype=str).set_index("item_id").recording_status
dj = dj[dj["判定"] != ""]
llm = ["general_dictionary" if rc[i] == "general_dictionary" else "other" for i in dj.item_id]
LLM2 = ["general_dictionary", "other"]
len_ = np.where(dj["判定"] == "番号つきの語義", "recorded", "not recorded")
str_ = np.where((dj["判定"] == "番号つきの語義") & (dj["一致度"] == "一致"), "recorded", "not recorded")
llm_b = np.array(["recorded" if x == "general_dictionary" else "not recorded" for x in llm])
assert f_int(len(dj)) == mac("NdictItems")
p_len = same(f_dec(100 * np.mean(len_ == llm_b), 1), "PdictAgreeLenient")
k_len = same(f_dec(cohen(len_, llm_b), 2), "KdictLenient")
p_str = same(f_dec(100 * np.mean(str_ == llm_b), 1), "PdictAgreeStrict")
k_str = same(f_dec(cohen(str_, llm_b), 2), "KdictStrict")
RB = ["recorded", "not recorded"]
LLMH = ["LLM: general\\_dictionary", "LLM: other"]
t_len = table("S3_dictionary_daijisen", "tab:S3-daijisen-lenient",
              f"\\ja{{大辞泉}} (Daijisen) against the LLM majority for the {len(dj)} Japanese headwords, lenient "
              f"criterion (recorded = a numbered sense, exact or close). Agreement {p_len}\\%, Cohen's $\\kappa$ = "
              f"{k_len}", ["dictionary_check/daijisen_judgments.csv", "coding/results/recording/recording_consensus.csv"],
              ["Daijisen"] + LLMH + ["Total"], xtab_rows(len_, llm, RB, LLM2), colspec="lrrr", midrules=(2,))
t_str = table("S3_dictionary_daijisen", "tab:S3-daijisen-strict",
              "As the previous table, strict criterion (recorded = a numbered sense that states the covert meaning "
              f"itself). Agreement {p_str}\\%, Cohen's $\\kappa$ = {k_str}",
              ["dictionary_check/daijisen_judgments.csv", "coding/results/recording/recording_consensus.csv"],
              ["Daijisen"] + LLMH + ["Total"], xtab_rows(str_, llm, RB, LLM2), colspec="lrrr", midrules=(2,))
sn_ = np.where(dj["判定"].isin(["番号つきの語義", "補説・用例"]), "recorded", "not recorded")
p_sn = f_dec(100 * np.mean(sn_ == llm_b), 1)
k_sn = f_dec(cohen(sn_, llm_b), 2)
t_sn = table("S3_dictionary_daijisen", "tab:S3-daijisen-sense-or-note",
             "As the first table, broadest criterion (recorded = a numbered sense, or the covert meaning in a "
             f"supplementary note or an example, \\ja{{補説・用例}}). Agreement {p_sn}\\%, Cohen's $\\kappa$ = {k_sn}",
             ["dictionary_check/daijisen_judgments.csv", "coding/results/recording/recording_consensus.csv"],
             ["Daijisen"] + LLMH + ["Total"], xtab_rows(sn_, llm, RB, LLM2), colspec="lrrr", midrules=(2,))
write_tables("S3_dictionary_daijisen", [t_len, t_str, t_sn],
             ["dictionary_check/daijisen_judgments.csv", "coding/results/recording/recording_consensus.csv"],
             "crosstabs recomputed; agreement and kappa checked against PdictAgree*, Kdict* (the third criterion has "
             "no macro); no dictionary text")

sk = read_csv("dictionary_check/sankoku_judgments.csv", dtype=str).fillna("")
sk = sk[sk["三国_掲載"] != ""]
SK4 = ["語義として載る", "注記・用例のみ", "載らない", "見出しなし"]
DJ3 = ["番号つきの語義", "補説・用例", "なし"]
djm = dj.set_index("item_id")["判定"]
sk_llm = ["general_dictionary" if rc[i] == "general_dictionary" else "other" for i in sk.item_id]
s_strict = (sk["三国_掲載"] == "語義として載る").astype(int).values
s_len = sk["三国_掲載"].isin(["語義として載る", "注記・用例のみ"]).astype(int).values
l_bin = np.array([x == "general_dictionary" for x in sk_llm]).astype(int)
d_strict = np.array([djm[i] == "番号つきの語義" for i in sk.item_id]).astype(int)
assert f_int(len(sk)) == mac("NskItems")
ps, ks = same(f_dec(100 * np.mean(s_strict == l_bin), 1), "PskLLMStrict"), same(f_dec(cohen(s_strict, l_bin), 2),
                                                                                 "KskLLMStrict")
pl, kl = same(f_dec(100 * np.mean(s_len == l_bin), 1), "PskLLMLenient"), same(f_dec(cohen(s_len, l_bin), 2),
                                                                              "KskLLMLenient")
pd_, kd = same(f_dec(100 * np.mean(s_strict == d_strict), 1), "PskDjStrict"), same(f_dec(cohen(s_strict, d_strict), 2),
                                                                                   "KskDjStrict")
nd = same(f_int(int((s_strict != d_strict).sum())), "NskDjDisagree")
SKSRC = ["dictionary_check/sankoku_judgments.csv", "coding/results/recording/recording_consensus.csv",
         "dictionary_check/daijisen_judgments.csv"]


def sankoku_tables(prefix, file):
    a = table(file, f"tab:{prefix}-sankoku-llm",
              f"\\ja{{三省堂国語辞典}} 8th ed. (author's check, 2026-10-29) against the LLM majority for the {len(sk)} "
              "Japanese headwords. Strict criterion (listed = \\ja{語義として載る}): agreement "
              f"{ps}\\%, Cohen's $\\kappa$ = {ks}; lenient (also \\ja{{注記・用例のみ}}): {pl}\\%, $\\kappa$ = {kl}. "
              "\\ja{見出しなし} counts as not listed", SKSRC[:2], ["\\ja{三国}"] + LLMH + ["Total"],
              xtab_rows(sk["三国_掲載"], sk_llm, SK4, LLM2), colspec="lrrr", midrules=(4,))
    b = table(file, f"tab:{prefix}-sankoku-daijisen",
              "\\ja{三省堂国語辞典} 8th ed. against \\ja{大辞泉}. Strict criteria (\\ja{語義として載る} vs a numbered "
              f"sense): agreement {pd_}\\%, Cohen's $\\kappa$ = {kd}; {nd} headwords differ",
              [SKSRC[0], SKSRC[2]], ["\\ja{三国} \\textbackslash{} \\ja{大辞泉}"] + [tex(x, "ja") for x in DJ3] + ["Total"],
              xtab_rows(sk["三国_掲載"], [djm[i] for i in sk.item_id], SK4, DJ3), colspec="lrrrr", midrules=(4,))
    return [a, b]


write_tables("S3_dictionary_sankoku", sankoku_tables("S3", "S3_dictionary_sankoku"), SKSRC,
             "crosstabs recomputed; agreement, kappa and disagreements checked against the Psk*/Ksk*/NskDjDisagree "
             "macros; author's notes not included")


# ================================================================== S4 sensitivity
def mname(x):
    return EMB_BY_NAME.get(x, x)


VERD = {"supported": "supported", "not supported": "not supported"}
st = load("results/sensitivity_tradition_unit.json")
GR = {"raw labels": "published", "strong candidates merged": "spelling variants merged",
      "all weak merged": "weak candidates merged too"}
rows = [[GR[r["grouping"]], r["hypothesis"], mname(r["model"]), str(r["n"]), str(r["n_groups"]), f_dec(r["z"], 2),
         p_cell(r["p_holm"]), r["verdict"]] for r in st["rows"]]
for r in st["rows"]:
    if r["grouping"] == "raw labels":
        k = [k for k, (t, n) in EMB.items() if n == r["model"]][0]
        pre = {"H-C1": "HCone", "H-C2": "HCtwo"}[r["hypothesis"]]
        assert f_dec(r["z"], 2) == mac(f"{pre}Z{k}")
write_tables("S4_tradition_groupings", [table(
    "S4_tradition_groupings", "tab:S4-tradition-groupings",
    "H-C1 and H-C2 with the tradition random intercept under three groupings of the labels: as published (the "
    "main analysis), spelling variants merged, and weak candidates merged as well (known false matches not "
    "merged). Mixed model pull $\\sim$ 1 + (1 | tradition), REML; one-sided Wald z; Holm across the three models",
    ["results/sensitivity_tradition_unit.json"],
    ["Grouping", "Hyp.", "Model", "n", "Groups", "z", "p (Holm)", "Verdict"], rows, colspec="p{0.2\\linewidth}lp{0.2\\linewidth}rrrrl",
    midrules=tuple(range(6, len(rows), 6)), size=r"\footnotesize")],
    ["results/sensitivity_tradition_unit.json"], "rows; z 2 decimals, p APA from unrounded p; published-label z "
    "checked against HConeZ*/HCtwoZ*")

sc = load("results/sensitivity_corrected_labels.json")
GC = {"published labels": "published", "corrected labels": "corrected (own group)",
      "corrected, joined to Anglo-Saxon kenning": "corrected, joined$^\\dagger$"}
rows = [[GC[r["grouping"]], r["hypothesis"], mname(r["model"]), str(r["n"]), str(r["n_groups"]), f_dec(r["z"], 2),
         p_cell(r["p_holm"]), r["verdict"]] for r in sc["rows"]]
for r in sc["rows"]:
    if r["grouping"] == "corrected labels" and r["hypothesis"] == "H-D2":
        k = [k for k, (t, n) in EMB.items() if n == r["model"]][0]
        assert f_p(r["p_holm"]) == mac(f"HDtwoP{k}Corr")
write_tables("S4_corrected_labels", [table(
    "S4_corrected_labels", "tab:S4-corrected-labels",
    "The four pre-specified tests with the tradition random intercept under the published labels, the corrected "
    "labels (NEU\\_001 as a group of its own; the reference of this sensitivity analysis, as requested before it "
    "was run) and the corrected labels with NEU\\_001 joined to the existing label Anglo-Saxon kenning (Beowulf). "
    "Holm across the three models; supported if p $<$ .05",
    ["results/sensitivity_corrected_labels.json"],
    ["Grouping", "Hyp.", "Model", "n", "Groups", "z", "p (Holm)", "Verdict"], rows,
    colspec="p{0.19\\linewidth}lp{0.2\\linewidth}rrrrl", long=True, size=r"\footnotesize",
    midrules=tuple(range(3, len(rows), 3)),
    notes=["$^\\dagger$ Chosen after the results were seen."])],
    ["results/sensitivity_corrected_labels.json"], "36 rows; H-D2 under the corrected labels checked against "
    "HDtwoP*Corr")

ss = aud["sensitivity"]
su = {(r["set"], r["hypothesis"], r["model"]): r for r in load("results/source_audit_sensitivity_unrounded.json")["rows"]}
for r in ss["rows"]:
    same4(su[(r["set"], r["hypothesis"], r["model"])], {k: r[k] for k in ["intercept", "se", "z", "p_holm"]})
SETN = {"supported": "supported only", "supported_or_partly": "supported or partly"}
rows = []
for r in ss["rows"]:
    k = [k for k, (t, n) in EMB.items() if n == r["model"]][0]
    pre = {"H-C1": "HCone", "H-C2": "HCtwo", "H-D1": "HDone", "H-D2": "HDtwo"}[r["hypothesis"]]
    suf = {"supported": "sup", "supported_or_partly": "supPart"}[r["set"]]
    ci = mac(f"{pre}CI{suf}{k}")
    rows.append([SETN[r["set"]], r["hypothesis"], mname(r["model"]), str(r["n"]), str(r["n_traditions"]),
                 f_dec(su[(r["set"], r["hypothesis"], r["model"])]["intercept"], 3), ci,
                 same(f_dec(r["z"], 2), f"{pre}Z{suf}{k}"), p_mac(f"{pre}P{suf}{k}")])
vr = [f"{SETN[k.split(' ')[0]]}, {k.split(' ')[1]}: {v}" for k, v in ss["verdicts"].items()]
write_tables("S4_supported_subsets", [table(
    "S4_supported_subsets", "tab:S4-supported-subsets",
    "The pre-specified tests H-C1, H-C2 (condition C) and H-D1, H-D2 (condition D) on the items whose source audit "
    "majority is supported, or supported or partly. 95\\% CI = intercept $\\pm$ 1.96 SE; one-sided p, Holm across "
    "the three models", ["results/source_audit_summary.json"],
    ["Items", "Hyp.", "Model", "n", "Trad.", "Intercept", "95\\% CI", "z", "p (Holm)"], rows,
    colspec="p{0.15\\linewidth}lp{0.19\\linewidth}rrrlrr", long=True, size=r"\footnotesize",
    midrules=tuple(range(3, len(rows), 3)), notes=["Verdicts (3 models supported: supported; 1--2: partly "
                                                   "supported; 0: not supported): " + "; ".join(vr) + "."])],
    ["results/source_audit_summary.json", "results/source_audit_sensitivity_unrounded.json"],
    "sensitivity.rows; intercept from the unrounded rows; CI, z and p = the HC*/HD* sup/supPart macros")

tb = load("results/target_blind_controls.json")
rows = []
for h in ["H-C1", "H-C2"]:
    for tag, name in EMB.values():
        r = tb["condition_C_reference"][h][tag]
        k = [k for k, (t, n) in EMB.items() if t == tag][0]
        pre = {"H-C1": "HCone", "H-C2": "HCtwo"}[h]
        rows.append(["C", h, name, str(r["n"]), str(r["n_traditions"]), same(f4(r["intercept"], 3), f"{pre}B{k}"),
                     mac(f"{pre}CI{k}"),
                     same(f_dec(r["z"], 2), f"{pre}Z{k}"), p_mac(f"{pre}P{k}")])
for r in tb["rows"]:
    k = [k for k, (t, n) in EMB.items() if t == r["model"]][0]
    pre = {"H-D1": "HDone", "H-D2": "HDtwo", "H-E1": "HEone", "H-E2": "HEtwo"}[r["hypothesis"]]
    ci = mac(f"{pre}CI{k}")
    rows.append([r["condition"], r["hypothesis"], EMB_BY_TAG[r["model"]], str(r["n"]), str(r["n_groups"]),
                 same(f4(r["intercept"], 3), f"{pre}B{k}"), ci, same(f_dec(r["z"], 2), f"{pre}Z{k}"),
                 p_mac(f"{pre}P{k}")])
write_tables("S4_conditions_D_E", [table(
    "S4_conditions_D_E", "tab:S4-conditions",
    "Intercepts of the pull under the three kinds of control words: C (LLM, covert target excluded), D (LLM, "
    "no covert target in the prompt), E (nearest vocabulary words in the embedding space; items in languages "
    "covered by the word list). Hypotheses 1: all items of the main set with 10 kept controls; 2: those with "
    "cpg\\_valid = yes. Mixed model pull $\\sim$ 1 + (1 | tradition), REML; 95\\% CI = intercept $\\pm$ 1.96 SE; "
    "one-sided p, Holm across the three models",
    ["results/target_blind_controls.json"], ["Cond.", "Hyp.", "Model", "n", "Trad.", "Intercept", "95\\% CI", "z",
                                             "p (Holm)"], rows,
    colspec="llp{0.2\\linewidth}rrrlrr", size=r"\footnotesize", midrules=tuple(range(3, len(rows), 3)))],
    ["results/target_blind_controls.json"], "condition_C_reference and rows; intercept, CI, z and p checked against the "
    "HC*/HD*/HE* macros")

ms5 = load_unrounded("results/model_size_e5.json", "results/model_size_e5_unrounded.json")
parts = []
for cond in ["C", "D"]:
    c = ms5["conditions"][cond]
    for r in c["per_size"]:
        assert f_int(hu(r["params"] / 1e6, 0)) == mac(f"Pe{r['size'].capitalize()}")
        assert f_dec(r["d"], 2) == mac(f"De{r['size'].capitalize()}{cond}")
    for r, k in zip(c["reference_earlier_models"], ["Oai", "Bge", "Gem"]):
        assert f_dec(r["d"], 2) == mac(f"De{k}{cond}")
    for h, pre in [("H-S1", "HSone"), ("H-S2", "HStwo")]:
        sfx = "C" if cond == "C" else ""
        x = c[h]
        assert (f_dec(x["b"], 3), f_dec(x["z"], 2), f_p(x["p_one_sided"]), f_ci(*x["ci95"])) == (
            mac(f"{pre}B{sfx}"), mac(f"{pre}Z{sfx}"), mac(f"{pre}P{sfx}"), mac(f"{pre}CI{sfx}")), (cond, h)
    rows = [[f"multilingual-e5-{r['size']}", f_int(hu(r["params"] / 1e6, 0)) + "M", str(r["n"]), f_dec(r["mean_pull"], 3),
             f_dec(r["d"], 2), f_dec(r["intercept"], 3), f_ci(*r["ci95"]), f_dec(r["z"], 2)] for r in c["per_size"]]
    rows += [[r["model"], "", str(r["n"]), f_dec(r["mean_pull"], 3), f_dec(r["d"], 2), "", "", ""]
             for r in c["reference_earlier_models"]]
    notes = []
    for h, d in [("H-S1", "slope of log10(parameters)"), ("H-S2", "interaction log10(parameters) $\\times$ hard "
                                                                  "(not inferable by an outsider)")]:
        s = c[h]
        notes.append(f"{h} ({d}; random effects {tex(s['random_effects'])}; {s['n_rows']} rows, {s['n_items']} items): "
                     f"b = {f_dec(s['b'], 3)}, 95\\% CI {f_ci(*s['ci95'])}, z = {f_dec(s['z'], 2)}, one-sided p "
                     f"{f_p(s['p_one_sided'])}.")
    parts.append(table(
        "S4_model_size", f"tab:S4-model-size-{cond}",
        f"Model size and pull, condition {cond}: multilingual-e5 small, base and large, with the three earlier "
        "models for reference (d = mean pull / SD). Intercept and 95\\% CI of pull $\\sim$ 1 + (1 | tradition)",
        ["results/model_size_e5.json"], ["Model", "Params", "n", "Mean pull", "d", "Intercept", "95\\% CI", "z"],
        rows, colspec="lrrrrrlr", size=r"\footnotesize", midrules=(3,), notes=notes))
write_tables("S4_model_size", parts, ["results/model_size_e5.json", "results/model_size_e5_unrounded.json"],
             "per_size, reference_earlier_models, H-S1, H-S2 from the unrounded results (checked against the stored "
             "ones); parameters, d and H-S values checked against the Pe*, De*, HSone*, HStwo* macros")


# ------------------------------------------------------------------ S4 additions (review, 2026-11-03)
from statsmodels.stats.multitest import multipletests


def holm(ps):
    return [float(a) for a in multipletests(ps, method="holm")[1]]


FEWER = []   # cells of values stored with 4 decimals shown with other decimals because the storage fixes no more


def fixed(ends, places, alt):
    """the value at `places` decimals if every possible unrounded value gives it, else at `alt` decimals (checked)"""
    for pl in (places, alt):
        if len({hu(e, pl) for e in ends}) == 1:
            if pl != places:
                FEWER.append(pl)
            return f_dec(ends[0], pl) + ("$^\\ddagger$" if pl != places else "")
    raise AssertionError(f"not fixed at {places} or {alt} decimals: {ends}")


def ci_stored(b, se):
    """95% CI b +- 1.96 SE from b and SE stored with 4 decimals: 3 decimals where the storage fixes them, else 2"""
    bounds = []
    for sign in (-1, 1):
        ends = [b + sign * 1.96 * se] + [b + db + sign * 1.96 * (se + ds) for db in (-5e-5, 5e-5) for ds in (-5e-5, 5e-5)]
        bounds.append(fixed(ends, 3, 2))
    return f"[{bounds[0]}, {bounds[1]}]"


def z_stored(z):
    """z stored with 4 decimals: 2 decimals unless the stored value is half-way, then 3 (fixed by the storage)"""
    return fixed([z, z - 5e-5, z + 5e-5], 2, 3)


T1SRC = ["results/t1_results.json", "results/t1_results_unrounded.json"]
t1u = load_unrounded(*T1SRC)
KEY = {t: k for k, (t, n) in EMB.items()}
HPRE = {"H-C1": "HCone", "H-C2": "HCtwo"}

# before and after the decisions of 2026-10-22
pre_h = load("results/before_decisions_2026-10-22/t1_results.json")["hypotheses"]
fr = load("results/final_results.json")
ba = fr["hypotheses_before_after_decisions"]
rows = []
for h in ["H-C1", "H-C2"]:
    for phase in ["before", "after"]:
        for i, (tag, name) in enumerate(EMB.values()):
            k = KEY[tag]
            st = ba[h][phase]["per_model"][tag]
            if phase == "before":
                r = pre_h[h]["per_model"][tag]
                assert (r["n"], r["z"], r["p_holm"]) == (st["n"], st["z"], st["p_holm"]), (h, tag)
                if h == "H-C1":
                    assert f_int(r["n"]) == mac("NkTenPre") and f4(r["z"], 2) == mac(f"HConeZ{k}Pre")
                cells = [f4(r["intercept"], 3), ci_stored(r["intercept"], r["se"]), z_stored(r["z"]),
                         p_cell(r["p_holm"], 4)]
            else:
                r = t1u["hypotheses"][h]["per_model"][tag]
                assert (r["n"], round(r["z"], 4), round(r["p_holm"], 4)) == (st["n"], st["z"], st["p_holm"])
                pre = HPRE[h]
                assert f_ci(r["intercept"] - 1.96 * r["se"], r["intercept"] + 1.96 * r["se"]) == mac(f"{pre}CI{k}")
                cells = [same(f_dec(r["intercept"], 3), f"{pre}B{k}"), mac(f"{pre}CI{k}"),
                         same(f_dec(r["z"], 2), f"{pre}Z{k}"), p_mac(f"{pre}P{k}")]
            v = ba[h][phase]
            rows.append([h if i == 0 else "", phase if i == 0 else "", name, str(r["n"]), str(r["n_traditions"])]
                        + cells + [f"{v['verdict']} ({v['n_models_holm_p05']}/3)" if i == 0 else ""])
assert not ba["H-C1"]["verdict_changed"] and not ba["H-C2"]["verdict_changed"]
write_tables("S4_before_after_decisions", [table(
    "S4_before_after_decisions", "tab:S4-before-after",
    "H-C1 and H-C2 (condition C, k = 10) before and after the author's decisions of 2026-10-22 (exclusions, "
    "adoption of the first form, the scope set by form\\_type). The decisions were taken after the first results "
    "had been seen. Mixed model pull $\\sim$ 1 + (1 | tradition), REML; 95\\% CI = intercept $\\pm$ 1.96 SE; "
    "one-sided p, Holm across the three models; supported if all three models have p $<$ .05",
    ["results/final_results.json", "results/before_decisions_2026-10-22/t1_results.json", "results/t1_results.json"],
    ["Hyp.", "Results", "Model", "n", "Trad.", "Intercept", "95\\% CI", "z", "p (Holm)", "Verdict"], rows,
    colspec="llp{0.19\\linewidth}rrrlrrl", size=r"\footnotesize", midrules=tuple(range(3, len(rows), 3)),
    notes=["Before: the results file fixed before the decisions (intercept, SE and z stored with 4 decimals; no "
           "unrounded results exist). After: the main analysis (the values of the main text)."]
    + (["$^\\ddagger$ The stored values fix this value only at the number of decimals shown."] if FEWER else []))],
    ["results/final_results.json", "results/before_decisions_2026-10-22/t1_results.json"] + T1SRC,
    "hypotheses_before_after_decisions; before from the fixed file (4 decimals; CI bounds checked to be fixed at 3 "
    "decimals), after from the unrounded results and checked against the HCone*/HCtwo* macros")

# k = all, 3, 5, 10
KS = [("10 (main)", lambda a: a["B_vs_C"]["C_k10_main"], "k10"), ("3", lambda a: a["sensitivity"]["k3_main"], "k3"),
      ("5", lambda a: a["sensitivity"]["k5_main"], "k5"), ("all kept", lambda a: a["sensitivity"]["kall_main"], "kall")]
rows = []
for lab, get, fk in KS:
    rs = [get(t1u["analyses"][tag]) for tag, _ in EMB.values()]
    adj = holm([r["mixed"]["p_one_sided"] for r in rs])
    for i, ((tag, name), r, a) in enumerate(zip(EMB.values(), rs, adj)):
        k, mx = KEY[tag], r["mixed"]
        st = fr["tests"]["all_main"][fk]["per_model"][k]
        assert (mx["n"], round(mx["z"], 4), round(a, 4)) == (st["n"], st["z"], st["p_holm"]), (fk, k)
        ci = f_ci(mx["intercept"] - 1.96 * mx["se"], mx["intercept"] + 1.96 * mx["se"])
        if fk == "k10":
            assert ci == mac(f"HConeCI{k}") and f_p(a) == mac(f"HConeP{k}")
            same(f_dec(mx["intercept"], 3), f"HConeB{k}")
        rows.append([lab if i == 0 else "", name, str(r["n"]), str(mx["n_traditions"]), f_dec(r["mean"], 3),
                     f"{r['pct_positive']:.1f}", f_dec(mx["intercept"], 3), ci, f_dec(mx["z"], 2), p_cell(a)])
write_tables("S4_k_sensitivity", [table(
    "S4_k_sensitivity", "tab:S4-k",
    "H-C1 (condition C, main set) with the pull computed from the first k kept controls. k = 10 is the main analysis "
    "(fixed in advance; only items with all ten controls kept); k = 3, 5 and all kept controls (up to ten) use every "
    "main-set item with at least that many. Mean and \\% $>$ 0 over the items; mixed model pull $\\sim$ 1 + "
    "(1 | tradition), REML; 95\\% CI = intercept $\\pm$ 1.96 SE; one-sided p, Holm across the three models",
    ["results/t1_results.json", "results/final_results.json"],
    ["k", "Model", "n", "Trad.", "Mean", "\\% $>$ 0", "Intercept", "95\\% CI", "z", "p (Holm)"], rows,
    colspec="lp{0.19\\linewidth}rrrrrlrr", size=r"\footnotesize", midrules=tuple(range(3, len(rows), 3)))],
    T1SRC + ["results/final_results.json"], "analyses.*.B_vs_C.C_k10_main and sensitivity.k3/k5/kall_main from the "
    "unrounded results; Holm recomputed and checked against final_results.json tests.all_main; k = 10 checked against "
    "the HCone* macros")

# hosts: condition U (host_original) against condition C (clean_host), 213 items with controls
GLOSS = re.compile(r"[(（]|\s—\s")


def strip_gloss(s):
    return re.sub(r"\s+—\s.*$", "", re.sub(r"\s*[(（][^)）]*[)）]", "", s)).strip()


def host_change(r):
    h, c = r.host_original, r.clean_host
    if h == c:
        return "same"
    if r.clean_host_source == "first_form_T3":
        return "first_form"
    if strip_gloss(h) == c:
        return "gloss"
    if c in [strip_gloss(p) for p in re.split(r"\s*/\s*", h)] + [p.strip() for p in re.split(r"\s*/\s*", strip_gloss(h))]:
        return "one_form"
    if re.sub(r"[\s/]+", "", h).casefold() == re.sub(r"[\s/]+", "", c).casefold():
        return "layout"
    return "other"


hb = read_csv("embeddings/results/nnpm_text_embedding_3_large.csv").rename(columns={"id": "item_id"})
hc = read_csv("embeddings/controls/clean_hosts.csv", dtype=str)
for tag, _ in EMB.values():   # the same host strings in the three models
    assert (read_csv(f"embeddings/results/nnpm_{tag}.csv").host_original.values == hb.host_original.values).all()
    cc = read_csv(f"embeddings/results/nnpm_C_{tag}.csv").set_index("item_id").clean_host
    assert (cc == hc.set_index("item_id").clean_host.reindex(cc.index)).all()
hx = hb.merge(hc, on="item_id")
assert len(hx) == 213
hx["change"] = hx.apply(host_change, axis=1)
HCH = hx.change.value_counts().to_dict()
N_GLOSS_U = int(hx.host_original.str.contains(GLOSS).sum())
OTHER_IDS = sorted(hx[hx.change == "other"].item_id)
excl_ids = set(read_csv("data/exclusions.csv").item_id)

# condition U, H-C1-type test
BLK = [("all items after the exclusions", lambda a: a["B_vs_C"]["B_k10_all_after_T3"]),
       ("main-set items (as H-C1)", lambda a: a["B_vs_C"]["same_items"]["B_k10"])]
rows = []
for lab, get in BLK:
    rs = [get(t1u["analyses"][tag]) for tag, _ in EMB.values()]
    adj = holm([r["mixed"]["p_one_sided"] for r in rs])
    for i, ((tag, name), r, a) in enumerate(zip(EMB.values(), rs, adj)):
        mx = r["mixed"]
        if lab.startswith("main"):
            same(f_dec(r["mean"], 3), f"MeanBsame{KEY[tag].capitalize()}")
        rows.append([lab if i == 0 else "", name, str(r["n"]), str(mx["n_traditions"]), f_dec(r["mean"], 3),
                     f"{r['pct_positive']:.1f}", f_dec(mx["intercept"], 3),
                     f_ci(mx["intercept"] - 1.96 * mx["se"], mx["intercept"] + 1.96 * mx["se"]), f_dec(mx["z"], 2),
                     p_cell(a)])
n_u_excl = int(hx.item_id.isin(excl_ids).sum())
write_tables("S4_condition_U", [table(
    "S4_condition_U", "tab:S4-condition-U",
    "The H-C1 test in condition U: the host as stored (host\\_original), the stored controls, the first ten "
    f"(pull\\_B\\_k10). All items: the {213 - n_u_excl} items with stored controls that remain after the exclusions "
    "(low\\_control items included); main-set items: the items of the H-C1 test. Mixed model pull $\\sim$ 1 + "
    "(1 | tradition), REML; 95\\% CI = intercept $\\pm$ 1.96 SE; one-sided p, Holm across the three models",
    ["results/t1_results.json"],
    ["Items", "Model", "n", "Trad.", "Mean", "\\% $>$ 0", "Intercept", "95\\% CI", "z", "p (Holm)"], rows,
    colspec="p{0.15\\linewidth}p{0.19\\linewidth}rrrrrlrr", size=r"\footnotesize", midrules=(3,),
    notes=[f"In {N_GLOSS_U} of the 213 items with stored controls, the stored host string of condition U contains an "
           "explanation in parentheses or after a dash (S5, Table~\\ref{tab:S5-inputs})."])],
    T1SRC + ["embeddings/results/nnpm_text_embedding_3_large.csv"],
    "analyses.*.B_vs_C (B_k10_all_after_T3, same_items.B_k10) from the unrounded results; Holm recomputed; main-set "
    "means checked against MeanBsame*")

# direction and primary channel (main set)
rows, mids = [], []
for var, lab in [("direction", "Direction"), ("primary_channel", "Primary channel")]:
    levels = sorted({lv for tag, _ in EMB.values() for lv, g in t1u["analyses"][tag][f"by_{var}"].items() if g["n"]},
                    key=lambda lv: (lv == "UNRESOLVED", lv))
    if rows:
        mids.append(len(rows))
    for j, lv in enumerate(levels):
        gs = [t1u["analyses"][tag][f"by_{var}"][lv] for tag, _ in EMB.values()]
        assert len({g["n"] for g in gs}) == 1
        rows.append([lab if j == 0 else "", tex(lv), str(gs[0]["n"])]
                    + [f"{f_dec(g['mean'], 3)} ({g['pct_positive']:.1f})" for g in gs])
write_tables("S4_direction_channel", [table(
    "S4_direction_channel", "tab:S4-direction-channel",
    "Mean pull (condition C, k = 10, main set) by the first-run consensus direction (R, E, S) and primary channel "
    "(codebook in S2), with the \\% of items with pull $>$ 0 in parentheses. Descriptive only",
    ["results/t1_results.json"],
    ["Variable", "Level", "n"] + [n for _, n in EMB.values()], rows,
    colspec="llrp{0.19\\linewidth}p{0.19\\linewidth}p{0.19\\linewidth}", size=r"\footnotesize", midrules=tuple(mids))],
    T1SRC, "analyses.*.by_direction, by_primary_channel from the unrounded results")

# traditions (main set)
itf = read_csv("results/items_final.csv")
ms = itf[(itf.low_control == 0) & itf.pull_C_k10_oai.notna()]
assert f_int(len(ms)) == mac("NkTen") and not ms.item_id.isin(excl_ids).any()
assert ms.groupby("tradition").tradition_pipeline.nunique().max() == 1
assert ms.groupby("tradition_pipeline").tradition.nunique().max() == 1
tested = {tag: {r["tradition"]: r for r in t1u["analyses"][tag]["by_tradition_n3"]["rows"]} for tag, _ in EMB.values()}
rows = []
for tr, g in sorted(ms.groupby("tradition"), key=lambda x: (-len(x[1]), x[0].casefold())):
    tp = g.tradition_pipeline.iloc[0]
    cells = []
    for tag, _ in EMB.values():
        x = g[f"pull_C_k10_{KEY[tag]}"]
        if tp in tested[tag]:
            assert len(x) == tested[tag][tp]["n"] and abs(x.mean() - tested[tag][tp]["mean"]) < 1e-12
        cells.append(f_dec(x.mean(), 3))
    tests = [f"{p_cell(tested[tag][tp]['p_one_sided'])} ({p_cell(tested[tag][tp]['q_bh'])})" if tp in tested[tag]
             else "--" for tag, _ in EMB.values()]
    rows.append([tex(tr), str(len(g))] + cells + tests)
assert all(len(tested[tag]) == sum(len(g) >= 3 for _, g in ms.groupby("tradition")) for tag, _ in EMB.values())
assert len(rows) == ms.tradition.nunique() == t1u["hypotheses"]["H-C1"]["per_model"]["text_embedding_3_large"]["n_traditions"]
summ_t = "; ".join(f"{n}: {b['tested']} tested, {b['p05']} with p $<$ .05, {b['bh_q05']} with q $<$ .05"
                   for (tag, n), b in zip(EMB.values(), [t1u["analyses"][t]["by_tradition_n3"] for t, _ in EMB.values()]))
SH = {"oai": "OpenAI", "bge": "bge-m3", "gem": "Gemini"}
write_tables("S4_traditions", [table(
    "S4_traditions", "tab:S4-traditions",
    f"Mean pull by tradition (condition C, k = 10, main set; {len(rows)} traditions, published labels). Traditions "
    "with at least three items were tested one by one: one-sided one-sample t test, p with the Benjamini--Hochberg q "
    "across the tested traditions of the model in parentheses",
    ["results/t1_results.json", "results/items_final.csv"],
    ["Tradition", "n"] + [f"Mean {SH[k]}" for k in EMB] + [f"p (q) {SH[k]}" for k in EMB], rows,
    colspec="p{0.27\\linewidth}rrrrlll", long=True, size=r"\scriptsize",
    notes=[f"Tests: {summ_t}. The q values were computed from the p values as stored (4 decimals)."])],
    T1SRC + ["results/items_final.csv"], "means recomputed from items_final.csv (checked against the tested rows); "
    "tests = analyses.*.by_tradition_n3 from the unrounded results")

# H1 and H2 in condition U (sensitivity, plan of 2026-11-03)
ru = load_unrounded("results/recording_conditionU.json", "results/recording_conditionU_unrounded.json")
rm = load("results/recording_results.json")
UC = {"pull_oai_U": EMB["oai"][1], "pull_bge_U": EMB["bge"][1], "pull_gem_U": EMB["gem"][1]}
LABS = [("majority", "main", "verdicts_main"), ("unanimous", "sensitivity_unanimous", "verdicts_sensitivity")]
h1rows, h2rows = [], []
yn = lambda b: "yes" if b else "no"
d4 = lambda x, pl=3: "--" if x is None else f_dec(x, pl)
p4 = lambda p: "--" if p is None else p_cell(p)
for lab, blk, vk in LABS:
    for i, (col, name) in enumerate(UC.items()):
        t, v1 = ru[blk][col]["trend"], ru[vk]["H1"]["per_model"][col]
        h1rows.append([lab if i == 0 else "", name, str(ru[blk][col]["n_existing_resolved"]),
                       f"{d4(t['slope'])} ({d4(t['se'])})", p4(t["p_one_sided"]), p4(v1["trend_p_holm"]),
                       yn(v1["order_ok"]), yn(v1["all_groups_positive"]), yn(v1["passes"])])
        h2, v2 = ru[blk][col]["h2_unrecorded_existing"], ru[vk]["H2"]["per_model"][col]
        h2rows.append([lab if i == 0 else "", name, str(h2["n"]), str(h2["n_traditions"]), d4(h2["mean"]),
                       f"{h2['pct_positive']:.1f}", f"{d4(h2.get('mixed_intercept'))} ({d4(h2.get('mixed_se'))})",
                       p4(h2.get("p_mixed_one_sided")), p4(v2["p_holm"])])
nose = [(lab, UC[col], ru[blk][col]["h2_unrecorded_existing"]) for lab, blk, _ in LABS for col in UC
        if ru[blk][col]["h2_unrecorded_existing"].get("mixed_se") is None]
NOSE = "".join(f"--: {name}, {lab} labels: the mixed model gave no standard error ({h['n']} items, {h['n_traditions']} "
               "traditions), so it has no p and Holm runs over the other two models. " for lab, name, h in nose).strip()
vline = lambda h, key: "; ".join(
    f"{lab}: {rm[vk][h]['verdict']} ({rm[vk][h][key]}/3) $\\to$ {ru[vk][h]['verdict']} ({ru[vk][h][key]}/3)"
    for lab, _, vk in LABS)
assert ru["openai_main_vs_U"]["n_identical"] == 0
RSRC = ["results/recording_conditionU.json", "results/recording_results.json"]
RSRC_M = RSRC + ["results/recording_conditionU_unrounded.json"]
CAPU = ("in condition U for all three models (pull\\_B\\_k10: the host as stored, the stored controls, the first "
        "ten; sensitivity analysis fixed on 2026-11-03 before it was run). The main analysis used, for OpenAI, the "
        "pull of the earlier pipeline. Items on an existing form with a resolved recording status; majority labels "
        "and the subset on which all three coders agree")
write_tables("S4_H1H2_conditionU", [
    table("S4_H1H2_conditionU", "tab:S4-H1-U",
          "H1 (recording gradient) " + CAPU + ". Trend: mixed model pull $\\sim$ level + (1 | tradition) with "
          "unrecorded = 0, cultural reference = 1, general dictionary = 2; one-sided p, Holm across the three models. "
          "A model passes if the trend is positive with Holm p $<$ .05, the means are ordered and every group mean is "
          "above 0 (one-sided t, p $<$ .05)", RSRC,
          ["Labels", "Model", "n", "Slope (SE)", "p", "p (Holm)", "Ordered", "All $>$ 0", "Passes"], h1rows,
          colspec="lp{0.19\\linewidth}rlrrlll", size=r"\footnotesize", midrules=(3,),
          notes=["Verdict, main analysis $\\to$ condition U (models passing): " + vline("H1", "models_passing") + "."]),
    table("S4_H1H2_conditionU", "tab:S4-H2-U",
          "H2 (unrecorded items have pull $>$ 0) " + CAPU + ". Mixed model pull $\\sim$ 1 + (1 | tradition) on the "
          "unrecorded items; one-sided p, Holm across the models with a result; supported if two or more models have "
          "Holm p $<$ .05", RSRC,
          ["Labels", "Model", "n", "Trad.", "Mean", "\\% $>$ 0", "Intercept (SE)", "p", "p (Holm)"], h2rows,
          colspec="lp{0.19\\linewidth}rrrrlrr", size=r"\footnotesize", midrules=(3,),
          notes=["Verdict, main analysis $\\to$ condition U (models with Holm p $<$ .05): "
                 + vline("H2", "models_sig_holm") + ". " + NOSE])],
    RSRC_M, "recording_conditionU from the unrounded results (checked against the stored ones) and the verdicts of "
    "recording_results.json")


# ================================================================== S5 models and settings
def log_dates(paths, key="time"):
    t = []
    for p in paths:
        for l in open(ROOT / p, encoding="utf-8"):
            a = json.loads(l)
            if a.get(key):
                t.append(a[key][:10])
    return (min(t) if min(t) == max(t) else f"{min(t)} to {max(t)}") if t else "not logged"


def temp_of(paths):
    v = {json.loads(l).get("temperature_applied") for p in paths for l in open(ROOT / p, encoding="utf-8")}
    assert len(v) == 1
    return "temperature 0" if v.pop() else "temperature not set (model default)"


def coding_rows(run, fmt):
    out = []
    for c, m in CODERS.items():
        p = [fmt.format(c=c, m=m)]
        extra = "; max\\_tokens 16000" if c == "claude" else ""
        extra += "; JSON response type" if c == "gemini" else ""
        out.append([run, m, temp_of(p) + extra, log_dates(p)])
    return out


src_s5 = []
gen_c = re.search(r'model="([^"]+)",\s*messages=.*?temperature=([\d.]+),\s*max_completion_tokens=(\d+)', leg, re.S)
dims = re.search(r"^DIMS = (\d+)", leg, re.M).group(1)
bge = load("embeddings/results/bge_m3_meta.json")
rows = [["Control words, conditions A--C (v1)", gen_c.group(1), f"temperature {gen_c.group(2)}; max\\_completion\\_tokens "
         f"{gen_c.group(3)}", "v1 study (not logged)"],
        ["Control words, condition D", d_ex["model"], f"temperature {d_ex['temperature']}; max\\_completion\\_tokens "
         f"{d_ex['max_completion_tokens']}", log_dates(["embeddings/results/logs/controls_D.jsonl"])],
        ["Embeddings, condition A (v1)", "text-embedding-3-large", f"{dims} dimensions", "v1 study (not logged)"]]
for tag, name in EMB.values():
    rows.append(["Embeddings, condition C", name,
                 {"oai": "default dimensions", "bge": f"local, revision {bge['revision'][:7]}; normalised",
                  "gem": "task\\_type SEMANTIC\\_SIMILARITY"}[[k for k, (t, n) in EMB.items() if t == tag][0]],
                 log_dates([f"embeddings/results/nnpm_C_{tag}.jsonl"])])
ab_date = re.search(r"Conditions A and B were run on (\d{4}-\d\d-\d\d)",
                    (ROOT / "README.md").read_text(encoding="utf-8")).group(1)
rows.append(["Embeddings, conditions A and B (v1 control words, re-embedded)", "BAAI/bge-m3; gemini-embedding-001",
             "as condition C", f"{ab_date} (no request log)"])
for tag, name in EMB.values():
    rows.append(["Embeddings, conditions D and E (control words)", name, "as condition C",
                 log_dates([f"embeddings/results/logs/nnpm_DE_{tag}.jsonl"])])
for tag, name in EMB.values():
    p = f"embeddings/results/logs/controls_E_{tag}.jsonl"
    rows.append(["Embeddings, condition E vocabulary", name, "wordfreq top 50{,}000 words per language",
                 log_dates([p])])
for size, r in ms5["models"].items():
    rows.append(["Embeddings, model size (C, D)", r["name"], f"local, revision {r['revision'][:7]}; prefix "
                 "``query: ''; normalised", r["time"][:10]])
for run, fmt in [("Coding, first run", "coding/results/first_run/codes_{c}.csv.jsonl"),
                 ("Coding, form and boundary", "coding/results/v1_1/v1_1_{c}.csv.jsonl"),
                 ("Coding, recording status", "coding/results/recording/recording_{c}.csv.jsonl"),
                 ("Coding, taboo domain", "coding/results/taboo/taboo_{c}.csv.jsonl"),
                 ("Source audit, stage B", "audit/runs/judge_{m}.jsonl"),
                 ("Source audit, judged again (1 item)", "audit/runs/judge_{m}_rejudge_2026-10-25.jsonl"),
                 ("Source audit, components", "audit/runs/judge_components_{m}.jsonl")]:
    rows += coding_rows(run, fmt)
acc = sorted(read_csv("dictionary_check/daijisen_judgments.csv", dtype=str)["取得日時"].str[:10])
dj_date = re.search(r"reading the extracted text on (\d{4}-\d\d-\d\d)",
                    (ROOT / "dictionary_check/scripts/apply_judgements.py").read_text(encoding="utf-8")).group(1)
att = []
for p in sorted((ROOT / "private/audit_attempts").glob("*.jsonl")):
    for l in open(p, encoding="utf-8"):
        a = json.loads(l)
        if a.get("accessed"):
            att.append(a["accessed"][:10])
rows += [["Source audit, evidence collection", "claude-opus-5-5 (agents with web search)", "instructions in S5",
          f"{min(att)} to {max(att)}"],
         ["Dictionary check, \\ja{大辞泉} judgement", "claude-opus-5-5", "reads the retrieved entry text",
          f"pages {acc[0]}" + ("" if acc[0] == acc[-1] else f" to {acc[-1]}") + f"; judged {dj_date}"]]
write_tables("S5_models", [table(
    "S5_models", "tab:S5-models",
    "Every model call: model as named in the call (API models have no further version string; local models "
    "with the revision of the model files), settings, and dates of the calls (from the request logs). Coders and "
    "embedding models were called in a fresh process for each run",
    ["request logs", "embeddings/results/*meta.json", "results/model_size_e5.json"],
    ["Step", "Model", "Settings", "Dates"], rows,
    colspec="p{0.27\\linewidth}p{0.22\\linewidth}p{0.27\\linewidth}p{0.17\\linewidth}", long=True,
    size=r"\footnotesize")],
    ["embeddings/results/logs/controls_D.jsonl", "embeddings/results/bge_m3_meta.json", "results/model_size_e5.json",
     "dictionary_check/daijisen_judgments.csv", "legacy/v1_submission/scripts/archaeology/cpg_archaeology.py"],
    "settings from the scripts and logs (temperature_applied, max_tokens); dates = min to max of the logged times")


# analysis inputs: host string, controls and columns per analysis
oai_b = read_csv("embeddings/results/nnpm_text_embedding_3_large.csv").rename(columns={"id": "item_id"})
iw = read_csv("results/items_with_consensus.csv").merge(oai_b[["item_id", "pull_openai", "pull_B_k10"]], on="item_id")
assert len(iw) == 213 and (iw.pull_index == iw.pull_openai).all()
dif = (iw.pull_index - iw.pull_B_k10).abs()
n_same3 = int((iw.pull_index.map(lambda x: hu(x, 3)) == iw.pull_B_k10.map(lambda x: hu(x, 3))).sum())
for col, tag in [("pull_bge_B", "BAAI_bge_m3"), ("pull_gem_B", "gemini_embedding_001")]:
    b = read_csv(f"embeddings/results/nnpm_{tag}.csv").rename(columns={"id": "item_id"})
    assert (iw[["item_id", col]].merge(b[["item_id", "pull_B_k10"]], on="item_id").pipe(
        lambda x: x[col] == x.pull_B_k10)).all()
HOST = {"C": "checked host (clean\\_host: the coders' majority form; first form where the author adopted it)",
        "U": "host as stored (host\\_original: original script; may contain an explanation)",
        "A": "host as embedded by the earlier pipeline (host\\_pipeline: often romanised, sometimes with a gloss)"}
CTRL = {"S": "stored controls, first ten", "C": "stored controls cleaned by R1--R4, first ten kept",
        "D": "controls generated without the covert target (condition D), cleaned, first ten kept",
        "E": "nearest vocabulary words in the model's space (condition E), cleaned, first ten kept"}
FILES = {"C": "nnpm\\_C\\_$\\langle$model$\\rangle$.csv", "U": "nnpm\\_$\\langle$model$\\rangle$.csv", "DE": "nnpm\\_DE\\_$\\langle$model$\\rangle$.csv"}
rows = [
    ["H-C1, H-C2 (main analysis)", HOST["C"], CTRL["C"], "pull\\_C\\_k10 (" + FILES["C"] + ")"],
    ["k sensitivity", HOST["C"], "as H-C1, first 3, 5 or all kept", "pull\\_C\\_k3, pull\\_C\\_k5, pull\\_C\\_kall ("
     + FILES["C"] + ")"],
    ["Direction, channel, traditions; tradition grouping and corrected labels; form type; source audit subsets",
     HOST["C"], CTRL["C"] + "; condition D where stated", "pull\\_C\\_k10; pull\\_D\\_k10 (" + FILES["DE"] + ")"],
    ["H-D1, H-D2", HOST["C"], CTRL["D"], "pull\\_D\\_k10 (" + FILES["DE"] + ")"],
    ["H-E1, H-E2", HOST["C"], CTRL["E"], "pull\\_E\\_k10 (" + FILES["DE"] + ")"],
    ["Model size (multilingual-e5)", HOST["C"], "conditions C and D, as above", "pull\\_C\\_k10, pull\\_D\\_k10 "
     "(nnpm\\_e5\\_$\\langle$size$\\rangle$.csv)"],
    ["Condition U test (S4)", HOST["U"], CTRL["S"], "pull\\_B\\_k10 (" + FILES["U"] + ")"],
    ["H1--H3, recording status (main analysis)", "OpenAI: " + HOST["A"] + "; bge-m3, Gemini: " + HOST["U"],
     "OpenAI: the controls of the earlier pipeline (data/control\\_cache.json); bge-m3, Gemini: " + CTRL["S"],
     "pull\\_index, pull\\_bge\\_B, pull\\_gem\\_B (results/items\\_with\\_consensus.csv)"],
    ["H1, H2 in condition U (S4)", HOST["U"], CTRL["S"], "pull\\_B\\_k10 (" + FILES["U"] + ")"],
    ["Condition A (earlier pipeline, S4 comparison)", HOST["A"], CTRL["S"], "pull\\_A\\_k10 (" + FILES["U"] + ")"],
]
cnt = lambda k: HCH.get(k, 0)
assert sum(HCH.values()) == 213 and set(HCH) <= {"same", "gloss", "one_form", "first_form", "layout", "other"}
notes = [
    f"Host strings, condition U $\\to$ C (213 items with stored controls): {cnt('same')} the same, "
    f"{213 - cnt('same')} changed: explanation in parentheses or after a dash removed {cnt('gloss')}; one of several "
    f"forms kept {cnt('one_form')}; first form adopted by the author {cnt('first_form')}; line breaks, slashes or "
    f"capitals only {cnt('layout')}; another form chosen by the coders {cnt('other')} ("
    + ", ".join(tex(i) for i in OTHER_IDS) + ").",
    f"Condition U: the stored host string contains an explanation in parentheses or after a dash in {N_GLOSS_U} of "
    "the 213 items.",
    f"OpenAI, recording status: pull\\_index (earlier pipeline) and pull\\_B\\_k10 (condition U) differ in all "
    f"{len(iw)} items (none identical; {int((dif < 1e-4).sum())} differ by less than 0.0001, {n_same3} agree "
    f"at 3 decimals; largest difference {f_dec(dif.max(), 3)}). For bge-m3 and Gemini, pull\\_bge\\_B and "
    "pull\\_gem\\_B are the condition U pulls (identical).",
    "Files in embeddings/results/; $\\langle$model$\\rangle$ = text\\_embedding\\_3\\_large, BAAI\\_bge\\_m3, gemini\\_embedding\\_001.",
]
write_tables("S5_analysis_inputs", [table(
    "S5_analysis_inputs", "tab:S5-inputs",
    "Inputs of each analysis: the host string embedded, the control words and the pull column used",
    ["embeddings/results", "results/items_with_consensus.csv", "embeddings/controls/clean_hosts.csv"],
    ["Analysis", "Host string", "Controls", "Column (file)"], rows,
    colspec="p{0.2\\linewidth}p{0.27\\linewidth}p{0.22\\linewidth}p{0.22\\linewidth}", long=True,
    size=r"\footnotesize", notes=notes)],
    ["embeddings/results/nnpm_text_embedding_3_large.csv", "embeddings/results/nnpm_BAAI_bge_m3.csv",
     "embeddings/results/nnpm_gemini_embedding_001.csv", "embeddings/controls/clean_hosts.csv",
     "results/items_with_consensus.csv"],
    "host strings compared per item (categories by rule, in order: first form, explanation removed, one of several "
    "forms, layout only, other); pull_index compared with pull_B_k10")


# ================================================================== S6 exploratory
tr = load("results/taboo_results.json")
REC3 = ["general_dictionary", "cultural_reference", "unrecorded"]
mt = tr["main"]
rows = [[t] + [str(tr["crosstab"][t][r]) for r in REC3] + [str(sum(tr["crosstab"][t].values())),
                                                            pct(tr["crosstab"][t]["general_dictionary"],
                                                                sum(tr["crosstab"][t].values()))]
        for t in ["taboo", "no taboo"]]
assert rows[0][-1] == mac("PgenTabooYes") and rows[1][-1] == mac("PgenTabooNo")
t_a = table("S6_taboo", "tab:S6-taboo-crosstab",
            f"Taboo domain (exploratory): taboo (consensus label not none) by recording status, {mt['n']} items "
            f"(left out: {mt['n_left_out_recording_unresolved']} with unresolved recording status, "
            f"{mt['n_left_out_taboo_unresolved']} with unresolved taboo label). Difference in the share of "
            f"general\\_dictionary: {same(f_dec(mt['diff_pct_points'], 1), 'DgenTaboo')} percentage points, 95\\% CI "
            f"{same(f_ci(*mt['ci95_pct_points_newcombe'], places=1), 'CIgenTaboo')} (Newcombe); Fisher's exact test, "
            f"one-sided (taboo lower) p {mac('PgenTabooFisher')}, two-sided p {mac('PgenTabooFisherTwo')}",
            ["results/taboo_results.json"], ["", "general\\_dict.", "cultural\\_ref.", "unrecorded", "Total",
                                             "\\% general"], rows, colspec="lrrrrr")
g = tr["glmm"]
t_b = table("S6_taboo", "tab:S6-taboo-domain",
            "Share of general\\_dictionary per taboo domain. Auxiliary logistic mixed model general $\\sim$ taboo + "
            f"(1 | tradition) ({tex(g['method'])}; {g['n']} items, {g['n_traditions']} traditions): taboo "
            f"coefficient posterior mean {f4(g['taboo_post_mean'], 2)}, SD {f4(g['taboo_post_sd'], 2)}, mean "
            f"$\\pm$ 1.96 SD {f4ci(*g['interval_mean_pm_1.96sd'], places=2)}",
            ["results/taboo_results.json"], ["Domain", "n", "general\\_dictionary", "\\%"],
            [[tex(r["taboo_domain"]), str(r["n"]), str(r["n_general"]),
              "--" if r["pct_general"] is None else f4(r["pct_general"], 1)] for r in tr["per_domain"]],
            colspec="lrrr")
t_c = table("S6_taboo", "tab:S6-taboo-refusals",
            "Items in a request refused by a coder, per taboo domain (consensus label, all 217 items). Only "
            "claude-opus-5-5 refused; no item was refused when sent alone",
            ["results/taboo_results.json"], ["Domain", "Items", "In a refused request", "Resent singly"],
            [[tex(r["taboo_domain"]), str(r["n_items_217"]), str(r["in_refused_request_any"]),
              str(r["resent_one_by_one_claude"])] for r in tr["refusals_by_label"]], colspec="lrrr")
write_tables("S6_taboo", [t_a, t_b, t_c], ["results/taboo_results.json"],
             "crosstab, main, glmm, per_domain, refusals_by_label; checked against the taboo macros")

ft = load_unrounded("results/form_type_pull.json", "results/form_type_pull_unrounded.json")
rows = []
for cond in ["C", "D"]:
    for r in ft["conditions"][cond]["per_model"]:
        rows.append([cond, r["model"], str(r["n"]), str(r["by_form_type"]["existing_form"]["n"]),
                     str(r["by_form_type"]["transparent_combination"]["n"]), f_dec(r["b_transparent"], 3),
                     f_ci(*r["ci95"]), f_dec(r["z"], 2), p_cell(r["p_holm"]), r["reading"]])
write_tables("S6_form_type", [table(
    "S6_form_type", "tab:S6-form-type",
    "Pull by form type (exploratory): coefficient of transparent\\_combination against existing\\_form in pull "
    "$\\sim$ form\\_type + (1 | tradition), conditions C and D. One-sided p (prediction: positive), Holm across "
    "the three models", ["results/form_type_pull.json"],
    ["Cond.", "Model", "n", "Existing", "Transp.", "b", "95\\% CI", "z", "p (Holm)", "Reading"], rows,
    colspec="lp{0.17\\linewidth}rrrrlrrp{0.17\\linewidth}", size=r"\footnotesize", midrules=(3,))],
    ["results/form_type_pull.json", "results/form_type_pull_unrounded.json"], "per_model rows from the unrounded "
    "results (checked against the stored ones)")

sk_parts = sankoku_tables("S6", "S6_sankoku")
tb_lab = ["taboo" if read_csv("coding/results/taboo/taboo_consensus.csv", dtype=str).set_index("item_id")
          .taboo_domain[i] != "none" else "no taboo" for i in sk.item_id]
sk_parts.append(table("S6_sankoku", "tab:S6-sankoku-taboo",
                      "Taboo (consensus taboo domain not none) by \\ja{三省堂国語辞典} 8th ed. (descriptive)",
                      ["dictionary_check/sankoku_judgments.csv", "coding/results/taboo/taboo_consensus.csv"],
                      [""] + [tex(x, "ja") for x in SK4] + ["Total"],
                      xtab_rows(tb_lab, sk["三国_掲載"], ["taboo", "no taboo"], SK4), colspec="lrrrrr", midrules=(2,)))
write_tables("S6_sankoku", sk_parts, SKSRC + ["coding/results/taboo/taboo_consensus.csv"],
             "as S3_dictionary_sankoku, plus taboo x 三国 (descriptive)")

# ------------------------------------------------------------------ checks on everything written, manifest
BAD = [r"/Users", r"/private/", "scratchpad", "API_KEY", r"\bsk-[A-Za-z0-9]{10}", r"\.env\b", "SPEC_", "internal/", "Language Sciences"]
for p in sorted(TAB.glob("*.tex")) + sorted(TXT.glob("*.txt")):
    s = p.read_text(encoding="utf-8")
    for b in BAD:
        assert not re.search(b, s), (p.name, b)
    if p.suffix == ".txt":
        assert all(len(x) <= 100 for x in s.split("\n")), p.name
assert not HALF, "values that need the unrounded result:\n" + "\n".join(HALF)
q = [r for r in MAN if r["file"].startswith("si/")]
pd.DataFrame(q).to_csv(SI / "si_manifest.csv", index=False, encoding="utf-8")
print(f"{len(list(TAB.glob('*.tex')))} tables, {len(list(TXT.glob('*.txt')))} texts")
for r in q:
    print(" ", r["file"])
