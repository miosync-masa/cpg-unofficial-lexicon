"""Cleaning rules R1-R4 of condition C (the same code as build_clean.py), for conditions D and E.

clean(controls, clean_host, host_keys) returns [(control, rule)] with rule "" for a kept control.
"""
import re
import unicodedata
from collections import Counter

HARAKAT = re.compile("[ً-ٰٟ]")


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold().replace("’", "'").replace("ʼ", "'")
    return HARAKAT.sub("", re.sub(r"\s+", " ", s)).strip()


def key(s):
    d = unicodedata.normalize("NFD", norm(s))
    return unicodedata.normalize("NFC", "".join(c for c in d if not unicodedata.combining(c))).replace(" ", "")


def char_script(c):
    o = ord(c)
    if 0x0400 <= o <= 0x052F or 0x1C80 <= o <= 0x1C8F or 0x2DE0 <= o <= 0x2DFF or 0xA640 <= o <= 0xA69F:
        return "Cyrillic"
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF or 0xFB50 <= o <= 0xFDFF \
            or 0xFE70 <= o <= 0xFEFF:
        return "Arabic"
    if 0x0900 <= o <= 0x097F or 0xA8E0 <= o <= 0xA8FF:
        return "Devanagari"
    if 0x1200 <= o <= 0x139F or 0x2D80 <= o <= 0x2DDF or 0xAB00 <= o <= 0xAB2F:
        return "Ethiopic"
    if 0xA980 <= o <= 0xA9DF:
        return "Javanese"
    if 0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF \
            or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F or 0x20000 <= o <= 0x2FA1F:
        return "CJK"
    if c.isascii() or 0x00C0 <= o <= 0x024F or 0x1E00 <= o <= 0x1EFF or 0x0250 <= o <= 0x02AF \
            or 0xA720 <= o <= 0xA7FF:
        return "Latin"
    return "Other"


def script(s):
    cs = [char_script(c) for c in str(s) if c.isalpha()]
    return Counter(cs).most_common(1)[0][0] if cs else "Other"


def clean(controls, clean_host, host_keys, limit=None):
    """R1-R4 in order, first rule that fires; stops once `limit` controls are kept (if given)."""
    hk, hs = key(clean_host), script(clean_host)
    has_letters = any(c.isalpha() for c in str(clean_host))
    seen, out = set(), []
    for c in controls:
        ck = key(c)
        if ck in host_keys:
            rule = "R1"
        elif len(ck) >= 2 and len(hk) >= 2 and (hk in ck or ck in hk):
            rule = "R2"
        elif has_letters and script(c) != hs:
            rule = "R3"
        elif ck in seen:
            rule = "R4"
        else:
            rule = ""
            seen.add(ck)
        out.append((c, rule))
        if limit and len(seen) >= limit:
            break
    return out
