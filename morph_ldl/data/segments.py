"""Segmentation of forms into space-separated symbols (docs/CONTRACT.md §2).

* orth: one symbol per Unicode character, combining marks (Mn/Me) attached to the
  preceding character.
* ipa_epitran (contract is silent; proposal recorded in DATA.md): as orth, and in
  addition IPA spacing modifiers for length / secondary articulation (ː ˑ ʰ ʲ ʷ ˠ ˤ ˀ ⁿ)
  are attached to the preceding symbol. epitran was run with ligatures=True, so
  affricates are already single code points (ʧ, ʤ, ʦ ...).
* phon_custom: the resource's own whitespace-separated tokens.
Word spaces inside a form become the symbol `_`. `#` is reserved; forms containing
it are flagged by `has_reserved`.
"""

from __future__ import annotations

import re
import unicodedata

from .util import nfc

WORD_SEP = "_"
BOUNDARY = "#"
IPA_ATTACH = set("ːˑʰʲʷˠˤˀⁿ")
_WS = re.compile(r"\s+")


def _graphemes(word: str, attach_modifiers: bool) -> list[str]:
    out: list[str] = []
    for ch in word:
        cat = unicodedata.category(ch)
        if out and (cat in ("Mn", "Me") or (attach_modifiers and ch in IPA_ATTACH)):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def segment(form: str, representation: str) -> str:
    form = nfc(form).strip()
    if not form:
        return ""
    if representation == "phon_custom":
        # Tokens are already space-separated; a word boundary cannot be told apart
        # from a token boundary in these resources, so none is inserted.
        return " ".join(_WS.split(form))
    attach = representation == "ipa_epitran"
    syms: list[str] = []
    for i, word in enumerate(_WS.split(form)):
        if i:
            syms.append(WORD_SEP)
        syms.extend(_graphemes(word, attach))
    return " ".join(syms)


def is_multiword(form: str, representation: str) -> bool:
    if representation == "phon_custom":
        return False
    return bool(_WS.search(form.strip()))


def has_reserved(form: str) -> bool:
    return BOUNDARY in form
