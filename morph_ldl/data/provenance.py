"""Derivation provenance of MGN paradigm files, read from the MGN build scripts.

`mgn_data/build-data/clean-paradigms.R` builds `mgn_data/data/*.csv` from UniMorph
(`/media/data/corpora/morphology/unimorph/<code>/<code>`), optionally running
`epitran` over the *form* column (`df_x$form <- epi_transliterate(...)`; lexeme labels
stay orthographic). `process_subset()` keeps cells with >100 forms, pivots to wide
format joining multiple forms per (lexeme, cell) as `paste(sort(unique(x)), collapse=";")`
(so variant order is alphabetical, not source order) and drops columns that duplicate an
earlier column across the whole file (lexicon-wide syncretism).
`mgn_data/build-data/clean-pol.R` builds Polish from PoliMorf 0.6.7 with epitran pol-Latn.
`mgn_data/data-custom/*.csv` have no build script in the repository.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_READ = re.compile(r'^(df_\w+)\s*<-\s*read_tsv\("[^"]*/unimorph/(\w+)/(\w+)"')
_EPI = re.compile(r'^(df_\w+)\$form\s*<-\s*epi_transliterate\(\s*df_\w+\$form\s*,\s*"([^"]+)"\)')
_WRITE = re.compile(r'write_csv\((df_\w+?)_(adj|n|v)\s*,\s*"\.\./data/([\w-]+)\.csv"\)')


@dataclass
class BuildProvenance:
    file_stem: str
    derivation: str                      # unimorph | polimorf | data_custom_undocumented
    upstream_code: Optional[str] = None
    epitran_code: Optional[str] = None
    script: Optional[str] = None
    script_lines: List[int] = field(default_factory=list)
    n_build_blocks: int = 0
    cell_min_forms: Optional[int] = None
    notes: List[str] = field(default_factory=list)


def parse_clean_paradigms(path: Path) -> Dict[str, BuildProvenance]:
    current: Dict[str, dict] = {}
    out: Dict[str, BuildProvenance] = {}
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        s = line.strip()
        m = _READ.match(s)
        if m:
            current[m.group(1)] = {"code": m.group(2), "epi": None, "line": i}
            continue
        m = _EPI.match(s)
        if m and m.group(1) in current:
            current[m.group(1)]["epi"] = m.group(2)
            continue
        for m in _WRITE.finditer(s):
            var, stem = m.group(1), m.group(3)
            blk = current.get(var, {})
            prev = out.get(stem)
            prov = BuildProvenance(
                file_stem=stem, derivation="unimorph", upstream_code=blk.get("code"),
                epitran_code=blk.get("epi"), script=path.name, cell_min_forms=101,
                script_lines=(prev.script_lines if prev else []) + [blk.get("line", i), i],
                n_build_blocks=(prev.n_build_blocks if prev else 0) + 1,
            )
            if prev is not None:
                prov.notes = prev.notes + [
                    f"written by {prov.n_build_blocks} blocks in {path.name}; the last block "
                    "(lines %s) determines the file" % prov.script_lines[-2:]]
                if prev.epitran_code != prov.epitran_code:
                    prov.notes.append("blocks disagree on epitran use")
            out[stem] = prov
    return out


def parse_clean_pol(path: Path) -> Dict[str, BuildProvenance]:
    text = path.read_text(encoding="utf-8")
    out = {}
    if "PoliMorf" in text:
        for stem, n in (("pol-adj", 401), ("pol-n", 101)):
            if f"{stem}.csv" in text:
                out[stem] = BuildProvenance(
                    file_stem=stem, derivation="polimorf", upstream_code="PoliMorf-0.6.7",
                    epitran_code="pol-Latn" if "pol-Latn" in text else None, script=path.name,
                    cell_min_forms=n, n_build_blocks=1,
                    notes=["built from PoliMorf 0.6.7, not UniMorph; script writes to ./data/"])
    return out


# Provisional source attributions for data-custom files, inferred from label and
# transcription conventions only (no build script in the repository). Unverified.
CUSTOM_SOURCE_HINTS = {
    "french-v": "label set (prs.1sg, ipfv.3pl, pst.ptcp.f.sg) and archiphonemes E/O match Flexique; unverified",
    "latin-v": "label format VERB:Fin+... matches LatInfLexi; unverified",
    "latin-n": "label format NOUN:Abl+Plur matches LatInfLexi nouns; unverified",
    "navajo-v": "Navajo verb paradigms with ':IPA' cell suffix; source undocumented here",
    "english-v": "8-cell English verb table (pres1s/pres3s/presothers/past13/pastnot13); CELEX-like layout; unverified",
}


def build_provenance(mgn_root: Path) -> Dict[str, BuildProvenance]:
    """Provenance for every file stem in mgn_data/data and mgn_data/data-custom."""
    prov = parse_clean_paradigms(mgn_root / "build-data" / "clean-paradigms.R")
    prov.update(parse_clean_pol(mgn_root / "build-data" / "clean-pol.R"))
    for f in sorted((mgn_root / "data").glob("*.csv")):
        if f.stem not in prov:
            prov[f.stem] = BuildProvenance(file_stem=f.stem, derivation="unknown",
                                           notes=["no write_csv for this file in build scripts"])
    for f in sorted((mgn_root / "data-custom").glob("*.csv")):
        prov[f.stem] = BuildProvenance(
            file_stem=f.stem, derivation="data_custom_undocumented",
            notes=[CUSTOM_SOURCE_HINTS.get(f.stem, "no build script in repository")])
    return prov


def representation_from_provenance(p: BuildProvenance, collection: str) -> str:
    if collection == "mgn_data-custom":
        return "phon_custom"
    if p.epitran_code:
        return "ipa_epitran"
    return "orth"


# --------------------------------------------------------------------------
# Heuristic cross-check of the declared representation against the forms.
# --------------------------------------------------------------------------

IPA_ONLY = set("ɐɑɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑ")


def representation_check(sample_forms: List[str], declared: str) -> dict:
    forms = [f for f in sample_forms if f]
    if not forms:
        return {"status": "no_forms", "ipa_char_rate": None, "spaced_token_rate": None, "upper_rate": None}
    ipa = sum(any(c in IPA_ONLY for c in f) for f in forms) / len(forms)
    spaced = sum((" " in f) and all(len(t) <= 3 for t in f.split()) for f in forms) / len(forms)
    upper = sum(any(c.isupper() for c in f) for f in forms) / len(forms)
    status = "consistent"
    if declared == "phon_custom" and spaced < 0.5:
        status = "warning: custom forms not space-tokenised"
    if declared == "orth" and spaced > 0.5:
        status = "warning: orth forms look space-tokenised"
    if declared == "ipa_epitran" and upper > 0.01:
        status = "warning: upper-case letters in epitran output (untransliterated material)"
    return {"status": status, "ipa_char_rate": round(ipa, 4), "spaced_token_rate": round(spaced, 4),
            "upper_rate": round(upper, 4)}
