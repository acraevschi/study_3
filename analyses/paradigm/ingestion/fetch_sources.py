#!/usr/bin/env python3
"""Programmatic acquisition and SHA-256 provenance logging for Track 2 morphological lexicons.

Downloads and verifies morphological paradigm datasets across three priority tiers:
  - Priority 1: Paralex v1 Zenodo releases (LeFFI, Vlexique, VeLeSpa, VeLeCa, VeLeRo,
    Eesthetic, ParaFin, FinCog, Zalilex, Parahungarian, NasLex) + documented CELEX English V.
  - Priority 2: Curated open national morphological lexicons (Morph-it! v0.48, PoliMorf 0.6.7,
    VESUM v6.8.6).
  - Priority 3: Official raw orthographic UniMorph repositories (github.com/unimorph/{code}.git)
    and SIGMORPHON 2020 Task 0 Uralic paradigm tables (mdf, myv, udm) for accepted GeLaTo
    Glottocodes in analyses/common/data/analysis_links.csv.

Writes exact URLs, versions / 40-char git commit SHAs, DOIs, licenses, academic citations,
and file SHA-256 checksums to `analyses/paradigm/data/provenance_manifest.csv`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "paradigm_sources"
DATA_OUT_DIR = PROJECT_ROOT / "analyses" / "paradigm" / "data"
MANIFEST_PATH = DATA_OUT_DIR / "provenance_manifest.csv"

USER_AGENT = "Study3-Track2-DataIngest/1.0 (Academic Morphological Predictability Research)"

CELEX_ZENODO_RECORD_ID = 11147171
CELEX_ENG_V_SHA256 = "a17ac27a620a500f051420d5a24dffd0c9c1d66db170c1a07738d27be5532ad6"

MORPHIT_UTF8_SHA256 = "7375216351ee59cee4326b455503f3a6c4e9542cd63ecb86ff5cd58ca5c8e800"
MORPHIT_ARCHIVE_URL = (
    "https://kielipankki.fi/download/hfst-morphologies/test/it/morph-it/hfst-it-morph-morph-it.zip"
)
MORPHIT_FALLBACK_UTF8_URL = (
    "https://raw.githubusercontent.com/TraMOOC-Impleval/lemmatize_italian_data/master/morph-it_048_utf8.txt"
)

MANIFEST_COLUMNS = [
    "source_id",
    "priority_tier",
    "dataset_name",
    "glottocodes",
    "iso_codes",
    "url",
    "version_or_commit",
    "doi",
    "license",
    "citation",
    "local_path",
    "sha256_target",
    "sha256",
    "status",
    "notes",
]


@dataclass(frozen=True)
class ZenodoParalexSpec:
    source_id: str
    record_id: int
    dataset_name: str
    glottocodes: str
    iso_codes: str
    doi: str
    version: str
    license: str
    citation: str
    notes: str = ""


PARALEX_SPECS: Tuple[ZenodoParalexSpec, ...] = (
    ZenodoParalexSpec(
        source_id="paralex_leffi",
        record_id=14442347,
        dataset_name="LeFFI: Lexicon of Italian Verb Forms",
        glottocodes="ital1282",
        iso_codes="ita",
        doi="10.5281/zenodo.14442347",
        version="v2.0.1",
        license="CC-BY-4.0",
        citation="Pellegrini, Matteo & Cignarella, Alessandra Teresa (2020/2024). LeFFI: Lexicon of Italian Verb Forms (v2.0.1). Zenodo. https://doi.org/10.5281/zenodo.14442347",
        notes="Paralex v1 Italian verbs (phon + sounds.csv; paired with Morph-it! v0.48 orth).",
    ),
    ZenodoParalexSpec(
        source_id="paralex_vlexique",
        record_id=22769916,
        dataset_name="Vlexique: French Verbal Lexicon",
        glottocodes="stan1290",
        iso_codes="fra",
        doi="10.5281/zenodo.22769916",
        version="2.1.1",
        license="CC-BY-NC-SA-4.0",
        citation="Beniamine, Sacha, Coavoux, Maximin, & Bonami, Olivier (2024/2026). Vlexique 2.1 (2.1.1). Zenodo. https://doi.org/10.5281/zenodo.22769916",
        notes="Paralex v1 French verbs derived from Lexique/Flexique (orth + phon + sounds.csv).",
    ),
    ZenodoParalexSpec(
        source_id="paralex_velespa",
        record_id=14202916,
        dataset_name="VeLeSpa: Verbal Lexicon of Spanish",
        glottocodes="stan1288",
        iso_codes="spa",
        doi="10.5281/zenodo.14202916",
        version="1.0",
        license="CC-BY-4.0",
        citation="Herce, Borja (2024). VeLeSpa: a Verbal Lexicon of Spanish. Language Resources and Evaluation / Zenodo. https://doi.org/10.5281/zenodo.14202916",
        notes="Paralex v1 Castilian Spanish verbs (phon + sounds.csv; paired with UniMorph spa orth).",
    ),
    ZenodoParalexSpec(
        source_id="paralex_veleca",
        record_id=14203025,
        dataset_name="VeLeCa: Verbal Lexicon of Catalan",
        glottocodes="stan1289",
        iso_codes="cat",
        doi="10.5281/zenodo.14203025",
        version="1.0",
        license="CC-BY-4.0",
        citation="Herce, Borja (2024). VeLeCa: a Verbal Lexicon of Catalan. Zenodo. https://doi.org/10.5281/zenodo.14203025",
        notes="Paralex v1 Central Catalan verbs (orth + phon + sounds.csv).",
    ),
    ZenodoParalexSpec(
        source_id="paralex_velero",
        record_id=21242751,
        dataset_name="VeLeRo: Verbal Lexicon of Romanian",
        glottocodes="roma1327",
        iso_codes="ron",
        doi="10.5281/zenodo.21242751",
        version="1.0",
        license="CC-BY-4.0",
        citation="Herce, Borja & Pricop, Bogdan (2025). VeLeRo: a Verbal Lexicon of Romanian. Zenodo. https://doi.org/10.5281/zenodo.21242751",
        notes="Paralex v1 Romanian verbs (phon + sounds.csv; paired with UniMorph ron orth).",
    ),
    ZenodoParalexSpec(
        source_id="paralex_eesthetic",
        record_id=14069724,
        dataset_name="Eesthetic: Estonian Paradigms in Phonemic Notation",
        glottocodes="esto1258",
        iso_codes="ekk",
        doi="10.5281/zenodo.14069724",
        version="v1.0.5",
        license="CC-BY-4.0",
        citation="Beniamine, Sacha, Baerman, Matthew, Aigro, Mari, Copot, Maria, & Bouton, Jules (2024). Eesthetic: Estonian Paradigms in Phonemic Notation (v1.0.5). Zenodo. https://doi.org/10.5281/zenodo.14069724",
        notes="Surrey/Paralex Standard Estonian nouns and verbs.",
    ),
    ZenodoParalexSpec(
        source_id="paralex_parafin",
        record_id=19219779,
        dataset_name="ParaFin: Finnish Paradigms in Phonemic Notation",
        glottocodes="finn1318",
        iso_codes="fin",
        doi="10.5281/zenodo.19219779",
        version="2.0.2",
        license="GPL-3.0-or-later",
        citation="Bouton, Jules (2024/2025). ParaFin: Finnish Paradigms in Phonemic Notation (2.0.2). Zenodo. https://doi.org/10.5281/zenodo.19219779",
        notes="Paralex Finnish nominal and verbal paradigms.",
    ),
    ZenodoParalexSpec(
        source_id="paralex_fincog",
        record_id=18624373,
        dataset_name="FinCog: Finnic Cognates in Paralex datasets",
        glottocodes="finn1318;esto1258",
        iso_codes="fin;ekk;olo",
        doi="10.5281/zenodo.18624373",
        version="1.0.0-alpha4",
        license="GPL-3.0-or-later",
        citation="Bouton, Jules (2025). FinCog: Finnic Cognates in Paralex datasets (1.0.0-alpha4). Zenodo. https://doi.org/10.5281/zenodo.18624373",
        notes="Supplementary Finnic cognate Paralex dataset.",
    ),
    ZenodoParalexSpec(
        source_id="paralex_parakar",
        record_id=19219021,
        dataset_name="ParaKar: Livvi Karelian Paradigms in Phonemic Notation",
        glottocodes="livv1243",
        iso_codes="olo",
        doi="10.5281/zenodo.19219021",
        version="1.0.4",
        license="CC-BY-SA-4.0",
        citation="Bouton, Jules (2024/2025). ParaKar: Livvi Karelian Paradigms in Phonemic Notation (1.0.4). Zenodo. https://doi.org/10.5281/zenodo.19219021",
        notes="Downloaded & verified per spec; note GeLaTo Karelian links to kare1335 (krl, Karelian Proper) rather than livv1243 (olo).",
    ),
    ZenodoParalexSpec(
        source_id="paralex_zalilex",
        record_id=15235590,
        dataset_name="Zalilex: Russian Nominal Paradigms Lexicon",
        glottocodes="russ1263",
        iso_codes="rus",
        doi="10.5281/zenodo.15235590",
        version="1.0.0",
        license="CC-BY-4.0",
        citation="Beniamine, Sacha et al. (2025). Zalilex: Russian Nominal Paradigms Lexicon (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.15235590",
        notes="Zaliznyak Russian nominal lexicon in Paralex standard (orth + phon + sounds.csv).",
    ),
    ZenodoParalexSpec(
        source_id="paralex_parahungarian",
        record_id=14218751,
        dataset_name="Parahungarian: Dataset of Hungarian Nouns",
        glottocodes="hung1274",
        iso_codes="hun",
        doi="10.5281/zenodo.14218751",
        version="1.0",
        license="CC-BY-4.0",
        citation="Parahungarian contributors (2024). Parahungarian: a dataset of Hungarian nouns. Zenodo. https://doi.org/10.5281/zenodo.14218751",
        notes="Paralex v1 Hungarian nouns (phon + sounds.csv; paired with UniMorph hun orth).",
    ),
    ZenodoParalexSpec(
        source_id="paralex_naslex",
        record_id=20039927,
        dataset_name="NasLex: Bosnian, Croatian, Montenegrin, Serbian Inflectional Lexicon",
        glottocodes="sout1528",
        iso_codes="hbs",
        doi="10.5281/zenodo.20039927",
        version="0.1",
        license="CC-BY-SA-4.0",
        citation="Sims, Andrea D. & Copot, Maria (2025). NasLex: A Bosnian, Croatian, Montenegrin, Serbian inflectional lexicon. Zenodo. https://doi.org/10.5281/zenodo.20039927",
        notes="Paralex v1 BCMS nominal paradigms (orth + phon + sounds.csv).",
    ),
)

# Canonical mapping from GeLaTo-linked Glottocodes (in analysis_links.csv) to official
# github.com/unimorph/{um_code} repositories.
# Verified against outputs/pcfp_v2_links_2026_10_20/typology/analysis_links.csv,
# external/glottolog-cldf/cldf/languages.csv, morph_ldl/data/identifiers.py, and
# morph_ldl/data/vendor/mgn_language_map.py.
UNIMORPH_SPECS: Tuple[Tuple[str, str, str], ...] = (
    # (um_repo_code, glottocode, canonical_iso639_3)
    ("ady", "adyg1241", "ady"),
    ("afb", "gulf1241", "afb"),
    ("amh", "amha1245", "amh"),
    ("arz", "egyp1253", "arz"),
    ("ast", "astu1245", "ast"),
    ("aym", "sout2996", "ayc"),
    ("aze", "nort2697", "azj"),
    ("bak", "bash1264", "bak"),
    ("bel", "bela1254", "bel"),
    ("ben", "beng1280", "ben"),
    ("bul", "bulg1262", "bul"),
    ("cat", "stan1289", "cat"),
    ("ceb", "cebu1242", "ceb"),
    ("ces", "czec1258", "ces"),
    ("ckt", "chuk1273", "ckt"),
    ("cre", "plai1258", "crk"),
    ("deu", "stan1295", "deu"),
    ("ell", "mode1248", "ell"),
    ("eng", "stan1293", "eng"),
    ("est", "esto1258", "ekk"),
    ("eus", "basq1248", "eus"),
    ("evn", "even1259", "evn"),
    ("fas", "west2369", "pes"),
    ("fin", "finn1318", "fin"),
    ("fra", "stan1290", "fra"),
    ("gal", "gali1258", "glg"),
    ("gla", "scot1245", "gla"),
    ("gle", "iris1253", "gle"),
    ("guj", "guja1252", "guj"),
    ("hbs", "sout1528", "hbs"),
    ("hun", "hung1274", "hun"),
    ("hye", "nucl1235", "hye"),
    ("isl", "icel1247", "isl"),
    ("ita", "ital1282", "ita"),
    ("itl", "itel1242", "itl"),
    ("jpn", "nucl1643", "jpn"),
    ("kat", "nucl1302", "kat"),
    ("kaz", "kaza1248", "kaz"),
    ("kbd", "kaba1278", "kbd"),
    ("ket", "kett1243", "ket"),
    ("khk", "halh1238", "khk"),
    ("kir", "kirg1245", "kir"),
    ("kjh", "khak1248", "kjh"),
    ("kmr", "nort2641", "kmr"),
    ("kor", "kore1280", "kor"),
    ("krl", "kare1335", "krl"),
    ("lit", "lith1251", "lit"),
    ("mlt", "malt1254", "mlt"),
    ("mon", "peri1253", "mvf"),  # Inspected in build_lexicons: excluded if script/variety duplicates khk
    ("nap", "neap1235", "nap"),
    ("nob", "norw1258", "nor"),
    ("nya", "nyan1308", "nya"),
    ("orm", "west2721", "gaz"),
    ("oss", "iron1242", "oss"),
    ("pol", "poli1260", "pol"),
    ("pus", "sout2649", "pbt"),
    ("que", "cusc1236", "quz"),
    ("ron", "roma1327", "ron"),
    ("rus", "russ1263", "rus"),
    ("sah", "yaku1245", "sah"),
    ("sme", "nort2671", "sme"),
    ("sot", "sout2807", "sot"),
    ("spa", "stan1288", "spa"),
    ("sqi", "gheg1238", "aln"),  # Linked to GeLaTo Albanian population via analysis_links.csv gheg1238
    ("tat", "tata1255", "tat"),
    ("tel", "telu1262", "tel"),
    ("tgk", "taji1245", "tgk"),
    ("tgl", "taga1270", "tgl"),
    ("tuk", "turk1304", "tuk"),
    ("tur", "nucl1301", "tur"),
    ("tyv", "tuvi1240", "tyv"),
    ("uig", "uigh1240", "uig"),
    ("ukr", "ukra1253", "ukr"),
    ("uzb", "nort2690", "uzn"),
    ("vep", "veps1250", "vep"),
    ("yid", "east2295", "ydd"),
    ("zul", "zulu1248", "zul"),
)

UNIMORPH_CITATION = (
    "Batsuren, Khuyagbaatar et al. (2022). UniMorph 4.0: Universal Morphology. "
    "Proceedings of LREC 2022, pp. 840-855. DOI: 10.5281/zenodo.7598038"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_path(path: Path, root: Path = PROJECT_ROOT) -> str:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        return str(resolved.relative_to(root_resolved))
    except ValueError:
        return str(resolved)


def download_url(url: str, dest: Path, timeout: int = 120, retries: int = 3) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Optional[BaseException] = None
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1024 * 256)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


class _HttpRangeReader(io.RawIOBase):
    """Seekable read-only stream backed by HTTP Range requests (206 Partial Content)."""

    def __init__(self, url: str, size: int, timeout: int = 60, retries: int = 3) -> None:
        self.url = url
        self.size = int(size)
        self.pos = 0
        self.timeout = timeout
        self.retries = max(1, retries)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        self.pos = max(0, min(self.size, self.pos))
        return self.pos

    def readinto(self, b: bytearray | memoryview) -> int:  # type: ignore[override]
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.size - 1, self.pos + len(b) - 1)
        last_err: Optional[BaseException] = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.url,
                    headers={"User-Agent": USER_AGENT, "Range": f"bytes={self.pos}-{end}"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    if resp.status != 206:
                        raise RuntimeError(f"Expected HTTP 206 Partial Content, got {resp.status}")
                    data = resp.read()
                n = len(data)
                b[:n] = data
                self.pos += n
                return n
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = exc
                if attempt + 1 < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        assert last_err is not None
        raise last_err


def _extract_zip_member_from_url(
    url: str,
    total_size: int,
    member_suffix: str,
    dest: Path,
    timeout: int = 120,
) -> None:
    """Extract a single file ending with `member_suffix` from a remote ZIP archive."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = dest.with_suffix(dest.suffix + ".part")
    try:
        buffered = io.BufferedReader(_HttpRangeReader(url, total_size, timeout=timeout), buffer_size=256 * 1024)
        with zipfile.ZipFile(buffered, "r") as zf:
            matches = [n for n in zf.namelist() if n.endswith(member_suffix)]
            if not matches:
                raise KeyError(f"No member ending with {member_suffix!r} in remote archive {url}")
            with zf.open(matches[0], "r") as src, open(tmp_out, "wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 256)
        tmp_out.replace(dest)
        return
    except RuntimeError:
        # Fallback if an intermediate HTTP proxy strips Range headers
        tmp_zip = dest.with_suffix(".archive.zip.part")
        try:
            download_url(url, tmp_zip, timeout=max(timeout, 600))
            with zipfile.ZipFile(tmp_zip, "r") as zf:
                matches = [n for n in zf.namelist() if n.endswith(member_suffix)]
                if not matches:
                    raise KeyError(f"No member ending with {member_suffix!r} in {url}")
                with zf.open(matches[0], "r") as src, open(tmp_out, "wb") as out:
                    shutil.copyfileobj(src, out, length=1024 * 256)
            tmp_out.replace(dest)
        finally:
            if tmp_zip.exists():
                tmp_zip.unlink()


def resolve_analysis_links_csv(project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve tracked `analyses/common/data/analysis_links.csv` (with legacy outputs/ fallback)."""
    candidates = (
        project_root / "analyses" / "common" / "data" / "analysis_links.csv",
        project_root / "outputs" / "pcfp_v2_links_2026_10_20" / "typology" / "analysis_links.csv",
    )
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Expected GeLaTo language-link table at {candidates[0]} (or {candidates[1]})"
    )


def _manifest_target_for_source(
    source_id: str,
    project_root: Path = PROJECT_ROOT,
    manifest_path: Optional[Path] = None,
) -> Optional[Path]:
    mpath = manifest_path or (project_root / "analyses" / "paradigm" / "data" / "provenance_manifest.csv")
    if not mpath.is_file():
        return None
    with open(mpath, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("source_id") == source_id and row.get("sha256_target"):
                t = Path(row["sha256_target"])
                cand = t if t.is_absolute() else (project_root / t)
                if cand.is_file():
                    return cand
    return None


def resolve_celex_english_path(
    project_root: Path = PROJECT_ROOT,
    external_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> Path:
    """Resolve local or downloaded CELEX English verbal phonemic lexicon (`english-v.csv`)."""
    ext_root = external_root or (project_root / "external" / "paradigm_sources")
    candidates = [
        project_root / "mgn_data" / "data-custom" / "english-v.csv",
        ext_root / "celex" / "english-v.csv",
    ]
    m_cand = _manifest_target_for_source("celex_eng_v", project_root=project_root, manifest_path=manifest_path)
    if m_cand is not None:
        candidates.append(m_cand)
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Could not find CELEX english-v.csv at {candidates[0]} or {candidates[1]}; run fetch_sources.py first."
    )


def resolve_morphit_path(
    project_root: Path = PROJECT_ROOT,
    external_root: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
) -> Path:
    """Resolve local or downloaded Morph-it! v0.48 UTF-8 lexicon (`morph-it_048.utf8.txt`)."""
    ext_root = external_root or (project_root / "external" / "paradigm_sources")
    candidates = [
        project_root / "external" / "morph-it" / "morph-it_048.utf8.txt",
        ext_root / "national" / "morphit" / "morph-it_048.utf8.txt",
    ]
    m_cand = _manifest_target_for_source("national_morphit", project_root=project_root, manifest_path=manifest_path)
    if m_cand is not None:
        candidates.append(m_cand)
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Could not find Morph-it! v0.48 at {candidates[0]} or {candidates[1]}; run fetch_sources.py first."
    )


def fetch_json(url: str, timeout: int = 45) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def git_rev_parse_head(repo_dir: Path) -> str:
    res = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return res.stdout.strip()


def find_unimorph_data_files(repo_dir: Path, code: str) -> List[Path]:
    """Locate canonical raw UniMorph paradigm TSV files inside a cloned repository.

    Strictly excludes auxiliary/sidecar files (`.args`, `.gloss`, `.derivations*`,
    `.segmentations*`, `.sm`, `.xz`, `.zip`, `.4`, conversion docs) and restricts
    multi-dialect Finnic repositories (`vep`, `krl`) to their standard literary tables.
    """
    if not repo_dir.exists():
        return []
    if code == "vep":
        p = repo_dir / "vep-new-written-veps"
        return [p] if p.is_file() else []
    if code == "krl":
        p = repo_dir / "krl-new-written-karelian"
        return [p] if p.is_file() else []
    if code == "kaz":
        return [p for p in (repo_dir / "kaz.noun.tsv", repo_dir / "kaz") if p.is_file()]
    if code == "fra":
        # `fra.um4` provides N (and v4 V); `fra` provides v3 V (never pooled into the same POS)
        return [p for p in (repo_dir / "fra.um4", repo_dir / "fra") if p.is_file()]

    part_re = re.compile(rf"^{re.escape(code)}(?:\.[123])?$")
    candidates: List[Path] = []
    for entry in sorted(repo_dir.iterdir(), key=lambda p: p.name):
        if entry.name.startswith(".") or not entry.is_file():
            continue
        if part_re.match(entry.name):
            candidates.append(entry)
    return candidates


def fetch_paralex_dataset(spec: ZenodoParalexSpec, force: bool = False) -> dict:
    base_dir = EXTERNAL_ROOT / "paralex" / spec.source_id
    raw_dir = base_dir / "raw"
    ext_dir = base_dir / "extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ext_dir.mkdir(parents=True, exist_ok=True)

    api_url = f"https://zenodo.org/api/records/{spec.record_id}"
    meta_cache = base_dir / "zenodo_record.json"
    if force or not meta_cache.exists():
        rec = fetch_json(api_url)
        with open(meta_cache, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)
    else:
        with open(meta_cache, encoding="utf-8") as f:
            rec = json.load(f)

    files = rec.get("files", [])
    if not files:
        raise RuntimeError(f"{spec.source_id}: Zenodo record {spec.record_id} lists no files")

    downloaded_paths: List[Path] = []
    for fobj in files:
        fname = Path(fobj["key"]).name
        f_url = fobj.get("links", {}).get("self") or fobj.get("links", {}).get("content")
        if not f_url:
            f_url = f"https://zenodo.org/api/records/{spec.record_id}/files/{fobj['key']}/content"
        dest = raw_dir / fname
        expected_checksum = fobj.get("checksum", "")
        need_dl = force or not dest.exists() or dest.stat().st_size == 0
        if not need_dl and expected_checksum.startswith("md5:"):
            if md5_file(dest) != expected_checksum.split(":", 1)[1]:
                need_dl = True
        if need_dl:
            download_url(f_url, dest, timeout=180)
            if expected_checksum.startswith("md5:"):
                got_md5 = md5_file(dest)
                want_md5 = expected_checksum.split(":", 1)[1]
                if got_md5 != want_md5:
                    raise RuntimeError(
                        f"{spec.source_id} ({fname}): MD5 mismatch got {got_md5}, expected {want_md5}"
                    )
        downloaded_paths.append(dest)

        if fname.endswith(".zip"):
            marker = ext_dir / f".unpacked_{fname}"
            if force or not marker.exists():
                with zipfile.ZipFile(dest, "r") as zf:
                    zf.extractall(ext_dir)
                marker.write_text("ok\n", encoding="utf-8")
        else:
            target_copy = ext_dir / fname
            if force or not target_copy.exists():
                shutil.copy2(dest, target_copy)

    # Choose primary archive or forms.csv as sha256_target for direct single-file auditability
    zips = [p for p in downloaded_paths if p.name.endswith(".zip")]
    forms_csvs = [p for p in downloaded_paths if "forms" in p.name.lower() and p.name.endswith(".csv")]
    primary_file = zips[0] if zips else (forms_csvs[0] if forms_csvs else sorted(downloaded_paths)[0])

    return {
        "source_id": spec.source_id,
        "priority_tier": "1_paralex",
        "dataset_name": spec.dataset_name,
        "glottocodes": spec.glottocodes,
        "iso_codes": spec.iso_codes,
        "url": f"https://zenodo.org/records/{spec.record_id}",
        "version_or_commit": spec.version,
        "doi": spec.doi,
        "license": spec.license,
        "citation": spec.citation,
        "local_path": rel_path(ext_dir),
        "sha256_target": rel_path(primary_file),
        "sha256": sha256_file(primary_file),
        "status": "downloaded",
        "notes": spec.notes,
    }


def fetch_celex_english(
    force: bool = False,
    *,
    project_root: Path = PROJECT_ROOT,
    external_root: Path = EXTERNAL_ROOT,
) -> dict:
    local_celex = project_root / "mgn_data" / "data-custom" / "english-v.csv"
    if not force and local_celex.is_file():
        got_sha = sha256_file(local_celex)
        if got_sha != CELEX_ENG_V_SHA256:
            raise RuntimeError(
                f"celex_eng_v ({local_celex}): SHA-256 mismatch got {got_sha}, expected {CELEX_ENG_V_SHA256}"
            )
        celex_path = local_celex
        status = "reused_local"
    else:
        celex_dir = external_root / "celex"
        celex_path = celex_dir / "english-v.csv"
        celex_dir.mkdir(parents=True, exist_ok=True)
        need_dl = force or not celex_path.is_file() or sha256_file(celex_path) != CELEX_ENG_V_SHA256
        if need_dl:
            api_url = f"https://zenodo.org/api/records/{CELEX_ZENODO_RECORD_ID}"
            meta_cache = celex_dir / "zenodo_record.json"
            if force or not meta_cache.is_file():
                rec = fetch_json(api_url)
                with open(meta_cache, "w", encoding="utf-8") as f:
                    json.dump(rec, f, indent=2, ensure_ascii=False)
            else:
                with open(meta_cache, encoding="utf-8") as f:
                    rec = json.load(f)
            files = rec.get("files", [])
            zip_obj = next((fo for fo in files if str(fo.get("key", "")).endswith(".zip")), None)
            if zip_obj is None:
                raise RuntimeError(f"celex_eng_v: Zenodo record {CELEX_ZENODO_RECORD_ID} contains no .zip archive")
            f_url = (
                zip_obj.get("links", {}).get("self")
                or zip_obj.get("links", {}).get("content")
                or f"https://zenodo.org/api/records/{CELEX_ZENODO_RECORD_ID}/files/{zip_obj['key']}/content"
            )
            f_size = int(zip_obj["size"])
            _extract_zip_member_from_url(
                f_url,
                total_size=f_size,
                member_suffix="data-custom/english-v.csv",
                dest=celex_path,
                timeout=180,
            )
        got_sha = sha256_file(celex_path)
        if got_sha != CELEX_ENG_V_SHA256:
            raise RuntimeError(
                f"celex_eng_v ({celex_path}): SHA-256 mismatch got {got_sha}, expected {CELEX_ENG_V_SHA256}"
            )
        status = "downloaded"

    return {
        "source_id": "celex_eng_v",
        "priority_tier": "1_celex",
        "dataset_name": "CELEX2 English Verbal Phonemic Lexicon (via MGN data-custom)",
        "glottocodes": "stan1293",
        "iso_codes": "eng",
        "url": f"https://doi.org/10.5281/zenodo.{CELEX_ZENODO_RECORD_ID}",
        "version_or_commit": "CELEX2-MGN2024",
        "doi": f"10.5281/zenodo.{CELEX_ZENODO_RECORD_ID}",
        "license": "LDC-CELEX2 / CC-BY-4.0 (Guzman Naranjo 2024)",
        "citation": (
            "Baayen, R. Harald, Piepenbrock, Richard, & Gulikers, Leon (1995). The CELEX Lexical Database "
            f"(Release 2), LDC96L14; packaged in Guzman Naranjo, Matias (2024), Zenodo 10.5281/zenodo.{CELEX_ZENODO_RECORD_ID}."
        ),
        "local_path": rel_path(celex_path, project_root),
        "sha256_target": rel_path(celex_path, project_root),
        "sha256": got_sha,
        "status": status,
        "notes": "Phonemic English verb paradigms (8 cells) for paired Orth-vs-IPA evaluation alongside unimorph_eng.",
    }


def fetch_morphit(
    force: bool = False,
    *,
    project_root: Path = PROJECT_ROOT,
    external_root: Path = EXTERNAL_ROOT,
) -> dict:
    local_morphit = project_root / "external" / "morph-it" / "morph-it_048.utf8.txt"
    if not force and local_morphit.is_file():
        got_sha = sha256_file(local_morphit)
        if got_sha != MORPHIT_UTF8_SHA256:
            raise RuntimeError(
                f"national_morphit ({local_morphit}): SHA-256 mismatch got {got_sha}, expected {MORPHIT_UTF8_SHA256}"
            )
        morphit_txt = local_morphit
        status = "reused_local"
    else:
        morphit_dir = external_root / "national" / "morphit"
        morphit_txt = morphit_dir / "morph-it_048.utf8.txt"
        morphit_dir.mkdir(parents=True, exist_ok=True)
        need_dl = force or not morphit_txt.is_file() or sha256_file(morphit_txt) != MORPHIT_UTF8_SHA256
        if need_dl:
            zip_dest = morphit_dir / "hfst-it-morph-morph-it.zip"
            tmp_txt = morphit_txt.with_suffix(".txt.part")
            try:
                download_url(MORPHIT_ARCHIVE_URL, zip_dest, timeout=180)
                with zipfile.ZipFile(zip_dest, "r") as zf:
                    members = [n for n in zf.namelist() if n.endswith("morph-it_048.txt")]
                    if not members:
                        raise KeyError("morph-it_048.txt not found inside hfst-it-morph-morph-it.zip")
                    raw_latin1 = zf.read(members[0])
                tmp_txt.write_bytes(raw_latin1.decode("iso-8859-1").encode("utf-8"))
                tmp_txt.replace(morphit_txt)
            except (urllib.error.URLError, TimeoutError, OSError, zipfile.BadZipFile, KeyError):
                download_url(MORPHIT_FALLBACK_UTF8_URL, morphit_txt, timeout=180)
        got_sha = sha256_file(morphit_txt)
        if got_sha != MORPHIT_UTF8_SHA256:
            raise RuntimeError(
                f"national_morphit ({morphit_txt}): SHA-256 mismatch got {got_sha}, expected {MORPHIT_UTF8_SHA256}"
            )
        status = "downloaded"

    return {
        "source_id": "national_morphit",
        "priority_tier": "2_national",
        "dataset_name": "Morph-it! v0.48 Italian Morphological Lexicon",
        "glottocodes": "ital1282",
        "iso_codes": "ita",
        "url": MORPHIT_ARCHIVE_URL,
        "version_or_commit": "0.48",
        "doi": "",
        "license": "CC-BY-SA-2.5-IT / LGPL-2.1 (Morph-it! v0.48)",
        "citation": "Zanchetta, Eros & Baroni, Marco (2005). Morph-it! A free corpus-based morphological resource for the Italian language. Proceedings of Corpus Linguistics 2005, University of Birmingham.",
        "local_path": rel_path(morphit_txt.parent, project_root),
        "sha256_target": rel_path(morphit_txt, project_root),
        "sha256": got_sha,
        "status": status,
        "notes": "Canonical Italian nominal (N) and verbal (V) orthographic lexicon (retaining essere/avere; parent lexicon of LeFFI).",
    }


def fetch_national_lexicons(
    force: bool = False,
    *,
    project_root: Path = PROJECT_ROOT,
    external_root: Path = EXTERNAL_ROOT,
) -> List[dict]:
    rows: List[dict] = []

    # 1. Morph-it! v0.48 (reuses external/morph-it/ when present, else auto-downloads & verifies SHA-256)
    rows.append(fetch_morphit(force=force, project_root=project_root, external_root=external_root))

    # 2. PoliMorf 0.6.7 (IPI PAN open download, clean orthographic Polish V and N)
    polimorf_dir = external_root / "national" / "polimorf"
    polimorf_gz = polimorf_dir / "PoliMorf-0.6.7.tab.gz"
    polimorf_url = "http://zil.ipipan.waw.pl/PoliMorf?action=AttachFile&do=get&target=PoliMorf-0.6.7.tab.gz"
    if force or not polimorf_gz.exists() or polimorf_gz.stat().st_size < 1_000_000:
        download_url(polimorf_url, polimorf_gz, timeout=300)
    rows.append({
        "source_id": "national_polimorf",
        "priority_tier": "2_national",
        "dataset_name": "PoliMorf 0.6.7 Polish Morphological Dictionary",
        "glottocodes": "poli1260",
        "iso_codes": "pol",
        "url": polimorf_url,
        "version_or_commit": "0.6.7",
        "doi": "",
        "license": "BSD-2-Clause",
        "citation": "Wolinski, Marcin, Milkowski, Marcin, & Ogrodniczuk, Maciej (2012). PoliMorf: a (not so) new open morphological dictionary for Polish. Proceedings of LREC 2012, pp. 860-864.",
        "local_path": rel_path(polimorf_dir, project_root),
        "sha256_target": rel_path(polimorf_gz, project_root),
        "sha256": sha256_file(polimorf_gz),
        "status": "downloaded",
        "notes": "Clean orthographic Polish verbs and nouns directly from IPI PAN (eliminating MGN epitran noise).",
    })

    # 3. VESUM v6.8.6 (brown-uk/dict_uk official release, clean orthographic Ukrainian V and N)
    vesum_dir = external_root / "national" / "vesum"
    vesum_bz2 = vesum_dir / "dict_corp_vis.txt.bz2"
    vesum_url = "https://github.com/brown-uk/dict_uk/releases/download/v6.8.6/dict_corp_vis.txt.bz2"
    if force or not vesum_bz2.exists() or vesum_bz2.stat().st_size < 1_000_000:
        download_url(vesum_url, vesum_bz2, timeout=300)
    rows.append({
        "source_id": "national_vesum",
        "priority_tier": "2_national",
        "dataset_name": "VESUM (dict_uk) v6.8.6 Ukrainian Electronic Morphological Dictionary",
        "glottocodes": "ukra1253",
        "iso_codes": "ukr",
        "url": vesum_url,
        "version_or_commit": "v6.8.6",
        "doi": "",
        "license": "GPL-3.0 / LGPL-2.1 (brown-uk/dict_uk)",
        "citation": "Rysin, Andriy & Starko, Vasyl (2024). VESUM: Large Electronic Morphological Dictionary of Ukrainian (v6.8.6). https://github.com/brown-uk/dict_uk",
        "local_path": rel_path(vesum_dir, project_root),
        "sha256_target": rel_path(vesum_bz2, project_root),
        "sha256": sha256_file(vesum_bz2),
        "status": "downloaded",
        "notes": "Clean orthographic Ukrainian verbs and nouns (replacing MGN epitran ukr-n/v).",
    })

    return rows


def fetch_one_unimorph(spec: Tuple[str, str, str], force: bool = False) -> dict:
    code, glottocode, canon_iso = spec
    repo_url = f"https://github.com/unimorph/{code}.git"

    # Reuse existing pinned pilot checkouts for ita and fin
    pinned_local = PROJECT_ROOT / "external" / f"unimorph-{code}"
    if code in ("ita", "fin") and pinned_local.exists():
        repo_dir = pinned_local
        status = "reused_local"
    else:
        repo_dir = EXTERNAL_ROOT / "unimorph" / code
        status = "downloaded"
        if force and repo_dir.exists():
            shutil.rmtree(repo_dir)
        if not repo_dir.exists() or not find_unimorph_data_files(repo_dir, code):
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            if repo_dir.exists():
                shutil.rmtree(repo_dir)
            subprocess.run(
                ["git", "clone", "-q", "--depth", "1", repo_url, str(repo_dir)],
                check=True,
                timeout=180,
            )

    commit_sha = git_rev_parse_head(repo_dir)
    data_files = find_unimorph_data_files(repo_dir, code)
    if not data_files:
        return {
            "source_id": f"unimorph_{code}",
            "priority_tier": "3_unimorph",
            "dataset_name": f"UniMorph {code} ({glottocode})",
            "glottocodes": glottocode,
            "iso_codes": canon_iso,
            "url": repo_url,
            "version_or_commit": commit_sha,
            "doi": "10.5281/zenodo.7598038",
            "license": "CC-BY-SA-3.0 / CC-BY-4.0 (UniMorph)",
            "citation": UNIMORPH_CITATION,
            "local_path": rel_path(repo_dir),
            "sha256_target": "",
            "sha256": "",
            "status": "empty_repo",
            "notes": "Official UniMorph repository contains no paradigm TSV file.",
        }

    # Check license text if present in repo
    lic_text = "CC-BY-SA-3.0 / CC-BY-4.0 (UniMorph)"
    for lic_name in ("LICENSE", "LICENSE.txt", "LICENSE.md"):
        lp = repo_dir / lic_name
        if lp.exists():
            first_line = lp.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
            if first_line:
                lic_text = first_line[0].strip()[:80]
            break

    primary_file = data_files[0]
    extra_note = ""
    if len(data_files) > 1:
        extra_hashes = "; ".join(f"{p.name}:{sha256_file(p)[:12]}" for p in data_files)
        extra_note = f"Multi-part TSVs ({len(data_files)} files: {extra_hashes})."

    return {
        "source_id": f"unimorph_{code}",
        "priority_tier": "3_unimorph",
        "dataset_name": f"UniMorph {code} ({glottocode})",
        "glottocodes": glottocode,
        "iso_codes": canon_iso,
        "url": repo_url,
        "version_or_commit": commit_sha,
        "doi": "10.5281/zenodo.7598038",
        "license": lic_text,
        "citation": UNIMORPH_CITATION,
        "local_path": rel_path(repo_dir),
        "sha256_target": rel_path(primary_file),
        "sha256": sha256_file(primary_file),
        "status": status,
        "notes": extra_note or "Official raw orthographic UniMorph paradigm repository.",
    }


def fetch_sigmorphon2020_uralic(force: bool = False) -> dict:
    repo_url = "https://github.com/sigmorphon2020/task0-data.git"
    repo_dir = EXTERNAL_ROOT / "sigmorphon2020" / "task0-data"
    if force and repo_dir.exists():
        shutil.rmtree(repo_dir)
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", repo_url, str(repo_dir)],
            check=True,
            timeout=180,
        )
    commit_sha = git_rev_parse_head(repo_dir)
    target = repo_dir / "DEVELOPMENT-LANGUAGES" / "uralic" / "myv.trn"
    if not target.exists():
        # Search recursively for myv.trn in case directory case differs
        matches = list(repo_dir.rglob("myv.trn"))
        if matches:
            target = matches[0]
    return {
        "source_id": "sigmorphon2020_uralic",
        "priority_tier": "3_sigmorphon",
        "dataset_name": "SIGMORPHON 2020 Task 0 Uralic Paradigm Tables (mdf, myv, udm)",
        "glottocodes": "moks1248;erzy1239;udmu1245",
        "iso_codes": "mdf;myv;udm",
        "url": repo_url,
        "version_or_commit": commit_sha,
        "doi": "10.18653/v1/2020.sigmorphon-1.1",
        "license": "CC-BY-SA-3.0 / CC-BY-4.0 (UniMorph / GiellaLT via SIGMORPHON 2020)",
        "citation": "Vylomova, Ekaterina et al. (2020). SIGMORPHON 2020 Shared Task 0: Typologically Diverse Morphological Inflection. Proceedings of SIGMORPHON 2020, pp. 1-39.",
        "local_path": rel_path(repo_dir),
        "sha256_target": rel_path(target) if target.exists() else "",
        "sha256": sha256_file(target) if target.exists() else "",
        "status": "downloaded",
        "notes": "Provides Moksha (mdf), Erzya (myv), and Udmurt (udm) UniMorph tables absent from standalone github.com/unimorph repos.",
    }


def run_all(max_workers: int = 4, force: bool = False) -> List[dict]:
    max_workers = max(1, min(4, int(max_workers)))
    EXTERNAL_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[dict] = []

    print(f"[fetch_sources] Fetching {len(PARALEX_SPECS)} Priority-1 Paralex Zenodo datasets...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(fetch_paralex_dataset, s, force): s.source_id for s in PARALEX_SPECS}
        for fut in concurrent.futures.as_completed(futs):
            sid = futs[fut]
            row = fut.result()
            print(f"  [ok] {sid} -> sha256={row['sha256'][:12]}...")
            manifest_rows.append(row)

    celex_row = fetch_celex_english(force=force)
    manifest_rows.append(celex_row)
    print(f"  [ok] celex_eng_v ({celex_row['status']}) -> sha256={celex_row['sha256'][:12]}...")

    print("[fetch_sources] Fetching Priority-2 Curated National Lexicons...")
    nat_rows = fetch_national_lexicons(force=force)
    for r in nat_rows:
        print(f"  [ok] {r['source_id']} -> sha256={r['sha256'][:12]}...")
    manifest_rows.extend(nat_rows)

    print(f"[fetch_sources] Cloning/verifying {len(UNIMORPH_SPECS)} Priority-3 UniMorph repositories...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(fetch_one_unimorph, spec, force): spec[0] for spec in UNIMORPH_SPECS}
        for fut in concurrent.futures.as_completed(futs):
            code = futs[fut]
            row = fut.result()
            print(f"  [{row['status']}] unimorph_{code} @ {row['version_or_commit'][:10]}")
            manifest_rows.append(row)

    print("[fetch_sources] Cloning/verifying SIGMORPHON 2020 Uralic supplement...")
    sig_row = fetch_sigmorphon2020_uralic(force=force)
    print(f"  [ok] {sig_row['source_id']} @ {sig_row['version_or_commit'][:10]}")
    manifest_rows.append(sig_row)

    # Sort deterministically by priority_tier, source_id
    manifest_rows.sort(key=lambda r: (r["priority_tier"], r["source_id"]))

    with open(MANIFEST_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[fetch_sources] Wrote {len(manifest_rows)} provenance records to {MANIFEST_PATH}")
    return manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify Track 2 morphological lexicon sources.")
    parser.add_argument("--workers", type=int, default=4, help="Max concurrent download threads (<=4).")
    parser.add_argument("--force", action="store_true", help="Re-download existing archives/clones.")
    args = parser.parse_args()
    run_all(max_workers=args.workers, force=args.force)


if __name__ == "__main__":
    main()
