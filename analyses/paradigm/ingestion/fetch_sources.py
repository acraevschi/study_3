#!/usr/bin/env python3
"""Programmatic acquisition and SHA-256 provenance logging for Track 2 morphological lexicons.

Downloads and verifies morphological paradigm datasets across three priority tiers:
  - Priority 1: Paralex v1 Zenodo releases (LeFFI, Vlexique, VeLeSpa, VeLeCa, VeLeRo,
    Eesthetic, ParaFin, FinCog, Zalilex, Parahungarian, NasLex) + documented CELEX English V.
  - Priority 2: Curated open national morphological lexicons (Morph-it! v0.48, PoliMorf 0.6.7,
    VESUM v6.8.6), with documented fallbacks where a login wall or multi-GB archive blocks fetch.
  - Priority 3: Official raw orthographic UniMorph repositories (github.com/unimorph/{code}.git)
    and SIGMORPHON 2020 Task 0 Uralic paradigm tables (mdf, myv, udm) for accepted GeLaTo
    Glottocodes in outputs/pcfp_v2_links_2026_10_20/typology/analysis_links.csv.

Writes exact URLs, versions / 40-char git commit SHAs, DOIs, licenses, academic citations,
and file SHA-256 checksums to `analyses/paradigm/data/provenance_manifest.csv`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
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


def rel_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def download_url(url: str, dest: Path, timeout: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, length=1024 * 256)
    tmp.replace(dest)


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


def fetch_celex_english() -> dict:
    celex_path = PROJECT_ROOT / "mgn_data" / "data-custom" / "english-v.csv"
    if not celex_path.exists():
        raise FileNotFoundError(f"Expected documented CELEX table at {celex_path}")
    return {
        "source_id": "celex_eng_v",
        "priority_tier": "1_celex",
        "dataset_name": "CELEX2 English Verbal Phonemic Lexicon (via MGN data-custom)",
        "glottocodes": "stan1293",
        "iso_codes": "eng",
        "url": "https://doi.org/10.5281/zenodo.10889598",
        "version_or_commit": "CELEX2-MGN2024",
        "doi": "10.5281/zenodo.10889598",
        "license": "LDC-CELEX2 / CC-BY-4.0 (Guzman Naranjo 2024)",
        "citation": "Baayen, R. Harald, Piepenbrock, Richard, & Gulikers, Leon (1995). The CELEX Lexical Database (Release 2), LDC96L14; packaged in Guzman Naranjo, Matias (2024), Zenodo 10.5281/zenodo.10889598.",
        "local_path": rel_path(celex_path),
        "sha256_target": rel_path(celex_path),
        "sha256": sha256_file(celex_path),
        "status": "reused_local",
        "notes": "Phonemic English verb paradigms (8 cells) for paired Orth-vs-IPA evaluation alongside unimorph_eng.",
    }


def fetch_national_lexicons(force: bool = False) -> List[dict]:
    rows: List[dict] = []

    # 1. Morph-it! v0.48 (already present in external/morph-it/)
    morphit_txt = PROJECT_ROOT / "external" / "morph-it" / "morph-it_048.utf8.txt"
    if not morphit_txt.exists():
        raise FileNotFoundError(f"Expected Morph-it! v0.48 at {morphit_txt}")
    rows.append({
        "source_id": "national_morphit",
        "priority_tier": "2_national",
        "dataset_name": "Morph-it! v0.48 Italian Morphological Lexicon",
        "glottocodes": "ital1282",
        "iso_codes": "ita",
        "url": "https://sslmitdev-online.sslmit.unibo.it/linguistics/morph-it.php",
        "version_or_commit": "0.48",
        "doi": "",
        "license": "CC-BY-SA-2.5-IT / LGPL-2.1 (Morph-it! v0.48)",
        "citation": "Zanchetta, Eros & Baroni, Marco (2005). Morph-it! A free corpus-based morphological resource for the Italian language. Proceedings of Corpus Linguistics 2005, University of Birmingham.",
        "local_path": rel_path(morphit_txt.parent),
        "sha256_target": rel_path(morphit_txt),
        "sha256": sha256_file(morphit_txt),
        "status": "reused_local",
        "notes": "Canonical Italian nominal (N) and verbal (V) orthographic lexicon (retaining essere/avere; parent lexicon of LeFFI).",
    })

    # 2. PoliMorf 0.6.7 (IPI PAN open download, clean orthographic Polish V and N)
    polimorf_dir = EXTERNAL_ROOT / "national" / "polimorf"
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
        "local_path": rel_path(polimorf_dir),
        "sha256_target": rel_path(polimorf_gz),
        "sha256": sha256_file(polimorf_gz),
        "status": "downloaded",
        "notes": "Clean orthographic Polish verbs and nouns directly from IPI PAN (eliminating MGN epitran noise).",
    })

    # 3. VESUM v6.8.6 (brown-uk/dict_uk official release, clean orthographic Ukrainian V and N)
    vesum_dir = EXTERNAL_ROOT / "national" / "vesum"
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
        "local_path": rel_path(vesum_dir),
        "sha256_target": rel_path(vesum_bz2),
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

    manifest_rows.append(fetch_celex_english())
    print("  [ok] celex_eng_v verified.")

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
