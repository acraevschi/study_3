#!/usr/bin/env python3
"""Programmatic acquisition of open academic token-frequency corpora for Study 3 Track 2.

Acquires and verifies three citable, non-proprietary frequency tiers for GeLaTo
languages without any hardcoded word counts:
  - Tier A: Universal Dependencies v2.15 (official GitHub repositories pinned at tag r2.15,
    recording exact 40-char commit SHA, tarball SHA-256, license, and (lemma, upos, form) counts).
  - Tier B: Leipzig Corpora Collection (official static tarballs from
    https://downloads.wortschatz-leipzig.de/corpora/, verifying tarball SHA-256 and retaining
    only compact *-words.txt and *-meta.txt / *-import.sql files on disk; Goldhahn et al. 2012).
  - Tier C: wordfreq v3.1.1 (Speer 2022, Zenodo DOI 10.5281/zenodo.7199437), recording exact
    on-disk msgpack wordlist SHA-256 hashes for overlapping major languages.

Strict resource bounds:
  - Maximum 2 worker threads by default.
  - Zero access to Anubis/anti-bot gated web pages; uses only direct static file archives
    and git tag resolution.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import unicodedata
import urllib.error
import urllib.request
import yaml

import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
FREQ_SOURCES_DIR = REPO_ROOT / "external" / "freq_sources"
PROVENANCE_CSV = REPO_ROOT / "analyses" / "paradigm" / "data" / "freq_provenance.csv"

USER_AGENT = "Study3-Track2-Academic-Frequency-Pipeline/1.0"

UD_CODES_YAML_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/docs-automation/master/codes_and_flags.yaml"
)
UD_AT_GLANCE_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/docs/pages-source/_includes/at_glance.html"
)
LCC_CATALOG_PARQUET_URL = (
    "https://huggingface.co/datasets/imvladikon/leipzig_corpora_collection/resolve/main/data/links.parquet"
)

UD_CITATION = (
    "Zeman, Daniel, Joakim Nivre, Mitchell Abrams, et al. (2024). "
    "Universal Dependencies 2.15. LINDAT/CLARIAH-CZ digital library at the Institute of "
    "Formal and Applied Linguistics (UFAL), Charles University. http://hdl.handle.net/11234/1-5787"
)
LCC_CITATION = (
    "Goldhahn, Dirk, Thomas Eckart, & Uwe Quasthoff (2012). "
    "Building Large Monolingual Dictionaries at the Leipzig Corpora Collection: "
    "From 100 to 200 Languages. Proceedings of LREC 2012, pp. 759-765."
)
WORDFREQ_CITATION = (
    "Speer, Robyn (2022). rspeer/wordfreq: v3.0 (v3.0.2). Zenodo. "
    "https://doi.org/10.5281/zenodo.7199437"
)
WORDFREQ_DOI_URL = "https://doi.org/10.5281/zenodo.7199437"

# Explicit, linguistically audited ISO-639-3 -> corpus ISO equivalences for standard/pluricentric
# varieties. Vernaculars (e.g. arz, afb) are NEVER mapped onto standard macrolanguages (ara).
ISO_CORPUS_ALIASES: Dict[str, List[str]] = {
    "ekk": ["ekk", "est"],
    "est": ["est", "ekk"],
    "pes": ["pes", "fas"],
    "fas": ["fas", "pes"],
    "arb": ["arb", "ara"],
    "ara": ["ara", "arb"],
    "azj": ["azj", "aze"],
    "aze": ["aze", "azj"],
    "azb": ["azb"],
    "als": ["als", "sqi"],
    "sqi": ["sqi", "als"],
    "aln": ["aln", "sqi", "als"],
    "hbs": ["hbs", "hrv", "srp", "bos"],
    "nob": ["nob", "nor"],
    "nor": ["nob", "nor"],
    "nno": ["nno"],
    "ydd": ["ydd", "yid"],
    "yid": ["yid", "ydd"],
    "lav": ["lav", "lvs"],
    "lvs": ["lvs", "lav"],
    "pbt": ["pbt", "pus", "pst"],
    "pst": ["pst", "pus", "pbt"],
    "pus": ["pus", "pbt", "pst"],
    "kmr": ["kmr"],
    "ckb": ["ckb", "kur"],
    "bxr": ["bxr", "bua"],
    "bua": ["bua", "bxr"],
    "khk": ["khk", "mon"],
    "mon": ["mon", "khk"],
    "uzn": ["uzn", "uzb"],
    "uzb": ["uzb", "uzn"],
    "kpv": ["kpv", "kom"],
    "koi": ["koi"],
    "mhr": ["mhr", "chm"],
    "mrj": ["mrj"],
    "krl": ["krl"],
    "olo": ["olo", "krl"],
    "swh": ["swh", "swa"],
    "swa": ["swa", "swh"],
    "gug": ["gug", "grn"],
    "grn": ["grn", "gug"],
    "ayc": ["ayc", "aym", "ayr"],
    "ayr": ["ayr", "aym", "ayc"],
    "aym": ["aym", "ayr", "ayc"],
    "quz": ["quz", "quy", "quh", "que"],
    "quy": ["quy", "quz", "que"],
    "quh": ["quh", "quz", "que"],
    "que": ["que", "quz", "quy", "quh"],
    "gaz": ["gaz", "orm"],
    "orm": ["orm", "gaz"],
    "plt": ["plt", "mlg"],
    "mlg": ["mlg", "plt"],
    "zsm": ["zsm", "msa"],
    "msa": ["msa", "zsm"],
    "ind": ["ind"],
    "npi": ["npi", "nep"],
    "nep": ["nep", "npi"],
    "ory": ["ory", "ori"],
    "ori": ["ori", "ory"],
    "ike": ["ike", "iku"],
    "iku": ["iku", "ike"],
    "fre": ["fra"],
    "gal": ["glg"],
}

# Exact 1-to-1 map from canonical ISO-639-3 to wordfreq v3.1.1 BCP-47 code (zero fuzzy fallback).
ISO_TO_WORDFREQ: Dict[str, str] = {
    "arb": "ar",
    "ara": "ar",
    "bul": "bg",
    "ben": "bn",
    "cat": "ca",
    "ces": "cs",
    "dan": "da",
    "deu": "de",
    "ell": "el",
    "eng": "en",
    "spa": "es",
    "pes": "fa",
    "fas": "fa",
    "fin": "fi",
    "tgl": "fil",
    "fil": "fil",
    "fra": "fr",
    "fre": "fr",
    "heb": "he",
    "hin": "hi",
    "hun": "hu",
    "ind": "id",
    "isl": "is",
    "ita": "it",
    "jpn": "ja",
    "kor": "ko",
    "lit": "lt",
    "lav": "lv",
    "lvs": "lv",
    "mkd": "mk",
    "zsm": "ms",
    "msa": "ms",
    "nob": "nb",
    "nor": "nb",
    "nld": "nl",
    "pol": "pl",
    "por": "pt",
    "ron": "ro",
    "rus": "ru",
    "hbs": "sh",
    "hrv": "sh",
    "bos": "sh",
    "slk": "sk",
    "slv": "sl",
    "swe": "sv",
    "tam": "ta",
    "tur": "tr",
    "ukr": "uk",
    "urd": "ur",
    "vie": "vi",
    "cmn": "zh",
    "zho": "zh",
}

EXTINCT_PREFIXES = (
    "UD_Ancient_",
    "UD_Classical_",
    "UD_Old_",
    "UD_Middle_",
    "UD_Gothic-",
    "UD_Latin-",
    "UD_Sanskrit-",
    "UD_Akkadian-",
    "UD_Hittite-",
    "UD_Coptic-",
    "UD_Ugaritic-",
    "UD_Sumerian-",
)

# Priority treebank suffixes for languages with many UD repositories, avoiding huge >50MB HDT
# downloads or tiny parallel PUD treebanks when large gold reference treebanks exist.
PREFERRED_UD_SUFFIXES = (
    "ISDT",
    "VIT",
    "TDT",
    "FTB",
    "SynTagRus",
    "Taiga",
    "GSD",
    "EWT",
    "Ancora",
    "PerDT",
    "Seraji",
    "IMST",
    "BOUN",
    "LVTB",
    "PDB",
    "LFG",
    "FicTree",
    "PDT",
    "CLTT",
    "SET",
    "ArmTDP",
    "BSUT",
    "EDT",
    "Bokmaal",
    "Talbanken",
    "Alpino",
    "LassySmall",
    "Bosque",
    "RRT",
    "Szeged",
    "GDT",
    "BDT",
    "MUDT",
    "HTB",
    "IAHLTwiki",
    "PADT",
    "NYUAD",
    "IU",
    "MDT",
    "SNK",
    "SSJ",
    "ALKSNIS",
    "BTB",
    "BDB",
    "NSE",
    "UFAL",
    "URDT",
    "HDTB",
    "GLC",
    "GNC",
    "Giella",
    "JR",
    "Lattice",
    "IKDP",
    "KKPP",
    "HSE",
    "YKTDT",
    "NMCTT",
    "Kurmanji",
    "YiTB",
    "TSA",
    "STAF",
    "GPS",
)

PROVENANCE_COLUMNS = [
    "source_id",
    "source_tier",
    "iso639_3",
    "corpus_id",
    "url",
    "version_or_commit",
    "license",
    "citation",
    "local_path",
    "sha256",
    "archive_sha256",
    "n_tokens",
    "n_types",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def http_get_bytes(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise
            time.sleep(1.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


def corpus_aliases_for_iso(iso: str) -> List[str]:
    iso = str(iso).strip().lower()
    if not iso:
        return []
    aliases = ISO_CORPUS_ALIASES.get(iso, [iso])
    seen: List[str] = []
    for code in [iso, *aliases]:
        if code and code not in seen:
            seen.append(code)
    return seen


def discover_target_isos(
    repo_root: Path = REPO_ROOT,
    unit_summary_path: Optional[Path] = None,
) -> List[str]:
    """Discover all GeLaTo-linked living ISO-639-3 codes plus any unit_summary.csv ISOs."""
    from morph_ldl.data.vendor.mgn_language_map import EXTINCT_BLACKLIST, MGN_LIVING_LANGS_64, canonical_iso

    links_path = repo_root / "analyses" / "common" / "data" / "analysis_links.csv"
    if not links_path.is_file():
        links_path = repo_root / "outputs" / "pcfp_v2_links_2026_10_20" / "typology" / "analysis_links.csv"
    glot_path = repo_root / "external" / "glottolog-cldf" / "cldf" / "languages.csv"

    links_df = pd.read_csv(links_path)
    glot_df = pd.read_csv(glot_path)
    gc_to_iso = dict(zip(glot_df["ID"], glot_df["ISO639P3code"]))
    gc_to_parent = dict(zip(glot_df["ID"], glot_df["Language_ID"]))

    target_isos: Set[str] = set()
    for gc in links_df["glottocode"].dropna().unique():
        iso = gc_to_iso.get(gc)
        if pd.isna(iso) or not iso:
            parent = gc_to_parent.get(gc)
            if pd.notna(parent):
                iso = gc_to_iso.get(parent)
        if pd.notna(iso) and iso:
            code = str(iso).strip().lower()
            if code not in EXTINCT_BLACKLIST:
                target_isos.add(code)

    for mgn_code in MGN_LIVING_LANGS_64:
        canon = canonical_iso(mgn_code)
        if canon not in EXTINCT_BLACKLIST:
            target_isos.add(canon)

    summary_file = unit_summary_path or (repo_root / "analyses" / "paradigm" / "data" / "unit_summary.csv")
    if summary_file.exists():
        udf = pd.read_csv(summary_file)
        if "iso639_3" in udf.columns:
            for code in udf["iso639_3"].dropna().unique():
                c = str(code).strip().lower()
                if c and c not in EXTINCT_BLACKLIST:
                    target_isos.add(c)

    return sorted(target_isos)


def ensure_metadata_catalogs(freq_dir: Path = FREQ_SOURCES_DIR) -> Tuple[dict, List[str], pd.DataFrame]:
    """Download and cache UD and LCC discovery catalogs in external/freq_sources/metadata/."""
    meta_dir = freq_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    cf_path = meta_dir / "ud_codes_and_flags.yaml"
    if not cf_path.exists():
        cf_path.write_bytes(http_get_bytes(UD_CODES_YAML_URL, timeout=30))
    cf_data = yaml.safe_load(cf_path.read_text(encoding="utf-8"))

    glance_path = meta_dir / "ud_at_glance.html"
    if not glance_path.exists():
        glance_path.write_bytes(http_get_bytes(UD_AT_GLANCE_URL, timeout=30))
    glance_html = glance_path.read_text(encoding="utf-8", errors="ignore")
    ud_repos = sorted(set(re.findall(r"UniversalDependencies/(UD_[A-Za-z0-9_-]+)", glance_html)))

    lcc_path = meta_dir / "lcc_links.parquet"
    if not lcc_path.exists():
        lcc_path.write_bytes(http_get_bytes(LCC_CATALOG_PARQUET_URL, timeout=45))
    lcc_df = pd.read_parquet(lcc_path)

    return cf_data, ud_repos, lcc_df


def build_ud_iso_repo_map(cf_data: dict, ud_repos: Sequence[str]) -> Dict[str, List[str]]:
    """Map ISO-639-3 codes to candidate UD repository names (excluding ancient/extinct)."""
    lang_to_repos: Dict[str, List[str]] = {}
    for repo in ud_repos:
        if repo.startswith(EXTINCT_PREFIXES):
            continue
        # Exclude HDT (>100MB split archive) since GSD provides 290K clean German tokens
        if repo in {"UD_German-HDT"}:
            continue
        lname = repo[3:].split("-")[0].replace("_", " ")
        lang_to_repos.setdefault(lname, []).append(repo)

    iso_map: Dict[str, List[str]] = {}
    for lname, info in cf_data.items():
        iso3 = info.get("iso3")
        if not iso3 or lname not in lang_to_repos:
            continue
        repos = lang_to_repos[lname]
        if iso3 == "nor":
            bok = [r for r in repos if "Nynorsk" not in r]
            nyn = [r for r in repos if "Nynorsk" in r]
            if bok:
                iso_map.setdefault("nob", []).extend(bok)
                iso_map.setdefault("nor", []).extend(bok)
            if nyn:
                iso_map.setdefault("nno", []).extend(nyn)
            continue
        iso_map.setdefault(str(iso3).lower(), []).extend(repos)

    def _repo_rank(repo: str) -> Tuple[int, str]:
        suffix = repo.split("-", 1)[1] if "-" in repo else repo
        if suffix == "PUD":
            return (500, repo)
        for idx, pref in enumerate(PREFERRED_UD_SUFFIXES):
            if suffix == pref:
                return (idx, repo)
        return (200, repo)

    return {k: sorted(set(v), key=_repo_rank) for k, v in iso_map.items()}


def resolve_ud_r215_commit(repo: str) -> Optional[str]:
    """Return the exact 40-char git commit SHA for tag r2.15, or None if not in UD v2.15."""
    git_url = f"https://github.com/UniversalDependencies/{repo}.git"
    try:
        out = subprocess.check_output(
            ["git", "ls-remote", "--tags", git_url, "refs/tags/r2.15", "refs/tags/r2.15^{}"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except Exception:  # noqa: BLE001
        return None
    sha_peeled: Optional[str] = None
    sha_direct: Optional[str] = None
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        sha, ref = parts
        if ref.endswith("^{}"):
            sha_peeled = sha
        elif ref == "refs/tags/r2.15":
            sha_direct = sha
    return sha_peeled or sha_direct


_INT_TOKEN_RE = re.compile(r"^\d+$")


def parse_conllu_stream(lines: Iterable[str]) -> Tuple[Counter, int]:
    """Parse CoNLL-U lines and return ((lemma, upos, form) Counter, total_word_tokens).

    Counts integer-ID syntactic word tokens with non-placeholder FORM != '_', excluding
    punctuation/symbols from total_word_tokens. Skips multiword token headers (1-2) and
    empty ellipsis nodes (1.1).
    """
    counts: Counter = Counter()
    total_word_tokens = 0
    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        tok_id, form, lemma, upos = cols[0], cols[1], cols[2], cols[3]
        if not _INT_TOKEN_RE.match(tok_id):
            continue
        if not form or form == "_":
            continue
        form_nfc = unicodedata.normalize("NFC", form)
        lemma_nfc = unicodedata.normalize("NFC", lemma) if lemma and lemma != "_" else form_nfc
        if upos in {"PUNCT", "SYM"}:
            continue
        total_word_tokens += 1
        counts[(lemma_nfc, upos, form_nfc)] += 1
    return counts, total_word_tokens


def extract_license_label(license_text: str) -> str:
    """Extract compact SPDX-like license identifier from a UD LICENSE.txt file."""
    txt = license_text.strip()
    for pat, label in [
        (r"CC[- ]BY[- ]SA[- ]4\.0|Attribution-ShareAlike 4\.0", "CC-BY-SA-4.0"),
        (r"CC[- ]BY[- ]NC[- ]SA[- ]4\.0|Attribution-NonCommercial-ShareAlike 4\.0", "CC-BY-NC-SA-4.0"),
        (r"CC[- ]BY[- ]NC[- ]SA[- ]3\.0|Attribution-NonCommercial-ShareAlike 3\.0", "CC-BY-NC-SA-3.0"),
        (r"CC[- ]BY[- ]SA[- ]3\.0|Attribution-ShareAlike 3\.0", "CC-BY-SA-3.0"),
        (r"CC[- ]BY[- ]4\.0|Attribution 4\.0", "CC-BY-4.0"),
        (r"CC[- ]BY[- ]3\.0|Attribution 3\.0", "CC-BY-3.0"),
        (r"CC0|Public Domain", "CC0-1.0"),
        (r"LGPL", "LGPL-LR"),
        (r"GPL", "GPL"),
        (r"Apache License", "Apache-2.0"),
        (r"MIT License", "MIT"),
    ]:
        if re.search(pat, txt, flags=re.IGNORECASE):
            return label
    first_line = next((ln.strip() for ln in txt.splitlines() if ln.strip()), "UD-Open-License")
    return first_line[:60]


def fetch_single_ud_repo(repo: str, iso_tag: str, freq_dir: Path = FREQ_SOURCES_DIR) -> Optional[dict]:
    """Download/verify one UD r2.15 archive, tabulate (lemma, upos, form) counts, return provenance."""
    ud_arch_dir = freq_dir / "ud" / "archives"
    ud_cnt_dir = freq_dir / "ud" / "counts"
    ud_lic_dir = freq_dir / "ud" / "licenses"
    ud_arch_dir.mkdir(parents=True, exist_ok=True)
    ud_cnt_dir.mkdir(parents=True, exist_ok=True)
    ud_lic_dir.mkdir(parents=True, exist_ok=True)

    commit_sha = resolve_ud_r215_commit(repo)
    if not commit_sha:
        return None

    url = f"https://github.com/UniversalDependencies/{repo}/archive/refs/tags/r2.15.tar.gz"
    arch_path = ud_arch_dir / f"{repo}-r2.15.tar.gz"
    cnt_path = ud_cnt_dir / f"{repo}-r2.15_counts.tsv.gz"
    lic_path = ud_lic_dir / f"{repo}-r2.15_LICENSE.txt"

    if not arch_path.exists():
        raw_bytes = http_get_bytes(url, timeout=90)
        arch_path.write_bytes(raw_bytes)
    arch_sha = sha256_file(arch_path)

    total_counts: Counter = Counter()
    total_tokens = 0
    license_label = "CC-BY-SA-4.0"

    with tarfile.open(arch_path, mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            bname = Path(member.name).name
            if bname.upper().startswith("LICENSE"):
                fobj = tf.extractfile(member)
                if fobj is not None:
                    lic_bytes = fobj.read()
                    lic_path.write_bytes(lic_bytes)
                    license_label = extract_license_label(lic_bytes.decode("utf-8", errors="replace"))
            elif bname.endswith(".conllu"):
                fobj = tf.extractfile(member)
                if fobj is not None:
                    text_lines = io.TextIOWrapper(fobj, encoding="utf-8", errors="replace")
                    c_part, n_part = parse_conllu_stream(text_lines)
                    total_counts.update(c_part)
                    total_tokens += n_part

    # Skip no-text stub UD treebanks (e.g., proprietary corpora with FORM='_')
    if total_tokens == 0 or not total_counts:
        arch_path.unlink(missing_ok=True)
        return None

    with gzip.open(cnt_path, "wt", encoding="utf-8", compresslevel=6) as gz:
        gz.write("lemma\tupos\tform\tcount\n")
        for (lemma, upos, form), cnt in sorted(total_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            gz.write(f"{lemma}\t{upos}\t{form}\t{cnt}\n")

    rel_local = arch_path.relative_to(REPO_ROOT).as_posix()
    return {
        "source_id": f"ud_v2.15:{repo}",
        "source_tier": "tier_a_ud_v2.15",
        "iso639_3": iso_tag,
        "corpus_id": f"{repo}@r2.15",
        "url": url,
        "version_or_commit": commit_sha,
        "license": license_label,
        "citation": UD_CITATION,
        "local_path": rel_local,
        "sha256": arch_sha,
        "archive_sha256": arch_sha,
        "n_tokens": int(total_tokens),
        "n_types": int(len(total_counts)),
    }


def select_lcc_corpora_for_iso(lcc_df: pd.DataFrame, corpus_iso: str, max_corpora: int = 2) -> List[dict]:
    """Select up to 2 compact official Leipzig Corpora tarballs (news/web + wikipedia/community)."""
    df = lcc_df.copy()
    df["prefix"] = df["data_id"].astype(str).str.split("_").str[0]
    df["base_iso"] = df["prefix"].str.split("-").str[0]

    exact = df[df["prefix"] == corpus_iso]
    sub = exact if len(exact) > 0 else df[df["base_iso"] == corpus_iso]
    if sub.empty:
        return []

    size_rank = {"300K": 0, "100K": 1, "1M": 2, "30K": 3, "10K": 4}

    def _parse_row(row: pd.Series) -> dict:
        data_id = str(row["data_id"])
        parts = data_id.split("_")
        genre = parts[1] if len(parts) > 1 else "misc"
        sz = str(row["size"])
        # community_2017 has year in 'size' column
        s_rank = size_rank.get(sz, 2 if genre == "community" else 5)
        # Skip 3M+ giant archives
        if sz in {"3M", "10M"}:
            s_rank = 99
        year_m = re.findall(r"20\d{2}", data_id)
        yr = int(year_m[-1]) if year_m else 2010
        is_news_web = 0 if any(g in genre for g in ("news", "web", "mixed")) else 1
        url = f"https://downloads.wortschatz-leipzig.de/corpora/{data_id}.tar.gz"
        return {
            "data_id": data_id,
            "url": url,
            "genre": genre,
            "size": sz,
            "s_rank": s_rank,
            "year": yr,
            "is_news_web": is_news_web,
        }

    parsed = [p for p in (_parse_row(r) for _, r in sub.iterrows()) if p["s_rank"] < 99]
    if not parsed:
        return []

    news_bucket = sorted(
        [p for p in parsed if p["is_news_web"] == 0],
        key=lambda p: (p["s_rank"], -p["year"], p["data_id"]),
    )
    wiki_bucket = sorted(
        [p for p in parsed if p["is_news_web"] == 1],
        key=lambda p: (p["s_rank"], -p["year"], p["data_id"]),
    )

    chosen: List[dict] = []
    if news_bucket:
        chosen.append(news_bucket[0])
    if wiki_bucket and len(chosen) < max_corpora:
        chosen.append(wiki_bucket[0])
    if len(chosen) < max_corpora:
        for p in sorted(parsed, key=lambda x: (x["s_rank"], -x["year"], x["data_id"])):
            if all(c["data_id"] != p["data_id"] for c in chosen):
                chosen.append(p)
                if len(chosen) >= max_corpora:
                    break
    return chosen


_PUNCT_ONLY_RE = re.compile(r"^[\W\d_]+$", flags=re.UNICODE)


def parse_lcc_words_bytes(raw_bytes: bytes) -> Tuple[int, int]:
    """Count total lexical word tokens and word types in a Leipzig *-words.txt file."""
    n_tokens = 0
    n_types = 0
    for line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        word = parts[1].strip()
        if not word or _PUNCT_ONLY_RE.match(word):
            continue
        try:
            # 3-col LCC files have (id, word, freq); 4-col community_2017 files have (id, word, norm, freq)
            cnt = int(parts[-1])
        except ValueError:
            continue
        if cnt > 0:
            n_tokens += cnt
            n_types += 1
    return n_tokens, n_types


def fetch_single_lcc_corpus(item: dict, iso_tag: str, freq_dir: Path = FREQ_SOURCES_DIR) -> Optional[dict]:
    """Download official LCC tarball, verify SHA-256, extract ONLY *-words.txt + *-meta.txt."""
    data_id = item["data_id"]
    url = item["url"]
    out_dir = freq_dir / "lcc" / data_id
    words_path = out_dir / f"{data_id}-words.txt"
    sha_record_path = out_dir / f"{data_id}.archive_sha256"

    if words_path.exists() and sha_record_path.exists():
        archive_sha = sha_record_path.read_text(encoding="utf-8").strip()
        words_bytes = words_path.read_bytes()
        words_sha = sha256_bytes(words_bytes)
        n_tokens, n_types = parse_lcc_words_bytes(words_bytes)
    else:
        try:
            tar_bytes = http_get_bytes(url, timeout=180)
        except Exception:  # noqa: BLE001
            return None
        archive_sha = sha256_bytes(tar_bytes)
        out_dir.mkdir(parents=True, exist_ok=True)

        extracted_words: Optional[bytes] = None
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                bname = Path(member.name).name
                if bname.endswith("-words.txt"):
                    fobj = tf.extractfile(member)
                    if fobj is not None:
                        extracted_words = fobj.read()
                        words_path.write_bytes(extracted_words)
                elif bname.endswith("-meta.txt") or bname.endswith("-import.sql"):
                    fobj = tf.extractfile(member)
                    if fobj is not None:
                        (out_dir / bname).write_bytes(fobj.read())

        # Free multi-MB tarball buffer immediately; sentence tables are never written to disk
        del tar_bytes
        if extracted_words is None:
            return None
        sha_record_path.write_text(archive_sha + "\n", encoding="utf-8")
        words_sha = sha256_bytes(extracted_words)
        n_tokens, n_types = parse_lcc_words_bytes(extracted_words)

    if n_tokens == 0:
        return None

    year_m = re.findall(r"20\d{2}", data_id)
    version_tag = year_m[-1] if year_m else "official"
    rel_local = words_path.relative_to(REPO_ROOT).as_posix()
    return {
        "source_id": f"lcc:{data_id}",
        "source_tier": "tier_b_leipzig_corpora",
        "iso639_3": iso_tag,
        "corpus_id": data_id,
        "url": url,
        "version_or_commit": f"{data_id} ({version_tag})",
        "license": "CC-BY-4.0",
        "citation": LCC_CITATION,
        "local_path": rel_local,
        "sha256": words_sha,
        "archive_sha256": archive_sha,
        "n_tokens": int(n_tokens),
        "n_types": int(n_types),
    }


def collect_wordfreq_provenance(target_isos: Iterable[str]) -> List[dict]:
    """Record provenance and exact SHA-256 hashes of installed wordfreq v3.1.1 msgpack files."""
    try:
        import wordfreq
    except ImportError:
        return []

    wf_version = getattr(wordfreq, "__version__", "3.1.1")
    wf_data_dir = Path(wordfreq.__file__).resolve().parent / "data"
    avail = set(wordfreq.available_languages("best"))
    rows: List[dict] = []
    seen_pairs: Set[Tuple[str, str]] = set()

    for iso in sorted(set(target_isos)):
        wf_lang = ISO_TO_WORDFREQ.get(iso)
        if not wf_lang or wf_lang not in avail:
            continue
        if (iso, wf_lang) in seen_pairs:
            continue
        seen_pairs.add((iso, wf_lang))

        data_file = wf_data_dir / f"large_{wf_lang}.msgpack.gz"
        variant = "large"
        if not data_file.exists():
            data_file = wf_data_dir / f"small_{wf_lang}.msgpack.gz"
            variant = "small"
        if not data_file.exists():
            continue

        file_sha = sha256_file(data_file)
        fdict = wordfreq.get_frequency_dict(wf_lang, wordlist="best")
        rel_path = data_file.relative_to(REPO_ROOT).as_posix()
        rows.append(
            {
                "source_id": f"wordfreq:{wf_lang}_{variant}",
                "source_tier": "tier_c_wordfreq",
                "iso639_3": iso,
                "corpus_id": f"wordfreq_{wf_version}_{wf_lang}_{variant}",
                "url": WORDFREQ_DOI_URL,
                "version_or_commit": f"{wf_version} (doi:10.5281/zenodo.7199437)",
                "license": "CC-BY-SA-4.0 / Apache-2.0",
                "citation": WORDFREQ_CITATION,
                "local_path": rel_path,
                "sha256": file_sha,
                "archive_sha256": file_sha,
                "n_tokens": 1_000_000,  # Normalized PMW probability distribution
                "n_types": int(len(fdict)),
            }
        )
    return rows


def run_fetch_pipeline(
    isos: Optional[Sequence[str]] = None,
    max_workers: int = 2,
    skip_ud: bool = False,
    skip_lcc: bool = False,
    skip_wordfreq: bool = False,
    freq_dir: Path = FREQ_SOURCES_DIR,
    provenance_csv: Path = PROVENANCE_CSV,
) -> pd.DataFrame:
    """Fetch all available Tier A (UD v2.15), Tier B (LCC), and Tier C (wordfreq) sources."""
    workers = max(1, min(int(max_workers), 2))
    target_isos = sorted(set(isos)) if isos else discover_target_isos()
    print(f"[fetch_frequencies] Target ISO count: {len(target_isos)} (workers={workers})")

    cf_data, ud_repos, lcc_df = ensure_metadata_catalogs(freq_dir=freq_dir)
    ud_iso_map = build_ud_iso_repo_map(cf_data, ud_repos)

    provenance_rows: List[dict] = []

    # --- Tier A: Universal Dependencies v2.15 ---
    if not skip_ud:
        ud_jobs: Dict[str, str] = {}
        for iso in target_isos:
            matched_repos: List[str] = []
            for alias in corpus_aliases_for_iso(iso):
                for repo in ud_iso_map.get(alias, []):
                    if repo not in matched_repos:
                        matched_repos.append(repo)
            # Keep up to top 2 ranked treebanks per language (3 for pluricentric hbs)
            cap = 3 if iso == "hbs" else 2
            for repo in matched_repos[:cap]:
                ud_jobs.setdefault(repo, iso)

        print(f"[fetch_frequencies] Checking/fetching {len(ud_jobs)} UD v2.15 candidate repositories...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(fetch_single_ud_repo, repo, iso_tag, freq_dir): repo
                for repo, iso_tag in sorted(ud_jobs.items())
            }
            done_cnt = 0
            for fut in as_completed(futs):
                done_cnt += 1
                repo = futs[fut]
                try:
                    res = fut.result()
                    if res is not None:
                        provenance_rows.append(res)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [UD skip] {repo}: {exc}")
                if done_cnt % 25 == 0 or done_cnt == len(futs):
                    print(f"  [UD progress] {done_cnt}/{len(futs)} checked ({len(provenance_rows)} active)")

    # --- Tier B: Leipzig Corpora Collection ---
    if not skip_lcc:
        lcc_jobs: Dict[str, Tuple[dict, str]] = {}
        for iso in target_isos:
            selected_for_iso: List[dict] = []
            max_cap = 3 if iso == "hbs" else 2
            # Exclude Cyrillic-script srp_* from LCC selection for Latin Gajica hbs
            lcc_aliases = ["hbs", "hrv", "bos"] if iso == "hbs" else corpus_aliases_for_iso(iso)
            for alias in lcc_aliases:
                per_alias_cap = 1 if iso == "hbs" else 2
                picks = select_lcc_corpora_for_iso(lcc_df, alias, max_corpora=per_alias_cap)
                for p in picks:
                    if all(x["data_id"] != p["data_id"] for x in selected_for_iso):
                        selected_for_iso.append(p)
                if len(selected_for_iso) >= max_cap and iso != "hbs":
                    break
            for item in selected_for_iso[:max_cap]:
                lcc_jobs.setdefault(item["data_id"], (item, iso))

        print(f"[fetch_frequencies] Downloading/verifying {len(lcc_jobs)} Leipzig Corpora wordlists...")
        lcc_added = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(fetch_single_lcc_corpus, item, iso_tag, freq_dir): data_id
                for data_id, (item, iso_tag) in sorted(lcc_jobs.items())
            }
            done_cnt = 0
            for fut in as_completed(futs):
                done_cnt += 1
                data_id = futs[fut]
                try:
                    res = fut.result()
                    if res is not None:
                        provenance_rows.append(res)
                        lcc_added += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  [LCC skip] {data_id}: {exc}")
                if done_cnt % 25 == 0 or done_cnt == len(futs):
                    print(f"  [LCC progress] {done_cnt}/{len(futs)} processed ({lcc_added} active)")

    # --- Tier C: wordfreq v3.1.1 ---
    if not skip_wordfreq:
        wf_rows = collect_wordfreq_provenance(target_isos)
        provenance_rows.extend(wf_rows)
        print(f"[fetch_frequencies] Recorded {len(wf_rows)} wordfreq v3.1.1 language packs.")

    # Merge with any prior rows in provenance_csv so incremental --isos runs never drop existing rows
    new_df = pd.DataFrame(provenance_rows, columns=PROVENANCE_COLUMNS)
    if provenance_csv.exists():
        old_df = pd.read_csv(provenance_csv)
        old_df = old_df[old_df["local_path"].map(lambda p: (REPO_ROOT / str(p)).exists())]
        # Clean up legacy cross-script alias tags if present
        old_df = old_df[
            ~(
                (old_df["source_id"].isin({"lcc:kur_community_2017", "lcc:kur_newscrawl_2011_30K"}))
                & (old_df["iso639_3"] == "kmr")
            )
            & ~(
                (old_df["source_id"] == "lcc:srp_news_2019_30K")
                & (old_df["iso639_3"] == "hbs")
            )
        ]
        combined = pd.concat([old_df, new_df], ignore_index=True)
    else:
        combined = new_df

    # Enforce strict primary-key uniqueness on `source_id`, merging shared alias ISOs with ';'
    if not combined.empty:
        iso_merged = (
            combined.groupby("source_id", sort=False)["iso639_3"]
            .agg(
                lambda s: ";".join(
                    sorted({tok.strip().lower() for val in s for tok in str(val).split(";") if tok.strip()})
                )
            )
            .to_dict()
        )
        combined = combined.drop_duplicates(subset=["source_id"], keep="last").copy()
        combined["iso639_3"] = combined["source_id"].map(iso_merged)
        assert combined["source_id"].is_unique, "source_id must be unique in freq_provenance.csv"

    combined = combined.sort_values(["source_tier", "iso639_3", "source_id"]).reset_index(drop=True)
    provenance_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(provenance_csv, index=False)
    print(f"[fetch_frequencies] Wrote {len(combined)} verified rows -> {provenance_csv}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch open academic frequency corpora for Track 2.")
    parser.add_argument("--isos", type=str, default="", help="Optional comma-separated ISO-639-3 list.")
    parser.add_argument("--workers", type=int, default=2, help="Max download threads (capped at 2).")
    parser.add_argument("--skip-ud", action="store_true", help="Skip Tier A UD v2.15.")
    parser.add_argument("--skip-lcc", action="store_true", help="Skip Tier B Leipzig Corpora.")
    parser.add_argument("--skip-wordfreq", action="store_true", help="Skip Tier C wordfreq.")
    args = parser.parse_args()

    iso_list = [x.strip().lower() for x in args.isos.split(",") if x.strip()] if args.isos else None
    run_fetch_pipeline(
        isos=iso_list,
        max_workers=min(args.workers, 2),
        skip_ud=args.skip_ud,
        skip_lcc=args.skip_lcc,
        skip_wordfreq=args.skip_wordfreq,
    )


if __name__ == "__main__":
    main()
