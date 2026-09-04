#!/usr/bin/env python3
"""
Demographic Data Ingestion & Cross-Linguistic Fusion Pipeline
=============================================================
Builds `global_demographic_registry.csv`, the master language-level demographic
and macro-ecological table for Study 3.

Data sources (all machine-readable, all shipped or downloaded; nothing hand-entered
except the two documented entries in MANUAL_POPULATION_OVERRIDES):

  Tier 1  Bromham et al. (2022/2025), `data.Rdata` -> bromham_extracted.csv (6,511 langs)
            population_l1            <- lang_L1.POP_lang
            div_bordering_languages  <- div_bordering.language.richness_lang
            area_km2                 <- lang_polygon.area_lang
            altitude_range           <- conn_altitude.range_nb
            roughness                <- conn_roughness_nb
          The only source of the four macro-ecological covariates. All four are
          derived from a language's geographic polygon and cannot be reconstructed
          for languages Bromham does not cover.

  Tier 1b Ethnologue multi-ISO table `all_multi_ISO_languages.csv` (1,061 langs),
          shipped in the same Bromham bundle. `LMP_POP1` is the SAME L1 variable
          as Bromham's lang_L1.POP_lang (verified identical for deu/als/azj/pbt/pes),
          and it covers the major world languages Bromham omits (eng, spa, rus,
          fra, por, hin, urd, hrv, srp, bos). Using it keeps population on one
          consistent definition instead of mixing Ethnologue editions.

  Tier 2  Koplenig (2019) / Ethnologue `ethnologue_population_data.csv` (2,143 langs)
            l2_proportion <- L2prop, vehicularity <- vehicularity
            population_l1 <- Population   (fallback only)
            area_km2      <- Rangesize    (fallback only)
          Koplenig's `Population` is also an L1 count: it agrees with Bromham
          exactly for the median of the 2,071 overlapping languages.

  Tier 3  Glottolog v5.3 via lingtypology + the Heti lookup table: glottocodes,
          families, macro-areas, coordinates.

Key invariants
--------------
- NO population is ever invented. Languages with no population in any source keep
  `population_l1 = NaN` and `population_source = NaN`. They stay in the registry
  (their taxonomy and coordinates are still useful) but downstream modelling
  filters them out.
- Macro-ecological covariates are imputed by family -> macro-area -> global median
  ONLY where the source has no polygon, and every such row is flagged in
  `eco_imputed` so the flag can be carried into the model and tested.
- Pluricentric standards (Serbo-Croatian, Standard Albanian, Modern Standard
  Arabic) sum L1 over their member varieties; see src/mgn_language_map.py.
- Extinct/ancient languages are excluded via EXTINCT_BLACKLIST.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from mgn_language_map import (
    BROMHAM_ISO_FIXUPS,
    EXTINCT_BLACKLIST,
    MANUAL_POPULATION_OVERRIDES,
    MGN_EXTINCT_LANGS_9,
    MGN_LIVING_LANGS_64,
    canonical_iso,
    dump_json,
    population_members,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("DemographicRegistry")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "data_sources"
OUTPUT_CSV_PATH = PROJECT_ROOT / "global_demographic_registry.csv"

ECO_COLS = ["div_bordering_languages", "area_km2", "altitude_range", "roughness"]

REGISTRY_COLS = [
    "iso_639_3",
    "glottocode",
    "language_name",
    "family",
    "macro_area",
    "population_l1",
    "population_source",
    "population_scope",
    "l2_proportion",
    "vehicularity",
    "div_bordering_languages",
    "area_km2",
    "altitude_range",
    "roughness",
    "eco_imputed",
    "latitude",
    "longitude",
]


class DemographicDataLoader:
    """Fetches, caches and loads the raw demographic and taxonomic sources."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_bromham_data(self) -> pd.DataFrame:
        """Bromham et al. L1 population + the four macro-ecological covariates."""
        extracted_csv = self.cache_dir / "bromham_extracted.csv"
        rdata_path = self.cache_dir / "data.Rdata"

        if extracted_csv.exists():
            logger.info("Loading Bromham data from cache: %s", extracted_csv)
            return pd.read_csv(extracted_csv)

        if not rdata_path.exists():
            logger.info("Downloading Bromham data.Rdata...")
            import requests
            url = "https://raw.githubusercontent.com/huaxia1985/LanguageEndangerment/main/data.Rdata"
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            rdata_path.write_bytes(r.content)

        logger.info("Extracting Bromham variables via Rscript...")
        r_script = f"""
        load('{rdata_path}')
        cols_to_keep <- c(
          'id_ISO_lang', 'id_name_lang', 'lang_L1.POP_lang',
          'div_bordering.language.richness_lang', 'lang_polygon.area_lang',
          'conn_altitude.range_nb', 'conn_roughness_nb'
        )
        df_sub <- as.data.frame(data[, cols_to_keep])
        colnames(df_sub) <- c('iso_639_3', 'language_name', 'population_l1',
                              'div_bordering_languages', 'area_km2',
                              'altitude_range', 'roughness')
        write.csv(df_sub, '{extracted_csv}', row.names=FALSE)
        """
        subprocess.run(["Rscript", "-e", r_script], check=True)
        return pd.read_csv(extracted_csv)

    def get_multi_iso_data(self) -> Optional[pd.DataFrame]:
        """Ethnologue multi-ISO table: L1 population for major world languages.

        `LMP_POP1` is the same L1 variable Bromham uses; this table covers the
        major languages (eng/spa/rus/fra/por/hin/urd/hrv/srp/bos) that Bromham omits.
        """
        path = self.cache_dir / "all_multi_ISO_languages.csv"
        if not path.exists():
            try:
                import requests
                url = ("https://raw.githubusercontent.com/rdinnager/language_endangerment/"
                       "main/data/all_multi_ISO_languages.csv")
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    path.write_bytes(r.content)
            except Exception as e:  # pragma: no cover - network dependent
                logger.warning("Could not download multi-ISO table: %s", e)
        if not path.exists():
            logger.warning("Ethnologue multi-ISO table unavailable; major world "
                           "languages may lack population.")
            return None
        logger.info("Loading Ethnologue multi-ISO table from %s", path)
        return pd.read_csv(path).drop_duplicates("LANG_ISO")

    def get_ethnologue_data(self) -> pd.DataFrame:
        """Koplenig (2019) / Ethnologue: L2 proportion, vehicularity, fallback pop."""
        eth_path = PROJECT_ROOT / "data_sources" / "ethnologue_population_data.csv"
        if not eth_path.exists():
            raise FileNotFoundError(f"Ethnologue population data not found at {eth_path}")
        logger.info("Loading Koplenig/Ethnologue data from %s", eth_path)
        return pd.read_csv(eth_path)

    def get_glottolog_reference(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Glottolog taxonomy: glottocodes, families, macro-areas, coordinates."""
        heti_path = self.cache_dir / "Glottolog_lookup_table_Heti_edition.tsv"
        glt_csv = self.cache_dir / "glottolog_extracted.csv"

        if not heti_path.exists():
            import requests
            url = ("https://raw.githubusercontent.com/rdinnager/language_endangerment/"
                   "main/data/Glottolog_lookup_table_Heti_edition.tsv")
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            heti_path.write_bytes(r.content)

        if not glt_csv.exists():
            r_script = f"""
            library(lingtypology)
            write.csv(lingtypology::glottolog, '{glt_csv}', row.names=FALSE)
            """
            subprocess.run(["Rscript", "-e", r_script], check=True)

        logger.info("Loading Glottolog references")
        return pd.read_csv(heti_path, sep="\t"), pd.read_csv(glt_csv)


class CodeResolver:
    """Coordinate normalisation helpers."""

    @staticmethod
    def normalize_longitude(lon: Optional[float]) -> Optional[float]:
        if lon is None or pd.isna(lon):
            return None
        lon = float(lon)
        if lon > 180.0:
            lon -= 360.0
        elif lon < -180.0:
            lon += 360.0
        return round(lon, 6)

    @staticmethod
    def normalize_latitude(lat: Optional[float]) -> Optional[float]:
        if lat is None or pd.isna(lat):
            return None
        return round(float(lat), 6)


class DemographicRegistryBuilder:
    """Builds, harmonises and exports global_demographic_registry.csv."""

    def __init__(self, loader: Optional[DemographicDataLoader] = None):
        self.loader = loader or DemographicDataLoader()
        self.code_resolver = CodeResolver()

    # -- taxonomy ----------------------------------------------------------

    def build_glottolog_dictionaries(
        self, heti_df: pd.DataFrame, glt_df: pd.DataFrame
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Lookup dictionaries indexed by glottocode and by ISO 639-3."""
        glotto_by_code: Dict[str, Dict[str, Any]] = {}
        glotto_by_iso: Dict[str, Dict[str, Any]] = {}

        glt_langs = glt_df[glt_df["level"] == "language"]
        for _, row in glt_langs.iterrows():
            gc = str(row["glottocode"]).strip()
            iso_raw = str(row["iso"]).strip() if pd.notna(row["iso"]) else ""
            iso_raw = iso_raw if len(iso_raw) == 3 else None
            aff = str(row["affiliation"]).strip() if pd.notna(row["affiliation"]) else ""
            rec = {
                "glottocode": gc,
                "iso_639_3": iso_raw,
                "language_name": str(row["language"]).strip() if pd.notna(row["language"]) else "",
                "family": aff.split(",")[0].strip() if aff else "Isolate",
                "macro_area": str(row["area"]).strip() if pd.notna(row["area"]) else None,
                "latitude": self.code_resolver.normalize_latitude(row["latitude"]),
                "longitude": self.code_resolver.normalize_longitude(row["longitude"]),
            }
            glotto_by_code[gc] = rec
            if iso_raw:
                glotto_by_iso[iso_raw] = rec

        heti_langs = heti_df[heti_df["level"] == "language"]
        for _, row in heti_langs.iterrows():
            gc = str(row["glottocode"]).strip()
            iso_raw = str(row["iso639_3"]).strip() if pd.notna(row["iso639_3"]) else ""
            iso = iso_raw if (len(iso_raw) == 3 and not iso_raw.startswith("NOCODE_")) else None
            fam = str(row["Family_name"]).strip() if pd.notna(row["Family_name"]) else "Isolate"
            area = str(row["Macroarea"]).strip() if pd.notna(row["Macroarea"]) else None
            lat = self.code_resolver.normalize_latitude(row["Latitude"])
            lon = self.code_resolver.normalize_longitude(row["Longitude"])

            if gc in glotto_by_code:
                rec = glotto_by_code[gc]
                if fam and fam != "Isolate":
                    rec["family"] = fam
                if area:
                    rec["macro_area"] = area
                if lat is not None:
                    rec["latitude"] = lat
                if lon is not None:
                    rec["longitude"] = lon
                if iso and not rec["iso_639_3"]:
                    rec["iso_639_3"] = iso
            else:
                glotto_by_code[gc] = {
                    "glottocode": gc,
                    "iso_639_3": iso,
                    "language_name": str(row["Name"]).strip() if pd.notna(row["Name"]) else "",
                    "family": fam,
                    "macro_area": area,
                    "latitude": lat,
                    "longitude": lon,
                }
            if iso and iso not in glotto_by_iso:
                glotto_by_iso[iso] = glotto_by_code[gc]

        return glotto_by_code, glotto_by_iso

    # -- registry ----------------------------------------------------------

    def _blank_record(self, iso: str, g_rec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        g_rec = g_rec or {}
        return {
            "iso_639_3": iso,
            "glottocode": g_rec.get("glottocode"),
            "language_name": g_rec.get("language_name", iso),
            "family": g_rec.get("family", "Isolate"),
            "macro_area": g_rec.get("macro_area"),
            "population_l1": np.nan,
            "population_source": None,
            "population_scope": None,
            "l2_proportion": np.nan,
            "vehicularity": 0,
            "div_bordering_languages": np.nan,
            "area_km2": np.nan,
            "altitude_range": np.nan,
            "roughness": np.nan,
            "eco_imputed": True,
            "latitude": g_rec.get("latitude"),
            "longitude": g_rec.get("longitude"),
        }

    def build_registry(self) -> pd.DataFrame:
        logger.info("Starting demographic registry fusion pipeline...")

        brom_df = self.loader.get_bromham_data()
        multi_df = self.loader.get_multi_iso_data()
        eth_df = self.loader.get_ethnologue_data()
        heti_df, glt_df = self.loader.get_glottolog_reference()
        _, glotto_by_iso = self.build_glottolog_dictionaries(heti_df, glt_df)

        # Raw per-ISO L1 populations, keyed independently of Glottolog level. The
        # registry only holds level=="language" units, but pluricentric members
        # such as hrv/srp/bos are Glottolog *dialects* and would otherwise be
        # invisible to the aggregation step below.
        raw_pop = self._collect_raw_populations(brom_df, multi_df, eth_df)

        # 1. Base registry: every living Glottolog language carrying an ISO code
        registry: Dict[str, Dict[str, Any]] = {}
        for iso, g_rec in glotto_by_iso.items():
            if iso in EXTINCT_BLACKLIST or len(iso) != 3:
                continue
            registry[iso] = self._blank_record(iso, g_rec)
        logger.info("Base registry: %d living Glottolog languages.", len(registry))

        # 2. Tier 1 - Bromham: L1 population AND the four macro-ecological covariates
        logger.info("Ingesting Bromham et al. global dataset...")
        n_eco = 0
        for _, row in brom_df.iterrows():
            raw_iso = str(row["iso_639_3"]).strip() if pd.notna(row["iso_639_3"]) else None
            if not raw_iso or raw_iso in EXTINCT_BLACKLIST:
                continue
            iso = BROMHAM_ISO_FIXUPS.get(raw_iso, raw_iso)

            if iso not in registry:
                registry[iso] = self._blank_record(
                    iso, glotto_by_iso.get(iso) or glotto_by_iso.get(raw_iso)
                )
                if pd.notna(row["language_name"]) and not registry[iso]["language_name"]:
                    registry[iso]["language_name"] = str(row["language_name"]).strip()
            rec = registry[iso]

            pop = float(row["population_l1"]) if pd.notna(row["population_l1"]) else None
            if pop is not None and pop > 0:
                rec["population_l1"] = pop
                rec["population_source"] = "Bromham"
                rec["population_scope"] = iso

            # Macro-ecological covariates: Bromham is the ONLY source.
            got_eco = False
            for src_col in ECO_COLS:
                val = float(row[src_col]) if pd.notna(row[src_col]) else None
                if val is not None and (src_col != "area_km2" or val > 0):
                    rec[src_col] = val
                    got_eco = True
            if got_eco:
                rec["eco_imputed"] = False
                n_eco += 1
        logger.info("Bromham: %d languages with measured macro-ecological covariates.", n_eco)

        # 3. Tier 1b - Ethnologue multi-ISO table (same L1 variable, wider on majors)
        if multi_df is not None:
            logger.info("Ingesting Ethnologue multi-ISO L1 populations...")
            n_multi = 0
            for _, row in multi_df.iterrows():
                iso = str(row["LANG_ISO"]).strip() if pd.notna(row["LANG_ISO"]) else None
                if not iso or len(iso) != 3 or iso in EXTINCT_BLACKLIST:
                    continue
                pop = row.get("LMP_POP1")
                pop = float(pop) if pd.notna(pop) else None
                if pop is None or pop <= 0:
                    continue
                if iso not in registry:
                    registry[iso] = self._blank_record(iso, glotto_by_iso.get(iso))
                rec = registry[iso]
                if pd.isna(rec["population_l1"]):
                    rec["population_l1"] = pop
                    rec["population_source"] = "Ethnologue_multiISO"
                    rec["population_scope"] = iso
                    n_multi += 1
                if not rec["language_name"] and pd.notna(row.get("NAME_PROP")):
                    rec["language_name"] = str(row["NAME_PROP"]).strip()
            logger.info("Ethnologue multi-ISO: filled %d additional populations.", n_multi)

        # 4. Tier 2 - Koplenig: L2 proportion + vehicularity (primary), pop/area fallback
        logger.info("Enriching with Koplenig (2019) L2 proportions and vehicularity...")
        n_kop_pop = 0
        for _, row in eth_df.iterrows():
            raw_iso = str(row["ISO"]).strip() if pd.notna(row["ISO"]) else None
            if not raw_iso or raw_iso in EXTINCT_BLACKLIST:
                continue
            iso = BROMHAM_ISO_FIXUPS.get(raw_iso, raw_iso)
            if iso not in registry:
                registry[iso] = self._blank_record(iso, glotto_by_iso.get(iso))
                if pd.notna(row.get("Language")):
                    registry[iso]["language_name"] = str(row["Language"]).strip()
            rec = registry[iso]

            if pd.notna(row["L2prop"]):
                rec["l2_proportion"] = float(row["L2prop"])
            if pd.notna(row["vehicularity"]):
                rec["vehicularity"] = int(row["vehicularity"])

            pop_eth = float(row["Population"]) if pd.notna(row["Population"]) else None
            if pd.isna(rec["population_l1"]) and pop_eth is not None and pop_eth > 0:
                rec["population_l1"] = pop_eth
                rec["population_source"] = "Koplenig"
                rec["population_scope"] = iso
                n_kop_pop += 1

            area_eth = float(row["Rangesize"]) if pd.notna(row["Rangesize"]) else None
            if pd.isna(rec["area_km2"]) and area_eth is not None and area_eth > 0:
                rec["area_km2"] = area_eth
        logger.info("Koplenig: filled %d additional populations.", n_kop_pop)

        df = pd.DataFrame(list(registry.values()))
        df = df[~df["iso_639_3"].isin(EXTINCT_BLACKLIST)].reset_index(drop=True)

        # 5. Pluricentric aggregation: sum L1 over member varieties
        df = self._apply_pluricentric_aggregation(df, raw_pop)

        # 6. Manual overrides (only for languages absent from every source)
        df = self._apply_manual_overrides(df)

        # 7. Backfill taxonomy/coordinates from Glottolog
        for idx, row in df.iterrows():
            iso = row["iso_639_3"]
            g = glotto_by_iso.get(iso)
            if not g:
                continue
            if pd.isna(row["glottocode"]) or not str(row["glottocode"]).strip():
                df.at[idx, "glottocode"] = g["glottocode"]
            if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
                df.at[idx, "latitude"] = g["latitude"]
                df.at[idx, "longitude"] = g["longitude"]
            if (pd.isna(row["family"]) or row["family"] == "Isolate") and g["family"] != "Isolate":
                df.at[idx, "family"] = g["family"]
            if pd.isna(row["macro_area"]) and g["macro_area"]:
                df.at[idx, "macro_area"] = g["macro_area"]

        # 8. Macro-ecological imputation: family -> macro-area -> global median.
        #    Population is NEVER imputed; these four are, and every imputed row is flagged.
        df = self._impute_eco_covariates(df)

        # 9. Housekeeping
        df["vehicularity"] = df["vehicularity"].fillna(0).astype(int)
        df["macro_area"] = df["macro_area"].fillna("Unknown")
        df = df[df["glottocode"].notna()].copy()
        df = df[df["glottocode"].astype(str).str.len() == 8].copy()
        df = df.drop_duplicates(subset=["iso_639_3"]).reset_index(drop=True)
        df = df[REGISTRY_COLS]

        n_pop = int(df["population_l1"].notna().sum())
        logger.info(
            "Registry complete: %d languages, %d with a real population (%d without), "
            "%d with measured macro-ecological covariates.",
            len(df), n_pop, len(df) - n_pop, int((~df["eco_imputed"]).sum()),
        )
        logger.info("Population provenance: %s", df["population_source"].value_counts().to_dict())
        return df

    @staticmethod
    def _collect_raw_populations(
        brom_df: pd.DataFrame,
        multi_df: Optional[pd.DataFrame],
        eth_df: pd.DataFrame,
    ) -> Dict[str, Tuple[float, str]]:
        """Per-ISO L1 population from every source, in hierarchy order.

        Returns {iso: (population, source_label)}. Keyed by raw ISO so that
        Glottolog dialects (hrv, srp, bos, lvs, ...) are still reachable.
        """
        raw: Dict[str, Tuple[float, str]] = {}

        def put(iso: Any, pop: Any, label: str) -> None:
            if pd.isna(iso) or pd.isna(pop):
                return
            iso = str(iso).strip()
            pop = float(pop)
            if len(iso) != 3 or pop <= 0 or iso in raw:
                return
            raw[iso] = (pop, label)

        for _, r in brom_df.iterrows():
            put(r["iso_639_3"], r["population_l1"], "Bromham")
        if multi_df is not None:
            for _, r in multi_df.iterrows():
                put(r["LANG_ISO"], r.get("LMP_POP1"), "Ethnologue_multiISO")
        for _, r in eth_df.iterrows():
            put(r["ISO"], r["Population"], "Koplenig")
        return raw

    def _apply_pluricentric_aggregation(
        self, df: pd.DataFrame, raw_pop: Dict[str, Tuple[float, str]]
    ) -> pd.DataFrame:
        """Sum L1 over member varieties for pluricentric standards."""
        pos_by_iso = {iso: i for i, iso in enumerate(df["iso_639_3"])}

        for anchor in sorted({canonical_iso(m) for m in MGN_LIVING_LANGS_64}):
            members = population_members(anchor)
            if len(members) == 1:
                continue
            have = {m: raw_pop[m][0] for m in members if m in raw_pop}
            if not have:
                logger.warning("Pluricentric %s: no member populations found (%s).",
                               anchor, members)
                continue
            if anchor not in pos_by_iso:
                logger.warning("Pluricentric anchor %s absent from registry; skipping.", anchor)
                continue
            total = float(sum(have.values()))
            i = df.index[pos_by_iso[anchor]]
            df.at[i, "population_l1"] = total
            df.at[i, "population_source"] = "Aggregate"
            df.at[i, "population_scope"] = "+".join(sorted(have))
            missing = sorted(set(members) - set(have))
            logger.info(
                "Pluricentric %s: summed %d/%d members (%s) -> %s%s",
                anchor, len(have), len(members), "+".join(sorted(have)),
                f"{total:,.0f}",
                f"  [no data for: {'+'.join(missing)}]" if missing else "",
            )
        return df

    def _apply_manual_overrides(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the two documented hand-entered populations."""
        pos_by_iso = {iso: i for i, iso in enumerate(df["iso_639_3"])}
        for iso, prof in MANUAL_POPULATION_OVERRIDES.items():
            if iso in pos_by_iso:
                i = df.index[pos_by_iso[iso]]
                if pd.notna(df.at[i, "population_l1"]):
                    logger.info("Manual override for %s skipped: real source present.", iso)
                    continue
                df.at[i, "population_l1"] = float(prof["population_l1"])
                df.at[i, "population_source"] = "Manual_Ethnologue"
                df.at[i, "population_scope"] = iso
            else:
                logger.warning("Manual override target %s not in registry.", iso)
                continue
            logger.info("Manual population override %s = %s (%s)",
                        iso, f"{prof['population_l1']:,.0f}", prof["note"])
        return df

    def _impute_eco_covariates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute the four polygon-derived covariates by family, then macro-area, then global."""
        # A row counts as imputed if ANY of the four is missing before filling.
        df["eco_imputed"] = df[ECO_COLS].isna().any(axis=1)
        n_before = int(df["eco_imputed"].sum())

        for group in ["family", "macro_area"]:
            med = df.groupby(group)[ECO_COLS].transform("median")
            for col in ECO_COLS:
                df[col] = df[col].fillna(med[col])
        global_med = df[ECO_COLS].median()
        for col in ECO_COLS:
            df[col] = df[col].fillna(global_med[col])

        logger.info("Macro-ecological covariates imputed for %d/%d languages (flagged in "
                    "`eco_imputed`).", n_before, len(df))
        return df


class RegistryValidator:
    """Validates the registry's structural and scientific invariants."""

    @staticmethod
    def validate(df: pd.DataFrame) -> bool:
        logger.info("Validating master registry...")

        assert len(df) > 6500, f"Expected >6,500 records, got {len(df)}"
        assert df["iso_639_3"].is_unique, "Duplicate ISO codes in registry"

        # No fabricated populations: nothing may sit on a single repeated constant.
        pops = df["population_l1"].dropna()
        modal_share = pops.value_counts(normalize=True).iloc[0] if len(pops) else 0.0
        assert modal_share < 0.05, (
            f"{modal_share:.1%} of populations share one value - looks like a constant "
            "imputation, which this pipeline must never do."
        )

        # Every declared provenance must be a real source label.
        allowed = {"Bromham", "Ethnologue_multiISO", "Koplenig", "Aggregate", "Manual_Ethnologue"}
        seen = set(df["population_source"].dropna().unique())
        assert seen <= allowed, f"Unknown population_source labels: {seen - allowed}"

        # Populations only where a source exists, and vice versa.
        assert not (df["population_l1"].notna() & df["population_source"].isna()).any(), \
            "Population present with no provenance"
        assert not (df["population_l1"].isna() & df["population_source"].notna()).any(), \
            "Provenance present with no population"

        # MGN coverage
        missing = []
        for mgn in sorted(MGN_LIVING_LANGS_64):
            iso = canonical_iso(mgn)
            sub = df[df["iso_639_3"] == iso]
            if sub.empty or pd.isna(sub.iloc[0]["population_l1"]):
                missing.append(f"{mgn}->{iso}")
        assert not missing, f"MGN languages without population: {missing}"
        logger.info("MGN coverage: %d/%d living languages resolved with a real population.",
                    len(MGN_LIVING_LANGS_64), len(MGN_LIVING_LANGS_64))

        # Coordinate and range sanity for MGN languages
        for mgn in sorted(MGN_LIVING_LANGS_64):
            row = df[df["iso_639_3"] == canonical_iso(mgn)].iloc[0]
            assert row["population_l1"] > 0, f"Non-positive population for {mgn}"
            assert -90.0 <= row["latitude"] <= 90.0, f"Latitude out of bounds for {mgn}"
            assert -180.0 <= row["longitude"] <= 180.0, f"Longitude out of bounds for {mgn}"
            assert len(str(row["glottocode"])) == 8, f"Invalid glottocode for {mgn}"

        # Macro-ecological covariates must stay inside the source distribution.
        assert df["roughness"].max() <= 10.0, "Roughness outside Bromham's observed scale"
        assert df["altitude_range"].max() <= 5000.0, \
            "Altitude range exceeds Bromham's observed maximum (4,858 m)"

        # Extinct exclusion
        present_extinct = MGN_EXTINCT_LANGS_9 & set(df["iso_639_3"])
        assert not present_extinct, f"Extinct languages present: {present_extinct}"

        # Chatino disambiguation: MGN 'yai'/'zen' must resolve to Oaxaca Chatino
        # languages, never to Yaghnobi (Tajikistan) or Zenaga (Mauritania).
        for mgn, gc in [("yai", "west2644"), ("zen", "zenz1235")]:
            iso = canonical_iso(mgn)
            sub = df[df["iso_639_3"] == iso]
            assert not sub.empty, f"{mgn}->{iso} missing from registry"
            row = sub.iloc[0]
            assert "Chatino" in str(row["language_name"]), \
                f"{mgn}->{iso} is not a Chatino language: {row['language_name']}"
            assert row["glottocode"] == gc, \
                f"{mgn}->{iso} glottocode {row['glottocode']} != {gc}"
            assert row["family"] == "Otomanguean", \
                f"{mgn}->{iso} family {row['family']} != Otomanguean"
            assert 15.0 <= row["latitude"] <= 18.5 and -98.5 <= row["longitude"] <= -95.0, \
                f"{mgn}->{iso} not located in Oaxaca"
            # The false friends must NOT be what we resolved to.
            assert iso not in {"yai", "zen"}, f"{mgn} still resolves to its false friend"
        logger.info("Chatino disambiguation verified (yai->ctp, zen->czn, both in Oaxaca).")

        logger.info("All registry invariants satisfied.")
        return True


def main() -> None:
    dump_json()
    builder = DemographicRegistryBuilder()
    df = builder.build_registry()
    df.to_csv(OUTPUT_CSV_PATH, index=False)
    logger.info("Saved %s", OUTPUT_CSV_PATH)
    RegistryValidator.validate(df)
    logger.info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()
