"""
Tier 5: scientific data-integrity tests.

Tiers 1-4 check that artefacts exist and have the right shape. These tests check
that the NUMBERS IN THEM ARE REAL. Each one corresponds to a defect that was found
in the pipeline and would have passed every structural test:

  * 1,442 languages carried a population of exactly 100.0, stamped with the
    provenance label "JoshuaProject" although no Joshua Project data was ever loaded.
  * Macro-ecological covariates for the largest languages were hand-invented
    (altitude = the country's highest mountain, area = the country's territory)
    and overwrote real Bromham measurements for nor/als/pbt.
  * Navajo was silently dropped because it has no num==200 condition, even though
    it is one of the paper's two headline high-complexity languages.
  * Two cell-label conventions were mixed, so 1,872 pairs got a spurious +1 distance.
  * A test named "all_tags_in_mgn_data_mapped" only checked that a JSON file had
    non-empty strings in it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import canonical_iso, MGN_LIVING_LANGS_64, PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from cell_normalization import normalize_cell  # noqa: E402
from mgn_language_map import (  # noqa: E402
    MANUAL_POPULATION_OVERRIDES,
    PLURICENTRIC_MEMBERS,
)

BROMHAM_CSV = PROJECT_ROOT / "data_sources" / "bromham_extracted.csv"

# Observed ranges in Bromham et al., the only source of these four covariates.
BROMHAM_MAX_ALTITUDE = 4858.0   # metres
BROMHAM_MAX_ROUGHNESS = 8.89
BROMHAM_MAX_AREA = 1_844_399.285  # km^2 largest language polygon in Bromham (kaz)
EARTH_LAND_AREA = 149_000_000.0  # km^2, absolute upper bound for any language range

VALID_POPULATION_SOURCES = {
    "Bromham", "Ethnologue_multiISO", "Koplenig", "Aggregate", "Manual_Ethnologue",
}


# ---------------------------------------------------------------------------
# Population provenance
# ---------------------------------------------------------------------------
class TestPopulationProvenance:

    def test_no_constant_population_imputation(self, demographic_registry_df):
        """No single population value may dominate the registry.

        The regression predictor is log10(population). A block of languages sharing
        one imputed constant is indistinguishable from real data downstream and
        silently anchors the low end of the predictor.
        """
        pops = demographic_registry_df["population_l1"].dropna()
        assert len(pops) > 1000, "Suspiciously few populations in registry"
        counts = pops.value_counts(normalize=True)
        top_value, top_share = counts.index[0], counts.iloc[0]
        assert top_share < 0.05, (
            f"{top_share:.1%} of populations are exactly {top_value} - that is a "
            "constant imputation, not data."
        )

    def test_population_sources_are_real(self, demographic_registry_df):
        """Every provenance label must name a source the pipeline actually loads."""
        seen = set(demographic_registry_df["population_source"].dropna().unique())
        unknown = seen - VALID_POPULATION_SOURCES
        assert not unknown, (
            f"Unknown population_source labels {unknown}. A label must correspond to "
            "a source that is genuinely ingested."
        )

    def test_population_and_provenance_agree(self, demographic_registry_df):
        """A population exists iff its provenance does."""
        df = demographic_registry_df
        orphan_pop = df["population_l1"].notna() & df["population_source"].isna()
        orphan_src = df["population_l1"].isna() & df["population_source"].notna()
        assert not orphan_pop.any(), f"{orphan_pop.sum()} populations without provenance"
        assert not orphan_src.any(), f"{orphan_src.sum()} provenance labels without population"

    def test_manual_overrides_are_minimal_and_declared(self, demographic_registry_df):
        """Hand-entered populations must be few and match the declared allowlist."""
        manual = demographic_registry_df[
            demographic_registry_df["population_source"] == "Manual_Ethnologue"]
        assert set(manual["iso_639_3"]) <= set(MANUAL_POPULATION_OVERRIDES), (
            "Manual_Ethnologue used for languages not declared in "
            "MANUAL_POPULATION_OVERRIDES"
        )
        assert len(manual) <= 5, (
            f"{len(manual)} hand-entered populations. Manual values are a last resort "
            "for languages absent from every machine-readable source."
        )

    def test_aggregate_scope_is_recorded(self, demographic_registry_df):
        """Pluricentric aggregates must record which varieties were summed."""
        agg = demographic_registry_df[
            demographic_registry_df["population_source"] == "Aggregate"]
        assert not agg.empty, "Expected at least one pluricentric aggregate"
        for _, row in agg.iterrows():
            scope = str(row["population_scope"])
            assert "+" in scope, (
                f"{row['iso_639_3']} is an Aggregate but population_scope='{scope}' "
                "does not list the summed varieties"
            )
            assert row["iso_639_3"] in PLURICENTRIC_MEMBERS, (
                f"{row['iso_639_3']} aggregated but not declared pluricentric"
            )


# ---------------------------------------------------------------------------
# Macro-ecological covariates
# ---------------------------------------------------------------------------
class TestMacroEcologicalIntegrity:

    def test_covariates_within_source_distribution(self, demographic_registry_df):
        """Values must stay inside the range the source can produce.

        Hand-invented values were detectable precisely because they left it:
        altitude 5,642 m (Mt Elbrus) against a Bromham maximum of 4,858, and
        area 17,098,242 km^2 (the territory of Russia) against a maximum
        language-polygon area of 1,844,399.
        """
        df = demographic_registry_df
        assert df["altitude_range"].max() <= BROMHAM_MAX_ALTITUDE, (
            f"altitude_range max {df['altitude_range'].max()} exceeds Bromham's "
            f"observed maximum {BROMHAM_MAX_ALTITUDE}"
        )
        assert df["roughness"].max() <= BROMHAM_MAX_ROUGHNESS + 0.1
        assert df["roughness"].min() >= 0.0
        # Area is NOT capped at Bromham's maximum: Bromham omits the global lingua
        # francas, and Koplenig's Rangesize legitimately reaches 13.99M km^2 for
        # English. The two sources agree exactly where they overlap (median ratio
        # 1.0000 over 2,058 languages), so both are the same variable. Only the
        # physical bound applies.
        assert df["area_km2"].max() <= EARTH_LAND_AREA, (
            f"area_km2 max {df['area_km2'].max()} exceeds Earth's land area"
        )
        assert (df["area_km2"] > 0).all()
        assert (df["div_bordering_languages"] >= 0).all()

    @pytest.mark.skipif(not BROMHAM_CSV.exists(), reason="Bromham cache unavailable")
    def test_oversized_areas_belong_to_languages_bromham_omits(self, demographic_registry_df):
        """Anything above Bromham's polygon maximum must be a language Bromham lacks.

        A value that large for a language Bromham DOES cover would mean a national
        territory had been substituted for the measured language range.
        """
        brom_isos = set(pd.read_csv(BROMHAM_CSV)["iso_639_3"].dropna())
        oversized = demographic_registry_df[
            demographic_registry_df["area_km2"] > BROMHAM_MAX_AREA * 1.001]
        offenders = sorted(set(oversized["iso_639_3"]) & brom_isos)
        assert not offenders, (
            f"Languages with a measured Bromham polygon carry an implausibly large "
            f"area: {offenders}"
        )

    @pytest.mark.skipif(not BROMHAM_CSV.exists(), reason="Bromham cache unavailable")
    def test_bromham_values_not_overwritten(self, demographic_registry_df):
        """Where Bromham has a measurement, the registry must reproduce it.

        Curated profiles previously overwrote real measurements: Southern Pashto's
        altitude range went from 1,746 m to 7,492 m (Noshaq), Norwegian's from 979 m
        to 2,469 m.
        """
        brom = pd.read_csv(BROMHAM_CSV).drop_duplicates("iso_639_3").set_index("iso_639_3")
        reg = demographic_registry_df.set_index("iso_639_3")
        shared = [i for i in reg.index if i in brom.index]
        assert len(shared) > 5000, "Too few overlapping languages to validate"

        mismatches = []
        for col in ["altitude_range", "roughness", "div_bordering_languages"]:
            a = brom.loc[shared, col]
            b = reg.loc[shared, col]
            both = a.notna() & b.notna()
            bad = (~np.isclose(a[both], b[both], rtol=1e-3, atol=1e-6))
            if bad.any():
                mismatches.append(f"{col}: {int(bad.sum())} languages "
                                  f"(e.g. {list(a[both][bad].index[:5])})")
        assert not mismatches, "Registry diverges from Bromham source: " + "; ".join(mismatches)

    def test_eco_imputed_flag_present_and_honest(self, demographic_registry_df):
        """The imputation flag must exist and mark a plausible minority of rows."""
        df = demographic_registry_df
        assert "eco_imputed" in df.columns, "eco_imputed flag missing"
        share = df["eco_imputed"].astype(bool).mean()
        assert 0.0 < share < 0.5, (
            f"{share:.1%} of rows flagged as imputed; expected a minority. A value of "
            "0 would mean the flag is not being set at all."
        )

    def test_mgn_languages_mostly_measured(self, demographic_registry_df):
        """Most modelled languages must have measured, not imputed, covariates."""
        isos = {canonical_iso(m) for m in MGN_LIVING_LANGS_64}
        sub = demographic_registry_df[demographic_registry_df["iso_639_3"].isin(isos)]
        assert len(sub) == len(isos)
        measured = int((~sub["eco_imputed"].astype(bool)).sum())
        assert measured >= 45, (
            f"Only {measured}/{len(isos)} MGN languages have measured macro-ecological "
            "covariates; the macro-ecological model would rest mostly on imputation."
        )


# ---------------------------------------------------------------------------
# Cell label normalisation
# ---------------------------------------------------------------------------
class TestCellNormalisation:

    def test_navajo_labels_parse(self):
        """Navajo's '<TAM>.<person><number>:IPA' convention must decompose correctly."""
        cases = {
            "FUT.1:IPA":     {"FUT", "1", "SG"},
            "IPFV.1dl:IPA":  {"IPFV", "1", "DU"},
            "PFV.3pl:IPA":   {"PFV", "3", "PL"},
            "OPT.3a:IPA":    {"OPT", "4", "SG"},     # 3a = fourth person
            "ITER.3i:IPA":   {"ITER", "INDF", "SG"},  # 3i = indefinite
            "FUT.3apl:IPA":  {"FUT", "4", "PL"},
        }
        for cell, expected in cases.items():
            assert normalize_cell(cell, {}) == expected, f"{cell} misparsed"

    def test_navajo_person_and_number_are_separable(self):
        """Changing only number must change exactly one feature."""
        a = normalize_cell("FUT.1:IPA", {})
        b = normalize_cell("FUT.1pl:IPA", {})
        assert len(a ^ b) == 2, "SG->PL should be a single feature swap (distance 2)"

    def test_polish_and_dotted_conventions_parse(self):
        assert normalize_cell("sg:nom.voc:f", {}) == {"SG", "NOM", "VOC", "FEM"}
        assert normalize_cell("pl:acc:m1.p1", {}) == {"PL", "ACC", "MASC"}
        assert normalize_cell("pst.ptcp.f.pl", {}) == {"PST", "V.PTCP", "FEM", "PL"}

    def test_curated_map_takes_precedence(self):
        assert normalize_cell("x", {"x": "V;PST;3;SG"}) == {"V", "PST", "3", "SG"}

    def test_unparseable_labels_rejected_not_opaque(self):
        """Labels with no recoverable structure must return None, never a single token.

        Treating 'prepositional plural' as one opaque feature gives every pair
        involving it the same meaningless distance.
        """
        for cell in ["prepositional plural", "pastnot13", "PartPasssm", "inst sg"]:
            assert normalize_cell(cell, {}) is None, f"{cell} should be unparseable"


# ---------------------------------------------------------------------------
# Modeling dataset
# ---------------------------------------------------------------------------
class TestModelingDatasetIntegrity:

    def test_all_64_languages_present(self, mgn_modeling_df):
        """Coverage must be checked on the MODELING dataset, not just the registry.

        Navajo resolved fine in the registry but vanished from the model because it
        has no num==200 condition. Asserting registry coverage alone missed that.
        """
        present = set(mgn_modeling_df["lang"].unique())
        missing = sorted(MGN_LIVING_LANGS_64 - present)
        assert not missing, f"Languages absent from the modelling dataset: {missing}"

    def test_navajo_present_via_num_fallback(self, mgn_modeling_df):
        nav = mgn_modeling_df[mgn_modeling_df["lang"] == "nav"]
        assert not nav.empty, "Navajo missing"
        assert nav["num_used"].nunique() == 1
        assert int(nav["num_used"].iloc[0]) == 500, "Navajo should fall back to num==500"

    def test_training_size_held_constant_within_language(self, mgn_modeling_df):
        """One training-set size per (language, POS), else accuracy is confounded."""
        counts = mgn_modeling_df.groupby(["lang", "pos"])["num_used"].nunique()
        bad = counts[counts > 1]
        assert bad.empty, f"Mixed training sizes within a language/POS: {bad.to_dict()}"

    def test_num_fallback_used_only_where_necessary(self, mgn_modeling_df):
        """The overwhelming majority must sit at the preferred num==200."""
        share_200 = (mgn_modeling_df["num_used"] == 200).mean()
        assert share_200 > 0.95, f"Only {share_200:.1%} of rows at num==200"

    def test_no_mixed_pos_tag_convention(self, mgn_modeling_df):
        """Bare POS features must be gone from every normalised bundle.

        Curated labels expanded to 'V;IMP;ACT;2;SG' while native UniMorph labels had
        no POS feature. Comparing the two added a spurious +1 to the distance for
        1,872 pairs across 15 languages.
        """
        bare_pos = {"V", "N", "ADJ", "ADV"}
        for col in ["unimorph_1", "unimorph_2"]:
            has_pos = mgn_modeling_df[col].astype(str).str.split(";").apply(
                lambda s: bool(bare_pos & set(s)))
            assert not has_pos.any(), (
                f"{int(has_pos.sum())} rows still carry a bare POS feature in {col}"
            )

    def test_distances_are_positive_and_bounded(self, mgn_modeling_df):
        d = mgn_modeling_df["distance"]
        assert d.min() >= 1, "Zero-distance pairs must be filtered out"
        assert d.max() <= 30, f"Implausible maximum distance {d.max()}"
        assert (mgn_modeling_df["distance_rel"] > 0).all()
        assert (mgn_modeling_df["distance_rel"] <= 1.0).all()

    def test_no_reflexive_pairs(self, mgn_modeling_df):
        assert not (mgn_modeling_df["cell_1"] == mgn_modeling_df["cell_2"]).any()

    def test_chance_level_control_carried(self, mgn_modeling_df):
        """nvar must be present and actually informative about accuracy.

        Raw accuracy is not comparable across cell pairs without it.
        """
        assert "nvar" in mgn_modeling_df.columns
        assert (mgn_modeling_df["nvar"] >= 0).all()
        acc = mgn_modeling_df["correct"] / mgn_modeling_df["total"]
        r = acc.corr(mgn_modeling_df["log10_nvar"])
        assert r < -0.1, (
            f"corr(accuracy, log10 nvar) = {r:.3f}; expected clearly negative, since "
            "more inflection classes means a lower chance baseline."
        )

    def test_l2_and_provenance_carried_through(self, mgn_modeling_df):
        """Columns the model may need must survive the merge."""
        for col in ["l2_proportion", "vehicularity", "population_source", "eco_imputed"]:
            assert col in mgn_modeling_df.columns, f"{col} dropped during merge"
        langs_with_l2 = mgn_modeling_df.dropna(subset=["l2_proportion"])["lang"].nunique()
        assert langs_with_l2 >= 20, f"Only {langs_with_l2} languages carry an L2 proportion"

    def test_language_level_zscores(self, mgn_modeling_df):
        """Language-level covariates standardised over languages, not trials.

        Trial counts range from ~6 to ~12,000 per language, so trial-level
        standardisation would weight the centring by paradigm size.
        """
        per_lang = mgn_modeling_df.drop_duplicates("iso_sanitized")
        for col in ["log10_pop_z", "contact_richness_scaled", "roughness_scaled"]:
            assert abs(per_lang[col].mean()) < 0.05, f"{col} not centred over languages"
            assert abs(per_lang[col].std() - 1.0) < 0.05, f"{col} not scaled over languages"

    def test_reproduces_paper_headline_finding(self, mgn_modeling_df):
        """Navajo and Yaitepec Chatino must be the least predictable languages.

        Guzman Naranjo (2024) abstract: I-complexity is low across the sample "with
        only two clear exceptions (Navajo and Yaitepec-Chatino)". Recovering that
        from the assembled dataset validates the language mapping (in particular
        yai -> ctp), the num fallback that rescued Navajo, and the merge as a whole.
        """
        acc = (mgn_modeling_df.groupby("lang")
               .apply(lambda g: g["correct"].sum() / g["total"].sum(),
                      include_groups=False)
               .sort_values())
        assert set(acc.index[:2]) == {"nav", "yai"}, (
            f"Expected Navajo and Yaitepec Chatino to be least predictable, got "
            f"{list(acc.index[:3])}"
        )


# ---------------------------------------------------------------------------
# Phylogenetic covariance matrix
# ---------------------------------------------------------------------------
class TestPhylogeneticMatrix:

    @pytest.fixture(scope="class")
    def A(self, phylo_cov_matrix_path):
        if not phylo_cov_matrix_path.exists():
            pytest.skip("phylo_cov_matrix.rds not built")
        try:
            import pyreadr
        except ImportError:
            pytest.skip("pyreadr not installed")
        return pyreadr.read_r(str(phylo_cov_matrix_path))[None]

    def test_is_a_correlation_matrix(self, A):
        """Brownian-motion correlation: unit diagonal, symmetric, entries in [0, 1]."""
        M = A.values
        assert M.shape[0] == M.shape[1] == 64, f"Expected 64x64, got {M.shape}"
        assert np.allclose(np.diag(M), 1.0), "Diagonal is not exactly 1"
        assert np.allclose(M, M.T, atol=1e-8), "Matrix is not symmetric"
        assert M.min() >= -1e-9 and M.max() <= 1.0 + 1e-9

    def test_positive_definite(self, A):
        ev = np.linalg.eigvalsh(A.values)
        assert ev.min() > 0, f"Not positive definite: min eigenvalue {ev.min()}"

    def test_relatedness_ordering(self, A):
        """Within-genus pairs must covary more than cross-family pairs."""
        M, idx = A.values, list(A.index)
        def cov(a, b):
            return M[idx.index(a), idx.index(b)]
        assert cov("ctp", "czn") > 0.5, "Two Chatino languages should covary strongly"
        assert cov("tur", "azj") > 0.4, "Turkic pair should covary"
        assert cov("rus", "pol") > 0.4, "Slavic pair should covary"
        assert cov("eng", "deu") > cov("eng", "rus"), "Germanic > cross-branch IE"
        for a, b in [("eng", "tur"), ("nav", "eng"), ("zul", "kat")]:
            assert cov(a, b) == pytest.approx(0.0, abs=1e-9), \
                f"Unrelated pair {a}/{b} should have zero covariance"

    def test_covers_all_modeling_languages(self, A, mgn_modeling_df):
        model_isos = set(mgn_modeling_df["iso_sanitized"].unique())
        missing = model_isos - set(A.index)
        assert not missing, f"Languages in the model but absent from A: {missing}"
