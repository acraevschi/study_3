"""Unit tests for Track 1 harmonized 5-domain Grambank data prep, Beta-Binomial, and 2PL IRT-SEM."""

from __future__ import annotations

import csv
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analyses.common.genetics import EXCLUDE_MORPH_IDS, compute_grambank_nonmorph_coverage
import analyses.grambank.prepare as gb_prepare
from analyses.grambank.prepare import (
    DOMAIN_FLOORS,
    DOMAIN_ITEM_SPECS,
    DOMAIN_ORDER,
    PREREQ_SPECS,
    derive_grambank_metadata_from_cldf,
    score_language_5domains,
)

ROOT = Path(__file__).resolve().parents[1]
GB_DATA = ROOT / "analyses/grambank/data"


class TestGrambankV2Pipeline(unittest.TestCase):
    def test_exclude_morph_ids_exact_44_and_nonmorph_151(self) -> None:
        self.assertEqual(len(EXCLUDE_MORPH_IDS), 44)
        used_in_items = {pid for spec in DOMAIN_ITEM_SPECS for pid in spec[4]}
        used_in_prereqs = {pid for pids in PREREQ_SPECS.values() for pid in pids}
        self.assertEqual(used_in_items | used_in_prereqs, set(EXCLUDE_MORPH_IDS))

        cov = compute_grambank_nonmorph_coverage()
        rows = cov if isinstance(cov, list) else cov.to_dict("records")
        self.assertEqual(len(rows), 225)
        self.assertTrue(all(int(r["n_nonmorph_total"]) == 151 for r in rows))

    def test_boolean_and_prerequisite_conditioning_rules(self) -> None:
        # 1. Exact OR vs lenient OR: missing component in OR_exact with 0s returns None,
        #    while OR_lenient with 0 and ? returns 0; any 1 returns 1 for both.
        raw_partial = {
            "GB082": "1",
            "GB089": "0", "GB090": "?", "GB091": "0", "GB092": "0",
            "GB093": "0", "GB094": "0",
            "GB300": "0", "GB402": "?", "GB403": "?",
            "GB111": "0", "GB110": "1", "GB099": "0", "GB109": "0",
            "GB042": "0", "GB043": "0", "GB044": "1", "GB165": "0", "GB166": "0",
            "GB039": "0", "GB041": "0", "GB321": "0",
            "GB051": "0", "GB052": "0", "GB053": "0", "GB054": "0", "GB192": "0",
            "GB170": "0", "GB171": "0", "GB172": "0", "GB198": "0",
        }
        items, counts = score_language_5domains(raw_partial)
        self.assertIsNone(items["subj_indexing"])      # OR_exact with ? -> None
        self.assertEqual(items["obj_indexing"], 0)     # OR_exact all 0 -> 0
        self.assertEqual(items["lex_verb_suppl"], 0)   # OR_lenient with 0 and ? -> 0 (any_v_morph==1)
        self.assertIsNone(items["pers_stem_alt"])      # person_idx is None (not 1) -> masked NA
        self.assertEqual(items["tam_suppletion"], 1)   # any_v_tam==1 -> unmasked
        self.assertEqual(items["noun_num_allom"], 0)   # any_noun_num==1 -> unmasked
        self.assertEqual(items["noun_num_suppl"], 0)
        self.assertIsNone(items["opaque_gender"])      # any_gender==0 -> masked NA
        self.assertEqual(counts["n_irreg_present"], 0)
        self.assertEqual(counts["n_irreg_coded"], 2)
        self.assertEqual(counts["n_irreg_raw_coded"], 3)

        # 2. Broad gender prerequisite catches opaque-only noun-class languages (GB171=1, GB321=1)
        raw_opaque = dict(raw_partial)
        raw_opaque["GB171"] = "1"
        raw_opaque["GB321"] = "1"
        items_op, counts_op = score_language_5domains(raw_opaque)
        self.assertEqual(items_op["opaque_gender"], 1)
        self.assertEqual(counts_op["n_irreg_present"], 1)
        self.assertEqual(counts_op["n_irreg_coded"], 3)

    def test_landmark_languages_and_item_observation_consistency(self) -> None:
        with open(GB_DATA / "language_data.csv", encoding="utf-8") as f:
            ld = {r["glottocode"]: r for r in csv.DictReader(f)}
        with open(GB_DATA / "items_metadata.csv", encoding="utf-8") as f:
            meta = list(csv.DictReader(f))
        with open(GB_DATA / "item_observations.csv", encoding="utf-8") as f:
            obs = list(csv.DictReader(f))

        # Dialect links resolved across all 225 languages
        self.assertEqual(ld["kare1335"]["grambank_entry"], "nort2673")
        self.assertEqual(ld["selk1253"]["grambank_entry"], "sout3262")
        self.assertEqual(ld["tere1279"]["grambank_entry"], "guan1270")

        # Typological landmark checks for deterministic prerequisite masking
        self.assertEqual((int(ld["nucl1301"]["n_irreg_present"]), int(ld["nucl1301"]["n_irreg_coded"])), (0, 2))
        self.assertEqual((int(ld["hung1274"]["n_irreg_present"]), int(ld["hung1274"]["n_irreg_coded"])), (0, 2))
        self.assertEqual(int(ld["mand1415"]["n_irreg_coded"]), 0)
        self.assertEqual(int(ld["basq1248"]["n_irreg_coded"]), 0)
        self.assertEqual((int(ld["fefe1239"]["n_irreg_present"]), int(ld["fefe1239"]["n_irreg_coded"])), (1, 3))

        # Testlet set matches specification
        self.assertEqual(
            {r["testlet"] for r in meta},
            {"verbal_tense", "noun_case", "gender_agr", "num_agr", "none"},
        )

        # Item observations sum exactly to language_data domain counts
        agg: dict[tuple[str, str], list[int]] = {}
        for r in obs:
            val = int(r["value"])
            self.assertIn(val, (0, 1))
            key = (r["glottocode"], r["item_group"])
            if key not in agg:
                agg[key] = [0, 0]
            agg[key][0] += val
            agg[key][1] += 1
        for gc, row in ld.items():
            for dom in DOMAIN_ORDER:
                pres, cod = agg.get((gc, dom), [0, 0])
                self.assertEqual(pres, int(row[f"{dom}_present"]), f"{gc} {dom}_present mismatch")
                self.assertEqual(cod, int(row[f"{dom}_coded"]), f"{gc} {dom}_coded mismatch")
                self.assertLessEqual(cod, DOMAIN_FLOORS[dom][1])

    def test_r_dry_runs_bb_and_sem(self) -> None:
        env_bb = {**os.environ, "BB_DRY": "1", "BB_OUTCOMES": "v_breadth,n_irreg"}
        proc_bb = subprocess.run(
            ["Rscript", "analyses/grambank/fit_stage2_bb.R", "--cores=2"],
            cwd=ROOT, env=env_bb, capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc_bb.returncode, 0, f"BB_DRY failed:\n{proc_bb.stderr}\n{proc_bb.stdout}")
        bb_out = proc_bb.stderr + proc_bb.stdout
        self.assertIn("BB_DRY=1 PASSED", bb_out)
        self.assertIn("1 Berkson 2-draw jobs queued", bb_out)
        self.assertNotIn("will not be used in the model", bb_out)

        # Verify generated brms Stan code actually emits inv_gamma(3, 0.4) on lscale
        r_chk = (
            'library(brms); '
            'df <- data.frame(n_pres=c(1,2,3,1,2,3), n_cod=rep(5,6), x=rnorm(6), '
            'sx=rnorm(6), sy=rnorm(6), sz=rnorm(6)); '
            'pr <- c(prior(normal(0, 1.5), class=Intercept), prior(normal(0, 0.5), class=b), '
            'prior(gamma(2, 0.1), class=phi), prior(exponential(1), class=sdgp), '
            'prior(inv_gamma(3, 0.4), class=lscale, coef=gpsxsysz)); '
            'sc <- make_stancode(n_pres | trials(n_cod) ~ 1 + x + gp(sx, sy, sz), '
            'data=df, family=beta_binomial(), prior=pr); '
            'stopifnot(grepl("inv_gamma_lpdf(lscale_1[1][1] | 3, 0.4)", sc, fixed=TRUE))'
        )
        proc_lsc = subprocess.run(["Rscript", "-e", r_chk], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(proc_lsc.returncode, 0, f"lscale Stan code check failed:\n{proc_lsc.stderr}")

        env_sem = {**os.environ, "SEM_DRY": "1", "SEM_DOMAINS": "n_irreg", "SEM_RUN_2FAC": "1"}
        proc_sem = subprocess.run(
            ["Rscript", "analyses/grambank/latent_sem/fit_domain_sem.R", "--cores=2"],
            cwd=ROOT, env=env_sem, capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc_sem.returncode, 0, f"SEM_DRY failed:\n{proc_sem.stderr}\n{proc_sem.stdout}")
        self.assertIn("SEM_DRY=1 PASSED", proc_sem.stderr + proc_sem.stdout)
        # Compiled binaries must live in gitignored latent_sem/fits/, never inside stan/
        self.assertFalse((ROOT / "analyses/grambank/latent_sem/stan/item_domain_sem").exists())
        self.assertFalse((ROOT / "analyses/grambank/latent_sem/stan/item_pos_2factor_sem").exists())

    def test_standalone_prepare_without_typ_dir(self) -> None:
        with open(GB_DATA / "language_data.csv", encoding="utf-8") as f:
            expected = {r["glottocode"]: r for r in csv.DictReader(f)}

        # Direct CLDF metadata derivation matches all 225 rows of canonical language_data.csv 100%
        derived = derive_grambank_metadata_from_cldf(sorted(expected.keys()))
        self.assertEqual(len(derived), 225)
        for row in derived:
            gc = str(row["glottocode"])
            exp = expected[gc]
            for col in ("grambank_entry", "name", "family", "macroarea"):
                self.assertEqual(str(row[col]), exp[col], f"{gc} {col} mismatch")
            self.assertEqual(int(row["n_present"]), int(exp["n_present"]), f"{gc} n_present mismatch")
            self.assertEqual(int(row["n_coded"]), int(exp["n_coded"]), f"{gc} n_coded mismatch")

        # Full prepare.main() succeeds into a temp dir even when TYP_DIR does not exist
        with tempfile.TemporaryDirectory() as tmp:
            tmp_out = Path(tmp) / "grambank_data"
            with mock.patch.object(gb_prepare, "TYP_DIR", Path(tmp) / "nonexistent_outputs_typology"):
                gb_prepare.main(out_dir=tmp_out)
            with open(tmp_out / "language_data.csv", encoding="utf-8") as f:
                actual = {r["glottocode"]: r for r in csv.DictReader(f)}
            self.assertEqual(set(actual.keys()), set(expected.keys()))
            for gc, exp in expected.items():
                for col in ("grambank_entry", "name", "family", "macroarea", "n_present", "n_coded",
                            "v_breadth_present", "v_breadth_coded", "n_breadth_present", "n_breadth_coded",
                            "v_irreg_present", "v_irreg_coded", "n_irreg_present", "n_irreg_coded",
                            "np_agr_present", "np_agr_coded"):
                    self.assertEqual(actual[gc][col], exp[col], f"{gc} {col} mismatch in standalone prepare")


if __name__ == "__main__":
    unittest.main()
