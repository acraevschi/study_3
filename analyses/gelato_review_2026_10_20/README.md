# First version of the GeLaTo link decision sheet (2026-10-20)

`seed_sheet.py` wrote the first version of `morph_ldl/data/gelato_link_review.csv`, the
only place where a person decides about a GeLaTo → Grambank link since 2026-10-20 (rules
and columns: `docs/TYPOLOGY.md` §2). From now on the sheet is edited by hand and rebuilt
with `python -m morph_ldl.cli link_review --config configs/pcfp_v2_links.yaml`, which keeps
every decision. This script is not run again.

What it put in the sheet:

* the 12 entries of `morph_ldl/data/gelato_review.yaml`, decisions and confirmations
  unchanged (10 confirmed `accept`; the two Sicilian rows are unconfirmed `exclude`
  proposals, user decision 2026-10-20);
* the Yukaghir swap: two `relink` proposals (Yukagir_Forest → sout2750, Yukagir_Tundra →
  nort2745), with the evidence in `reason`;
* doubt labels for the sensitivity analyses (user decision 2026-10-20), only for
  populations with a main-set link:
  * `diaspora`: Graff et al. mapping location `migrants_collected_in_…` (8 populations:
    3 Jewish communities sampled in Israel, 5 migrant groups sampled in Kuwait);
  * `wrong_code`: `shift_status = wrong_code` in
    `analyses/gelato_review_2026_10_16/provenance_check.csv` (7 populations).

The regeneration then added the 5 group codes without a Graff choice and filled the
evidence columns. Run order:

```bash
PYTHONPATH=. .venv/bin/python analyses/gelato_review_2026_10_20/seed_sheet.py --links outputs/pcfp_v2_review_2026_10_16/typology/grambank_population_links.csv
.venv/bin/python -m morph_ldl.cli typology --config configs/pcfp_v2_links.yaml
PYTHONPATH=. .venv/bin/python analyses/gelato_review_2026_10_20/seed_sheet.py --links outputs/pcfp_v2_links_2026_10_20/typology/grambank_population_links.csv --prune-with outputs/pcfp_v2_links_2026_10_20/typology/analysis_links.csv
.venv/bin/python -m morph_ldl.cli link_review --config configs/pcfp_v2_links.yaml
.venv/bin/python -m morph_ldl.cli typology --config configs/pcfp_v2_links.yaml
```

No ancestry file is read.
