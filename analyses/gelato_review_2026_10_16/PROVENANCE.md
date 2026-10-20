# Provenance check of the 64 flagged GeLaTo links (2026-10-16)

Companion to `provenance_check.csv` (one row per flagged population). This note says what was
checked, what the original papers actually document, and which of the earlier flags survived.

**Correction (2026-10-16, main session).** The first version of this note said that the three
Lazaridis 2016 French samples (French_East, French_Northwest, French_West) come from other
départements than the sheet says. That came from the département names in Supplementary Data
Table 2. These names are wrong: each one is the département numbered one above or below the one
at the row's coordinates (Morbihan 56 for Moselle 57, Orne 61 for Pas-de-Calais 62, Corse-du-Sud
2A for Finistère 29; 10 of 11 locations). The coordinates agree with the population labels and
with GeLaTo, so they are taken as correct. The original flags stand: French_East is mostly from
Moselle (Lorraine Franconian area), French_Northwest from Nord, Pas-de-Calais and Aisne (Picard
area) plus Seine-Maritime, and French_West from Brittany (5 of 10 in Finistère, Breton area). The
sections below and the CSV are corrected.

Nothing here is a decision. `proposed_reading` in the CSV is a proposal; only a person fills
`decision`, `confirmed_by`, `confirmed_date` in `gelato_review.yaml`.

## What was checked, and how

For each of the 64 rows marked `requires review` in `main_set_links.csv`:

1. the GeLaTo row (`analyses/gelato_feasibility_2026_10_01/sources/gelato_c625fdc/cldf/populations.csv`):
   Glottocode, `Language_Name`, `curation_notes_linguistics`, `Source`; the GeLaTo README; and the
   Graff et al. 2025 mapping (`GeLaTo-population-glottocode-mapping.csv`, `tables/tableS1.csv`);
2. the original genetics paper named in `reference`: its supplement, sample tables and sampling
   paragraph (downloaded PDFs/XLSX from Nature/Springer, PMC full texts; see `source_url`,
   `source_section` per row);
3. the sociolinguistic situation of the community and place, from Glottolog 5.3 (AES), census
   figures and standard descriptions;
4. Glottocodes and Grambank coverage from `external/glottolog-cldf` and `external/grambank`.

**Ancestry firewall respected.** No `.Q` file, `best_runs/`, `F2results/`, `FST.csv`,
`MegaAdmixtureCatalogue/`, GeLaTo `datasets/*/data.csv`, `population_crosswalk.csv` or any file of
the draft admixture model was opened. Only population metadata, language tables and published
sample descriptions were read. No genotype data was downloaded.

**Not accessible:** Vyas et al. 2017 (Am J Phys Anthropol, Yemen) is paywalled with no PMC copy, so
the two Yemeni rows rest on the mapping's governorate field alone. The Barbieri et al. 2022 PNAS
SI appendix could not be downloaded (PMC blocked the file); the GeLaTo README and the paper's
Materials and Methods were used instead.

## How GeLaTo assigned languages

The GeLaTo README states the rule: every genetic population is matched to one Glottocode, the
"main language spoken by their people", taken from the original genetics publication and derived
"either from direct sampling observation, cultural/linguistic self-identification, or geographical
characterization", with help from linguists. Two further rules matter here:

* colonial and trans-national languages are deliberately **not** used, "to exclude the wave of
  historical language shift documented in the past ~2 centuries" — this is why Aleut, Nganasan,
  Veps and so on are coded by the heritage language even where the community speaks Russian;
* for multilingual populations only one language is chosen, with alternatives sometimes noted in
  the curation notes.

The per-row `curation_notes_linguistics` says who did it: 23 of the 64 are
"automatically assigned by Robert F" (label-to-Glottocode matching, never checked by a linguist),
the rest are "checked by Chiara/Damian", "correct by Harald/Chiara/Søren" or "assigned by Chiara".
The automatic ones hold most of the problems: Cypriot, Yukagir_Forest, Sardinian, Tu, Tujia, She,
Guarani_GN/KW.

**Nine of the 64 populations are not in the released GeLaTo population table at all**
(Aleut_Tlingit, TaiwanHan, Negidal_Orochi, Nganasan_Dudinka, Yemen_Ad_Dali, Yemen_Ibb, Scottish,
Malawi_Ngoni, Chane). Their links come only from the Graff et al. 2025 mapping, so there is no
GeLaTo curation note to appeal to.

## How the source papers selected individuals

**Lazaridis et al. 2014** (24 rows). SI 9, Table S9.4 lists population label, region, country,
coordinates, sample counts and the contributing collector — nothing else. There is no language
criterion, no ancestry rule and no locality for most entries. The verbose population IDs are
informative in two cases: `Georgian_Megrels` (the Georgian sample is Mingrelian) and
`Spanish_*_IBS` / `Scottish_Argyll_Bute_GBR` (Coriell panels, i.e. selection by region only).

**Lazaridis et al. 2016** (11 rows). SI 2 plus Supplementary Data Table 2, which lists the
town/province of each of the 238 new donors. The Irish samples are the Irish DNA Atlas, whose
donors must have all eight great-grandparents born within 50 km of each other (Gilbert et al. 2017,
Sci Rep); so selection is by great-grandparental ancestry and place, never by language. The
Scottish regional samples are recorded by region only.

**Patterson et al. 2012 / HGDP** (9 rows). These are HGDP-CEPH cell lines (Cann et al. 2002). The
paper takes the HGDP population labels unchanged; the CEPH panel documentation says the information
per cell line "is limited to sex of the individual, population and geographic origin". So for Han,
She, Tujia, Hezhen, Tu, Pathan, Sardinian, Basque, Mbuti and Bergamo there is no documented
language, no subgroup and usually no locality. Everything in those rows is inference from the
ethnic label.

**Jeong et al. 2019** (7 rows). The best-documented source: individuals were included whose
grandparents were all self-identified members of the ethnic group and who were born in the sampled
districts, and subgroups were sampled where they were known to speak different dialects. This
documents *ethnicity*, not language use — Nivkh, Nanai, Negidal, Shor and Veps donors are selected
as ethnic members of moribund-language communities.

**The rest, briefly.** Flegontov et al. 2019 (Nganasan): saliva from four West Siberian ethnic
groups with local approval; no selection rule stated. Barbieri et al. 2019 (Cocama, Wayku): the
only source that records the language each donor speaks, and that filtered out donors reporting a
non-Native parent or grandparent. Skoglund et al. 2015 (Guarani): Table S1.1 gives population
label, state, family-level "Language" (Tupi) and coordinates. Skoglund et al. 2016: contains no
description of a Taiwan Han sample. Skoglund et al. 2017 (Malawi): 34 Malawians by ethnic label
only. Mallick et al. 2016 (SGDP): populations chosen to span linguistic and cultural variation, no
per-donor criteria; note that GeLaTo attributes Bergamo to Patterson 2012 (HGDP Bergamo) while the
review sheet attributes it to Mallick 2016.

## Proposed readings by case type

### Language shift: heritage language only (the community now speaks another language)

| population | linked | proposed | heritage / current | conf. |
|---|---|---|---|---|
| Aleut, Aleut_Tlingit | Aleut | link_heritage_only | aleu1260 / russ1263 | high / medium |
| Nivh | Amur Nivkh | link_heritage_only | gily1242 / russ1263 | high |
| Even | Even | link_heritage_only | even1260 / russ1263 | medium |
| Nanai | Nanai | link_heritage_only | nana1257 / russ1263 | medium |
| Hezhen | Nanai | link_heritage_only | nana1257 / mand1415 | high |
| Negidal_Orochi | Negidal | link_heritage_only | negi1245 / russ1263 | medium |
| Ulchi | Ulch | link_heritage_only | ulch1241 / russ1263 | high |
| Veps | Veps | link_heritage_only | veps1250 / russ1263 | medium |
| Nganasan, _Dudinka, _UstAvam, _Volochanka | Nganasan | link_heritage_only | ngan1291 / russ1263 | high |
| Shor_Khakassia, Shor_Mountain | Mrassu Shor-Tutal | link_heritage_only | shor1247 / russ1263 | medium |
| Kusunda | Kusunda | link_heritage_only | kusu1250 / nepa1254 | high |
| Hoan | Amkoe | link_heritage_only | hoaa1235 / Gui-Gana (no Grambank) | low |
| Tujia | Northern Tujia | link_heritage_only | nort2732 / mand1415 | low |

These are exactly the cases GeLaTo intends to code by heritage language, so the links are
"correct" by GeLaTo's own rule; the question for this study is whether a heritage code may be paired
with the genetics of a community that has shifted.

### Regional samples whose traditional vernacular differs from today's language

| population | proposed | note |
|---|---|---|
| French_East | link_correct (current) | Moselle: Lorraine Franconian heritage (pala1330, no Grambank); donors probably speak French |
| French_Northwest | link_correct (current) | Picard area (pica1241, no Grambank) plus 2 Normandy donors |
| French_West | link_correct (current) | Brittany: Breton heritage (bret1244, in Grambank) for the Finistère donors |
| French_South | link_correct | Lourdes/Béarn, Gascon lost in the 20th c.; donors speak French |
| Irish_Connacht, _Cork_Kerry, _Leinster, _Munster, _Ulster | link_heritage_only | Irish DNA Atlas, English-speaking donors, Irish as heritage |
| Scottish_Highlands, Scottish (Argyll & Bute) | link_heritage_only | Gaelic heartlands, now a minority language |
| Scottish_Borders, _Dumfries_Galloway, _Fife, _Grampian, _Shetlandic | link_wrong_language | Scots/English areas; Shetland never Gaelic |
| Bergamo | link_correct | Italian is right; Bergamasco (east2276 → lomb1257) is the heritage alternative |
| Spanish_Cataluna, Spanish_Valencia, Spanish_PaisVasco, Spanish_Galicia | unclear / link_heritage_only / link_correct | Coriell regional panels; first language not recorded |
| Basque (French) | unclear | HGDP ethnic Basque sample, selection rule undocumented |
| Sardinian | unclear | HGDP Sardinians, variety unrecorded; map-down to Campidanese is a data choice |
| Georgian | link_correct | the source says Megrels: Georgian is the written/second language, Mingrelian the home language |

### Diaspora samples

Jew_Ethiopian, Jew_Georgian, Jew_Cochin: all `diaspora`, all `link_heritage_only`. Collected in
Israel from the National Laboratory for the Genetics of Israeli Populations; GeLaTo places them at
origin by its migrant rule. Heritage codes (Amharic, Judeo-Georgian → Georgian, Jewish Malayalam →
Malayalam) all have Grambank entries; the current language is Modern Hebrew (hebr1245, Grambank).

### Group map-downs that picked a language the sample does not speak

Mixe (`link_wrong_language`): Tamazulápam is solidly Mixe-speaking, but the sampled variety is
Tlahuitoltepec/Tamazulápam Mixe (tlah1239, no Grambank); Oluta Popoluca (olut1240) is a nearly
extinct Veracruz language 300 km away, chosen only because it is the one Mixean language in
Grambank. Pathan (`link_wrong_language`): Northern Pashto (nort2646) is the regional variety, and
Graff et al. themselves chose it for the TLI side while keeping Southern Pashto (sout2649) for
Grambank. Tu (`unclear`): Mongghul (Huzhu) is likelier than Mangghuer (Minhe), but the county is
not recorded. Malawi_Ngoni (`link_wrong_language`) and Chane (`link_wrong_language`): both groups
shifted long ago (to Tumbuka/Chichewa and to Chaco Guaraní), and the mapped languages (Tanzanian
Ngoni, Terena-Kinikinao-Chane) were picked for data availability. Sardinian and Tujia as above.

### Codes that point at the wrong language

Cypriot → Cypriot Arabic (a few hundred Maronites) instead of Cypriot Greek; **Yukagir_Forest →
Northern (Tundra) Yukaghir** although Nelemnoye on the Kolyma is Southern/Forest Yukaghir
(sout2750) and a separate Yukagir_Tundra population exists; Kaitag → North-Central Dargwa although
Glottolog has Kajtak (kajt1238) as a separate language; Guarani_GN → Mbyá although GN is Chiripá
(chir1286); Guarani_KW → Mbyá although KW is Kaiwá (kaiw1246); Yemen_Ad_Dali and Yemen_Ibb →
Sanaani although both governorates are in the Ta'izzi-Adeni area (taiz1242); She → the Hmong-Mien
She language although ~99% of the She speak Shehua, a Hakka-like Sinitic variety; Han_HGDP and
TaiwanHan → Mandarin for samples of mixed Sinitic heritage (both left `unclear`); MbutiPygmy →
Lese, defensible for Efe but the subgroup is unrecorded (`unclear`).

## Flags that turned out unfounded

* **French_East, French_Northwest, French_West.** Not unfounded (see the correction at the
  top). The coordinates confirm the regions of the original flags. GeLaTo records the wrong
  source paper for them (Lazaridis 2014 instead of 2016).
* **Georgian.** Flagged because the place is Megrelia; the source is in fact explicit
  (`Georgian_Megrels`), and the Georgian code is still the right one for a Mingrelian community,
  since Mingrelian has no Grambank entry and Megrels are Georgian-speaking bilinguals.
* **Cocama** and **Wayku.** Barbieri et al. record the language of each donor; both are described
  as speaker communities. The AES-rule flag (nearly extinct / moribund) is not borne out for these
  particular samples.
* **Kharia** (low confidence). The flag rests on the Jaunpur coordinate, which looks like the
  contributing laboratory's reference point; Kharia itself is a normally transmitted language of
  ~300,000 speakers and the donors were selected as ethnic Kharia.
* **Bergamo** and **Spanish_Galicia**: the linked language is plausibly the donors' own.

## Flags that were confirmed, or turned out worse

Worse than flagged: **Yukagir_Forest** (not merely moribund — the code looks like the wrong
Yukaghir language, the one belonging to the other GeLaTo sample; GeLaTo has the two Yukaghir codes
swapped, see question 4); **Chane** (n = 1) and **Nganasan_Dudinka** (n = 2) are
below GeLaTo's own five-individual rule; **Negidal_Orochi** (n = 4) mixes two peoples whose
languages both have codes (negi1245, oroc1248). **TaiwanHan** has no documentation at all in its
cited source. **Sardinian** and **Tu** are worse than flagged in the sense that the sampled variety
is probably not the mapped one *and* cannot be determined from the source.

## Cases where the current-language scenario drops the population

A "current language" scenario needs a Grambank entry for that language. It is missing for:

* **Spanish** (stan1288): Spanish_Cataluna, Spanish_Valencia, Spanish_PaisVasco, Spanish_Galicia;
* **Scots** (scot1243) and **Scottish Gaelic** (scot1245): all seven Scottish rows keep a value
  only under an English reading (stan1293);
* **Ta'izzi-Adeni Arabic** (taiz1242): both Yemeni rows;
* **Chiripá** (chir1286), **Kaiwá** (kaiw1246), **Kajtak** (kajt1238), **Tlahuitoltepec Mixe**
  (tlah1239), **Southern Yukaghir** (sout2750), **Northern Pashto** (nort2646), **Tumbuka**
  (tumb1250), **Mongghul** (huzh1238), **Cypriot Greek** (cypr1249), **Gui-Gana** (gana1274):
  these drop under a corrected link, in both scenarios where the heritage language is the same.

The heritage scenario drops Sardinian (logu1236), Cypriot (cypr1249), Kaitag, the Yemeni rows,
Guarani_GN/KW, Mixe, Pathan, Yukagir_Forest, Chane (tere1279 at language level; the dialect
tere1281 does have Grambank data), the five Scots-heritage Scottish rows, French_East
(Lothringisch) and French_Northwest (Picard). The exact pairs are in
the CSV columns `scenario_heritage_glottocode` and `scenario_current_glottocode` (empty = drops).

## Open questions for the user

1. **The GeLaTo rule versus this study's question.** GeLaTo deliberately codes the pre-colonial
   heritage language. If admixture is to be paired with the grammar people actually speak, 29 of
   these rows change language (mostly to Russian, English, Mandarin or Spanish) and several then
   drop for lack of Grambank data. Which scenario is the primary analysis?
2. **Dialect entries as stand-ins.** Taibei Hokkien (taib1242) and Terena (tere1281) have Grambank
   entries although their language-level codes do not. Use them for TaiwanHan and Chane, or treat
   those populations as unmatched?
3. **The three French rows.** The locations are confirmed by the coordinates (see the correction at
   the top). Only the source paper in GeLaTo is wrong (2014 for 2016). Treat them like the other
   regional samples (three scenarios)?
4. **Yukaghir.** GeLaTo has the two codes swapped: Yukagir_Forest (Nelemnoye, Kolyma) carries
   nort2745 (Northern = Tundra Yukaghir) and Yukagir_Tundra (Chokurdakh) carries sout2750
   (Southern = Kolyma/Forest Yukaghir). Only Northern Yukaghir has Grambank data, so after a swap
   Yukagir_Tundra would link to it and Yukagir_Forest would drop.
5. **Kaitag.** Glottolog splits Kajtak from Dargwa; Russian-tradition classifications do not. Keep
   darg1241 as a lumping decision, or drop the population?
6. **Nine populations missing from the released GeLaTo table.** They rely on the Graff et al. 2025
   mapping; is a newer GeLaTo release available that documents them?
7. **Vyas et al. 2017** could not be read. If you have institutional access, the Yemeni rows
   deserve a look at the actual sampling description.
8. **Small samples.** Chane (n = 1), Nganasan_Dudinka (n = 2), Negidal_Orochi and Aleut_Tlingit
   (n = 4), Scottish (n = 4) fall below or at GeLaTo's five-individual threshold.
