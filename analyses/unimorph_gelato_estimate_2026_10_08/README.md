# Crude UniMorph x GeLaTo verb coverage estimate (2026-10-08)

Question: how many languages outside MGN does UniMorph add that (a) have at least 100
verbal lemmas, (b) average more than 4 distinct verbal forms per lemma and (c) match a
GeLaTo population?

Method: GitHub repos of the `unimorph` org with 3-letter names were mapped to Glottocodes
with the pipeline's pinned languages-of-the-world layer and project ISO corrections
(`morph_ldl.data.identifiers.resolve_identifier`); macrolanguage codes were mapped by
hand (column `gelato_match`). Languages whose Glottocode is in the MGN registry were
dropped. GeLaTo = base Glottocodes of the main 397 and expanded 558 panels
(`analyses/gelato_feasibility_2026_10_01/outputs/population_crosswalk.csv`). Matched repos
were shallow-cloned at their default branch on 2026-10-08 (not pinned) and verbal rows
(tag `V` or `V.*`) counted per lemma. For Veps the largest dialect file was used; for
Uzbek `uzb_verbs`.

Not checked: population/community review, the 8-cell verb panel, infinitive availability,
multiword forms, data quality. Exact Glottocode equality is an initial match only.

## Addendum: Paralex and MorphyNet (2026-10-08)

Paralex: the 31 records of the Zenodo `paralex` community (API listing 2026-10-08).
Languages not already in MGN or UniMorph (Livonian, Livvi, eight Kuki-Chin languages,
Ngkolmpu, Pitjantjatjara/Yankunytjatjara, Kasem, Classical Tibetan) have no GeLaTo
population with the same Glottocode; Livvi is only a neighbour of GeLaTo's
Karelian_Northern/Southern samples, which Glottolog classifies as a different language.
Paralex does fill MGN verb gaps for GeLaTo-matched languages (`other_sources_verbs.csv`).

MorphyNet (github kbatsuren/MorphyNet, main @ 2023-04-02): Mongolian verbs are 55 lemmas
(the V rows are heavily duplicated; 902 distinct verb rows), the same size as UniMorph `khk`;
Mongolian nouns have 2,062 lemmas. Czech, Hungarian and Russian verb files were counted.

Counts are raw lemmas with any verbal form; the source+8-cell panel is not checked.

Level mismatches found afterwards: UniMorph `krl` (Karelian, kare1335) and `jpn`
(Japanese, nucl1643) match GeLaTo populations coded at dialect level (Karelian_Northern,
Karelian_Southern) or group level (Japanese, japa1256), so the exact join missed them.
Both pass the crude rule; see the last rows of `other_sources_verbs.csv`.

## Addendum: Grambank inflection extent for GeLaTo languages (2026-10-08)

`grambank_inflection_gelato.csv`: GeLaTo populations (main + expanded panels) rolled up
from dialect to language level with Glottolog 5.3 (`Language_ID`), joined to Grambank
v1.0.3. Inflection extent = share of 35 core inflectional features coded present
(verbal TAM/other affixes GB079 GB080 GB082 GB083 GB084 GB086 GB312; person indexing
GB089-GB094; negation/interrogation morphology GB107 GB286; nominal number GB042 GB043
GB044 GB165 GB166; case GB070-GB073; possession affixes GB430-GB433; agreement GB170
GB171 GB172 GB184 GB185 GB186 GB198). Derivation, auxiliaries, valency and diminutives
were left out. Grambank counts clitics as bound, which inflates some largely analytic
languages (e.g. Burmese 15/35). `covered` = already reachable through MGN, UniMorph,
Paralex or MorphyNet paradigms.
