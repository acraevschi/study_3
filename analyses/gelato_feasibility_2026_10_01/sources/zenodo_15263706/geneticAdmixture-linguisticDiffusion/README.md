# admixture/diffusion

Code and data for Graff et al. 2025, Science Advances.

## figs
Contains all figures in the main text and SI of the manuscript.

## input
- MegaAdmixtureCatalogue: files from ADMIXTURE runs relevant to this analysis. 
- FST.csv: contains all pairwise FST-values for the GeLaTo populations relevant to this analysis.
- F2results: a zip-file containing all pairwise precomputed F2 values (necessary for F3-analysis). Unzip after download.
- GBI: contains the raw GBI data from Graff et al. 2025. The parameters.csv file is from the GBI-statistical CLDF. 
- GeLaTo-population-glottocode-mapping.csv: contains manual mappings of all GeLaTo populations to GBI and TLI.
- gelatoLanguageMapping.R: validates the automated columns in GeLaTo-population-glottocode-mapping.csv expressing coding densities per dataset per population.
- lang-metadata.csv: contains metadata on all languages in GBI and TLI, from Graff et al. 2025
- TLI: contains the raw TLI data from Graff et al. 2025. The parameters.csv file is from the TLI-statistical CLDF. 

## output
- all-features-metadata.csv: contains meta-data on all features from GBI and TLI, including number of admixed pair groups attested per feature.
- brms: contains one subdirectory for the brms outputs of the main analysis, one for the brms outputs of the sensitivity analysis and one for the brms outputs of the F3-sensitivity analysis. Each subdirectory hosts prior fits and model fits (including model comparison outputs). The main and sensitivity analysis directories further include scales of contact effects on state sharing and meta-analyses of feature group effect sizes under contact. The directory also hosts the fixed effect meta-analysis model.
- contact-pairs-gbi.csv: contains all 348 contact pairs in GBI, including their assignments to pair ids and broad pair ids.
- contact-pairs-tli.csv: contains all 818 contact pairs in TLI, including their assignments to pair ids and broad pair ids.
- longlists: stores genetic triplet longlists and support files containing FST-information for manual curation at three different thresholds (70%, 80%, 90%) in the respective threshold folders. The file longlist_for_manual_curation.csv contains condensed and filtered information from these files, appended with language availability data.
- pairs-allGBI-allTLI-seed26-N300.rda: contains information per feature (for binary features) or feature state (for features with >2 states) on all available genetic contact language pairs plus 300 random baseline language pairs. Subsets of this data are inputs for the logistic regression models.
- pairs-allGBI-allTLI-seed26-N300-F3.rda: same as pairs-allGBI-allTLI-seed26-N300.rda, but excluding pairs that do not pass the thresholds of the F3-analysis for sensitivity analysis

## scripts
- 1-admixtureTripletsFST.R: generates long lists of admixture triplets given specific thresholds specified in the manuscript, saved in output/longlists. Produces table S1.
- 2-admixtureTripletsLongListForManualCuration.R: filters triplets to those involving populations speaking languages available in GBI or TLI. Produces the single file "longlist_for_manual_curation.csv" in output/longlists, condensing information from the triplets generated at different thresholds (70%, 80%, 90%).
- 3-F3.R: contains the F3-analysis
- 4-extractPairData.R: assembles language pairs from Table S2 (generating the files contact-pairs-gbi.csv and contact-pairs-tli.csv in output), tabulates genetic pair availabilities for all GBI and TLI features (generating the file all-features-metadata.csv in output), extracts pairs for all feature states for analysis (pairs-allGBI-allTLI-seed26-N300.rda in output) and F3-sensitivity analysis (pairs-allGBI-allTLI-seed26-N300-F3.rda in output).
- 5-brmsModelling.R: sampling from prior and model fits for all models of the main, sensitivity and F3-sensitivity analyses. Generation of scales of contact effects on state sharing for all contact types and meta-analyses on feature group behavior. Model comparisons of m1-m3.
- 6-exploringAndPlotting.R: generates and saves all plots and remaining tables.


## tables
Contains all tables for the SI of the manuscript.
