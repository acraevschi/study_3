# Data and tree preparation for Within-Family Phylogenetic Branch-Rate & Contrast Analysis
# (PhD Study 3 — GeLaTo-to-Grambank branch)
#
# Prepares:
#   1. Pruned within-family Bouckaert v2.0.0 time-calibrated trees (in kyr) for:
#      - min5: families with >= 5 languages (10 families, N = 144 on cat12)
#      - min3: families with >= 3 languages (13 families, N = 153 on cat12)
#   2. Six genetic representations at the language tips (with tip measurement-error SDs):
#      - ent:           Stage-1 ADMIXTURE Shannon entropy (d = 1)
#      - sqrt_het:      Stage-1 square-root within-population ancestry heterogeneity (d = 1)
#      - f3k_est_pos:   Stage-1 clamped 5-nearest admixture f3 statistic (d = 1)
#      - q_within_d3:   Within-family ADMIXTURE Q-matrix (K=12..30) top 3 PCoA axes (d = 3)
#      - fst_within_d3: Within-family pairwise FST top 3 PCoA axes (d = 3)
#      - fst_global_d3: Global pairwise FST top 3 PCoA axes (supervisor's literal sketch, d = 3)
#   3. GeLaTo demographic covariates per language (sample size n_individuals, harmonicMean Ne,
#      medianFST) to test whether FST branch jumps reflect contact vs. bottleneck drift.

suppressPackageStartupMessages({
  library(ape)
})

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
script_dir <- if (length(file_arg) > 0L) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE))
} else if (file.exists("gen_rrw.stan")) {
  normalizePath(".", mustWork = FALSE)
} else if (dir.exists("analyses/within_family_rates_2026_10_30")) {
  normalizePath("analyses/within_family_rates_2026_10_30", mustWork = FALSE)
} else {
  normalizePath(".", mustWork = FALSE)
}
repo_root <- if (file.exists(file.path(script_dir, "../../analyses/admixture_links_2026_10_21/language_data.csv"))) {
  normalizePath(file.path(script_dir, "../.."), mustWork = FALSE)
} else if (file.exists("analyses/admixture_links_2026_10_21/language_data.csv")) {
  normalizePath(".", mustWork = FALSE)
} else {
  normalizePath(file.path(script_dir, "../.."), mustWork = FALSE)
}
adm_dir   <- file.path(repo_root, "analyses/admixture_links_2026_10_21")
src_dir   <- file.path(repo_root, "analyses/gelato_feasibility_2026_10_01/sources")
out_dir   <- Sys.getenv("RRW_OUT_DIR", script_dir)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "data"), recursive = TRUE, showWarnings = FALSE)

# --- 1. Load Bouckaert v2.0.0 tree and map the 4 Glottolog 5.3 split codes ----------------
nwk_file <- file.path(out_dir, "data/bouckaert_v2.nwk")
if (!file.exists(nwk_file)) {
  gz_candidates <- c(
    file.path(src_dir, "trees/v2.0.0_global-language-summary-tree-relabelled_cartoonised.tree.gz"),
    file.path(repo_root, "v2.0.0_global-language-summary-tree-relabelled_cartoonised.tree.gz")
  )
  gz_path <- gz_candidates[file.exists(gz_candidates)][1]
  if (is.na(gz_path)) {
    stop("Cannot find v2.0.0_global-language-summary-tree-relabelled_cartoonised.tree.gz to extract data/bouckaert_v2.nwk")
  }
  cat("Extracting data/bouckaert_v2.nwk from", gz_path, "...\n")
  con <- gzfile(gz_path, "rt")
  nex_lines <- suppressWarnings(readLines(con))
  close(con)
  tr_start <- grep("^\\s*Translate", nex_lines, ignore.case = TRUE)[1]
  tree_line_idx <- grep("tree STATE_1", nex_lines, fixed = TRUE)[1]
  trans_lines <- trimws(sub(",$", "", nex_lines[(tr_start + 1L):(tree_line_idx - 1L)]))
  trans_lines <- trans_lines[nzchar(trans_lines)]
  parts <- strsplit(trans_lines, "\\s+")
  id_map <- setNames(vapply(parts, function(x) substr(gsub("'", "", x[2]), 1L, 8L), ""),
                     vapply(parts, `[`, "", 1L))
  clean_nwk <- gsub("\\[&[^\\]]*\\]", "", nex_lines[tree_line_idx], perl = TRUE)
  state1_chunk <- strsplit(clean_nwk, ";", fixed = TRUE)[[1]][2]
  state1_nwk <- paste0(sub("^\\s*tree\\s+STATE_1\\s*=\\s*", "", state1_chunk, perl = TRUE), ";")
  m_ids <- gregexpr("(?<=[(,])[0-9]+(?=:)", state1_nwk, perl = TRUE)
  regmatches(state1_nwk, m_ids) <- list(unname(id_map[regmatches(state1_nwk, m_ids)[[1]]]))
  writeLines(state1_nwk, nwk_file)
}
stopifnot(file.exists(nwk_file))
global_tree <- read.tree(nwk_file)

# Verified 1-to-1 non-colliding mappings from Bouckaert v2.0.0 pre-split/sister tips to Glottolog 5.3:
#   osse1243 (Ossetian)   -> iron1242 (Iron Ossetian)
#   tuuu1240 (Tu/Monguor) -> minh1238 (Mangghuer)
#   ngon1269 (Ngoni)      -> tanz1241 (Tanzanian Ngoni)
#   turo1239 (Turoyo)     -> mlah1239 (Mlahso)
split_map <- c(
  osse1243 = "iron1242",
  tuuu1240 = "minh1238",
  ngon1269 = "tanz1241",
  turo1239 = "mlah1239"
)
for (old_c in names(split_map)) {
  idx <- which(global_tree$tip.label == old_c)
  stopifnot(length(idx) == 1L)
  global_tree$tip.label[idx] <- unname(split_map[old_c])
}

# --- 2. Load language_data.csv, populations.csv, and GeLaTo Ne / FST metadata -------------
lang_df <- read.csv(file.path(adm_dir, "language_data.csv"), stringsAsFactors = FALSE)
lang_df <- lang_df[lang_df$in_main == 1, ]
stopifnot(all(lang_df$glottocode %in% global_tree$tip.label))
cat("All", nrow(lang_df), "main-set Grambank languages matched in Bouckaert v2.0.0 tree (100%).\n")

pop_df <- read.csv(file.path(adm_dir, "populations.csv"), stringsAsFactors = FALSE)
pop_df <- pop_df[pop_df$in_main == 1, ]

gelato_raw <- read.csv(file.path(src_dir, "gelato_c625fdc/datasets/HumanOrigins_AutosomalSNP/data.csv"),
                       stringsAsFactors = FALSE, check.names = FALSE)
names(gelato_raw)[names(gelato_raw) == "PopName"] <- "population"
for (col in c("medianFST", "medianFSTregion", "MedianFSTAdjustedNeighbors", "harmonicMean")) {
  gelato_raw[[col]] <- suppressWarnings(as.numeric(gelato_raw[[col]]))
}
pop_df <- merge(pop_df, gelato_raw[, c("population", "medianFST", "medianFSTregion", "harmonicMean")],
                by = "population", all.x = TRUE)

# Aggregate GeLaTo demographic indicators to language level (weighted by n_individuals)
w_mean <- function(x, w) {
  ok <- !is.na(x) & !is.na(w) & w > 0
  if (!any(ok)) return(NA_real_)
  sum(x[ok] * w[ok]) / sum(w[ok])
}
demog_rows <- lapply(split(pop_df, pop_df$glottocode), function(df) {
  data.frame(
    glottocode       = df$glottocode[1],
    ne_harmonic_mean = w_mean(df$harmonicMean, df$n_individuals),
    ne_harmonic_min  = if (any(!is.na(df$harmonicMean))) min(df$harmonicMean, na.rm = TRUE) else NA_real_,
    median_fst_global = w_mean(df$medianFST, df$n_individuals),
    median_fst_region = w_mean(df$medianFSTregion, df$n_individuals),
    graff_target_any  = as.integer(any(df$graff_target == 1, na.rm = TRUE)),
    stringsAsFactors = FALSE
  )
})
demog_df <- do.call(rbind, demog_rows)
lang_df <- merge(lang_df, demog_df, by = "glottocode", all.x = TRUE)

# --- 3. Load Stage-1 Admixture Exposures (ent, sqrt_het, f3k_est_pos) ---------------------
e_main <- read.csv(file.path(adm_dir, "results/exposure_main.csv"), stringsAsFactors = FALSE)
e_tail <- read.csv(file.path(adm_dir, "results/exposure_main_tailcheck.csv"), stringsAsFactors = FALSE)
e_f3   <- read.csv(file.path(adm_dir, "results/exposure_main_f3check.csv"), stringsAsFactors = FALSE)
expo_all <- rbind(
  e_main[e_main$measure %in% c("ent", "het", "f3_est"), c("glottocode", "measure", "eta_mean", "eta_sd")],
  e_tail[e_tail$measure %in% c("sqrt_het", "f3k_est_pos"), c("glottocode", "measure", "eta_mean", "eta_sd")],
  e_f3[e_f3$measure %in% c("f3k_est"), c("glottocode", "measure", "eta_mean", "eta_sd")]
)
for (m in unique(expo_all$measure)) {
  sub <- expo_all[expo_all$measure == m, ]
  idx <- match(lang_df$glottocode, sub$glottocode)
  lang_df[[paste0("s1_", m, "_mean")]] <- sub$eta_mean[idx]
  lang_df[[paste0("s1_", m, "_sd")]]   <- sub$eta_sd[idx]
}

# --- 4. Construct ADMIXTURE Q-matrix ensemble (K = 12..30) per language -------------------
adm_q_dir <- file.path(src_dir, "zenodo_15263706/geneticAdmixture-linguisticDiffusion/input/MegaAdmixtureCatalogue/ADMIXTURE")
info_id <- read.csv(file.path(adm_q_dir, "GeneticInfoID.csv"), stringsAsFactors = FALSE)
info_id$population <- info_id$Population
stopifnot(all(pop_df$population %in% info_id$population))

ks <- 12:30
q_list <- lapply(ks, function(k) {
  qf <- file.path(adm_q_dir, sprintf("best_runs/GelatoHO_mergedSetMarchBEDnorelatives_pruned_autosomes_K%d.Q", k))
  m <- as.matrix(read.table(qf))
  m / rowSums(m)
})
# Concatenate K=12..30 scaled by 1/sqrt(length(ks)) so Euclidean distance = RMS distance across K
q_concat <- do.call(cbind, lapply(q_list, function(m) m / sqrt(length(ks))))
rownames(q_concat) <- info_id$population

# Language-level mean Q vector and individual-level SE per coordinate
set.seed(20261030)
lang_pops <- split(pop_df$population, pop_df$glottocode)
q_lang_mean <- matrix(NA_real_, nrow = nrow(lang_df), ncol = ncol(q_concat),
                      dimnames = list(lang_df$glottocode, NULL))
q_lang_se   <- matrix(NA_real_, nrow = nrow(lang_df), ncol = ncol(q_concat),
                      dimnames = list(lang_df$glottocode, NULL))

for (g in lang_df$glottocode) {
  pops_g <- lang_pops[[g]]
  ind_idx <- which(info_id$population %in% pops_g)
  q_sub <- q_concat[ind_idx, , drop = FALSE]
  q_lang_mean[g, ] <- colMeans(q_sub)
  if (nrow(q_sub) >= 2L) {
    q_lang_se[g, ] <- apply(q_sub, 2, sd) / sqrt(nrow(q_sub))
  } else {
    q_lang_se[g, ] <- NA_real_
  }
}
# Fill n=1 language SE with median per-individual SD across multi-individual languages
med_col_sd <- apply(q_lang_se, 2, median, na.rm = TRUE) * sqrt(median(lang_df$n_individuals[lang_df$n_individuals >= 2]))
for (g in lang_df$glottocode) {
  if (any(is.na(q_lang_se[g, ]))) {
    q_lang_se[g, ] <- med_col_sd / sqrt(max(1, lang_df$n_individuals[lang_df$glottocode == g]))
  }
}

# --- 5. Load Pairwise FST matrix (558 x 558) or precomputed FST embeddings ----------------
fst_candidates <- c(
  file.path(src_dir, "zenodo_15263706/geneticAdmixture-linguisticDiffusion/input/FST.csv"),
  file.path(out_dir, "data/FST.csv")
)
fst_file <- fst_candidates[file.exists(fst_candidates)][1]
fst_emb_file <- file.path(out_dir, "data/fst_embeddings.csv")
has_raw_fst <- !is.na(fst_file)

if (has_raw_fst) {
  fst_long <- read.csv(fst_file, stringsAsFactors = FALSE)
  all_pops <- sort(unique(c(fst_long$Pop1, fst_long$Pop2)))
  fst_pop <- matrix(0, nrow = length(all_pops), ncol = length(all_pops),
                    dimnames = list(all_pops, all_pops))
  i1 <- match(fst_long$Pop1, all_pops)
  i2 <- match(fst_long$Pop2, all_pops)
  v  <- pmax(0, as.numeric(fst_long$FST))
  fst_pop[cbind(i1, i2)] <- v
  fst_pop[cbind(i2, i1)] <- v

  # Language-by-language FST matrix (mean across population pairs, plus within-language SE)
  G <- nrow(lang_df)
  gcodes <- lang_df$glottocode
  fst_lang <- matrix(0, nrow = G, ncol = G, dimnames = list(gcodes, gcodes))
  fst_within_lang_sd <- setNames(rep(NA_real_, G), gcodes)

  for (a in seq_len(G)) {
    pa <- lang_pops[[gcodes[a]]]
    if (length(pa) >= 2L) {
      sub_aa <- fst_pop[pa, pa, drop = FALSE]
      fst_within_lang_sd[a] <- mean(sub_aa[upper.tri(sub_aa)], na.rm = TRUE)
    }
    for (b in a:G) {
      if (a == b) next
      pb <- lang_pops[[gcodes[b]]]
      d_ab <- mean(fst_pop[pa, pb], na.rm = TRUE)
      fst_lang[a, b] <- d_ab
      fst_lang[b, a] <- d_ab
    }
  }
  # 14 populations have 91 pairwise NAs among themselves in FST.csv while having valid FST to the other 544 populations;
  # fill any remaining NA language pair (a, b) using the closest proxy language a' (minimal FST to a with non-NA FST to b)
  na_idx <- which(is.na(fst_lang), arr.ind = TRUE)
  if (nrow(na_idx) > 0L) {
    cat("Filling", nrow(na_idx) / 2, "NA language pairs in fst_lang via nearest-neighbor FST proxy.\n")
    for (k in seq_len(nrow(na_idx))) {
      i <- na_idx[k, 1]; j <- na_idx[k, 2]
      if (i < j) {
        cand <- which(!is.na(fst_lang[i, ]) & !is.na(fst_lang[, j]) & seq_len(G) != i & seq_len(G) != j)
        best_c <- cand[which.min(fst_lang[i, cand])]
        val <- fst_lang[best_c, j]
        fst_lang[i, j] <- val
        fst_lang[j, i] <- val
      }
    }
  }
  stopifnot(!any(is.na(fst_lang)))

  # Global 3D PCoA of FST across the 202 cat12 main languages (supervisor's literal embedding)
  cat12_codes <- lang_df$glottocode[lang_df$n_coded >= 8]
  pcoa_glob <- cmdscale(as.dist(fst_lang[cat12_codes, cat12_codes]), k = 3, eig = TRUE)
  # Standardize each global axis to unit SD across the 202 languages
  glob_coords <- scale(pcoa_glob$points)
  for (d in 1:3) {
    lang_df[[paste0("fst_global_pc", d)]] <- NA_real_
    lang_df[[paste0("fst_global_pc", d, "_se")]] <- NA_real_
    idx <- match(cat12_codes, lang_df$glottocode)
    lang_df[[paste0("fst_global_pc", d)]][idx] <- glob_coords[, d]
    # Tip SE inversely proportional to sqrt(n_individuals), calibrated from multi-population languages
    lang_df[[paste0("fst_global_pc", d, "_se")]][idx] <- 0.15 / sqrt(pmax(1, lang_df$n_individuals[idx] / 10))
  }
} else {
  stopifnot(file.exists(fst_emb_file))
  fst_emb_cache <- read.csv(fst_emb_file, stringsAsFactors = FALSE)
  sub_g <- fst_emb_cache[fst_emb_cache$cutoff == "min3" & fst_emb_cache$outcome == "cat12", ]
  for (d in 1:3) {
    lang_df[[paste0("fst_global_pc", d)]] <- NA_real_
    lang_df[[paste0("fst_global_pc", d, "_se")]] <- NA_real_
    idx <- match(sub_g$glottocode, lang_df$glottocode)
    lang_df[[paste0("fst_global_pc", d)]][idx] <- sub_g[[paste0("fst_global_pc", d)]]
    lang_df[[paste0("fst_global_pc", d, "_se")]][idx] <- sub_g[[paste0("fst_global_pc", d, "_se")]]
  }
}

# --- 6. Helper to build within-family trees, incidence matrices, and within-family PCoAs ---
outcome_defs <- list(
  cat12   = list(pres = "n_present",       coded = "n_coded",       total = 12L, min_cov = 0.6),
  flex    = list(pres = "flex_present",    coded = "flex_coded",    total = 30L, min_cov = 0.6),
  irreg13 = list(pres = "irreg13_present", coded = "irreg13_coded", total = 13L, min_cov = 0.6)
)

build_dataset <- function(min_fam_size, out_name) {
  cut_label <- if (min_fam_size == 5L) "min5" else "min3"
  odef <- outcome_defs[[out_name]]
  # First identify families meeting min_fam_size in the primary cat12 202-language set
  # AND having at least 3 coded languages in the active outcome
  d_cat12 <- lang_df[lang_df$n_coded >= 8 & lang_df$family != "isolate", ]
  fam_counts_cat12 <- table(d_cat12$family)
  eligible_fams <- names(fam_counts_cat12[fam_counts_cat12 >= min_fam_size])

  ok_cov <- (lang_df[[odef$coded]] / odef$total) >= (odef$min_cov - 1e-9)
  d_sub <- lang_df[ok_cov & (lang_df$family %in% eligible_fams), ]
  fam_counts_out <- table(d_sub$family)
  keep_fams <- names(fam_counts_out[fam_counts_out >= min(3L, min_fam_size)])
  d_sub <- d_sub[d_sub$family %in% keep_fams, ]
  # Order families from largest to smallest, and tips in tree order within each family
  fam_order <- names(sort(table(d_sub$family), decreasing = TRUE))

  fam_list <- list()
  ordered_rows <- list()

  for (f_idx in seq_along(fam_order)) {
    fam_name <- fam_order[f_idx]
    df_f <- d_sub[d_sub$family == fam_name, ]
    tr_f <- keep.tip(global_tree, df_f$glottocode)
    tr_f <- collapse.singles(tr_f)
    # Put df_f in exact tr_f$tip.label order
    df_f <- df_f[match(tr_f$tip.label, df_f$glottocode), ]
    Tf <- Ntip(tr_f)
    Bf <- nrow(tr_f$edge)
    # Guard against any zero branch length
    tr_f$edge.length <- pmax(tr_f$edge.length, 1e-4)

    # Tip-by-branch path incidence matrix A (Tf x Bf)
    A_f <- matrix(0, nrow = Tf, ncol = Bf)
    paths <- nodepath(tr_f)
    for (i in seq_len(Tf)) {
      nodes_i <- paths[[i]]
      edges_i <- which(tr_f$edge[, 2] %in% nodes_i[-1])
      A_f[i, edges_i] <- 1
    }
    A_scaled_f <- sweep(A_f, 2, sqrt(tr_f$edge.length), `*`)
    A_time_f   <- sweep(A_f, 2, tr_f$edge.length, `*`)

    # Within-family PCA/PCoA of ADMIXTURE Q-vectors (K=12..30) in natural Q units
    k_max <- min(3L, Tf - 1L)
    q_f <- q_lang_mean[df_f$glottocode, , drop = FALSE]
    q_f_centered <- scale(q_f, center = TRUE, scale = FALSE)
    svd_q <- svd(q_f_centered)
    q_pcs_sc <- q_f_centered %*% svd_q$v[, seq_len(k_max), drop = FALSE]
    while (ncol(q_pcs_sc) < 3L) q_pcs_sc <- cbind(q_pcs_sc, NA_real_)

    # Propagate individual-level Q SE onto the within-family PCs (in natural Q units)
    q_se_f <- q_lang_se[df_f$glottocode, , drop = FALSE]
    q_pc_se <- sqrt((q_se_f^2) %*% (svd_q$v[, seq_len(k_max), drop = FALSE]^2))
    while (ncol(q_pc_se) < 3L) q_pc_se <- cbind(q_pc_se, NA_real_)

    if (has_raw_fst) {
      # Within-family PCoA of FST with Cailliez correction so all k_max = min(3, Tf-1) axes are Euclidean
      # Keep natural FST units across families (do NOT force every family to unit variance, which would
      # artificially inflate rates in young/homogeneous families); scale globally after looping over families.
      d_fst_f <- fst_lang[df_f$glottocode, df_f$glottocode]
      mds_res <- cmdscale(as.dist(d_fst_f), k = k_max, eig = TRUE, add = TRUE)
      mds_fst <- mds_res$points
      if (!is.matrix(mds_fst)) mds_fst <- matrix(mds_fst, ncol = 1)
      mds_fst_sc <- mds_fst
      while (ncol(mds_fst_sc) < 3L) mds_fst_sc <- cbind(mds_fst_sc, NA_real_)

      # Mean FST to linguistic sisters in the same family
      diag(d_fst_f) <- NA_real_
      df_f$mean_fst_sisters <- rowMeans(d_fst_f, na.rm = TRUE)

      for (d in 1:3) {
        df_f[[paste0("fst_within_pc", d)]]    <- mds_fst_sc[, d]
        # Within-language FST noise floor (calibrated from multi-population languages + 1/sqrt(n))
        df_f[[paste0("fst_within_pc", d, "_se")]] <- 0.005 / sqrt(pmax(1, df_f$n_individuals / 10))
      }
    } else {
      sub_e <- fst_emb_cache[fst_emb_cache$cutoff == cut_label & fst_emb_cache$outcome == out_name, ]
      m_idx <- match(df_f$glottocode, sub_e$glottocode)
      df_f$mean_fst_sisters <- sub_e$mean_fst_sisters[m_idx]
      for (d in 1:3) {
        df_f[[paste0("fst_global_pc", d)]]        <- sub_e[[paste0("fst_global_pc", d)]][m_idx]
        df_f[[paste0("fst_global_pc", d, "_se")]] <- sub_e[[paste0("fst_global_pc", d, "_se")]][m_idx]
        df_f[[paste0("fst_within_pc", d)]]        <- sub_e[[paste0("fst_within_pc", d)]][m_idx]
        df_f[[paste0("fst_within_pc", d, "_se")]] <- sub_e[[paste0("fst_within_pc", d, "_se")]][m_idx]
      }
    }

    for (d in 1:3) {
      df_f[[paste0("q_within_pc", d)]]      <- q_pcs_sc[, d]
      df_f[[paste0("q_within_pc", d, "_se")]] <- q_pc_se[, d]
    }
    df_f$y_pres  <- df_f[[odef$pres]]
    df_f$y_coded <- df_f[[odef$coded]]
    df_f$y_share <- df_f$y_pres / df_f$y_coded
    # Empirical logit for continuous contrast diagnostics
    df_f$y_elogit <- log((df_f$y_pres + 0.5) / (df_f$y_coded - df_f$y_pres + 0.5))
    df_f$family_idx <- f_idx

    fam_list[[fam_name]] <- list(
      family     = fam_name,
      family_idx = f_idx,
      tree       = tr_f,
      T_f        = Tf,
      B_f        = Bf,
      A          = A_f,
      A_scaled   = A_scaled_f,
      A_time     = A_time_f,
      edge_len   = tr_f$edge.length,
      crown_kyr  = max(node.depth.edgelength(tr_f)),
      data       = df_f
    )
    ordered_rows[[f_idx]] <- df_f
  }

  full_df <- do.call(rbind, ordered_rows)
  rownames(full_df) <- NULL
  # Apply a SINGLE global scale factor across all families so relative genetic distances between
  # homogeneous vs. heterogeneous families are preserved while keeping Stan inputs O(1)
  glob_q_sd <- sqrt(mean(as.matrix(full_df[, c("q_within_pc1", "q_within_pc2", "q_within_pc3")])^2, na.rm = TRUE))
  if (has_raw_fst) {
    glob_fst_sd <- sqrt(mean(as.matrix(full_df[, c("fst_within_pc1", "fst_within_pc2", "fst_within_pc3")])^2, na.rm = TRUE))
    for (d in 1:3) {
      full_df[[paste0("fst_within_pc", d)]]    <- full_df[[paste0("fst_within_pc", d)]] / glob_fst_sd
      full_df[[paste0("fst_within_pc", d, "_se")]] <- full_df[[paste0("fst_within_pc", d, "_se")]] / glob_fst_sd
    }
  }
  for (d in 1:3) {
    full_df[[paste0("q_within_pc", d)]]      <- full_df[[paste0("q_within_pc", d)]] / glob_q_sd
    full_df[[paste0("q_within_pc", d, "_se")]] <- pmax(0.02, full_df[[paste0("q_within_pc", d, "_se")]] / glob_q_sd)
  }
  for (f_idx in seq_along(fam_list)) {
    f_name <- fam_list[[f_idx]]$family
    fam_list[[f_idx]]$data <- full_df[full_df$family == f_name, ]
  }
  list(
    min_fam_size = min_fam_size,
    outcome      = out_name,
    n_families   = length(fam_list),
    n_languages  = nrow(full_df),
    n_branches   = sum(vapply(fam_list, `[[`, 0L, "B_f")),
    families     = fam_list,
    data         = full_df
  )
}

summary_rows <- list()
fst_emb_rows <- list()
fst_cols <- c("fst_global_pc1", "fst_global_pc1_se",
              "fst_global_pc2", "fst_global_pc2_se",
              "fst_global_pc3", "fst_global_pc3_se",
              "mean_fst_sisters",
              "fst_within_pc1", "fst_within_pc1_se",
              "fst_within_pc2", "fst_within_pc2_se",
              "fst_within_pc3", "fst_within_pc3_se")
for (cut_label in c("min5", "min3")) {
  mfs <- if (cut_label == "min5") 5L else 3L
  for (out_name in names(outcome_defs)) {
    ds <- build_dataset(mfs, out_name)
    saveRDS(ds, file.path(out_dir, "data", sprintf("dataset_%s_%s.rds", cut_label, out_name)))
    if (has_raw_fst) {
      emb_sub <- ds$data[, c("glottocode", fst_cols)]
      emb_sub$cutoff  <- cut_label
      emb_sub$outcome <- out_name
      fst_emb_rows[[length(fst_emb_rows) + 1L]] <- emb_sub[, c("cutoff", "outcome", "glottocode", fst_cols)]
    }
    summary_rows[[length(summary_rows) + 1L]] <- data.frame(
      cutoff      = cut_label,
      min_langs   = mfs,
      outcome     = out_name,
      n_families  = ds$n_families,
      n_languages = ds$n_languages,
      n_branches  = ds$n_branches,
      n_contrasts = ds$n_languages - ds$n_families,
      stringsAsFactors = FALSE
    )
    if (out_name == "cat12") {
      write.csv(ds$data, file.path(out_dir, "data", sprintf("languages_%s_cat12.csv", cut_label)), row.names = FALSE)
    }
  }
}
if (has_raw_fst && length(fst_emb_rows) > 0L) {
  fst_emb_df <- do.call(rbind, fst_emb_rows)
  for (col in fst_cols) {
    fst_emb_df[[col]] <- ifelse(is.na(fst_emb_df[[col]]), NA_character_, sprintf("%.17g", fst_emb_df[[col]]))
  }
  write.csv(fst_emb_df, fst_emb_file, row.names = FALSE)
}

sum_df <- do.call(rbind, summary_rows)
write.csv(sum_df, file.path(out_dir, "data/dataset_summary.csv"), row.names = FALSE)
print(sum_df, row.names = FALSE)

# Also print family-level breakdown for min3 cat12
ds_m3 <- readRDS(file.path(out_dir, "data/dataset_min3_cat12.rds"))
fam_tab <- do.call(rbind, lapply(ds_m3$families, function(f) {
  data.frame(
    family     = f$family,
    n_tips     = f$T_f,
    n_branches = f$B_f,
    n_contrasts = f$T_f - 1L,
    crown_kyr  = round(f$crown_kyr, 2),
    mean_edge_kyr = round(mean(f$edge_len), 2),
    stringsAsFactors = FALSE
  )
}))
write.csv(fam_tab, file.path(out_dir, "data/family_tree_summary.csv"), row.names = FALSE)
cat("\n=== Within-Family Tree Summary (Bouckaert v2.0.0, kyr) ===\n")
print(fam_tab, row.names = FALSE)
