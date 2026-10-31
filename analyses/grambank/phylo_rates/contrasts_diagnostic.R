# Component 1: Within-Family Phylogenetic Independent Contrasts (PIC) & Branch-Jump Diagnostic
#
# Runs two empirical tests across both family cutoffs (min5: 10 families, min3: 13 families)
# and all three morphological outcomes (cat12, flex, irreg13):
#   1. Demographic & Contact Audit of Genetic Branch Jumps:
#      Reconstructs within-family ancestral states (PIC/ML) and computes terminal branch jump rates
#      |Delta x_b| / sqrt(t_b) for all 6 genetic representations, testing correlations with:
#        - GeLaTo effective population size Ne (ne_harmonic_mean) -> tests bottleneck confounding
#        - Sample size (n_individuals) -> tests small-n noise inflation
#        - Graff et al. 2025 curated inter-family admixture targets (graff_target_any)
#        - Stage-1 admixture entropy (ent) and f3 (f3k_est_pos)
#   2. Within-Family Phylogenetic Independent Contrasts (Felsenstein 1985; Garland 1992):
#      Across all internal nodes within families:
#        - Rate Coupling Test:      |c_morph| vs. ||c_gen||_2 (Spearman rho, p-value, OLS slope)
#        - Directional Drift Test:  c_morph ~ 0 + c_gen (regression through the origin;
#                                   for 3D embeddings, c_gen is the contrast of Euclidean distance
#                                   from the family root/centroid so >0 means diverging from family)

suppressPackageStartupMessages({
  library(ape)
})

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
script_dir <- if (length(file_arg) > 0L) {
  dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE))
} else if (file.exists("gen_rrw.stan")) {
  normalizePath(".", mustWork = FALSE)
} else if (dir.exists("analyses/grambank/phylo_rates")) {
  normalizePath("analyses/grambank/phylo_rates", mustWork = FALSE)
} else {
  normalizePath(".", mustWork = FALSE)
}
out_dir <- Sys.getenv("RRW_OUT_DIR", script_dir)
res_dir <- file.path(out_dir, "results")
dir.create(res_dir, recursive = TRUE, showWarnings = FALSE)

gen_specs <- list(
  ent           = list(cols = "s1_ent_mean",                                           d = 1L, type = "scalar"),
  sqrt_het      = list(cols = "s1_sqrt_het_mean",                                      d = 1L, type = "scalar"),
  f3k_est_pos   = list(cols = "s1_f3k_est_pos_mean",                                   d = 1L, type = "scalar"),
  q_within_d3   = list(cols = c("q_within_pc1", "q_within_pc2", "q_within_pc3"),       d = 3L, type = "coord"),
  fst_within_d3 = list(cols = c("fst_within_pc1", "fst_within_pc2", "fst_within_pc3"), d = 3L, type = "coord"),
  fst_global_d3 = list(cols = c("fst_global_pc1", "fst_global_pc2", "fst_global_pc3"), d = 3L, type = "coord")
)

# Helper: compute exact Brownian/PIC nodal reconstructions and branch jump rates within one family tree
compute_family_jumps <- function(tr, mat_x) {
  # mat_x: Tf x D matrix in tr$tip.label order (drop columns that are all NA)
  keep_cols <- which(colSums(!is.na(mat_x)) == nrow(mat_x))
  mat_x <- mat_x[, keep_cols, drop = FALSE]
  Tf <- Ntip(tr)
  D  <- ncol(mat_x)
  rownames(mat_x) <- tr$tip.label
  # Nodal states (1:Tf are tips, (Tf+1):(2*Tf-1) are internal nodes via exact Brownian ML/GLS on tree)
  # Note: for any internal node u, we can use ape::ace(..., method="pic", CI=FALSE) on multi2di(tr)
  node_vals <- matrix(NA_real_, nrow = Tf + Nnode(tr), ncol = D)
  node_vals[seq_len(Tf), ] <- mat_x
  for (d in seq_len(D)) {
    if (sd(mat_x[, d]) < 1e-10) {
      node_vals[(Tf + 1):(Tf + Nnode(tr)), d] <- mean(mat_x[, d])
    } else {
      ac <- tryCatch(ace(mat_x[, d], tr, type = "continuous", method = "pic", CI = FALSE)$ace,
                     error = function(e) NULL)
      if (is.null(ac) || any(!is.finite(ac))) {
        # Exact analytical Brownian motion conditional expectation at internal nodes
        C_all <- vcvPhylo(tr, anc.nodes = TRUE)
        C_tt  <- C_all[seq_len(Tf), seq_len(Tf), drop = FALSE]
        C_nt  <- C_all[(Tf + 1):nrow(C_all), seq_len(Tf), drop = FALSE]
        one   <- rep(1, Tf)
        invC  <- solve(C_tt + diag(1e-8, Tf))
        mu_hat <- as.numeric(crossprod(one, invC %*% mat_x[, d]) / crossprod(one, invC %*% one))
        ac <- mu_hat + as.numeric(C_nt %*% (invC %*% (mat_x[, d] - mu_hat)))
      }
      node_vals[(Tf + 1):(Tf + Nnode(tr)), d] <- ac
    }
  }
  # For each edge b: parent -> child
  parent <- tr$edge[, 1]
  child  <- tr$edge[, 2]
  t_b    <- tr$edge.length
  delta  <- node_vals[child, , drop = FALSE] - node_vals[parent, , drop = FALSE]
  jump_l2 <- sqrt(rowSums(delta^2))
  rate_b  <- jump_l2 / sqrt(t_b)
  # Signed shift for scalar, or shift in distance from family root state for coordinates
  root_node <- Tf + 1L
  if (D == 1L) {
    signed_delta <- as.numeric(delta[, 1])
    tip_dev      <- as.numeric(mat_x[, 1] - node_vals[root_node, 1])
  } else {
    dist_from_root <- sqrt(rowSums(sweep(node_vals, 2, node_vals[root_node, ], `-`)^2))
    signed_delta   <- dist_from_root[child] - dist_from_root[parent]
    tip_dev        <- dist_from_root[seq_len(Tf)]
  }
  list(
    edge_parent  = parent,
    edge_child   = child,
    t_b          = t_b,
    jump_l2      = jump_l2,
    rate_b       = rate_b,
    signed_delta = signed_delta,
    tip_dev      = tip_dev
  )
}

# --- PART 1: Terminal Branch Jump Audit (Contact vs. Low-Ne Bottleneck Drift) -------------
ds_m3_cat12 <- readRDS(file.path(out_dir, "data/dataset_min3_cat12.rds"))

term_rows <- list()
for (f in ds_m3_cat12$families) {
  tr <- f$tree
  df <- f$data
  Tf <- f$T_f
  # Terminal edges leading to tips 1..Tf
  term_edge_idx <- match(seq_len(Tf), tr$edge[, 2])
  row_f <- df[, c("glottocode", "name", "family", "macroarea", "n_individuals", "n_populations",
                  "ne_harmonic_mean", "ne_harmonic_min", "median_fst_global", "median_fst_region",
                  "graff_target_any", "s1_ent_mean", "s1_sqrt_het_mean", "s1_f3k_est_pos_mean",
                  "y_pres", "y_coded", "y_share", "y_elogit")]
  row_f$term_branch_kyr <- tr$edge.length[term_edge_idx]

  for (gname in names(gen_specs)) {
    gs <- gen_specs[[gname]]
    mat_x <- as.matrix(df[, gs$cols, drop = FALSE])
    keep_cols <- which(colSums(!is.na(mat_x)) > 0)
    mat_x <- mat_x[, keep_cols, drop = FALSE]
    ok_tips <- which(rowSums(is.na(mat_x)) == 0)
    row_f[[paste0("rate_", gname)]] <- NA_real_
    row_f[[paste0("jump_", gname)]] <- NA_real_
    if (length(ok_tips) < 3L) next
    if (length(ok_tips) < Tf) {
      tr_sub <- collapse.singles(keep.tip(tr, df$glottocode[ok_tips]))
      ord_sub <- match(tr_sub$tip.label, df$glottocode)
      res_j <- compute_family_jumps(tr_sub, mat_x[ord_sub, , drop = FALSE])
      sub_term_idx <- match(seq_len(Ntip(tr_sub)), tr_sub$edge[, 2])
      row_f[[paste0("rate_", gname)]][ord_sub] <- res_j$rate_b[sub_term_idx]
      row_f[[paste0("jump_", gname)]][ord_sub] <- res_j$jump_l2[sub_term_idx]
    } else {
      res_j <- compute_family_jumps(tr, mat_x)
      row_f[[paste0("rate_", gname)]] <- res_j$rate_b[term_edge_idx]
      row_f[[paste0("jump_", gname)]] <- res_j$jump_l2[term_edge_idx]
    }
  }
  term_rows[[f$family]] <- row_f
}
term_df <- do.call(rbind, term_rows)
rownames(term_df) <- NULL
write.csv(term_df, file.path(res_dir, "terminal_branch_jumps_min3_cat12.csv"), row.names = FALSE)

# Correlate terminal branch jump rates with demographic & contact benchmarks
audit_rows <- list()
for (cut_label in c("min5", "min3")) {
  sub_t <- if (cut_label == "min5") term_df[term_df$family %in% names(readRDS(file.path(out_dir, "data/dataset_min5_cat12.rds"))$families), ] else term_df
  for (gname in names(gen_specs)) {
    r_col <- sub_t[[paste0("rate_", gname)]]
    cor_s <- function(v) {
      ok <- !is.na(r_col) & !is.na(v)
      if (sum(ok) < 5) return(c(rho = NA_real_, p = NA_real_))
      ct <- suppressWarnings(cor.test(r_col[ok], v[ok], method = "spearman"))
      c(rho = unname(ct$estimate), p = ct$p.value)
    }
    c_ne    <- cor_s(sub_t$ne_harmonic_mean)
    c_nind  <- cor_s(sub_t$n_individuals)
    c_ent   <- cor_s(sub_t$s1_ent_mean)
    c_f3    <- cor_s(sub_t$s1_f3k_est_pos_mean)
    c_graff <- cor_s(sub_t$graff_target_any)
    audit_rows[[length(audit_rows) + 1L]] <- data.frame(
      cutoff          = cut_label,
      genetic_measure = gname,
      n_langs         = sum(!is.na(r_col)),
      rho_Ne          = round(c_ne["rho"], 3),
      p_Ne            = signif(c_ne["p"], 3),
      rho_n_ind       = round(c_nind["rho"], 3),
      p_n_ind         = signif(c_nind["p"], 3),
      rho_ent         = round(c_ent["rho"], 3),
      p_ent           = signif(c_ent["p"], 3),
      rho_f3k_pos     = round(c_f3["rho"], 3),
      p_f3k_pos       = signif(c_f3["p"], 3),
      rho_graff_target = round(c_graff["rho"], 3),
      p_graff_target  = signif(c_graff["p"], 3),
      stringsAsFactors = FALSE
    )
  }
}
audit_df <- do.call(rbind, audit_rows)
write.csv(audit_df, file.path(res_dir, "jump_demographic_audit.csv"), row.names = FALSE)
cat("=== Terminal Branch Jump Rate Correlations with Ne (Bottlenecks) vs. Admixture Benchmarks ===\n")
print(audit_df, row.names = FALSE)

# Top 15 terminal branch jumps under Within-Family FST vs Global FST vs Q-within
for (gname in c("fst_within_d3", "fst_global_d3", "q_within_d3")) {
  ord <- order(term_df[[paste0("rate_", gname)]], decreasing = TRUE)
  top15 <- head(term_df[ord, c("glottocode", "name", "family", "n_individuals", "ne_harmonic_mean",
                               "s1_ent_mean", "s1_f3k_est_pos_mean", "graff_target_any",
                               "term_branch_kyr", paste0("rate_", gname), "y_share")], 15)
  write.csv(top15, file.path(res_dir, sprintf("top15_jumps_%s.csv", gname)), row.names = FALSE)
  cat(sprintf("\n=== Top 10 Terminal Branch Jump Rates: %s ===\n", gname))
  print(head(top15, 10), row.names = FALSE)
}

# --- PART 2: Within-Family Phylogenetic Independent Contrasts (PIC) -----------------------
pic_summary_rows <- list()
all_contrasts_list <- list()

for (cut_label in c("min5", "min3")) {
  for (out_name in c("cat12", "flex", "irreg13")) {
    ds <- readRDS(file.path(out_dir, "data", sprintf("dataset_%s_%s.rds", cut_label, out_name)))
    for (gname in names(gen_specs)) {
      gs <- gen_specs[[gname]]
      c_morph_elogit <- numeric(0)
      c_morph_share  <- numeric(0)
      c_gen_rate     <- numeric(0)
      c_gen_dir      <- numeric(0)
      c_fam          <- character(0)

      for (f in ds$families) {
        df <- f$data
        tr <- f$tree
        mat_x <- as.matrix(df[, gs$cols, drop = FALSE])
        keep_cols <- which(colSums(!is.na(mat_x)) > 0)
        mat_x <- mat_x[, keep_cols, drop = FALSE]
        # Some measures (sqrt_het, f3k_est_pos) have a few missing tips; prune family tree to non-NA tips
        ok_tips <- which(rowSums(is.na(mat_x)) == 0)
        if (length(ok_tips) < 3L) next
        if (length(ok_tips) < nrow(df)) {
          tr_sub <- collapse.singles(keep.tip(tr, df$glottocode[ok_tips]))
          df_sub <- df[match(tr_sub$tip.label, df$glottocode), ]
          mat_x  <- as.matrix(df_sub[, gs$cols[keep_cols], drop = FALSE])
        } else {
          tr_sub <- tr
          df_sub <- df
        }

        # Morphological contrasts (elogit and share)
        pic_m_el <- as.numeric(pic(setNames(df_sub$y_elogit, df_sub$glottocode), tr_sub))
        pic_m_sh <- as.numeric(pic(setNames(df_sub$y_share,  df_sub$glottocode), tr_sub))

        # Genetic contrasts:
        # 1) Multi-dimensional Euclidean contrast norm ||c_gen||_2 (for rate coupling test)
        pic_g_mat <- sapply(seq_len(ncol(mat_x)), function(d) {
          as.numeric(pic(setNames(mat_x[, d], df_sub$glottocode), tr_sub))
        })
        if (!is.matrix(pic_g_mat)) pic_g_mat <- matrix(pic_g_mat, ncol = ncol(mat_x))
        rate_g <- sqrt(rowSums(pic_g_mat^2))

        # 2) Signed directional contrast:
        #    For scalar admixture exposures (ent, sqrt_het, f3k_est_pos), c_gen itself.
        #    For 3D coordinate embeddings, contrast of tip distance from the family root/centroid
        #    (so positive contrast = the sister clade that diverged farther from the family profile).
        if (gs$type == "scalar") {
          dir_g <- as.numeric(pic_g_mat[, 1])
        } else {
          res_j <- compute_family_jumps(tr_sub, mat_x)
          dir_g <- as.numeric(pic(setNames(res_j$tip_dev, df_sub$glottocode), tr_sub))
        }

        c_morph_elogit <- c(c_morph_elogit, pic_m_el)
        c_morph_share  <- c(c_morph_share,  pic_m_sh)
        c_gen_rate     <- c(c_gen_rate,     rate_g)
        c_gen_dir      <- c(c_gen_dir,      dir_g)
        c_fam          <- c(c_fam,          rep(f$family, length(pic_m_el)))
      }

      # Test A: Rate coupling (|c_morph| vs ||c_gen||_2)
      ct_rate_el <- suppressWarnings(cor.test(abs(c_morph_elogit), c_gen_rate, method = "spearman"))
      ct_rate_sh <- suppressWarnings(cor.test(abs(c_morph_share),  c_gen_rate, method = "spearman"))
      lm_rate    <- lm(scale(abs(c_morph_elogit)) ~ scale(c_gen_rate))
      sm_rate    <- summary(lm_rate)$coefficients[2, ]

      # Test B: Directional simplification (c_morph ~ 0 + c_gen_dir, Felsenstein 1985 through origin)
      # Standardize by RMS contrast SD so slopes are comparable per SD of contrast
      sd_m_el <- sqrt(mean(c_morph_elogit^2))
      sd_g_dir <- sqrt(mean(c_gen_dir^2))
      m_std <- c_morph_elogit / sd_m_el
      g_std <- c_gen_dir / sd_g_dir
      lm_dir <- lm(m_std ~ 0 + g_std)
      sm_dir <- summary(lm_dir)$coefficients[1, ]
      # Also Spearman on signed contrasts (orienting c_gen_dir >= 0 by symmetry of contrasts)
      sgn <- ifelse(c_gen_dir >= 0, 1, -1)
      ct_dir <- suppressWarnings(cor.test(c_morph_elogit * sgn, abs(c_gen_dir), method = "spearman"))

      pic_summary_rows[[length(pic_summary_rows) + 1L]] <- data.frame(
        cutoff          = cut_label,
        outcome         = out_name,
        genetic_measure = gname,
        n_contrasts     = length(c_morph_elogit),
        n_families      = length(unique(c_fam)),
        rate_spearman_elogit = round(unname(ct_rate_el$estimate), 3),
        rate_p_elogit        = signif(ct_rate_el$p.value, 3),
        rate_spearman_share  = round(unname(ct_rate_sh$estimate), 3),
        rate_p_share         = signif(ct_rate_sh$p.value, 3),
        rate_slope_std       = round(sm_rate["Estimate"], 3),
        rate_se_std          = round(sm_rate["Std. Error"], 3),
        dir_slope_std        = round(sm_dir["Estimate"], 3),
        dir_se_std           = round(sm_dir["Std. Error"], 3),
        dir_t                = round(sm_dir["t value"], 2),
        dir_p                = signif(sm_dir["Pr(>|t|)"], 3),
        stringsAsFactors = FALSE
      )

      if (cut_label == "min5") {
        all_contrasts_list[[paste(out_name, gname, sep = "__")]] <- data.frame(
          outcome = out_name, genetic_measure = gname, family = c_fam,
          c_morph_elogit = c_morph_elogit, c_morph_share = c_morph_share,
          c_gen_rate = c_gen_rate, c_gen_dir = c_gen_dir, stringsAsFactors = FALSE
        )
      }
    }
  }
}

pic_df <- do.call(rbind, pic_summary_rows)
write.csv(pic_df, file.path(res_dir, "pic_contrasts_summary.csv"), row.names = FALSE)
write.csv(do.call(rbind, all_contrasts_list), file.path(res_dir, "pic_contrasts_min5_nodes.csv"), row.names = FALSE)

cat("\n=== Within-Family Phylogenetic Independent Contrasts Summary (min5, >= 5 langs/family) ===\n")
print(pic_df[pic_df$cutoff == "min5", c("outcome", "genetic_measure", "n_contrasts",
                                         "rate_spearman_elogit", "rate_p_elogit",
                                         "dir_slope_std", "dir_se_std", "dir_p")], row.names = FALSE)

cat("\n=== Within-Family Phylogenetic Independent Contrasts Summary (min3, >= 3 langs/family) ===\n")
print(pic_df[pic_df$cutoff == "min3", c("outcome", "genetic_measure", "n_contrasts",
                                         "rate_spearman_elogit", "rate_p_elogit",
                                         "dir_slope_std", "dir_se_std", "dir_p")], row.names = FALSE)
