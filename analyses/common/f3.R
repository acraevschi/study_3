# Population-level f3 admixture and f4 SI check across all linked GeLaTo populations.
#
# For each target population A in `analyses/common/data/analysis_links.csv` (main):
#   f3(A; B, C)
#     B: up to K_NEAR (default 5) language-deduplicated, Glottolog clade-stratified
#        (alternating deepest-basal and closest non-conspecific sister clades),
#        strictly non-conspecific sister populations from A's family (n_ind >= MIN_N_REF).
#     C: up to K_NEAR language-deduplicated, donor-family-stratified geographically
#        nearest different-family populations within MAX_KM (default 3000 km; n_ind >= MIN_N_REF).
#
# Conspecificity uses the full set of language-level Glottocodes across ALL `main` rows
# in `analysis_links.csv` plus non-family `glottocodeBase`, preventing dual-link leaks.
# Triples are ranked by conservative point estimate min_j(est_j + se_j), i.e.
# max_j(-est_j - se_j), with deterministic tie-breaking on (est, pop2, pop3).
# In a single F2 pass, writes `analyses/common/data/f3_fixed_scores.csv` with:
#   - Primary strict non-conspecific clade-stratified f3 (`f3_fixed = max(0, -est - se)`)
#   - Isolate sub-population fallback (`f3_fixed_iso`, used only when n_B_nonconspec == 0)
#   - All-triples non-conspecific benchmark (`f3_all_fixed`)
#   - Non-African f4(MbutiPygmy, C*; B*, A) SI check on the primary selected (B*, C*) triple
#
# Usage (from repository root):
#   Rscript analyses/common/f3.R [f2dir] [max_km] [k_near] [min_n_ref]

suppressPackageStartupMessages(library(admixtools))

args <- commandArgs(trailingOnly = TRUE)
f2dir_candidates <- c(
  if (length(args) >= 1L && nzchar(args[1])) args[1] else character(0),
  "external/zenodo_15263706/input/F2results",
  "external/zenodo_15263706/F2results"
)
f2dir <- f2dir_candidates[dir.exists(f2dir_candidates)][1]
if (is.na(f2dir) || !dir.exists(f2dir)) {
  stop("F2results directory not found. Pass path as first argument or populate external/zenodo_15263706/input/F2results.")
}
MAX_KM    <- if (length(args) >= 2L && nzchar(args[2])) as.numeric(args[2]) else 3000
K_NEAR    <- if (length(args) >= 3L && nzchar(args[3])) as.integer(args[3]) else 5L
MIN_N_REF <- if (length(args) >= 4L && nzchar(args[4])) as.integer(args[4]) else 5L

out_dir <- "analyses/common/data"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
src <- "analyses/gelato_feasibility_2026_10_01/sources"
adm_src <- file.path(src, "zenodo_15263706/geneticAdmixture-linguisticDiffusion/input")

m <- read.csv(file.path(adm_src, "GeLaTo-population-glottocode-mapping.csv"), stringsAsFactors = FALSE)
g <- read.csv(file.path(src, "gelato_c625fdc/cldf/populations.csv"), stringsAsFactors = FALSE)
gid <- read.csv(file.path(adm_src, "MegaAdmixtureCatalogue/ADMIXTURE/GeneticInfoID.csv"), stringsAsFactors = FALSE)
glot <- read.csv("external/glottolog-cldf/cldf/languages.csv", stringsAsFactors = FALSE)
gvals <- read.csv("external/glottolog-cldf/cldf/values.csv", stringsAsFactors = FALSE)
gvals <- gvals[gvals$Parameter_ID == "classification", c("Language_ID", "Value")]

n_ind_tab <- table(gid$Population)
pops_all <- setdiff(
  sub("\\.DS_Store", "", list.files(f2dir)),
  c("", grep("^block_lengths", list.files(f2dir), value = TRUE))
)

al_path <- if (file.exists(file.path(out_dir, "analysis_links.csv"))) {
  file.path(out_dir, "analysis_links.csv")
} else {
  "outputs/pcfp_v2_links_2026_10_20/typology/analysis_links.csv"
}
al <- read.csv(al_path, stringsAsFactors = FALSE)
al_main <- al[al$analysis == "main", ]

# Primary 1-to-1 glottocode per population (non-proxy takes precedence) for metadata/paths
al_main$is_proxy_bool <- tolower(as.character(al_main$is_proxy)) == "true"
al_main_ord <- al_main[order(al_main$population, al_main$is_proxy_bool, al_main$glottocode), ]
al_dedup <- al_main_ord[!duplicated(al_main_ord$population), ]
targets <- sort(unique(al_dedup$population))

meta <- m[m$PopName %in% pops_all, c("PopName", "lat", "lon", "glottologFamily", "glottocodeBase")]
names(meta) <- c("pop", "lat", "lon", "family", "glottocode_base")
meta$family[is.na(meta$family)] <- ""
meta$glottocode_base[is.na(meta$glottocode_base)] <- ""
meta$n_ind <- as.integer(n_ind_tab[meta$pop])
meta$n_ind[is.na(meta$n_ind)] <- 0L
stopifnot(all(targets %in% meta$pop))

al_gc_idx <- match(meta$pop, al_dedup$population)
meta$glottocode_main <- ifelse(!is.na(al_gc_idx), al_dedup$glottocode[al_gc_idx], "")
meta$eff_gc <- ifelse(nzchar(meta$glottocode_main), meta$glottocode_main, meta$glottocode_base)

gl_lang_id <- setNames(ifelse(nzchar(glot$Language_ID), glot$Language_ID, glot$ID), glot$ID)
gl_level   <- setNames(glot$Level, glot$ID)
gl_macro   <- setNames(glot$Macroarea, glot$ID)
cls_map    <- setNames(gvals$Value, gvals$Language_ID)

resolve_lang_id <- function(gc) {
  out <- unname(gl_lang_id[gc])
  ifelse(is.na(out) | !nzchar(out), gc, out)
}
meta$lang_id <- resolve_lang_id(meta$eff_gc)
meta$base_is_family <- unname(gl_level[meta$glottocode_base]) == "family"
meta$base_is_family[is.na(meta$base_is_family)] <- FALSE
meta$base_lang_id <- ifelse(meta$base_is_family, "", resolve_lang_id(meta$glottocode_base))
meta$macroarea <- unname(gl_macro[meta$lang_id])
meta$macroarea[is.na(meta$macroarea)] <- ""

# Full conspecific language-ID set per population across ALL main rows in analysis_links.csv
# plus non-family glottocodeBase (closes multi-glottocode proxy leaks such as Taa_North/West)
al_gcs_by_pop <- split(al_main$glottocode, al_main$population)
lang_sets_by_pop <- setNames(lapply(seq_len(nrow(meta)), function(i) {
  p <- meta$pop[i]
  gcs <- c(al_gcs_by_pop[[p]], if (nzchar(meta$base_lang_id[i])) meta$base_lang_id[i] else character(0))
  gcs <- unique(gcs[nzchar(gcs)])
  lids <- resolve_lang_id(gcs)
  unique(lids[nzchar(lids)])
}), meta$pop)

is_conspecific_vec <- function(pop_a, cand_pops) {
  la <- lang_sets_by_pop[[pop_a]]
  if (!length(la)) return(rep(FALSE, length(cand_pops)))
  vapply(cand_pops, function(pb) any(lang_sets_by_pop[[pb]] %in% la), logical(1))
}

# Coordinate fallback hierarchy: graff_mapping -> gelato -> glottolog_base_language -> glottolog_linked_language
meta$coord_source <- ifelse(is.na(meta$lat) | is.na(meta$lon), NA_character_, "graff_mapping")
fb <- g[match(meta$pop, g$Name), c("Latitude", "Longitude")]
need <- is.na(meta$coord_source) & !is.na(fb$Latitude) & !is.na(fb$Longitude)
meta$lat[need] <- fb$Latitude[need]
meta$lon[need] <- fb$Longitude[need]
meta$coord_source[need] <- "gelato"

use_glottolog_coords <- function(codes, rows, label) {
  i <- match(codes, glot$ID)
  lat <- suppressWarnings(as.numeric(glot$Latitude[i]))
  lon <- suppressWarnings(as.numeric(glot$Longitude[i]))
  ok <- rows & !is.na(lat) & !is.na(lon)
  meta$lat[ok] <<- lat[ok]
  meta$lon[ok] <<- lon[ok]
  meta$coord_source[ok] <<- label
}
use_glottolog_coords(meta$glottocode_base, is.na(meta$coord_source), "glottolog_base_language")
use_glottolog_coords(meta$glottocode_main, is.na(meta$coord_source) & meta$pop %in% targets, "glottolog_linked_language")

message(
  "Targets: ", length(targets), " | coordinate sources: ",
  paste(names(table(meta$coord_source)), table(meta$coord_source), collapse = ", "),
  "; missing coords among targets: ", sum(is.na(meta$coord_source) & meta$pop %in% targets)
)

# Precompute Glottolog classification ancestor paths for clade stratification
get_path <- function(lid, raw_gc) {
  p <- unname(cls_map[lid])
  if (is.na(p) || !nzchar(p)) p <- unname(cls_map[raw_gc])
  toks <- if (!is.na(p) && nzchar(p)) strsplit(p, "/", fixed = TRUE)[[1]] else character(0)
  toks <- toks[nzchar(toks)]
  if (nzchar(lid)) c(toks, lid) else toks
}
paths_by_pop <- setNames(
  lapply(seq_len(nrow(meta)), function(i) get_path(meta$lang_id[i], meta$eff_gc[i])),
  meta$pop
)

hav <- function(lat1, lon1, lat2, lon2) {
  r <- pi / 180
  a <- sin((lat2 - lat1) * r / 2)^2 + cos(lat1 * r) * cos(lat2 * r) * sin((lon2 - lon1) * r / 2)^2
  12742 * asin(pmin(1, sqrt(a)))
}

select_clade_stratified_B <- function(pop_a, cand_pops, k) {
  if (!length(cand_pops)) return(character(0))
  pa <- paths_by_pop[[pop_a]]
  idx_m <- match(cand_pops, meta$pop)
  lids <- meta$lang_id[idx_m]
  lids[!nzchar(lids)] <- cand_pops[!nzchar(lids)]
  n_ind_c <- meta$n_ind[idx_m]

  # Step 1: Deduplicate to 1 representative population per distinct lang_id (highest n_ind, tie-break pop)
  ord_dedup <- order(lids, -n_ind_c, cand_pops)
  c_ord <- cand_pops[ord_dedup]
  l_ord <- lids[ord_dedup]
  is_prim <- !duplicated(l_ord)
  prim_pops <- c_ord[is_prim]
  extra_pops <- c_ord[!is_prim]

  # Step 2: Compute LCA depth, patristic distance, and clade key for language-deduplicated candidates
  n_p <- length(prim_pops)
  p_idx <- match(prim_pops, meta$pop)
  lca_depth <- integer(n_p)
  pdist <- integer(n_p)
  bkey <- character(n_p)
  p_nind <- meta$n_ind[p_idx]

  for (j in seq_len(n_p)) {
    pb <- paths_by_pop[[prim_pops[j]]]
    ml <- min(length(pa), length(pb))
    d <- 0L
    if (ml > 0L) {
      for (t in seq_len(ml)) {
        if (pa[t] == pb[t]) d <- t else break
      }
    }
    lca_depth[j] <- d
    pdist[j] <- (length(pa) - d) + (length(pb) - d)
    bkey[j] <- if (d < length(pb)) pb[d + 1L] else meta$lang_id[p_idx[j]]
    if (!nzchar(bkey[j])) bkey[j] <- prim_pops[j]
  }

  # Step 3: Order distinct clades deepest-to-closest, then alternate deepest basal & closest sister clades
  df_c <- data.frame(
    pop = prim_pops, clade = paste0(lca_depth, "::", bkey),
    bkey = bkey, lca = lca_depth, pdist = pdist, n_ind = p_nind,
    stringsAsFactors = FALSE
  )
  clade_ids <- unique(df_c$clade)
  clade_meta <- do.call(rbind, lapply(clade_ids, function(cid) {
    sub <- df_c[df_c$clade == cid, , drop = FALSE]
    data.frame(
      clade = cid, bkey = sub$bkey[1L],
      min_lca = min(sub$lca), max_pdist = max(sub$pdist),
      stringsAsFactors = FALSE
    )
  }))
  clade_meta <- clade_meta[order(clade_meta$min_lca, -clade_meta$max_pdist, clade_meta$bkey), ]

  n_cl <- nrow(clade_meta)
  alt_idx <- integer(0)
  lo <- 1L
  hi <- n_cl
  while (lo <= hi) {
    alt_idx <- c(alt_idx, lo)
    if (hi > lo) alt_idx <- c(alt_idx, hi)
    lo <- lo + 1L
    hi <- hi - 1L
  }
  clade_order <- clade_meta$clade[alt_idx]

  by_clade <- lapply(clade_order, function(cid) {
    sub <- df_c[df_c$clade == cid, , drop = FALSE]
    sub <- sub[order(-sub$n_ind, sub$pop), ]
    sub$pop
  })

  chosen <- character(0)
  rnd <- 1L
  while (length(chosen) < k) {
    added <- FALSE
    for (cl in by_clade) {
      if (rnd <= length(cl)) {
        chosen <- c(chosen, cl[rnd])
        added <- TRUE
        if (length(chosen) == k) break
      }
    }
    if (!added) break
    rnd <- rnd + 1L
  }

  # Fallback only if fewer than k distinct languages exist in the family
  if (length(chosen) < k && length(extra_pops)) {
    ex_idx <- match(extra_pops, meta$pop)
    ex_ord <- extra_pops[order(-meta$n_ind[ex_idx], extra_pops)]
    chosen <- c(chosen, head(ex_ord, k - length(chosen)))
  }
  chosen
}

select_nearest_C <- function(lat_a, lon_a, cand_pops, k) {
  if (!length(cand_pops)) return(character(0))
  idx_m <- match(cand_pops, meta$pop)
  dk <- hav(lat_a, lon_a, meta$lat[idx_m], meta$lon[idx_m])
  lids <- meta$lang_id[idx_m]
  lids[!nzchar(lids)] <- cand_pops[!nzchar(lids)]
  n_ind_c <- meta$n_ind[idx_m]
  fams <- meta$family[idx_m]

  # Step 1: Deduplicate to 1 geographically nearest population per distinct lang_id
  ord_all <- order(dk, -n_ind_c, cand_pops)
  c_ord <- cand_pops[ord_all]
  d_ord <- dk[ord_all]
  l_ord <- lids[ord_all]
  n_ord <- n_ind_c[ord_all]
  f_ord <- fams[ord_all]

  is_prim <- !duplicated(l_ord)
  p_pops <- c_ord[is_prim]
  p_dk   <- d_ord[is_prim]
  p_nind <- n_ord[is_prim]
  p_fam  <- f_ord[is_prim]
  extra_pops <- c_ord[!is_prim]

  # Step 2: Stratify round-robin across distinct donor families (ordered by min distance to A)
  df_c <- data.frame(pop = p_pops, dk = p_dk, n_ind = p_nind, fam = p_fam, stringsAsFactors = FALSE)
  ufams <- unique(df_c$fam)
  fam_meta <- do.call(rbind, lapply(ufams, function(fm) {
    sub <- df_c[df_c$fam == fm, , drop = FALSE]
    data.frame(fam = fm, min_dk = min(sub$dk), max_n = max(sub$n_ind), stringsAsFactors = FALSE)
  }))
  fam_meta <- fam_meta[order(fam_meta$min_dk, -fam_meta$max_n, fam_meta$fam), ]

  by_fam <- lapply(fam_meta$fam, function(fm) {
    sub <- df_c[df_c$fam == fm, , drop = FALSE]
    sub <- sub[order(sub$dk, -sub$n_ind, sub$pop), ]
    sub$pop
  })

  chosen <- character(0)
  rnd <- 1L
  while (length(chosen) < k) {
    added <- FALSE
    for (fl in by_fam) {
      if (rnd <= length(fl)) {
        chosen <- c(chosen, fl[rnd])
        added <- TRUE
        if (length(chosen) == k) break
      }
    }
    if (!added) break
    rnd <- rnd + 1L
  }

  # Fallback only if fewer than k distinct neighbor languages exist within MAX_KM
  if (length(chosen) < k && length(extra_pops)) {
    chosen <- c(chosen, head(extra_pops, k - length(chosen)))
  }
  chosen
}

pick_best_triple <- function(r_df) {
  r_df <- r_df[is.finite(r_df$est) & is.finite(r_df$se) & is.finite(r_df$z), , drop = FALSE]
  if (!nrow(r_df)) return(NULL)
  score <- r_df$est + r_df$se
  ord <- order(score, r_df$est, r_df$pop2, r_df$pop3)
  best <- r_df[ord[1L], , drop = FALSE]
  list(
    n_triples = nrow(r_df),
    n_z_lt_m3 = sum(r_df$z < -3),
    pop2 = as.character(best$pop2[1L]),
    pop3 = as.character(best$pop3[1L]),
    est = as.numeric(best$est[1L]),
    se = as.numeric(best$se[1L]),
    z = as.numeric(best$z[1L])
  )
}

t0 <- Sys.time()
f2 <- f2_from_precomp(f2dir, pops = meta$pop, verbose = FALSE)
message(
  "F2 blocks loaded: ", paste(dim(f2), collapse = " x "), " in ",
  round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1), " min"
)

OUTGROUP_F4 <- "MbutiPygmy"
has_outgroup <- OUTGROUP_F4 %in% meta$pop
if (!has_outgroup) warning("Outgroup ", OUTGROUP_F4, " not present in F2 blocks; f4 columns will be NA.")

is_pop_african <- function(pop_vec) {
  vapply(pop_vec, function(p) {
    if (is.na(p) || !nzchar(p)) return(FALSE)
    if (p == OUTGROUP_F4) return(TRUE)
    idx <- match(p, meta$pop)
    if (is.na(idx)) return(FALSE)
    mac <- meta$macroarea[idx]
    if (identical(mac, "Africa")) return(TRUE)
    grepl("Africa", mac, fixed = TRUE) &&
      !is.na(meta$lat[idx]) && !is.na(meta$lon[idx]) &&
      meta$lat[idx] <= 37.5 && meta$lon[idx] >= -20.0 && meta$lon[idx] <= 34.0
  }, logical(1))
}

out <- vector("list", length(targets))
for (ti in seq_along(targets)) {
  a <- targets[ti]
  ma <- meta[meta$pop == a, , drop = FALSE]

  row <- data.frame(
    population = a,
    glottocode = ma$eff_gc,
    family = ma$family,
    macroarea = ma$macroarea,
    n_ind = ma$n_ind,
    coord_source = ma$coord_source,
    n_B_avail = 0L,
    n_C_avail = 0L,
    n_B = 0L,
    n_C = 0L,
    n_triples = 0L,
    n_z_lt_m3 = 0L,
    best_pop2 = NA_character_,
    best_pop3 = NA_character_,
    best_est = NA_real_,
    best_se = NA_real_,
    best_z = NA_real_,
    f3_est_neg = NA_real_,
    f3_pen = NA_real_,
    f3_fixed = NA_real_,
    f3_pos = NA_real_,
    iso_fallback_used = 0L,
    iso_n_B = 0L,
    iso_best_pop2 = NA_character_,
    iso_best_pop3 = NA_character_,
    iso_best_est = NA_real_,
    iso_best_se = NA_real_,
    iso_best_z = NA_real_,
    f3_fixed_iso = NA_real_,
    all_n_triples = 0L,
    all_best_pop2 = NA_character_,
    all_best_pop3 = NA_character_,
    all_best_est = NA_real_,
    all_best_se = NA_real_,
    all_best_z = NA_real_,
    f3_all_fixed = NA_real_,
    f4_outgroup = NA_character_,
    f4_est = NA_real_,
    f4_se = NA_real_,
    f4_z = NA_real_,
    max_km = MAX_KM,
    k_near = K_NEAR,
    min_n_ref = MIN_N_REF,
    stringsAsFactors = FALSE
  )

  if (nzchar(ma$family) && !is.na(ma$lat) && !is.na(ma$lon)) {
    # Non-conspecific same-family candidate B (n_ind >= MIN_N_REF) using full multi-link lang_set
    same_fam <- (meta$family == ma$family) & (meta$pop != a) & (meta$n_ind >= MIN_N_REF)
    same_fam_pops <- meta$pop[same_fam]
    B_all <- same_fam_pops[!is_conspecific_vec(a, same_fam_pops)]

    # Different-family candidate C within MAX_KM (n_ind >= MIN_N_REF)
    d_all <- hav(ma$lat, ma$lon, meta$lat, meta$lon)
    ok_C <- nzchar(meta$family) & (meta$family != ma$family) &
      (meta$n_ind >= MIN_N_REF) & is.finite(d_all) & (d_all <= MAX_KM)
    C_all <- meta$pop[ok_C]

    row$n_B_avail <- length(B_all)
    row$n_C_avail <- length(C_all)

    B_sel <- select_clade_stratified_B(a, B_all, K_NEAR)
    C_sel <- select_nearest_C(ma$lat, ma$lon, C_all, K_NEAR)
    row$n_B <- length(B_sel)
    row$n_C <- length(C_sel)

    # 1. Primary strict non-conspecific clade-stratified f3
    if (length(B_sel) && length(C_sel)) {
      r_prim <- suppressWarnings(f3(f2, pop1 = a, pop2 = B_sel, pop3 = C_sel, verbose = FALSE))
      bp <- pick_best_triple(r_prim)
      if (!is.null(bp)) {
        row$n_triples <- bp$n_triples
        row$n_z_lt_m3 <- bp$n_z_lt_m3
        row$best_pop2 <- bp$pop2
        row$best_pop3 <- bp$pop3
        row$best_est  <- bp$est
        row$best_se   <- bp$se
        row$best_z    <- bp$z
        row$f3_est_neg <- -bp$est
        row$f3_pen     <- -bp$est - bp$se
        row$f3_fixed   <- max(0, -bp$est - bp$se)
        row$f3_pos     <- max(0, -bp$est)

        # Copy primary into iso columns when non-conspecific B is available
        row$iso_n_B       <- row$n_B
        row$iso_best_pop2 <- bp$pop2
        row$iso_best_pop3 <- bp$pop3
        row$iso_best_est  <- bp$est
        row$iso_best_se   <- bp$se
        row$iso_best_z    <- bp$z
        row$f3_fixed_iso  <- row$f3_fixed

        # 4. Non-African f4(MbutiPygmy, C*; B*, A) SI check
        is_african <- any(is_pop_african(c(a, bp$pop2, bp$pop3)))
        if (has_outgroup && !is_african) {
          r_f4 <- suppressWarnings(
            f4(f2, pop1 = OUTGROUP_F4, pop2 = bp$pop3, pop3 = bp$pop2, pop4 = a, verbose = FALSE)
          )
          if (nrow(r_f4) >= 1L && is.finite(r_f4$est[1L])) {
            row$f4_outgroup <- OUTGROUP_F4
            row$f4_est <- as.numeric(r_f4$est[1L])
            row$f4_se  <- as.numeric(r_f4$se[1L])
            row$f4_z   <- as.numeric(r_f4$z[1L])
          }
        }
      }
    } else if (!length(B_sel) && length(C_sel)) {
      # 2. Isolate sub-population fallback (ONLY when n_B_nonconspec == 0)
      if (length(same_fam_pops)) {
        B_iso_sel <- head(same_fam_pops[order(-meta$n_ind[match(same_fam_pops, meta$pop)], same_fam_pops)], K_NEAR)
        row$iso_fallback_used <- 1L
        row$iso_n_B <- length(B_iso_sel)
        r_iso <- suppressWarnings(f3(f2, pop1 = a, pop2 = B_iso_sel, pop3 = C_sel, verbose = FALSE))
        bi <- pick_best_triple(r_iso)
        if (!is.null(bi)) {
          row$iso_best_pop2 <- bi$pop2
          row$iso_best_pop3 <- bi$pop3
          row$iso_best_est  <- bi$est
          row$iso_best_se   <- bi$se
          row$iso_best_z    <- bi$z
          row$f3_fixed_iso  <- max(0, -bi$est - bi$se)
        }
      }
    }

    # 3. All-triples non-conspecific benchmark
    if (length(B_all) && length(C_all)) {
      r_all <- suppressWarnings(f3(f2, pop1 = a, pop2 = B_all, pop3 = C_all, verbose = FALSE))
      ba <- pick_best_triple(r_all)
      if (!is.null(ba)) {
        row$all_n_triples <- ba$n_triples
        row$all_best_pop2 <- ba$pop2
        row$all_best_pop3 <- ba$pop3
        row$all_best_est  <- ba$est
        row$all_best_se   <- ba$se
        row$all_best_z    <- ba$z
        row$f3_all_fixed  <- max(0, -ba$est - ba$se)
      }
    }
  }

  out[[ti]] <- row
  if (ti %% 50L == 0L || ti == length(targets)) {
    message(format(Sys.time(), "%H:%M"), " ", ti, "/", length(targets))
  }
}

res <- do.call(rbind, out)
out_csv <- file.path(out_dir, "f3_fixed_scores.csv")
write.csv(res, out_csv, row.names = FALSE)
message(
  "Done: ", sum(res$n_triples > 0L), " primary + ",
  sum(res$iso_fallback_used == 1L & !is.na(res$f3_fixed_iso)),
  " isolate-fallback of ", nrow(res), " targets written to ", out_csv
)
