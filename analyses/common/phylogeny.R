# Shared phylogenetic correlation matrix and tree utilities for Grambank and LDL/Wug tracks.
#
# Provides:
#   - build_grafen_corr(tree_or_nwk, glottocodes): prunes a Glottolog topology tree to
#     `glottocodes`, collapses singles, computes Grafen branch lengths, and returns
#     `ape::vcv(tree, corr = TRUE)`.
#   - load_bouckaert_tree(nwk_path, gz_path, apply_split_map): loads the time-calibrated
#     Bouckaert v2.0.0 global language tree (extracting STATE_1 from the Nexus .tree.gz
#     archive if needed) and applies the 4 verified Glottolog 5.3 split mappings.
#   - build_bouckaert_corr(glottocodes, families, tree, nwk_path, within_family, crown_only):
#     prunes the Bouckaert v2.0.0 tree to `glottocodes` and returns the time-calibrated
#     correlation matrix (global, block-diagonal crown-MRCA per family when
#     `within_family = TRUE, crown_only = TRUE`, or block-diagonal global-stem per family
#     when `within_family = TRUE, crown_only = FALSE`).

suppressPackageStartupMessages(library(ape))

# Verified 1-to-1 non-colliding mappings from Bouckaert v2.0.0 pre-split/sister tips to Glottolog 5.3:
#   osse1243 (Ossetian)   -> iron1242 (Iron Ossetian)
#   tuuu1240 (Tu/Monguor) -> minh1238 (Mangghuer)
#   ngon1269 (Ngoni)      -> tanz1241 (Tanzanian Ngoni)
#   turo1239 (Turoyo)     -> mlah1239 (Mlahso)
BOUCKAERT_SPLIT_MAP <- c(
  osse1243 = "iron1242",
  tuuu1240 = "minh1238",
  ngon1269 = "tanz1241",
  turo1239 = "mlah1239"
)

build_grafen_corr <- function(tree_or_nwk, glottocodes) {
  raw_tree <- if (inherits(tree_or_nwk, "phylo")) {
    tree_or_nwk
  } else {
    read.tree(as.character(tree_or_nwk))
  }
  pruned <- collapse.singles(keep.tip(raw_tree, glottocodes))
  tree <- compute.brlen(pruned, method = "Grafen")
  A <- vcv(tree, corr = TRUE)
  stopifnot(all(glottocodes %in% rownames(A)))
  A
}

load_bouckaert_tree <- function(nwk_path = NULL, gz_path = NULL, apply_split_map = TRUE) {
  if (is.null(nwk_path)) {
    cand_nwk <- c(
      "analyses/common/data/bouckaert_v2.nwk",
      "../../common/data/bouckaert_v2.nwk",
      "../common/data/bouckaert_v2.nwk",
      "data/bouckaert_v2.nwk"
    )
    found <- cand_nwk[file.exists(cand_nwk)]
    nwk_path <- if (length(found)) found[1] else "analyses/common/data/bouckaert_v2.nwk"
  }
  if (!file.exists(nwk_path)) {
    if (is.null(gz_path)) {
      gz_candidates <- c(
        "analyses/gelato_feasibility_2026_10_01/sources/trees/v2.0.0_global-language-summary-tree-relabelled_cartoonised.tree.gz",
        "../../gelato_feasibility_2026_10_01/sources/trees/v2.0.0_global-language-summary-tree-relabelled_cartoonised.tree.gz",
        "v2.0.0_global-language-summary-tree-relabelled_cartoonised.tree.gz"
      )
      gz_path <- gz_candidates[file.exists(gz_candidates)][1]
    }
    if (is.na(gz_path) || !file.exists(gz_path)) {
      stop("Cannot find Bouckaert v2.0.0 tree at ", nwk_path)
    }
    dir.create(dirname(nwk_path), recursive = TRUE, showWarnings = FALSE)
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
    writeLines(state1_nwk, nwk_path)
  }
  tr <- read.tree(nwk_path)
  if (isTRUE(apply_split_map)) {
    for (old_c in names(BOUCKAERT_SPLIT_MAP)) {
      idx <- which(tr$tip.label == old_c)
      if (length(idx) == 1L && !(BOUCKAERT_SPLIT_MAP[old_c] %in% tr$tip.label)) {
        tr$tip.label[idx] <- unname(BOUCKAERT_SPLIT_MAP[old_c])
      }
    }
  }
  tr
}

build_bouckaert_corr <- function(glottocodes, families = NULL, tree = NULL,
                                 nwk_path = NULL, within_family = FALSE,
                                 crown_only = TRUE, scale_by_age = FALSE) {
  if (is.null(tree)) {
    tree <- load_bouckaert_tree(nwk_path = nwk_path, apply_split_map = TRUE)
  }
  glottocodes <- as.character(glottocodes)
  stopifnot(all(glottocodes %in% tree$tip.label))

  if (!isTRUE(within_family)) {
    pruned <- collapse.singles(keep.tip(tree, glottocodes))
    pruned$edge.length <- pmax(pruned$edge.length, 1e-4)
    A <- vcv(pruned, corr = !isTRUE(scale_by_age))
    if (isTRUE(scale_by_age)) {
      A <- A / mean(diag(A))
    }
    stopifnot(all(glottocodes %in% rownames(A)))
    return(A[glottocodes, glottocodes, drop = FALSE])
  }

  if (is.null(families)) {
    stop("`families` must be provided when `within_family = TRUE`")
  }
  stopifnot(length(families) == length(glottocodes))
  fam_chr <- as.character(families)
  fam_eff <- ifelse(is.na(fam_chr) | !nzchar(fam_chr) | fam_chr == "isolate",
                    paste0("isolate__", glottocodes), fam_chr)

  if (!isTRUE(crown_only)) {
    # Block-diagonal global-stem Bouckaert correlation matrix:
    # zeroes out cross-family correlations while retaining each family's root-to-crown stem + kyr crown topology
    pruned <- collapse.singles(keep.tip(tree, glottocodes))
    pruned$edge.length <- pmax(pruned$edge.length, 1e-4)
    A_glob <- vcv(pruned, corr = TRUE)[glottocodes, glottocodes, drop = FALSE]
    same_fam <- outer(fam_eff, fam_eff, "==")
    A_stem <- A_glob * same_fam
    dimnames(A_stem) <- list(glottocodes, glottocodes)
    return(A_stem)
  }

  # Block-diagonal crown-MRCA Bouckaert matrix per multi-language family (1.0 diagonal for singletons)
  N <- length(glottocodes)
  A <- diag(1.0, nrow = N, ncol = N)
  dimnames(A) <- list(glottocodes, glottocodes)

  # Precompute mean crown age (in kyr) across multi-language families when `scale_by_age = TRUE`
  mean_crown_kyr <- 1.0
  if (isTRUE(scale_by_age)) {
    crown_ages <- numeric(0)
    for (f in unique(fam_eff)) {
      idx <- which(fam_eff == f)
      if (length(idx) >= 2L) {
        tr_f <- collapse.singles(keep.tip(tree, glottocodes[idx]))
        tr_f$edge.length <- pmax(tr_f$edge.length, 1e-4)
        crown_ages <- c(crown_ages, max(node.depth.edgelength(tr_f)))
      }
    }
    if (length(crown_ages)) mean_crown_kyr <- mean(crown_ages)
  }

  for (f in unique(fam_eff)) {
    idx <- which(fam_eff == f)
    if (length(idx) >= 2L) {
      gf <- glottocodes[idx]
      tr_f <- collapse.singles(keep.tip(tree, gf))
      tr_f$edge.length <- pmax(tr_f$edge.length, 1e-4)
      if (isTRUE(scale_by_age)) {
        A_f <- (vcv(tr_f, corr = FALSE) / mean_crown_kyr)[gf, gf, drop = FALSE]
      } else {
        A_f <- vcv(tr_f, corr = TRUE)[gf, gf, drop = FALSE]
      }
      eig_min <- min(eigen(A_f, symmetric = TRUE, only.values = TRUE)$values)
      if (eig_min < 1e-6) {
        A_f <- A_f + diag(1e-5, nrow = length(gf))
      }
      A[gf, gf] <- A_f
    }
  }
  A
}
