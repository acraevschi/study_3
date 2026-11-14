# Visual QC helper for Study 3 Track 2 figures.
# Generates 2x2 contact sheets (Greyscale, Protanopia, Deuteranopia, Box-blur squint test)
# and verifies font-size floor (>= 7.0 pt at 6.5 in), ink share, and colour share.
# Usage: Rscript run_qc.R <png_path> <source_width_in> <smallest_pt> <qc_out_dir>

suppressMessages(library(png))

args <- commandArgs(TRUE)
if (length(args) < 4L) {
  stop("Usage: Rscript run_qc.R <png_path> <source_width_in> <smallest_pt> <qc_out_dir>")
}

file_path <- args[1]
source_width_in <- as.numeric(args[2])
smallest_pt <- as.numeric(args[3])
out_dir <- args[4]
final_width_in <- 6.5
floor_pt <- 7.0

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
img <- readPNG(file_path)[, , 1:3]
H <- dim(img)[1]; W <- dim(img)[2]

# Downsample to 150 dpi at final_width_in for realistic print/screen preview
final_w <- round(final_width_in * 150)
final_h <- round(H * final_w / W)
yi <- round(seq(1, H, length.out = final_h))
xi <- round(seq(1, W, length.out = final_w))
fin <- img[yi, xi, ]

bg <- apply(matrix(fin, ncol = 3), 2, median)
diff_bg <- pmax(abs(fin[, , 1] - bg[1]), abs(fin[, , 2] - bg[2]), abs(fin[, , 3] - bg[3]))
ink_share <- mean(diff_bg > 0.08)
sat <- pmax(fin[, , 1], fin[, , 2], fin[, , 3]) - pmin(fin[, , 1], fin[, , 2], fin[, , 3])
col_share <- mean(sat > 0.25)

scale_fac <- final_width_in / source_width_in
final_pt <- smallest_pt * scale_fac
flag <- if (final_pt < floor_pt) "BELOW FLOOR" else "ok"

stem <- sub("\\.png$", "", basename(file_path))
final_path <- file.path(out_dir, paste0(stem, "_final.png"))
qc_path <- file.path(out_dir, paste0(stem, "_qc.png"))
writePNG(fin, final_path)

# CVD simulation (Machado et al. 2009)
to_lin <- function(c) ifelse(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055)^2.4)
to_srgb <- function(c) {
  c <- pmin(pmax(c, 0), 1)
  ifelse(c <= 0.0031308, c * 12.92, 1.055 * c^(1 / 2.4) - 0.055)
}
sim_cvd <- function(rgb, mat) {
  lin <- to_lin(rgb)
  out <- array(0, dim = dim(rgb))
  for (i in 1:3) out[, , i] <- lin[, , 1] * mat[i, 1] + lin[, , 2] * mat[i, 2] + lin[, , 3] * mat[i, 3]
  to_srgb(out)
}
P_mat <- rbind(c(0.152286, 1.052583, -0.204868), c(0.114503, 0.786281, 0.099216), c(-0.003882, -0.048116, 1.051998))
D_mat <- rbind(c(0.367322, 0.860646, -0.227968), c(0.280085, 0.672501, 0.047413), c(-0.011820, 0.042940, 0.968881))

grey_val <- 0.2126 * fin[, , 1] + 0.7152 * fin[, , 2] + 0.0722 * fin[, , 3]
grey_img <- array(rep(grey_val, 3), dim = dim(fin))
prot_img <- sim_cvd(fin, P_mat)
deut_img <- sim_cvd(fin, D_mat)

# Box blur squint test
r <- 4L
blur_img <- fin
for (ch in 1:3) {
  mat <- fin[, , ch]
  acc <- matrix(0, final_h, final_w)
  cnt <- 0L
  for (dy in c(-r, 0L, r)) {
    for (dx in c(-r, 0L, r)) {
      yy <- pmin(pmax(seq_len(final_h) + dy, 1L), final_h)
      xx <- pmin(pmax(seq_len(final_w) + dx, 1L), final_w)
      acc <- acc + mat[yy, xx]
      cnt <- cnt + 1L
    }
  }
  blur_img[, , ch] <- acc / cnt
}

sheet <- array(1, dim = c(final_h * 2, final_w * 2, 3))
sheet[1:final_h, 1:final_w, ] <- grey_img
sheet[1:final_h, (final_w + 1):(2 * final_w), ] <- prot_img
sheet[(final_h + 1):(2 * final_h), 1:final_w, ] <- deut_img
sheet[(final_h + 1):(2 * final_h), (final_w + 1):(2 * final_w), ] <- blur_img
writePNG(sheet, qc_path)

cat(sprintf("[%s] final=%dx%d | ink=%.1f%% | colour=%.1f%% | smallest=%.1f pt x %.2f = %.1f pt (%s)\n",
            stem, final_w, final_h, ink_share * 100, col_share * 100,
            smallest_pt, scale_fac, final_pt, flag))
