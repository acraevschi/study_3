# Forest plot of the admixture slopes (logit scale, per SD; graff: target vs not).
# Run after fit_models.R: Rscript analyses/draft_admixture_model_2026_10_10/plot_slopes.R
# or after fit_hier.R:     Rscript analyses/draft_admixture_model_2026_10_10/plot_slopes.R hier
suppressPackageStartupMessages(library(ggplot2))
here <- "analyses/draft_admixture_model_2026_10_10"
hier <- identical(commandArgs(TRUE)[1], "hier")
r <- read.csv(file.path(here, if (hier) "hier_model_results.csv" else "model_results_raw.csv"))
r <- r[!is.na(r$b_mean), ]
lab <- c(N = "x only", P = "+ phylogeny", S = "+ space", F = "+ phylogeny + space")
r$controls <- factor(lab[r$ctrl], levels = rev(lab))
r$measure <- factor(r$measure, levels = c("ent", "ent_me", "pop_ent", "nonmax", "ent_K12", "ent_K30", "graff", "neigh", "support", "amount"))
p <- ggplot(r, aes(x = b_mean, y = controls)) +
  geom_vline(xintercept = 0, colour = "grey50") +
  geom_errorbarh(aes(xmin = b_q025, xmax = b_q975), height = 0, colour = "#2b6cb0") +
  geom_errorbarh(aes(xmin = b_q05, xmax = b_q95), height = 0, linewidth = 1.2, colour = "#2b6cb0") +
  geom_point(size = 2, colour = "#1a365d") +
  facet_wrap(~ measure, ncol = 2) +
  labs(x = "Slope on logit(share of inflectional categories present); 90% and 95% intervals",
       y = NULL, title = "Draft: admixture measure -> Grambank inflectional extent",
       subtitle = paste0(if (hier) "Hierarchical exposure with measurement error. " else "",
                         "Negative = more admixture, fewer categories")) +
  theme_minimal(base_size = 10)
ggsave(file.path(here, if (hier) "slopes_hier.png" else "slopes.png"), p, width = 8, height = if (hier) 9 else 7, dpi = 150)
