// Stage 2: Within-Family Morphological Beta-Binomial Model with Coupled RRW & Directional Drift
// Models morphological counts y_pres | trials(y_coded) across F multi-language families.
// Each family f has its own root intercept mu_fam[f] (centred around global mu_0 with SD sd_fam).
// Supports 4 nested specifications via data flags (use_dir, use_rate):
//   1. clock (use_dir=0, use_rate=0): Strict clock Brownian covariance within each family
//   2. rate  (use_dir=0, use_rate=1): Branch variance scaled by exp(2 * gamma_rate * J_hat_b)
//   3. dir   (use_dir=1, use_rate=0): Cumulative branch jump drift beta_dir * z_jump + strict clock
//   4. both  (use_dir=1, use_rate=1): Both directional drift beta_dir and rate scaling gamma_rate

data {
  int<lower=1> T;                       // Total tips across all F families
  int<lower=1> B;                       // Total within-family branches across all F families
  int<lower=1> F;                       // Number of multi-language families
  int<lower=1> tip_start[F];            // Starting row in 1..T for family f
  int<lower=1> T_fam[F];                // Number of tips in family f
  int<lower=1> br_start[F];             // Starting column in 1..B for family f
  int<lower=1> B_fam[F];                // Number of branches in family f
  matrix[T, B] A_scaled;                // Block-diagonal root-to-tip incidence matrix weighted by sqrt(t_b)
  int<lower=0> y_pres[T];               // Number of present morphological features/categories
  int<lower=1> y_coded[T];              // Number of coded morphological features/categories
  vector[B] J_hat;                      // Standardized Stage-1 branch genetic jump estimates
  vector[T] z_jump;                     // Standardized cumulative root-to-tip genetic jump exposure
  int<lower=0, upper=1> use_dir;        // 1 = estimate directional slope beta_dir, 0 = fix to 0
  int<lower=0, upper=1> use_rate;       // 1 = estimate rate coupling gamma_rate, 0 = fix to 0
}

parameters {
  real mu_0;                            // Global mean intercept across families (logit scale)
  real<lower=0> sd_fam;                 // Between-family root SD
  vector[F] z_fam;                      // Standardized family root offsets
  real<lower=0> sigma_morph;            // Within-family morphological Brownian diffusion SD (per sqrt(kyr))
  vector[T] z_phylo;                    // Standardized within-family phylogenetic effects
  real<lower=0> phi;                    // Beta-binomial overdispersion concentration parameter
  vector[use_dir ? 1 : 0] beta_dir_raw;     // Directional drift slope per SD of cumulative genetic jump
  vector[use_rate ? 1 : 0] gamma_rate_raw;  // Rate coupling exponent on branch SD: sigma_b = sigma_morph * exp(gamma * J_hat_b)
}

transformed parameters {
  vector[F] mu_fam = mu_0 + sd_fam * z_fam;
  real beta_dir = use_dir ? beta_dir_raw[1] : 0.0;
  real gamma_rate = use_rate ? gamma_rate_raw[1] : 0.0;
  vector[T] eta;

  for (f in 1:F) {
    int ts = tip_start[f];
    int tf = T_fam[f];
    int bs = br_start[f];
    int bf = B_fam[f];
    matrix[tf, bf] A_f = block(A_scaled, ts, bs, tf, bf);
    vector[bf] j_f = segment(J_hat, bs, bf);
    vector[bf] s_f = sigma_morph * exp(gamma_rate * j_f);
    matrix[tf, bf] L_B = diag_post_multiply(A_f, s_f);
    matrix[tf, tf] C_f = add_diag(tcrossprod(L_B), 1e-5);
    matrix[tf, tf] L_f = cholesky_decompose(C_f);
    vector[tf] z_p = segment(z_phylo, ts, tf);
    vector[tf] z_j = segment(z_jump, ts, tf);

    eta[ts:(ts + tf - 1)] = rep_vector(mu_fam[f], tf) + beta_dir * z_j + L_f * z_p;
  }
}

model {
  // Same weakly-informative priors as analyses/admixture_links_2026_10_21/fit.R
  mu_0 ~ normal(0, 1.5);
  sd_fam ~ exponential(1);
  z_fam ~ std_normal();
  sigma_morph ~ exponential(1);
  z_phylo ~ std_normal();
  phi ~ gamma(2, 0.1);
  if (use_dir) beta_dir_raw ~ normal(0, 0.5);
  if (use_rate) gamma_rate_raw ~ normal(0, 0.5);

  for (i in 1:T) {
    real p_i = inv_logit(eta[i]);
    y_pres[i] ~ beta_binomial(y_coded[i], p_i * phi, (1.0 - p_i) * phi);
  }
}

generated quantities {
  vector[T] log_lik;
  for (i in 1:T) {
    real p_i = inv_logit(eta[i]);
    log_lik[i] = beta_binomial_lpmf(y_pres[i] | y_coded[i], p_i * phi, (1.0 - p_i) * phi);
  }
}
