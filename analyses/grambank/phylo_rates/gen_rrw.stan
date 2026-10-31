// Stage 1: Within-Family Genetic Relaxed Random Walk (RRW)
// Models D genetic coordinates across F independent language family trees (no global root stems).
// Each branch b in family f has diffusion SD sigma_b = sigma_0 * exp(J_b), where J_b ~ Normal(0, tau_J).
// Marginal tip covariance within family f is:
//   C_tree_f = (A_scaled_f * diag(sigma_b)) * (A_scaled_f * diag(sigma_b))'
//   C_obs_f,d = C_tree_f + diag(X_se[f,d]^2 + sigma_tip_extra^2)

data {
  int<lower=1> T;                       // Total tips across all F families
  int<lower=1> B;                       // Total within-family branches across all F families
  int<lower=1> F;                       // Number of multi-language families
  int<lower=1> D;                       // Maximum number of genetic dimensions (1 or 3)
  int<lower=1> tip_start[F];            // Starting row in 1..T for family f
  int<lower=1> T_fam[F];                // Number of tips in family f
  int<lower=1> br_start[F];             // Starting column in 1..B for family f
  int<lower=1> B_fam[F];                // Number of branches in family f
  int<lower=1, upper=D> D_fam[F];       // Active dimensions in family f: min(D, T_fam[f] - 1)
  matrix[T, B] A_scaled;                // Block-diagonal root-to-tip incidence matrix weighted by sqrt(t_b)
  matrix[T, D] X;                       // Genetic tip coordinates (0-padded where d > D_fam[f])
  matrix<lower=0>[T, D] X_se;           // Known tip measurement-error SDs
}

parameters {
  vector[B] J_raw;                      // Standardized log-rate shift per branch
  real<lower=0> tau_J;                  // SD of log-rate shifts across branches
  real<lower=0> sigma_0;                // Baseline genetic clock diffusion SD (per sqrt(kyr))
  real<lower=0> sigma_tip_extra;        // Residual tip-level SD beyond known X_se
  matrix[F, D] mu_fam;                  // Family-specific root states per dimension
}

transformed parameters {
  vector[B] J = tau_J * J_raw;
  vector<lower=0>[B] sigma_branch = sigma_0 * exp(J);
}

model {
  J_raw ~ std_normal();
  tau_J ~ normal(0, 0.75);
  sigma_0 ~ normal(0, 1.5);
  sigma_tip_extra ~ normal(0, 0.5);
  to_vector(mu_fam) ~ normal(0, 2.0);

  for (f in 1:F) {
    int ts = tip_start[f];
    int tf = T_fam[f];
    int bs = br_start[f];
    int bf = B_fam[f];
    matrix[tf, bf] A_f = block(A_scaled, ts, bs, tf, bf);
    vector[bf] s_f = segment(sigma_branch, bs, bf);
    matrix[tf, bf] L_B = diag_post_multiply(A_f, s_f);
    matrix[tf, tf] C_tree = tcrossprod(L_B);

    for (d in 1:D_fam[f]) {
      vector[tf] x_fd = segment( col(X, d), ts, tf );
      vector[tf] se_fd = segment( col(X_se, d), ts, tf );
      matrix[tf, tf] C_fd = C_tree;
      for (i in 1:tf) {
        C_fd[i, i] += square(se_fd[i]) + square(sigma_tip_extra) + 1e-6;
      }
      x_fd ~ multi_normal_cholesky(rep_vector(mu_fam[f, d], tf), cholesky_decompose(C_fd));
    }
  }
}

generated quantities {
  vector<lower=0>[B] rate_mult = exp(J);
}
