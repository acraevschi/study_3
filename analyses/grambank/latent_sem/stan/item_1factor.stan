// 1-Factor Item-Level Morphological Models (M0_1pl, M1_1pl, MA0_2pl, MA_2pl, MB_1pl, MB_2pl)
//
// Operates on the N_lang x K_item binary observation grid (missing '?' cells omitted):
//   - M0_1pl:  with_x = 0, use_2pl = 0, use_var_slopes = 0
//   - M1_1pl:  with_x = 1, use_2pl = 0, use_var_slopes = 0  (equal weights lambda_k = 1)
//   - MA0_2pl: with_x = 0, use_2pl = 1, use_var_slopes = 0  (2PL baseline without x)
//   - MA_2pl:  with_x = 1, use_2pl = 1, use_var_slopes = 0  (2PL IRT / MIMIC: learned lambda_k)
//   - MB_1pl:  with_x = 1, use_2pl = 0, use_var_slopes = 1  (varying slopes on 1PL baseline)
//   - MB_2pl:  with_x = 1, use_2pl = 1, use_var_slopes = 1  (varying slopes on 2PL baseline)

functions {
  // Isometric zero-sum basis matrix Q in R^(K x (K-1)):
  // Columns are orthogonal Helmert vectors scaled by sqrt(K / (K - 1)) so that for
  // z ~ N(0, I_{K-1}), delta = Q * z satisfies sum(delta) = 0 and Var(delta_k) = 1 for all k,
  // with equicorrelated off-diagonals -1/(K-1) (strictly invariant to item ordering).
  matrix make_zero_sum_basis(int K) {
    matrix[K, K - 1] Q = rep_matrix(0.0, K, K - 1);
    real scale = sqrt(1.0 * K / (K - 1.0));
    for (j in 1:(K - 1)) {
      real denom = sqrt(1.0 * j * (j + 1.0));
      for (i in 1:j) {
        Q[i, j] = scale / denom;
      }
      Q[j + 1, j] = -scale * j / denom;
    }
    return Q;
  }

  // Squared-exponential Gaussian Process matching brms::gp(sx, sy, sz)
  vector gp_Iso(data array[] vector x, real sdgp, real lscale, vector zgp) {
    int N = size(x);
    matrix[N, N] cov = gp_exp_quad_cov(x, sdgp, lscale);
    for (n in 1:N) {
      cov[n, n] += 1e-12;
    }
    return cholesky_decompose(cov) * zgp;
  }

  // Numerical integration helper for PSIS-LOO (Golub-Welsch Gauss-Hermite rule):
  // Conditional and marginal LOO target the same predictive density p(y_i | y_{-i}), but when
  // a language is held out its unit-specific random shock z_lang[i] reverts from a narrow
  // posterior to its Normal(0, 1) prior, causing high PSIS importance-sampling variance
  // (Pareto-k > 0.7 and finite-sample bias). Integrating out the Normal(0, 1) shocks in
  // `generated quantities` stabilizes PSIS-LOO using M deterministic nodes (col 1: `gh_q`) and
  // log-weights (col 2: `gh_log_w`, exact for polynomials up to degree 2M - 1), computed from
  // the eigenvalues and eigenvectors of the M x M Golub-Welsch tridiagonal matrix.
  matrix make_gauss_hermite(int M) {
    matrix[M, M] J = rep_matrix(0.0, M, M);
    for (m in 1:(M - 1)) {
      real s = sqrt(1.0 * m);
      J[m, m + 1] = s;
      J[m + 1, m] = s;
    }
    vector[M] nodes = eigenvalues_sym(J);
    matrix[M, M] V = eigenvectors_sym(J);
    matrix[M, 2] out;
    out[, 1] = nodes;
    for (m in 1:M) {
      out[m, 2] = 2.0 * log(abs(V[1, m]));
    }
    return out;
  }
}

data {
  int<lower=1> N_lang;
  int<lower=2> K_item;
  int<lower=2> D_dom;
  int<lower=0> T_test;
  int<lower=1> N_obs;

  array[N_obs] int<lower=1, upper=N_lang> lang_id;
  array[N_obs] int<lower=1, upper=K_item> item_id;
  array[N_obs] int<lower=0, upper=1> Y;

  array[K_item] int<lower=1, upper=D_dom> item_dom;
  array[K_item] int<lower=0, upper=T_test> item_testlet;
  array[K_item] int<lower=0, upper=1> item_is_agree;

  vector[N_lang] x;

  int<lower=0, upper=1> with_x;
  int<lower=0, upper=1> use_2pl;
  int<lower=0, upper=1> use_var_slopes;
  int<lower=0, upper=1> use_phylo;
  int<lower=0, upper=1> use_space;
  int<lower=0, upper=1> use_testlets;

  matrix[N_lang, N_lang] L_phylo;
  array[N_lang] vector[3] Xgp;
  real<lower=0> lscale_prior_shape;
  real<lower=0> lscale_prior_scale;
}

transformed data {
  matrix[K_item, K_item - 1] Q_zf = make_zero_sum_basis(K_item);
  matrix[D_dom, D_dom - 1] Q_zf_dom = make_zero_sum_basis(D_dom);
  int has_testlets = (use_testlets == 1 && T_test > 0) ? 1 : 0;
  int n_agree = sum(item_is_agree);
  int n_inherent = K_item - n_agree;

  // Count observed items per (language, testlet) for exact LOO marginalization
  array[N_lang, T_test > 0 ? T_test : 1] int lang_testlet_n = rep_array(0, N_lang, T_test > 0 ? T_test : 1);
  // Dense lookup of observed outcomes per (language, item): -1 if missing, 0 or 1 if observed
  array[N_lang, K_item] int Y_mat = rep_array(-1, N_lang, K_item);

  // 9-point Gauss-Hermite grid (nodes `gh_q` and log-weights `gh_log_w`) used only in
  // `generated quantities` to average over unknown Normal(0, 1) random effects for LOO-CV.
  int M_gh = 9;
  matrix[M_gh, 2] gh = make_gauss_hermite(M_gh);
  vector[M_gh] gh_q = gh[, 1];
  vector[M_gh] gh_log_w = gh[, 2];

  for (n in 1:N_obs) {
    int i = lang_id[n];
    int k = item_id[n];
    int t = item_testlet[k];
    Y_mat[i, k] = Y[n];
    if (t > 0) {
      lang_testlet_n[i, t] += 1;
    }
  }
}

parameters {
  // Hierarchical item easiness intercepts: alpha_k = mu_alpha + sd_alpha * z_alpha[k]
  real mu_alpha;
  real<lower=0> sd_alpha;
  vector[K_item] z_alpha;

  // Main admixture slope (mean item slope across all K_item categories)
  vector[with_x ? 1 : 0] b_x_vec;

  // 2PL item discrimination weights: arithmetic mean = 1 via K_item * softmax(delta)
  vector<lower=0>[use_2pl ? 1 : 0] sd_lambda;
  vector[use_2pl ? K_item - 1 : 0] z_lambda;

  // Architecture B: zero-sum varying admixture slopes by domain and multi-item category
  vector<lower=0>[use_var_slopes ? 1 : 0] sd_dom_b;
  vector[use_var_slopes ? D_dom - 1 : 0] z_dom_b;
  vector<lower=0>[use_var_slopes ? 1 : 0] sd_item_b;
  vector[use_var_slopes ? K_item : 0] z_item_b;

  // Language-level baseline random effects (residual + phylogeny + spatial GP)
  real<lower=0> sd_lang;
  vector[N_lang] z_lang;

  vector<lower=0>[use_phylo ? 1 : 0] sd_phylo;
  vector[use_phylo ? N_lang : 0] z_phylo;

  vector<lower=0>[use_space ? 1 : 0] sd_gp;
  vector<lower=0>[use_space ? 1 : 0] lscale_gp;
  vector[use_space ? N_lang : 0] z_gp;

  // Within-language domain testlet random effects u_{i, testlet(k)}
  vector<lower=0>[has_testlets ? 1 : 0] sd_testlet;
  matrix[has_testlets ? N_lang : 0, has_testlets ? T_test : 0] z_testlet;
}

transformed parameters {
  vector[K_item] alpha = mu_alpha + sd_alpha * z_alpha;
  vector[K_item] lambda = rep_vector(1.0, K_item);
  real b_x = with_x ? b_x_vec[1] : 0.0;
  vector[D_dom] beta_domain = rep_vector(b_x, D_dom);
  vector[K_item] beta_item = rep_vector(b_x, K_item);
  vector[N_lang] theta_struct = rep_vector(0.0, N_lang);
  vector[N_lang] theta_0;
  vector[N_lang] theta;

  // Arithmetic-mean-1 normalization: lambda_k > 0 and mean(lambda) == 1.0 on every draw
  if (use_2pl) {
    lambda = K_item * softmax(sd_lambda[1] * (Q_zf * z_lambda));
  }

  if (use_phylo) {
    theta_struct += sd_phylo[1] * (L_phylo * z_phylo);
  }
  if (use_space) {
    theta_struct += gp_Iso(Xgp, sd_gp[1], lscale_gp[1], z_gp);
  }
  theta_0 = theta_struct + sd_lang * z_lang;

  if (use_var_slopes) {
    vector[D_dom] dom_off = sd_dom_b[1] * (Q_zf_dom * z_dom_b);
    vector[K_item] raw_b;
    theta = theta_0;
    for (k in 1:K_item) {
      // Only add item-within-domain slope offset when domain has >= 2 items
      real item_off = (item_testlet[k] > 0) ? (sd_item_b[1] * z_item_b[k]) : 0.0;
      raw_b[k] = dom_off[item_dom[k]] + item_off;
    }
    // Center raw_b so mean(beta_item) == b_x identically on every draw
    raw_b = raw_b - mean(raw_b);
    beta_item = b_x + raw_b;
    for (d in 1:D_dom) {
      beta_domain[d] = b_x + dom_off[d];
    }
  } else {
    theta = b_x * x + theta_0;
    beta_item = lambda * b_x;
  }
}

model {
  // Priors (matching fit.R scales)
  mu_alpha ~ normal(0, 1.5);
  sd_alpha ~ exponential(1);
  z_alpha ~ std_normal();

  if (with_x) {
    b_x_vec ~ normal(0, 0.5);
  }

  if (use_2pl) {
    sd_lambda ~ normal(0, 0.7);
    z_lambda ~ std_normal();
  }

  if (use_var_slopes) {
    sd_dom_b ~ exponential(2);
    z_dom_b ~ std_normal();
    sd_item_b ~ exponential(2);
    z_item_b ~ std_normal();
  }

  sd_lang ~ exponential(1);
  z_lang ~ std_normal();

  if (use_phylo) {
    sd_phylo ~ exponential(1);
    z_phylo ~ std_normal();
  }

  if (use_space) {
    sd_gp ~ exponential(1);
    lscale_gp ~ inv_gamma(lscale_prior_shape, lscale_prior_scale);
    z_gp ~ std_normal();
  }

  if (has_testlets) {
    sd_testlet ~ exponential(1);
    to_vector(z_testlet) ~ std_normal();
  }

  // Likelihood over observed (language, item) cells
  {
    vector[N_obs] eta;
    for (n in 1:N_obs) {
      int i = lang_id[n];
      int k = item_id[n];
      int t = item_testlet[k];
      real u_test = (has_testlets && t > 0) ? (sd_testlet[1] * z_testlet[i, t]) : 0.0;
      if (use_var_slopes) {
        eta[n] = alpha[k] + beta_item[k] * x[i] + lambda[k] * theta_0[i] + u_test;
      } else {
        eta[n] = alpha[k] + lambda[k] * theta[i] + u_test;
      }
    }
    Y ~ bernoulli_logit(eta);
  }
}

generated quantities {
  vector[N_obs] log_lik_obs;
  vector[N_lang] log_lik_lang;
  real b_inherent = 0.0;
  real b_agreement = 0.0;
  real b_inherent_minus_agreement = 0.0;

  if (n_inherent > 0 && n_agree > 0) {
    for (k in 1:K_item) {
      if (item_is_agree[k] == 1) {
        b_agreement += beta_item[k] / n_agree;
      } else {
        b_inherent += beta_item[k] / n_inherent;
      }
    }
    b_inherent_minus_agreement = b_inherent - b_agreement;
  }

  // 1. Cell-level log_lik_obs (marginalizing z_testlet[i, t] if only 1 item of testlet t was coded in lang i)
  for (n in 1:N_obs) {
    int i = lang_id[n];
    int k = item_id[n];
    int t = item_testlet[k];
    real eta_base = use_var_slopes
      ? (alpha[k] + beta_item[k] * x[i] + lambda[k] * theta_0[i])
      : (alpha[k] + lambda[k] * theta[i]);
    if (has_testlets && t > 0) {
      if (lang_testlet_n[i, t] >= 2) {
        log_lik_obs[n] = bernoulli_logit_lpmf(Y[n] | eta_base + sd_testlet[1] * z_testlet[i, t]);
      } else {
        vector[M_gh] lp_gh;
        for (r in 1:M_gh) {
          lp_gh[r] = gh_log_w[r] + bernoulli_logit_lpmf(Y[n] | eta_base + sd_testlet[1] * gh_q[r]);
        }
        log_lik_obs[n] = log_sum_exp(lp_gh);
      }
    } else {
      log_lik_obs[n] = bernoulli_logit_lpmf(Y[n] | eta_base);
    }
  }

  // 2. Language-level marginal log_lik_lang (integrating out i.i.d. z_lang[i] and z_testlet[i, 1:T_test]
  //    via nested 9-point Gauss-Hermite quadrature conditioned on theta_struct[i] and x[i])
  for (i in 1:N_lang) {
    vector[M_gh] lp_outer;
    real mu_theta_i = theta_struct[i] + (use_var_slopes ? 0.0 : b_x * x[i]);
    for (m in 1:M_gh) {
      real theta_im = mu_theta_i + sd_lang * gh_q[m];
      real ll_m = gh_log_w[m];

      // Non-testlet items (t == 0 or has_testlets == 0)
      for (k in 1:K_item) {
        if (Y_mat[i, k] >= 0 && (!has_testlets || item_testlet[k] == 0)) {
          real eta_ik = alpha[k] + lambda[k] * theta_im + (use_var_slopes ? beta_item[k] * x[i] : 0.0);
          ll_m += bernoulli_logit_lpmf(Y_mat[i, k] | eta_ik);
        }
      }

      // Multi-item testlets: 1D Gauss-Hermite integral per testlet t
      if (has_testlets) {
        for (t in 1:T_test) {
          if (lang_testlet_n[i, t] > 0) {
            vector[M_gh] lp_inner;
            for (r in 1:M_gh) {
              real ll_r = gh_log_w[r];
              real u_r = sd_testlet[1] * gh_q[r];
              for (k in 1:K_item) {
                if (Y_mat[i, k] >= 0 && item_testlet[k] == t) {
                  real eta_ik = alpha[k] + lambda[k] * theta_im + (use_var_slopes ? beta_item[k] * x[i] : 0.0) + u_r;
                  ll_r += bernoulli_logit_lpmf(Y_mat[i, k] | eta_ik);
                }
              }
              lp_inner[r] = ll_r;
            }
            ll_m += log_sum_exp(lp_inner);
          }
        }
      }
      lp_outer[m] = ll_m;
    }
    log_lik_lang[i] = log_sum_exp(lp_outer);
  }
}
