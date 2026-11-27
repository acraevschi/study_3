// Joint Within-POS 2-Factor 2PL IRT-SEM (Breadth vs. Irregularity within Verb or Noun)
//
// Jointly models Factor 1 (Breadth: K1 items) and Factor 2 (Irregularity: K2 items)
// within a single part of speech (`Verb_2fac`: K1=9, K2=5; `Noun_2fac`: K1=5, K2=3).
// Languages passing the >=60% floor on Breadth supply Factor-1 observations and languages
// passing the >=60% floor on domain-conditioned Irregularity supply Factor-2 observations.
// Estimates:
//   - `b_breadth`, `b_irreg`, `b_diff = b_breadth - b_irreg`
//   - `rho_latent` (residual correlation between latent Breadth and Irregularity)
//   - `rho_theta0` (empirical correlation of baseline structural+tip latent scores)
//   - Arbitrary language-level covariates `Z_cov` (`z_pop`, `z_area`, `z_doc`) on both factors
//   - Regularized spatial GP (`inv_gamma(lscale_prior_shape, lscale_prior_scale)`), Grafen/Bouckaert
//     phylogeny, optional family root intercepts, and local-dependence testlets.

functions {
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

  vector gp_Iso(data array[] vector x, real sdgp, real lscale, vector zgp) {
    int N = size(x);
    matrix[N, N] cov = gp_exp_quad_cov(x, sdgp, lscale);
    for (n in 1:N) {
      cov[n, n] += 1e-12;
    }
    return cholesky_decompose(cov) * zgp;
  }

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
  int<lower=2> K1;                  // Factor 1 (Breadth) items
  int<lower=2> K2;                  // Factor 2 (Irregularity) items
  int<lower=4> K_item;              // K1 + K2
  int<lower=0> T_test;
  int<lower=1> N_obs;
  int<lower=1> N_fam;
  int<lower=0> P_cov;

  array[N_obs] int<lower=1, upper=N_lang> lang_id;
  array[N_obs] int<lower=1, upper=K_item> item_id;
  array[N_obs] int<lower=0, upper=1> Y;

  array[K_item] int<lower=0, upper=T_test> item_testlet;

  array[N_lang] int<lower=1, upper=N_fam> fam_id;
  array[N_lang] int<lower=1> fam_size;

  vector[N_lang] x;
  vector[N_lang] x_within;
  vector[N_lang] x_between;
  matrix[N_lang, P_cov] Z_cov;

  int<lower=0, upper=1> with_x;
  int<lower=0, upper=1> use_2pl;
  int<lower=0, upper=1> use_phylo;
  int<lower=0, upper=1> use_space;
  int<lower=0, upper=1> use_testlets;
  int<lower=0, upper=1> use_fam_int;
  int<lower=0, upper=1> use_mundlak;

  matrix[N_lang, N_lang] L_phylo;
  array[N_lang] vector[3] Xgp;
  real<lower=0> lscale_prior_shape;
  real<lower=0> lscale_prior_scale;
}

transformed data {
  matrix[K1, K1 - 1] Q_zf1 = make_zero_sum_basis(K1);
  matrix[K2, K2 - 1] Q_zf2 = make_zero_sum_basis(K2);
  int has_testlets = (use_testlets == 1 && T_test > 0) ? 1 : 0;
  vector[N_lang] x_main = use_mundlak ? x_within : x;

  array[N_lang, T_test > 0 ? T_test : 1] int lang_testlet_n = rep_array(0, N_lang, T_test > 0 ? T_test : 1);
  array[N_lang, K_item] int Y_mat = rep_array(-1, N_lang, K_item);

  array[T_test > 0 ? T_test : 1] int testlet_factor = rep_array(0, T_test > 0 ? T_test : 1);
  array[T_test > 0 ? T_test : 1] int testlet_k_len = rep_array(0, T_test > 0 ? T_test : 1);
  array[T_test > 0 ? T_test : 1, K_item] int testlet_k_list = rep_array(0, T_test > 0 ? T_test : 1, K_item);

  int M_gh = 7;
  matrix[M_gh, 2] gh = make_gauss_hermite(M_gh);
  vector[M_gh] gh_q = gh[, 1];
  vector[M_gh] gh_log_w = gh[, 2];

  int M_rel = 15;
  matrix[M_rel, 2] gh_rel = make_gauss_hermite(M_rel);
  vector[M_rel] gh_rel_q = gh_rel[, 1];
  vector[M_rel] gh_rel_log_w = gh_rel[, 2];

  int M_out = 25;
  matrix[M_out, 2] gh_out = make_gauss_hermite(M_out);
  vector[M_out] gh_out_q = gh_out[, 1];
  vector[M_out] gh_out_log_w = gh_out[, 2];

  matrix[N_lang, N_lang] L_inv = mdivide_left_tri_low(L_phylo, diag_matrix(rep_vector(1.0, N_lang)));
  vector[N_lang] A_inv_diag;
  for (i in 1:N_lang) {
    A_inv_diag[i] = dot_self(L_inv[, i]);
  }

  for (k in 1:K_item) {
    int t = item_testlet[k];
    if (t > 0) {
      testlet_k_len[t] += 1;
      testlet_k_list[t, testlet_k_len[t]] = k;
      testlet_factor[t] = (k <= K1) ? 1 : 2;
    }
  }

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
  vector[2] mu_alpha;
  vector<lower=0>[2] sd_alpha;
  vector[K_item] z_alpha;

  vector[with_x ? 2 : 0] b_x_vec;
  vector[(with_x && use_mundlak) ? 2 : 0] b_between_vec;
  matrix[P_cov, 2] b_cov;

  vector<lower=0>[use_2pl ? 2 : 0] sd_lambda;
  vector[use_2pl ? K1 - 1 : 0] z_lambda1;
  vector[use_2pl ? K2 - 1 : 0] z_lambda2;

  vector<lower=0>[use_fam_int ? 2 : 0] sd_fam;
  matrix[use_fam_int ? N_fam : 0, use_fam_int ? 2 : 0] z_fam;

  cholesky_factor_corr[2] L_corr_lang;

  vector<lower=0>[2] sd_lang;
  matrix[N_lang, 2] z_lang;

  vector<lower=0>[use_phylo ? 2 : 0] sd_phylo;
  matrix[use_phylo ? N_lang : 0, use_phylo ? 2 : 0] z_phylo;

  vector<lower=0>[use_space ? 2 : 0] sd_gp;
  vector<lower=0>[use_space ? 2 : 0] lscale_gp;
  matrix[use_space ? N_lang : 0, use_space ? 2 : 0] z_gp;

  vector<lower=0>[has_testlets ? 1 : 0] sd_testlet;
  matrix[has_testlets ? N_lang : 0, has_testlets ? T_test : 0] z_testlet;
}

transformed parameters {
  vector[K_item] alpha;
  vector[K1] lambda1 = rep_vector(1.0, K1);
  vector[K2] lambda2 = rep_vector(1.0, K2);
  real b_breadth = with_x ? b_x_vec[1] : 0.0;
  real b_irreg = with_x ? b_x_vec[2] : 0.0;
  real b_bet_breadth = (with_x && use_mundlak) ? b_between_vec[1] : 0.0;
  real b_bet_irreg = (with_x && use_mundlak) ? b_between_vec[2] : 0.0;

  matrix[2, 2] L_lang = diag_pre_multiply(sd_lang, L_corr_lang);
  matrix[N_lang, 2] theta_struct = rep_matrix(0.0, N_lang, 2);
  matrix[N_lang, 2] cov_shift = rep_matrix(0.0, N_lang, 2);
  matrix[N_lang, 2] theta_0;
  matrix[N_lang, 2] theta;

  for (k in 1:K1) {
    alpha[k] = mu_alpha[1] + sd_alpha[1] * z_alpha[k];
  }
  for (m in 1:K2) {
    alpha[K1 + m] = mu_alpha[2] + sd_alpha[2] * z_alpha[K1 + m];
  }

  if (use_2pl) {
    lambda1 = K1 * softmax(sd_lambda[1] * (Q_zf1 * z_lambda1));
    lambda2 = K2 * softmax(sd_lambda[2] * (Q_zf2 * z_lambda2));
  }

  if (use_fam_int) {
    matrix[2, 2] L_fam_2 = diag_pre_multiply(sd_fam, L_corr_lang);
    matrix[N_fam, 2] u_fam = z_fam * L_fam_2';
    for (i in 1:N_lang) {
      theta_struct[i, ] += u_fam[fam_id[i], ];
    }
  }

  if (use_phylo) {
    matrix[2, 2] L_phy_2 = diag_pre_multiply(sd_phylo, L_corr_lang);
    theta_struct += (L_phylo * z_phylo) * L_phy_2';
  }

  if (use_space) {
    theta_struct[, 1] += gp_Iso(Xgp, sd_gp[1], lscale_gp[1], z_gp[, 1]);
    theta_struct[, 2] += gp_Iso(Xgp, sd_gp[2], lscale_gp[2], z_gp[, 2]);
  }

  theta_0 = theta_struct + z_lang * L_lang';

  if (with_x) {
    cov_shift[, 1] += b_breadth * x_main;
    cov_shift[, 2] += b_irreg * x_main;
  }
  if (with_x && use_mundlak) {
    cov_shift[, 1] += b_bet_breadth * x_between;
    cov_shift[, 2] += b_bet_irreg * x_between;
  }
  if (P_cov > 0) {
    cov_shift += Z_cov * b_cov;
  }

  theta = theta_0 + cov_shift;
}

model {
  mu_alpha ~ normal(0, 1.5);
  sd_alpha ~ exponential(1);
  z_alpha ~ std_normal();

  if (with_x) {
    b_x_vec ~ normal(0, 0.5);
  }
  if (with_x && use_mundlak) {
    b_between_vec ~ normal(0, 0.5);
  }
  if (P_cov > 0) {
    to_vector(b_cov) ~ normal(0, 0.5);
  }

  if (use_2pl) {
    sd_lambda ~ normal(0, 0.7);
    z_lambda1 ~ std_normal();
    z_lambda2 ~ std_normal();
  }

  if (use_fam_int) {
    sd_fam ~ exponential(1);
    to_vector(z_fam) ~ std_normal();
  }

  L_corr_lang ~ lkj_corr_cholesky(2.0);
  sd_lang ~ exponential(1);
  to_vector(z_lang) ~ std_normal();

  if (use_phylo) {
    sd_phylo ~ exponential(1);
    to_vector(z_phylo) ~ std_normal();
  }

  if (use_space) {
    sd_gp ~ exponential(1);
    lscale_gp ~ inv_gamma(lscale_prior_shape, lscale_prior_scale);
    to_vector(z_gp) ~ std_normal();
  }

  if (has_testlets) {
    sd_testlet ~ exponential(1);
    to_vector(z_testlet) ~ std_normal();
  }

  {
    vector[N_obs] eta;
    for (n in 1:N_obs) {
      int i = lang_id[n];
      int k = item_id[n];
      int t = item_testlet[k];
      real u_test = (has_testlets && t > 0) ? (sd_testlet[1] * z_testlet[i, t]) : 0.0;
      if (k <= K1) {
        eta[n] = alpha[k] + lambda1[k] * theta[i, 1] + u_test;
      } else {
        int m = k - K1;
        eta[n] = alpha[k] + lambda2[m] * theta[i, 2] + u_test;
      }
    }
    Y ~ bernoulli_logit(eta);
  }
}

generated quantities {
  real b_diff = b_breadth - b_irreg;
  real b_bet_diff = b_bet_breadth - b_bet_irreg;
  real rho_latent = multiply_lower_tri_self_transpose(L_corr_lang)[1, 2];
  real rho_theta0;
  vector[N_obs] log_lik_obs;
  vector[N_lang] log_lik_lang;
  vector[N_lang] log_lik_lang_tip;

  {
    vector[N_lang] c1 = theta_0[, 1] - mean(theta_0[, 1]);
    vector[N_lang] c2 = theta_0[, 2] - mean(theta_0[, 2]);
    real denom = sqrt(dot_self(c1) * dot_self(c2));
    rho_theta0 = (denom > 0) ? (dot_product(c1, c2) / denom) : 0.0;
  }

  for (n in 1:N_obs) {
    int i = lang_id[n];
    int k = item_id[n];
    int t = item_testlet[k];
    real eta_base = (k <= K1)
      ? (alpha[k] + lambda1[k] * theta[i, 1])
      : (alpha[k] + lambda2[k - K1] * theta[i, 2]);
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

  {
    matrix[use_phylo ? N_lang : 0, 2] A_inv_u;
    matrix[2, 2] Sigma_lang = multiply_lower_tri_self_transpose(L_lang);
    matrix[2, 2] Sigma_phy = rep_matrix(0.0, 2, 2);
    matrix[2, 2] Sigma_fam = rep_matrix(0.0, 2, 2);
    matrix[use_fam_int ? N_fam : 0, 2] u_fam_gq;

    real s2_l = sqrt(Sigma_lang[2, 2]);
    real c12_l = Sigma_lang[1, 2] / s2_l;
    real s1_2_l = sqrt(fmax(1e-12, Sigma_lang[1, 1] - square(c12_l)));

    if (use_phylo) {
      matrix[2, 2] L_phy_2 = diag_pre_multiply(sd_phylo, L_corr_lang);
      A_inv_u = (L_inv' * z_phylo) * L_phy_2';
      Sigma_phy = multiply_lower_tri_self_transpose(L_phy_2);
    }
    if (use_fam_int) {
      matrix[2, 2] L_fam_2 = diag_pre_multiply(sd_fam, L_corr_lang);
      u_fam_gq = z_fam * L_fam_2';
      Sigma_fam = multiply_lower_tri_self_transpose(L_fam_2);
    }

    for (i in 1:N_lang) {
      real mu1 = theta_struct[i, 1] + cov_shift[i, 1];
      real mu2 = theta_struct[i, 2] + cov_shift[i, 2];

      {
        vector[M_gh * M_gh] lp_outer;
        int idx = 0;
        for (m1 in 1:M_gh) {
          real q1 = gh_q[m1];
          real lw1 = gh_log_w[m1];
          real th2 = mu2 + s2_l * q1;
          real ll_ir = 0.0;
          for (k in (K1 + 1):K_item) {
            if (Y_mat[i, k] >= 0 && (!has_testlets || item_testlet[k] == 0)) {
              ll_ir += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda2[k - K1] * th2);
            }
          }
          if (has_testlets) {
            for (t in 1:T_test) {
              if (testlet_factor[t] == 2 && lang_testlet_n[i, t] > 0) {
                vector[M_gh] lp_inner;
                int nk = testlet_k_len[t];
                for (r in 1:M_gh) {
                  real ll_r = gh_log_w[r];
                  real u_r = sd_testlet[1] * gh_q[r];
                  for (pos in 1:nk) {
                    int k = testlet_k_list[t, pos];
                    if (Y_mat[i, k] >= 0) {
                      ll_r += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda2[k - K1] * th2 + u_r);
                    }
                  }
                  lp_inner[r] = ll_r;
                }
                ll_ir += log_sum_exp(lp_inner);
              }
            }
          }
          for (m2 in 1:M_gh) {
            real q2 = gh_q[m2];
            real lw12 = lw1 + gh_log_w[m2];
            real th1 = mu1 + c12_l * q1 + s1_2_l * q2;
            real ll_b = 0.0;
            idx += 1;
            for (k in 1:K1) {
              if (Y_mat[i, k] >= 0 && (!has_testlets || item_testlet[k] == 0)) {
                ll_b += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda1[k] * th1);
              }
            }
            if (has_testlets) {
              for (t in 1:T_test) {
                if (testlet_factor[t] == 1 && lang_testlet_n[i, t] > 0) {
                  vector[M_gh] lp_inner;
                  int nk = testlet_k_len[t];
                  for (r in 1:M_gh) {
                    real ll_r = gh_log_w[r];
                    real u_r = sd_testlet[1] * gh_q[r];
                    for (pos in 1:nk) {
                      int k = testlet_k_list[t, pos];
                      if (Y_mat[i, k] >= 0) {
                        ll_r += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda1[k] * th1 + u_r);
                      }
                    }
                    lp_inner[r] = ll_r;
                  }
                  ll_b += log_sum_exp(lp_inner);
                }
              }
            }
            lp_outer[idx] = lw12 + ll_b + ll_ir;
          }
        }
        log_lik_lang[i] = log_sum_exp(lp_outer);
      }

      {
        matrix[2, 2] Sigma_tip = Sigma_lang
          + (use_phylo ? Sigma_phy / A_inv_diag[i] : rep_matrix(0.0, 2, 2))
          + ((use_fam_int && fam_size[i] == 1) ? Sigma_fam : rep_matrix(0.0, 2, 2));
        real s2_t = sqrt(Sigma_tip[2, 2]);
        real c12_t = Sigma_tip[1, 2] / s2_t;
        real s1_2_t = sqrt(fmax(1e-12, Sigma_tip[1, 1] - square(c12_t)));

        real mu1_tip = mu1
          - (use_phylo ? A_inv_u[i, 1] / A_inv_diag[i] : 0.0)
          - ((use_fam_int && fam_size[i] == 1) ? u_fam_gq[fam_id[i], 1] : 0.0);
        real mu2_tip = mu2
          - (use_phylo ? A_inv_u[i, 2] / A_inv_diag[i] : 0.0)
          - ((use_fam_int && fam_size[i] == 1) ? u_fam_gq[fam_id[i], 2] : 0.0);

        int use_dense = (fam_size[i] <= 2 || A_inv_diag[i] < 1.5) ? 1 : 0;
        int M_grid = use_dense ? M_out : M_rel;
        vector[M_grid * M_grid] lp_outer_tip;
        int idx = 0;

        for (m1 in 1:M_grid) {
          real q1 = use_dense ? gh_out_q[m1] : gh_rel_q[m1];
          real lw1 = use_dense ? gh_out_log_w[m1] : gh_rel_log_w[m1];
          real th2 = mu2_tip + s2_t * q1;
          real ll_ir = 0.0;
          for (k in (K1 + 1):K_item) {
            if (Y_mat[i, k] >= 0 && (!has_testlets || item_testlet[k] == 0)) {
              ll_ir += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda2[k - K1] * th2);
            }
          }
          if (has_testlets) {
            for (t in 1:T_test) {
              if (testlet_factor[t] == 2 && lang_testlet_n[i, t] > 0) {
                vector[M_gh] lp_inner;
                int nk = testlet_k_len[t];
                for (r in 1:M_gh) {
                  real ll_r = gh_log_w[r];
                  real u_r = sd_testlet[1] * gh_q[r];
                  for (pos in 1:nk) {
                    int k = testlet_k_list[t, pos];
                    if (Y_mat[i, k] >= 0) {
                      ll_r += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda2[k - K1] * th2 + u_r);
                    }
                  }
                  lp_inner[r] = ll_r;
                }
                ll_ir += log_sum_exp(lp_inner);
              }
            }
          }
          for (m2 in 1:M_grid) {
            real q2 = use_dense ? gh_out_q[m2] : gh_rel_q[m2];
            real lw12 = lw1 + (use_dense ? gh_out_log_w[m2] : gh_rel_log_w[m2]);
            real th1 = mu1_tip + c12_t * q1 + s1_2_t * q2;
            real ll_b = 0.0;
            idx += 1;
            for (k in 1:K1) {
              if (Y_mat[i, k] >= 0 && (!has_testlets || item_testlet[k] == 0)) {
                ll_b += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda1[k] * th1);
              }
            }
            if (has_testlets) {
              for (t in 1:T_test) {
                if (testlet_factor[t] == 1 && lang_testlet_n[i, t] > 0) {
                  vector[M_gh] lp_inner;
                  int nk = testlet_k_len[t];
                  for (r in 1:M_gh) {
                    real ll_r = gh_log_w[r];
                    real u_r = sd_testlet[1] * gh_q[r];
                    for (pos in 1:nk) {
                      int k = testlet_k_list[t, pos];
                      if (Y_mat[i, k] >= 0) {
                        ll_r += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda1[k] * th1 + u_r);
                      }
                    }
                    lp_inner[r] = ll_r;
                  }
                  ll_b += log_sum_exp(lp_inner);
                }
              }
            }
            lp_outer_tip[idx] = lw12 + ll_b + ll_ir;
          }
        }
        log_lik_lang_tip[i] = log_sum_exp(lp_outer_tip);
      }
    }
  }
}
