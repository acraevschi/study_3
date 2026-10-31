// 2-Factor Joint Morphological SEM / IRT Models (MC0_strict, MC_strict, MC0_cross, MC_cross)
//
// Jointly models Factor 1 (Enumerative Breadth: cat12, items 1..K1) and
// Factor 2 (Paradigmatic Irregularity/Flexivity: irreg13 or flex26, items K1+1..K1+K2).
//   - MC_strict (allow_agree_cross = 0): items 1..K1 load strictly on theta_1 (Breadth);
//                                        items K1+1..K_item load strictly on theta_2 (Irreg/Flex).
//   - MC_cross  (allow_agree_cross = 1): contextual agreement categories (gender_agreement,
//                                        number_agreement) also receive cross-loadings gamma_cross
//                                        onto theta_2 (Irregularity/Flexivity).

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
}

data {
  int<lower=1> N_lang;
  int<lower=2> K1;                  // number of Factor 1 (cat12) items (= 12)
  int<lower=2> K2;                  // number of Factor 2 (irreg13 = 13 or flex26 = 26) items
  int<lower=4> K_item;              // K1 + K2
  int<lower=0> T_test;              // number of multi-item testlets across all K_item items
  int<lower=1> K_agree;             // number of agreement items in Factor 1 (= 2)
  int<lower=1> N_obs;

  array[N_obs] int<lower=1, upper=N_lang> lang_id;
  array[N_obs] int<lower=1, upper=K_item> item_id;
  array[N_obs] int<lower=0, upper=1> Y;

  array[K_item] int<lower=0, upper=T_test> item_testlet;
  array[K_item] int<lower=0, upper=K_agree> item_agree_idx; // 0 if not agreement, 1..K_agree if agreement

  vector[N_lang] x;

  int<lower=0, upper=1> with_x;
  int<lower=0, upper=1> use_2pl;
  int<lower=0, upper=1> allow_agree_cross;
  int<lower=0, upper=1> use_phylo;
  int<lower=0, upper=1> use_space;
  int<lower=0, upper=1> use_testlets;

  matrix[N_lang, N_lang] L_phylo;
  array[N_lang] vector[3] Xgp;
  real<lower=0> lscale_prior_shape;
  real<lower=0> lscale_prior_scale;
}

transformed data {
  matrix[K1, K1 - 1] Q_zf1 = make_zero_sum_basis(K1);
  matrix[K2, K2 - 1] Q_zf2 = make_zero_sum_basis(K2);
  int has_testlets = (use_testlets == 1 && T_test > 0) ? 1 : 0;

  array[N_lang, T_test > 0 ? T_test : 1] int lang_testlet_n = rep_array(0, N_lang, T_test > 0 ? T_test : 1);
  array[N_lang, K_item] int Y_mat = rep_array(-1, N_lang, K_item);

  // 7-point probabilists' Gauss-Hermite quadrature nodes and log-weights for z ~ N(0, 1)
  int M_gh = 7;
  vector[7] gh_q = [
    -3.750439717726, -2.366759410735, -1.154405394740,
     0.000000000000,  1.154405394740,  2.366759410735,  3.750439717726
  ]';
  vector[7] gh_log_w = [
    -7.508744778240, -3.481633637819, -1.426603243117,
    -0.782759339250, -1.426603243117, -3.481633637819, -7.508744778240
  ]';

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
  // Separate hierarchical item intercept distributions for Factor 1 (Breadth) and Factor 2 (Irreg/Flex)
  vector[2] mu_alpha;
  vector<lower=0>[2] sd_alpha;
  vector[K_item] z_alpha;

  // Admixture slopes on Factor 1 (b_breadth) and Factor 2 (b_irreg)
  vector[with_x ? 2 : 0] b_x_vec;

  // 2PL discrimination weights within Factor 1 and Factor 2 (arithmetic mean = 1 within each factor)
  vector<lower=0>[use_2pl ? 2 : 0] sd_lambda;
  vector[use_2pl ? K1 - 1 : 0] z_lambda1;
  vector[use_2pl ? K2 - 1 : 0] z_lambda2;

  // Cross-loadings of Agreement items (gender_agreement, number_agreement) onto Factor 2
  vector[allow_agree_cross ? K_agree : 0] gamma_cross;

  // Shared 2x2 Cholesky correlation factor across the two latent dimensions (Breadth, Irreg/Flex)
  cholesky_factor_corr[2] L_corr_lang;

  // Correlated language-level residual innovations
  vector<lower=0>[2] sd_lang;
  matrix[N_lang, 2] z_lang;

  // Correlated phylogenetic random effects per factor
  vector<lower=0>[use_phylo ? 2 : 0] sd_phylo;
  matrix[use_phylo ? N_lang : 0, use_phylo ? 2 : 0] z_phylo;

  // Spatial GP random effects per factor
  vector<lower=0>[use_space ? 2 : 0] sd_gp;
  vector<lower=0>[use_space ? 2 : 0] lscale_gp;
  matrix[use_space ? N_lang : 0, use_space ? 2 : 0] z_gp;

  // Within-language domain testlet random effects
  vector<lower=0>[has_testlets ? 1 : 0] sd_testlet;
  matrix[has_testlets ? N_lang : 0, has_testlets ? T_test : 0] z_testlet;
}

transformed parameters {
  vector[K_item] alpha;
  vector[K1] lambda1 = rep_vector(1.0, K1);
  vector[K2] lambda2 = rep_vector(1.0, K2);
  real b_breadth = with_x ? b_x_vec[1] : 0.0;
  real b_irreg = with_x ? b_x_vec[2] : 0.0;
  matrix[2, 2] L_lang = diag_pre_multiply(sd_lang, L_corr_lang);
  matrix[N_lang, 2] theta_struct = rep_matrix(0.0, N_lang, 2);
  matrix[N_lang, 2] theta_0;
  matrix[N_lang, 2] theta;

  for (k in 1:K1) {
    alpha[k] = mu_alpha[1] + sd_alpha[1] * z_alpha[k];
  }
  for (m in 1:K2) {
    alpha[K1 + m] = mu_alpha[2] + sd_alpha[2] * z_alpha[K1 + m];
  }

  // Arithmetic-mean-1 normalization within each factor: mean(lambda1) == 1 and mean(lambda2) == 1
  if (use_2pl) {
    lambda1 = K1 * softmax(sd_lambda[1] * (Q_zf1 * z_lambda1));
    lambda2 = K2 * softmax(sd_lambda[2] * (Q_zf2 * z_lambda2));
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
  theta = theta_0;

  if (with_x) {
    theta[, 1] += b_breadth * x;
    theta[, 2] += b_irreg * x;
  }
}

model {
  mu_alpha ~ normal(0, 1.5);
  sd_alpha ~ exponential(1);
  z_alpha ~ std_normal();

  if (with_x) {
    b_x_vec ~ normal(0, 0.5);
  }

  if (use_2pl) {
    sd_lambda ~ normal(0, 0.7);
    z_lambda1 ~ std_normal();
    z_lambda2 ~ std_normal();
  }

  if (allow_agree_cross) {
    gamma_cross ~ normal(0, 0.75);
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

  // Likelihood over observed (language, item) cells
  {
    vector[N_obs] eta;
    for (n in 1:N_obs) {
      int i = lang_id[n];
      int k = item_id[n];
      int t = item_testlet[k];
      real u_test = (has_testlets && t > 0) ? (sd_testlet[1] * z_testlet[i, t]) : 0.0;
      if (k <= K1) {
        int ag = item_agree_idx[k];
        real cross_term = (allow_agree_cross && ag > 0) ? (gamma_cross[ag] * theta[i, 2]) : 0.0;
        eta[n] = alpha[k] + lambda1[k] * theta[i, 1] + cross_term + u_test;
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
  real rho_latent = multiply_lower_tri_self_transpose(L_corr_lang)[1, 2];
  real rho_theta0;
  vector[N_obs] log_lik_obs;
  vector[N_lang] log_lik_lang;
  vector[N_lang] log_lik_lang_breadth;
  vector[N_lang] log_lik_lang_irreg;

  // Empirical correlation between total baseline factors theta_0[,1] and theta_0[,2]
  {
    vector[N_lang] c1 = theta_0[, 1] - mean(theta_0[, 1]);
    vector[N_lang] c2 = theta_0[, 2] - mean(theta_0[, 2]);
    real denom = sqrt(dot_self(c1) * dot_self(c2));
    rho_theta0 = (denom > 0) ? (dot_product(c1, c2) / denom) : 0.0;
  }

  // 1. Cell-level log_lik_obs (marginalizing z_testlet[i, t] if only 1 item of testlet t was coded in lang i)
  for (n in 1:N_obs) {
    int i = lang_id[n];
    int k = item_id[n];
    int t = item_testlet[k];
    real eta_base;
    if (k <= K1) {
      int ag = item_agree_idx[k];
      real cross_term = (allow_agree_cross && ag > 0) ? (gamma_cross[ag] * theta[i, 2]) : 0.0;
      eta_base = alpha[k] + lambda1[k] * theta[i, 1] + cross_term;
    } else {
      int m = k - K1;
      eta_base = alpha[k] + lambda2[m] * theta[i, 2];
    }
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

  // 2. Language-level marginal log_lik_lang (integrating out 2D z_lang[i, 1:2] over M_gh x M_gh grid
  //    and 1D z_testlet[i, 1:T_test] conditioned on theta_struct[i, 1:2] and x[i])
  for (i in 1:N_lang) {
    vector[M_gh * M_gh] lp_outer;
    vector[M_gh * M_gh] lp_outer_b;
    vector[M_gh * M_gh] lp_outer_ir;
    real mu1 = theta_struct[i, 1] + (with_x ? b_breadth * x[i] : 0.0);
    real mu2 = theta_struct[i, 2] + (with_x ? b_irreg * x[i] : 0.0);
    int idx = 0;

    for (m1 in 1:M_gh) {
      real q1 = gh_q[m1];
      real lw1 = gh_log_w[m1];
      for (m2 in 1:M_gh) {
        real q2 = gh_q[m2];
        real lw12 = lw1 + gh_log_w[m2];
        real th1 = mu1 + L_lang[1, 1] * q1;
        real th2 = mu2 + L_lang[2, 1] * q1 + L_lang[2, 2] * q2;
        real ll_b = 0.0;
        real ll_ir = 0.0;
        idx += 1;

        // Non-testlet items (t == 0 or has_testlets == 0)
        for (k in 1:K_item) {
          if (Y_mat[i, k] >= 0 && (!has_testlets || item_testlet[k] == 0)) {
            if (k <= K1) {
              int ag = item_agree_idx[k];
              real cross_term = (allow_agree_cross && ag > 0) ? (gamma_cross[ag] * th2) : 0.0;
              ll_b += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda1[k] * th1 + cross_term);
            } else {
              int m = k - K1;
              ll_ir += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda2[m] * th2);
            }
          }
        }

        // Multi-item testlets (each testlet belongs strictly to either Factor 1 or Factor 2)
        if (has_testlets) {
          for (t in 1:T_test) {
            if (lang_testlet_n[i, t] > 0) {
              vector[M_gh] lp_inner;
              int is_f1_testlet = 0;
              for (r in 1:M_gh) {
                real ll_r = gh_log_w[r];
                real u_r = sd_testlet[1] * gh_q[r];
                for (k in 1:K_item) {
                  if (Y_mat[i, k] >= 0 && item_testlet[k] == t) {
                    if (k <= K1) {
                      int ag = item_agree_idx[k];
                      real cross_term = (allow_agree_cross && ag > 0) ? (gamma_cross[ag] * th2) : 0.0;
                      ll_r += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda1[k] * th1 + cross_term + u_r);
                      is_f1_testlet = 1;
                    } else {
                      int m = k - K1;
                      ll_r += bernoulli_logit_lpmf(Y_mat[i, k] | alpha[k] + lambda2[m] * th2 + u_r);
                    }
                  }
                }
                lp_inner[r] = ll_r;
              }
              if (is_f1_testlet == 1) {
                ll_b += log_sum_exp(lp_inner);
              } else {
                ll_ir += log_sum_exp(lp_inner);
              }
            }
          }
        }

        lp_outer[idx] = lw12 + ll_b + ll_ir;
        lp_outer_b[idx] = lw12 + ll_b;
        lp_outer_ir[idx] = lw12 + ll_ir;
      }
    }
    log_lik_lang[i] = log_sum_exp(lp_outer);
    log_lik_lang_breadth[i] = log_sum_exp(lp_outer_b);
    log_lik_lang_irreg[i] = log_sum_exp(lp_outer_ir);
  }
}
