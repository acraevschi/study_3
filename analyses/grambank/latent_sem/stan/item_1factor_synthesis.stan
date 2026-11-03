// 1-Factor Item-Level Morphological Synthesis Models (Step 2 Unifying Synthesis Checks)
//
// Extends stan/item_1factor.stan with modular switches for:
//   - Check 1.1: Family-varying admixture slopes (`use_fam_slopes = 1`: beta_fam[f] = b_x + sd_fam_b * z_fam_b[f]
//                multiplying within-family centered exposure `x_within[i] = x[i] - mean_f(x)`).
//   - Check 1.1b: Mundlak within- vs. between-family decomposition (`use_mundlak = 1`: separates
//                 within-family slope `b_x` on `x_within` from between-family slope `b_between` on `x_between`).
//   - Check 2:   Explicit family random intercepts (`use_fam_int = 1`: u_fam[f] = sd_fam * z_fam[f])
//                for crown-MRCA Bouckaert phylogenetic covariance (`P_b_crown` and `P_b_kyr`).
//   - Check 3.1: Demographic controls on the latent factor (`with_pop = 1`: b_pop * z_pop[i];
//                `with_int_pop = 1`: b_x_pop * x[i] * z_pop[i]).
//   - Tip-marginalized language LOO (`log_lik_lang_tip`, M_out = 21) alongside `log_lik_obs` and `log_lik_lang`.

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
  int<lower=2> K_item;
  int<lower=2> D_dom;
  int<lower=0> T_test;
  int<lower=1> N_obs;
  int<lower=1> N_fam;

  array[N_obs] int<lower=1, upper=N_lang> lang_id;
  array[N_obs] int<lower=1, upper=K_item> item_id;
  array[N_obs] int<lower=0, upper=1> Y;

  array[K_item] int<lower=1, upper=D_dom> item_dom;
  array[K_item] int<lower=0, upper=T_test> item_testlet;
  array[K_item] int<lower=0, upper=1> item_is_agree;

  array[N_lang] int<lower=1, upper=N_fam> fam_id;
  array[N_lang] int<lower=1> fam_size;

  vector[N_lang] x;
  vector[N_lang] x_within;
  vector[N_lang] x_between;
  vector[N_lang] z_pop;

  int<lower=0, upper=1> with_x;
  int<lower=0, upper=1> use_2pl;
  int<lower=0, upper=1> use_var_slopes;
  int<lower=0, upper=1> use_phylo;
  int<lower=0, upper=1> use_space;
  int<lower=0, upper=1> use_testlets;
  int<lower=0, upper=1> use_fam_int;
  int<lower=0, upper=1> use_fam_slopes;
  int<lower=0, upper=1> use_mundlak;
  int<lower=0, upper=1> with_pop;
  int<lower=0, upper=1> with_int_pop;

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
  vector[N_lang] x_z_pop = x .* z_pop;
  // When use_mundlak == 1, the main slope b_x / beta_item multiplies x_within while b_between multiplies x_between
  vector[N_lang] x_main = use_mundlak ? x_within : x;

  array[N_lang, T_test > 0 ? T_test : 1] int lang_testlet_n = rep_array(0, N_lang, T_test > 0 ? T_test : 1);
  array[N_lang, K_item] int Y_mat = rep_array(-1, N_lang, K_item);

  int M_gh = 9;
  matrix[M_gh, 2] gh = make_gauss_hermite(M_gh);
  vector[M_gh] gh_q = gh[, 1];
  vector[M_gh] gh_log_w = gh[, 2];

  int M_out = 21;
  matrix[M_out, 2] gh_out = make_gauss_hermite(M_out);
  vector[M_out] gh_out_q = gh_out[, 1];
  vector[M_out] gh_out_log_w = gh_out[, 2];

  // A^-1 = L_inv' * L_inv for conditional phylogenetic tip variance sd_phylo^2 / A_inv_diag[i]
  matrix[N_lang, N_lang] L_inv = mdivide_left_tri_low(L_phylo, diag_matrix(rep_vector(1.0, N_lang)));
  vector[N_lang] A_inv_diag;
  for (i in 1:N_lang) {
    A_inv_diag[i] = dot_self(L_inv[, i]);
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
  real mu_alpha;
  real<lower=0> sd_alpha;
  vector[K_item] z_alpha;

  vector[with_x ? 1 : 0] b_x_vec;
  vector[(with_x && use_mundlak) ? 1 : 0] b_between_vec;
  vector[with_pop ? 1 : 0] b_pop_vec;
  vector[with_int_pop ? 1 : 0] b_x_pop_vec;

  vector<lower=0>[use_2pl ? 1 : 0] sd_lambda;
  vector[use_2pl ? K_item - 1 : 0] z_lambda;

  vector<lower=0>[use_var_slopes ? 1 : 0] sd_dom_b;
  vector[use_var_slopes ? D_dom - 1 : 0] z_dom_b;
  vector<lower=0>[use_var_slopes ? 1 : 0] sd_item_b;
  vector[use_var_slopes ? K_item : 0] z_item_b;

  vector<lower=0>[use_fam_slopes ? 1 : 0] sd_fam_b;
  vector[use_fam_slopes ? N_fam : 0] z_fam_b;

  vector<lower=0>[use_fam_int ? 1 : 0] sd_fam;
  vector[use_fam_int ? N_fam : 0] z_fam;

  real<lower=0> sd_lang;
  vector[N_lang] z_lang;

  vector<lower=0>[use_phylo ? 1 : 0] sd_phylo;
  vector[use_phylo ? N_lang : 0] z_phylo;

  vector<lower=0>[use_space ? 1 : 0] sd_gp;
  vector<lower=0>[use_space ? 1 : 0] lscale_gp;
  vector[use_space ? N_lang : 0] z_gp;

  vector<lower=0>[has_testlets ? 1 : 0] sd_testlet;
  matrix[has_testlets ? N_lang : 0, has_testlets ? T_test : 0] z_testlet;
}

transformed parameters {
  vector[K_item] alpha = mu_alpha + sd_alpha * z_alpha;
  vector[K_item] lambda = rep_vector(1.0, K_item);
  real b_x = with_x ? b_x_vec[1] : 0.0;
  real b_between = (with_x && use_mundlak) ? b_between_vec[1] : 0.0;
  real b_pop = with_pop ? b_pop_vec[1] : 0.0;
  real b_x_pop = with_int_pop ? b_x_pop_vec[1] : 0.0;
  vector[N_fam] delta_fam_b = use_fam_slopes ? (sd_fam_b[1] * z_fam_b) : rep_vector(0.0, N_fam);
  vector[N_fam] beta_fam = b_x + delta_fam_b;
  vector[D_dom] beta_domain = rep_vector(b_x, D_dom);
  vector[K_item] beta_item = rep_vector(b_x, K_item);
  vector[N_lang] theta_struct = rep_vector(0.0, N_lang);
  vector[N_lang] cov_shift = rep_vector(0.0, N_lang);
  // Direct logit-scale family slope offset when use_var_slopes == 1 (so MB_2pl keeps beta_item and delta_fam_b on the same scale)
  vector[N_lang] var_fam_shift = rep_vector(0.0, N_lang);
  vector[N_lang] theta_0;
  vector[N_lang] theta;

  if (use_2pl) {
    lambda = K_item * softmax(sd_lambda[1] * (Q_zf * z_lambda));
  }

  if (use_fam_int) {
    for (i in 1:N_lang) {
      theta_struct[i] += sd_fam[1] * z_fam[fam_id[i]];
    }
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
    for (k in 1:K_item) {
      real item_off = (item_testlet[k] > 0) ? (sd_item_b[1] * z_item_b[k]) : 0.0;
      raw_b[k] = dom_off[item_dom[k]] + item_off;
    }
    raw_b = raw_b - mean(raw_b);
    beta_item = b_x + raw_b;
    for (d in 1:D_dom) {
      beta_domain[d] = b_x + dom_off[d];
    }
    if (use_fam_slopes) {
      for (i in 1:N_lang) {
        var_fam_shift[i] = delta_fam_b[fam_id[i]] * x_within[i];
      }
    }
  } else {
    if (with_x) {
      cov_shift += b_x * x_main;
    }
    if (use_fam_slopes) {
      for (i in 1:N_lang) {
        cov_shift[i] += delta_fam_b[fam_id[i]] * x_within[i];
      }
    }
    beta_item = lambda * b_x;
  }

  if (with_x && use_mundlak) {
    cov_shift += b_between * x_between;
  }
  if (with_pop) {
    cov_shift += b_pop * z_pop;
  }
  if (with_int_pop) {
    cov_shift += b_x_pop * x_z_pop;
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
  if (with_pop) {
    b_pop_vec ~ normal(0, 0.5);
  }
  if (with_int_pop) {
    b_x_pop_vec ~ normal(0, 0.5);
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

  if (use_fam_slopes) {
    sd_fam_b ~ exponential(2);
    z_fam_b ~ std_normal();
  }

  if (use_fam_int) {
    sd_fam ~ exponential(1);
    z_fam ~ std_normal();
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

  {
    vector[N_obs] eta;
    for (n in 1:N_obs) {
      int i = lang_id[n];
      int k = item_id[n];
      int t = item_testlet[k];
      real u_test = (has_testlets && t > 0) ? (sd_testlet[1] * z_testlet[i, t]) : 0.0;
      if (use_var_slopes) {
        eta[n] = alpha[k] + beta_item[k] * x_main[i] + var_fam_shift[i] + lambda[k] * theta[i] + u_test;
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
  vector[N_lang] log_lik_lang_tip;
  real b_inherent = 0.0;
  real b_agreement = 0.0;
  real b_inherent_minus_agreement = 0.0;
  real b_within_minus_between = (with_x && use_mundlak) ? (b_x - b_between) : 0.0;

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

  // 1. Cell-level log_lik_obs
  for (n in 1:N_obs) {
    int i = lang_id[n];
    int k = item_id[n];
    int t = item_testlet[k];
    real eta_base = use_var_slopes
      ? (alpha[k] + beta_item[k] * x_main[i] + var_fam_shift[i] + lambda[k] * theta[i])
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

  // 2. Language-level marginal log_lik_lang and tip-marginalized log_lik_lang_tip
  {
    vector[use_phylo ? N_lang : 0] A_inv_u;
    if (use_phylo) {
      A_inv_u = sd_phylo[1] * (L_inv' * z_phylo);
    }

    for (i in 1:N_lang) {
      vector[M_gh] lp_outer;
      vector[M_out] lp_outer_tip;
      real mu_theta_i = theta_struct[i] + cov_shift[i];

      real fam_sing_shift = (use_fam_int && fam_size[i] == 1) ? (sd_fam[1] * z_fam[fam_id[i]]) : 0.0;
      real fam_sing_var = (use_fam_int && fam_size[i] == 1) ? square(sd_fam[1]) : 0.0;
      real tip_shift = (use_phylo ? A_inv_u[i] / A_inv_diag[i] : 0.0) + fam_sing_shift;
      real sd_out_i = sqrt(
        square(sd_lang)
        + (use_phylo ? square(sd_phylo[1]) / A_inv_diag[i] : 0.0)
        + fam_sing_var
      );
      real mu_theta_tip_i = mu_theta_i - tip_shift;

      // 2a. Standard log_lik_lang (M_gh = 9)
      for (m in 1:M_gh) {
        real theta_im = mu_theta_i + sd_lang * gh_q[m];
        real ll_m = gh_log_w[m];
        for (k in 1:K_item) {
          if (Y_mat[i, k] >= 0 && (!has_testlets || item_testlet[k] == 0)) {
            real eta_ik = alpha[k] + lambda[k] * theta_im + (use_var_slopes ? (beta_item[k] * x_main[i] + var_fam_shift[i]) : 0.0);
            ll_m += bernoulli_logit_lpmf(Y_mat[i, k] | eta_ik);
          }
        }
        if (has_testlets) {
          for (t in 1:T_test) {
            if (lang_testlet_n[i, t] > 0) {
              vector[M_gh] lp_inner;
              for (r in 1:M_gh) {
                real ll_r = gh_log_w[r];
                real u_r = sd_testlet[1] * gh_q[r];
                for (k in 1:K_item) {
                  if (Y_mat[i, k] >= 0 && item_testlet[k] == t) {
                    real eta_ik = alpha[k] + lambda[k] * theta_im + (use_var_slopes ? (beta_item[k] * x_main[i] + var_fam_shift[i]) : 0.0) + u_r;
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

      // 2b. Tip-marginalized log_lik_lang_tip (M_out = 21)
      for (m in 1:M_out) {
        real theta_im = mu_theta_tip_i + sd_out_i * gh_out_q[m];
        real ll_m = gh_out_log_w[m];
        for (k in 1:K_item) {
          if (Y_mat[i, k] >= 0 && (!has_testlets || item_testlet[k] == 0)) {
            real eta_ik = alpha[k] + lambda[k] * theta_im + (use_var_slopes ? (beta_item[k] * x_main[i] + var_fam_shift[i]) : 0.0);
            ll_m += bernoulli_logit_lpmf(Y_mat[i, k] | eta_ik);
          }
        }
        if (has_testlets) {
          for (t in 1:T_test) {
            if (lang_testlet_n[i, t] > 0) {
              vector[M_gh] lp_inner;
              for (r in 1:M_gh) {
                real ll_r = gh_log_w[r];
                real u_r = sd_testlet[1] * gh_q[r];
                for (k in 1:K_item) {
                  if (Y_mat[i, k] >= 0 && item_testlet[k] == t) {
                    real eta_ik = alpha[k] + lambda[k] * theta_im + (use_var_slopes ? (beta_item[k] * x_main[i] + var_fam_shift[i]) : 0.0) + u_r;
                    ll_r += bernoulli_logit_lpmf(Y_mat[i, k] | eta_ik);
                  }
                }
                lp_inner[r] = ll_r;
              }
              ll_m += log_sum_exp(lp_inner);
            }
          }
        }
        lp_outer_tip[m] = ll_m;
      }
      log_lik_lang_tip[i] = log_sum_exp(lp_outer_tip);
    }
  }
}
