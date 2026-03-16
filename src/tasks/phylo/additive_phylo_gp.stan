data {
  int<lower=1> N;
  int<lower=1> D;              // 5 continuous predictors
  matrix[N, D] X;
  vector[N] y;
  matrix[N, N] Sigma_phylo;
  int<lower=1> N_int;          // 5 interactions
  array[N_int, 2] int int_idx; // {{1,2}, {1,3}, {1,5}, {2,3}, {2,5}}
}

transformed data {
  array[D] matrix[N, N] sq_dist;
  for (d in 1:D) {
    for (i in 1:N) {
      for (j in i:N) {
        sq_dist[d][i, j] = square(X[i, d] - X[j, d]);
        sq_dist[d][j, i] = sq_dist[d][i, j];
      }
    }
  }
}

parameters {
  vector<lower=0>[D] alpha_main;
  vector<lower=0>[D] rho_main;
  vector<lower=0>[N_int] alpha_int;
  array[N_int] vector<lower=0>[2] rho_int;
  real<lower=0> sigma_phylo;
  real<lower=0> sigma_noise;
}

model {
  alpha_main ~ student_t(4, 0, 1);
  rho_main ~ inv_gamma(5, 5);
  alpha_int ~ student_t(4, 0, 0.5);
  for (m in 1:N_int) rho_int[m] ~ inv_gamma(5, 5);
  sigma_phylo ~ student_t(4, 0, 1);
  sigma_noise ~ student_t(4, 0, 1);

  matrix[N, N] K = rep_matrix(0, N, N);
  for (d in 1:D)
    K += square(alpha_main[d]) * exp(-0.5 * sq_dist[d] / square(rho_main[d]));
  for (m in 1:N_int) {
    int d1 = int_idx[m, 1];
    int d2 = int_idx[m, 2];
    K += square(alpha_int[m])
         * exp(-0.5 * sq_dist[d1] / square(rho_int[m][1]))
         .* exp(-0.5 * sq_dist[d2] / square(rho_int[m][2]));
  }
  K += square(sigma_phylo) * Sigma_phylo;
  for (i in 1:N) K[i, i] += square(sigma_noise);

  y ~ multi_normal(rep_vector(0, N), K);
}

generated quantities {
  // Build K_signal = K_fixed + K_phylo (no noise)
  matrix[N, N] K_signal = rep_matrix(0, N, N);
  for (d in 1:D)
    K_signal += square(alpha_main[d]) * exp(-0.5 * sq_dist[d] / square(rho_main[d]));
  for (m in 1:N_int) {
    int d1 = int_idx[m, 1];
    int d2 = int_idx[m, 2];
    K_signal += square(alpha_int[m])
                * exp(-0.5 * sq_dist[d1] / square(rho_int[m][1]))
                .* exp(-0.5 * sq_dist[d2] / square(rho_int[m][2]));
  }
  K_signal += square(sigma_phylo) * Sigma_phylo;

  // K_total = K_signal + sigma_noise^2 * I
  matrix[N, N] K_total = K_signal;
  for (i in 1:N) K_total[i, i] += square(sigma_noise) + 1e-10;

  // Posterior of f|y: mean = K_signal * K_total^{-1} * y
  //                   cov  = K_signal - K_signal * K_total^{-1} * K_signal
  vector[N] f_mu = K_signal * mdivide_left_spd(K_total, y);

  // Posterior covariance: C = K_signal - K_signal * K_total^{-1} * K_signal
  matrix[N, N] W = mdivide_left_spd(K_total, K_signal);  // K_total^{-1} * K_signal
  matrix[N, N] C = K_signal - K_signal * W;
  // Symmetrize and add jitter for numerical stability
  C = 0.5 * (C + C');
  for (i in 1:N) C[i, i] += 1e-10;
  matrix[N, N] L_C = cholesky_decompose(C);

  // Sample f from posterior: f = f_mu + L_C * z, z ~ N(0, I)
  vector[N] z;
  for (i in 1:N) z[i] = std_normal_rng();
  vector[N] f_draw = f_mu + L_C * z;

  // Decompose f into fixed + phylo using K_signal^{-1} * f_draw
  // K_signal needs jitter for inversion
  matrix[N, N] K_signal_jit = K_signal;
  for (i in 1:N) K_signal_jit[i, i] += 1e-10;
  vector[N] alpha_signal = mdivide_left_spd(K_signal_jit, f_draw);

  // Component vectors: mu_d = K_d * alpha_signal
  array[D] vector[N] mu_main;
  for (d in 1:D) {
    matrix[N, N] K_d = square(alpha_main[d])
                        * exp(-0.5 * sq_dist[d] / square(rho_main[d]));
    mu_main[d] = K_d * alpha_signal;
  }

  array[N_int] vector[N] mu_int;
  for (m in 1:N_int) {
    int d1 = int_idx[m, 1];
    int d2 = int_idx[m, 2];
    matrix[N, N] K_m = square(alpha_int[m])
                        * exp(-0.5 * sq_dist[d1] / square(rho_int[m][1]))
                        .* exp(-0.5 * sq_dist[d2] / square(rho_int[m][2]));
    mu_int[m] = K_m * alpha_signal;
  }

  // Phylogenetic component
  vector[N] mu_phylo = square(sigma_phylo) * Sigma_phylo * alpha_signal;

  // f_draw = sum(mu_main) + sum(mu_int) + mu_phylo (exact by construction)

  // Independent noise: sample from prior, not y - f
  vector[N] eps;
  for (i in 1:N) eps[i] = normal_rng(0, sigma_noise);
}
