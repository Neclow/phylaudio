#!/usr/bin/env Rscript

# run_phylo_regression_nonlinear.R
#
# Additive GP phylogenetic regression with Shapley variance decomposition.
# Fits an exact additive GP in Stan (cmdstanr), then runs the same
# component-variance Shapley decomposition as the linear model.
#
# USAGE:
#   Rscript src/tasks/phylo/run_phylo_regression_nonlinear.R \
#       --beast_dir <path> [--variant with_inventory] \
#       [--iter_sampling 1000] [--iter_warmup 1000] \
#       [--adapt_delta 0.95] [--max_treedepth 12] [--seed 20231103]

# 0. Libraries
suppressPackageStartupMessages({
    library(cmdstanr)
    library(ape)
    library(dplyr)
    library(parallel)
})

source("src/tasks/phylo/phylo_regression_helpers.R")

# 1. Helper functions

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
    defaults <- list(
        beast_dir     = NULL,
        tree_file     = NULL,
        variant       = "with_inventory",
        iter_sampling = 1000L,
        iter_warmup   = 1000L,
        adapt_delta   = 0.95,
        max_treedepth = 12L,
        seed          = 20231103L
    )

    args <- args[args != "--"]
    i <- 1
    while (i <= length(args)) {
        key <- sub("^--", "", args[i])
        if (i + 1 > length(args)) stop(sprintf("Missing value for --%s", key))
        val <- args[i + 1]
        if (key %in% c("iter_sampling", "iter_warmup", "max_treedepth", "seed"))
            val <- as.integer(val)
        if (key == "adapt_delta") val <- as.numeric(val)
        defaults[[key]] <- val
        i <- i + 2
    }

    if (is.null(defaults$beast_dir)) stop("--beast_dir is required")
    defaults
}

# 2. Parse args & load data
cfg <- parse_args()
beast_dir    <- cfg$beast_dir
variant      <- cfg$variant
model_type   <- "gp_geo"

if (!variant %in% c("with_inventory", "no_inventory"))
    stop(sprintf("--variant must be 'with_inventory' or 'no_inventory', got '%s'", variant))

cat(sprintf("======================================================================\n"))
cat(sprintf("  model_type:     %s (additive GP)\n", model_type))
cat(sprintf("  beast_dir:      %s\n", beast_dir))
cat(sprintf("  variant:        %s\n", variant))
cat(sprintf("  iter_sampling:  %d\n", cfg$iter_sampling))
cat(sprintf("  iter_warmup:    %d\n", cfg$iter_warmup))
cat(sprintf("  adapt_delta:    %s\n", cfg$adapt_delta))
cat(sprintf("  max_treedepth:  %d\n", cfg$max_treedepth))
cat(sprintf("  seed:           %d\n", cfg$seed))
cat(sprintf("======================================================================\n\n"))

out_dir <- file.path(beast_dir, "phyloregression", variant)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

loaded <- load_regression_data(beast_dir, variant, cfg$tree_file)
df    <- loaded$df
tr    <- loaded$tr
V_raw <- loaded$V_raw

# 3. Normalize predictors & prepare Stan data
df <- normalize_predictors(df, variant)
use_segments <- variant == "with_inventory"

# Response (center for zero-mean GP)
y_raw <- log(df$rate_median)
y_mean <- mean(y_raw)
y <- y_raw - y_mean
cat(sprintf("Response centered: mean(y_raw) = %.4f removed\n", y_mean))

# Predictor matrix
if (use_segments) {
    pred_cols <- c("longitude_norm", "latitude_norm", "log_n_speakers_norm",
                   "n_phonemes_norm", "delta_norm")
    int_idx <- matrix(c(1,2, 1,3, 1,5, 2,3, 2,5), ncol = 2, byrow = TRUE)
    int_names <- c("longitude_norm:latitude_norm",
                   "longitude_norm:log_n_speakers_norm",
                   "longitude_norm:delta_norm",
                   "latitude_norm:log_n_speakers_norm",
                   "latitude_norm:delta_norm")
} else {
    pred_cols <- c("longitude_norm", "latitude_norm", "log_n_speakers_norm",
                   "delta_norm")
    int_idx <- matrix(c(1,2, 1,3, 1,4, 2,3, 2,4), ncol = 2, byrow = TRUE)
    int_names <- c("longitude_norm:latitude_norm",
                   "longitude_norm:log_n_speakers_norm",
                   "longitude_norm:delta_norm",
                   "latitude_norm:log_n_speakers_norm",
                   "latitude_norm:delta_norm")
}

X_pred <- as.matrix(df[, pred_cols])
N <- nrow(X_pred)
D <- ncol(X_pred)
N_int <- nrow(int_idx)

# Phylogenetic covariance
language_levels <- rownames(df)
Sigma_phylo <- V_raw[language_levels, language_levels]

stan_data <- list(
    N = N, D = D, X = X_pred, y = y,
    Sigma_phylo = Sigma_phylo,
    N_int = N_int, int_idx = int_idx
)

# 4. Compile and fit
stan_file <- "src/tasks/phylo/additive_phylo_gp.stan"
if (!file.exists(stan_file))
    stop("Cannot find additive_phylo_gp.stan")

cat(sprintf("\nCompiling Stan model: %s\n", stan_file))
mod <- cmdstan_model(stan_file)

cat(sprintf("\nFitting GP model (%s, %s)...\n", beast_dir, variant))
fit <- mod$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = min(4L, detectCores()),
    iter_warmup = cfg$iter_warmup,
    iter_sampling = cfg$iter_sampling,
    adapt_delta = cfg$adapt_delta,
    max_treedepth = cfg$max_treedepth,
    seed = cfg$seed
)

# 5. Diagnostics
cat("\n======================================================================\n")
cat("                    MCMC DIAGNOSTICS\n")
cat("======================================================================\n")

diag_summary <- fit$diagnostic_summary()
cat(sprintf("Divergent transitions: %s\n",
            paste(diag_summary$num_divergent, collapse = ", ")))
cat(sprintf("Max treedepth reached: %s\n",
            paste(diag_summary$num_max_treedepth, collapse = ", ")))
cat(sprintf("EBFMI: %s\n",
            paste(sprintf("%.3f", diag_summary$ebfmi), collapse = ", ")))

summ <- fit$summary()
cat(sprintf("\nMax Rhat:     %.4f\n", max(summ$rhat, na.rm = TRUE)))
cat(sprintf("Min ESS bulk: %.0f\n", min(summ$ess_bulk, na.rm = TRUE)))
cat(sprintf("Min ESS tail: %.0f\n", min(summ$ess_tail, na.rm = TRUE)))
cat("======================================================================\n\n")

n_div <- sum(diag_summary$num_divergent)

# 6. Extract component vectors from posterior
S <- cfg$iter_sampling * 4  # total posterior samples

cat("Extracting component vectors...\n")

mu_main_draws <- array(dim = c(S, N, D))
for (d in 1:D) {
    for (i in 1:N) {
        mu_main_draws[, i, d] <- as.numeric(
            fit$draws(paste0("mu_main[", d, ",", i, "]"), format = "matrix"))
    }
}

mu_int_draws <- array(dim = c(S, N, N_int))
for (m in 1:N_int) {
    for (i in 1:N) {
        mu_int_draws[, i, m] <- as.numeric(
            fit$draws(paste0("mu_int[", m, ",", i, "]"), format = "matrix"))
    }
}

mu_phylo_draws <- matrix(nrow = S, ncol = N)
for (i in 1:N) {
    mu_phylo_draws[, i] <- as.numeric(
        fit$draws(paste0("mu_phylo[", i, "]"), format = "matrix"))
}

f_draw_draws <- matrix(nrow = S, ncol = N)
for (i in 1:N) {
    f_draw_draws[, i] <- as.numeric(
        fit$draws(paste0("f_draw[", i, "]"), format = "matrix"))
}

sigma_noise_draws <- as.numeric(fit$draws("sigma_noise", format = "matrix"))

# 7. Variance decomposition
cat("Computing variance decomposition...\n")

mu_fixed_draws <- matrix(0, nrow = S, ncol = N)
for (d in 1:D) mu_fixed_draws <- mu_fixed_draws + mu_main_draws[, , d]
for (m in 1:N_int) mu_fixed_draws <- mu_fixed_draws + mu_int_draws[, , m]

recon_err <- max(abs(mu_fixed_draws[1, ] + mu_phylo_draws[1, ] - f_draw_draws[1, ]))
cat(sprintf("Reconstruction check |mu_fixed + mu_phylo - f_draw|: %.2e (should be ~0)\n",
            recon_err))

eps_draws <- matrix(nrow = S, ncol = N)
for (i in 1:N) {
    eps_draws[, i] <- as.numeric(
        fit$draws(paste0("eps[", i, "]"), format = "matrix"))
}

var_fixed <- apply(mu_fixed_draws, 1, var)
var_phylo <- apply(mu_phylo_draws, 1, var)
var_resid <- apply(eps_draws, 1, var)

cov_fix_phy <- sapply(1:S, function(s) cov(mu_fixed_draws[s, ], mu_phylo_draws[s, ]))

var_signal <- apply(f_draw_draws, 1, var)
V_total <- var_signal + var_resid

prop_fixed       <- var_fixed / V_total
prop_phylo       <- var_phylo / V_total
prop_cov_fix_phy <- 2 * cov_fix_phy / V_total
prop_residual    <- var_resid / V_total

R2_full <- prop_fixed + prop_phylo + prop_cov_fix_phy

decomp_check <- prop_fixed + prop_phylo + prop_cov_fix_phy + prop_residual
cat(sprintf("Decomposition check: %.4f +/- %.4f (ideally = 1)\n",
            mean(decomp_check), sd(decomp_check)))

# 8. Shapley decomposition of fixed effects
all_comp_names <- c(pred_cols, int_names)
n_comps <- length(all_comp_names)

cat(sprintf("\nComputing Shapley decomposition (%d terms, %d samples)...\n", n_comps, S))

results_shapley <- mclapply(1:S, function(s) {
    comps <- setNames(
        c(
            lapply(1:D, function(d) mu_main_draws[s, , d]),
            lapply(1:N_int, function(m) mu_int_draws[s, , m])
        ),
        all_comp_names
    )
    shapley_one_sample(comps)
}, mc.cores = max(1L, detectCores() - 1))

shap <- collect_shapley_results(results_shapley, all_comp_names)

cat(sprintf("V_fixed: %.4f (95%% CI: %.4f - %.4f)\n",
            shap$vfix_summary$mean, shap$vfix_summary$q2.5, shap$vfix_summary$q97.5))
cat(sprintf("Shapley sum check: %.6f (should be 1.0)\n", mean(rowSums(shap$shap_mat))))

# 9. Save outputs
file_prefix <- model_type

vardecomp <- list(
    R2_full = R2_full, prop_fixed = prop_fixed, prop_linear = prop_fixed,
    prop_phylo = prop_phylo, prop_residual = prop_residual,
    prop_cov_fix_phy = prop_cov_fix_phy, V_total = V_total
)

# 9a. Variance samples CSV
save_variance_samples(out_dir, file_prefix, beast_dir, model_type,
                       vardecomp, shap, all_comp_names)

# 9b. Summary CSV
csv_results <- data.frame(
    model             = "cmdstanr_gp",
    model_type        = model_type,
    beast_dir         = beast_dir,
    stringsAsFactors  = FALSE
)

csv_results$n_divergent    <- n_div
csv_results$max_rhat       <- max(summ$rhat, na.rm = TRUE)
csv_results$min_bulk_ess   <- min(summ$ess_bulk, na.rm = TRUE)
csv_results$min_tail_ess   <- min(summ$ess_tail, na.rm = TRUE)

csv_results$method                  <- "GP_Variance_Decomposition"
csv_results$total_response_variance <- mean(V_total)

csv_results$V_trace_norm <- mean(diag(Sigma_phylo))
csv_results$n_obs        <- N

csv_results <- add_decomp_columns(csv_results, vardecomp, shap)

csv_path <- file.path(out_dir, paste0("phylolm_", file_prefix, ".csv"))
write.csv(csv_results, csv_path, row.names = FALSE)
cat("Summary CSV written to", csv_path, "\n")

# 10. Print summary
print_variance_summary(model_type, beast_dir, vardecomp, shap)
