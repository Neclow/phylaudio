#!/usr/bin/env Rscript

# run_phylo_regression_nonlinear.R
#
# Additive GP phylogenetic regression with Shapley variance decomposition.
# Fits an exact additive GP in Stan (cmdstanr), then runs the same
# component-variance Shapley decomposition as the linear model.
#
# USAGE:
#   Rscript final_scripts/run_phylo_regression_nonlinear.R \
#       --tree <tree_name> [--variant with_inventory] \
#       [--iter_sampling 1000] [--iter_warmup 1000] \
#       [--adapt_delta 0.95] [--max_treedepth 12] [--seed 20231103]

# ═══════════════════════════════════════════════════════════════════════════════
# 0. Libraries
# ═══════════════════════════════════════════════════════════════════════════════
suppressPackageStartupMessages({
    library(cmdstanr)
    library(ape)
    library(dplyr)
    library(parallel)
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
    defaults <- list(
        tree          = NULL,
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

    if (is.null(defaults$tree)) stop("--tree is required")
    defaults
}

apply_coord_scaling <- function(vec, shift = 10) {
    scaled <- scale(vec)
    list(
        values = as.numeric(scaled),
        center = as.numeric(attr(scaled, "scaled:center")),
        scale  = as.numeric(attr(scaled, "scaled:scale"))
    )
}

summarize_posterior <- function(x) {
    list(
        mean  = mean(x, na.rm = TRUE),
        sd    = sd(x, na.rm = TRUE),
        q2.5  = unname(quantile(x, 0.025, na.rm = TRUE)),
        q50   = unname(quantile(x, 0.50, na.rm = TRUE)),
        q97.5 = unname(quantile(x, 0.975, na.rm = TRUE))
    )
}

shapley_one_sample <- function(components) {
    D <- length(components)
    comp_names <- names(components)
    comp_mat <- do.call(cbind, components)

    V_fixed <- var(rowSums(comp_mat))

    n_coal <- 2^D
    v <- numeric(n_coal)
    for (bits in 1:(n_coal - 1)) {
        members <- which(as.logical(intToBits(bits)[1:D]))
        v[bits + 1] <- var(rowSums(comp_mat[, members, drop = FALSE]))
    }

    shapley_raw <- numeric(D)
    names(shapley_raw) <- comp_names

    for (j in 1:D) {
        j_bit <- 2^(j - 1)
        phi <- 0
        for (bits in 0:(n_coal - 1)) {
            if (bitwAnd(bits, j_bit) != 0) next
            s_size <- sum(as.logical(intToBits(bits)[1:D]))
            weight <- factorial(s_size) * factorial(D - s_size - 1) / factorial(D)
            phi <- phi + weight * (v[bitwOr(bits, j_bit) + 1] - v[bits + 1])
        }
        shapley_raw[j] <- phi
    }

    marginal_raw <- apply(comp_mat, 2, var)
    names(marginal_raw) <- comp_names

    list(
        shapley  = shapley_raw / V_fixed,
        marginal = marginal_raw / V_fixed,
        V_fixed  = V_fixed
    )
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Parse args & load data
# ═══════════════════════════════════════════════════════════════════════════════
cfg <- parse_args()
tree_name    <- cfg$tree
variant      <- cfg$variant
model_type   <- "gp_geo"

if (!variant %in% c("with_inventory", "no_inventory"))
    stop(sprintf("--variant must be 'with_inventory' or 'no_inventory', got '%s'", variant))

cat(sprintf("======================================================================\n"))
cat(sprintf("  model_type:     %s (additive GP)\n", model_type))
cat(sprintf("  tree:           %s\n", tree_name))
cat(sprintf("  variant:        %s\n", variant))
cat(sprintf("  iter_sampling:  %d\n", cfg$iter_sampling))
cat(sprintf("  iter_warmup:    %d\n", cfg$iter_warmup))
cat(sprintf("  adapt_delta:    %s\n", cfg$adapt_delta))
cat(sprintf("  max_treedepth:  %d\n", cfg$max_treedepth))
cat(sprintf("  seed:           %d\n", cfg$seed))
cat(sprintf("======================================================================\n\n"))

out_dir <- file.path("data/phyloregression", variant)
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

source("src/tasks/phylo/beast.R")

tree_mcc <- list(
    input_v12_combined_resampled = "data/trees/beast/input_v12_combined_resampled.mcc",
    heggarty2024_raw             = "data/trees/beast/IECoR_Main_M3_Binary_Covarion_Rates_By_Mg_Bin_mcc.tree"
)
tree_path <- tree_mcc[[tree_name]]
if (is.null(tree_path)) stop(sprintf("Unknown tree name: %s", tree_name))
tr <- read.annot.beast(tree_path)

tree_to_csv <- if (variant == "with_inventory") {
    list(
        input_v12_combined_resampled = "data/phyloregression/speech_metadata_with_inventory.csv",
        heggarty2024_raw             = "data/phyloregression/cognate_metadata_with_inventory.csv"
    )
} else {
    list(
        input_v12_combined_resampled = "data/phyloregression/speech_metadata.csv",
        heggarty2024_raw             = "data/phyloregression/cognate_metadata.csv"
    )
}
reg_data_path <- tree_to_csv[[tree_name]]
if (is.null(reg_data_path))
    stop(sprintf("No metadata CSV mapped for tree '%s'", tree_name))
df <- read.csv(reg_data_path, stringsAsFactors = FALSE)
rownames(df) <- df$language

tr    <- drop.tip(tr, setdiff(tr$tip.label, rownames(df)))
V_raw <- vcv(unroot(tr))
V_raw <- V_raw[rownames(df), rownames(df)]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Normalize predictors & prepare Stan data
# ═══════════════════════════════════════════════════════════════════════════════
log_n_scaled <- scale(log(df$n_speakers))
df$log_n_speakers_norm <- log_n_scaled[, 1]

lon_stats <- apply_coord_scaling(df$longitude)
lat_stats <- apply_coord_scaling(df$latitude)
df$longitude_norm <- lon_stats$values
df$latitude_norm  <- lat_stats$values

use_segments <- variant == "with_inventory"
if (use_segments) {
    df$n_phonemes_norm <- as.numeric(scale(df$n_phonemes))
}
df$delta_norm <- as.numeric(scale(df$delta))

# Response (center for zero-mean GP)
y_raw <- log(df$rate_median)
y_mean <- mean(y_raw)
y <- y_raw - y_mean
cat(sprintf("Response centered: mean(y_raw) = %.4f removed\n", y_mean))

# Predictor matrix
if (use_segments) {
    pred_cols <- c("longitude_norm", "latitude_norm", "log_n_speakers_norm",
                   "n_phonemes_norm", "delta_norm")
    # Interactions: {lon,lat}, {lon,lns}, {lon,delta}, {lat,lns}, {lat,delta}
    # Column indices:  1,2      1,3       1,5          2,3        2,5
    int_idx <- matrix(c(1,2, 1,3, 1,5, 2,3, 2,5), ncol = 2, byrow = TRUE)
    int_names <- c("longitude_norm:latitude_norm",
                   "longitude_norm:log_n_speakers_norm",
                   "longitude_norm:delta_norm",
                   "latitude_norm:log_n_speakers_norm",
                   "latitude_norm:delta_norm")
} else {
    pred_cols <- c("longitude_norm", "latitude_norm", "log_n_speakers_norm",
                   "delta_norm")
    # Column indices:  1,2      1,3       1,4          2,3        2,4
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

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Compile and fit
# ═══════════════════════════════════════════════════════════════════════════════
stan_file <- "src/tasks/phylo/additive_phylo_gp.stan"
if (!file.exists(stan_file))
    stop("Cannot find additive_phylo_gp.stan")

cat(sprintf("\nCompiling Stan model: %s\n", stan_file))
mod <- cmdstan_model(stan_file)

cat(sprintf("\nFitting GP model (%s, %s)...\n", tree_name, variant))
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

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════
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

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Extract component vectors from posterior
# ═══════════════════════════════════════════════════════════════════════════════
S <- cfg$iter_sampling * 4  # total posterior samples

cat("Extracting component vectors...\n")

# mu_main[d, i] -> S x N x D array
mu_main_draws <- array(dim = c(S, N, D))
for (d in 1:D) {
    for (i in 1:N) {
        mu_main_draws[, i, d] <- as.numeric(
            fit$draws(paste0("mu_main[", d, ",", i, "]"), format = "matrix"))
    }
}

# mu_int[m, i] -> S x N x N_int array
mu_int_draws <- array(dim = c(S, N, N_int))
for (m in 1:N_int) {
    for (i in 1:N) {
        mu_int_draws[, i, m] <- as.numeric(
            fit$draws(paste0("mu_int[", m, ",", i, "]"), format = "matrix"))
    }
}

# mu_phylo[i] -> S x N matrix (extracted from Stan GQ)
mu_phylo_draws <- matrix(nrow = S, ncol = N)
for (i in 1:N) {
    mu_phylo_draws[, i] <- as.numeric(
        fit$draws(paste0("mu_phylo[", i, "]"), format = "matrix"))
}

# f_draw[i] -> S x N matrix (posterior sample of signal f)
f_draw_draws <- matrix(nrow = S, ncol = N)
for (i in 1:N) {
    f_draw_draws[, i] <- as.numeric(
        fit$draws(paste0("f_draw[", i, "]"), format = "matrix"))
}

# sigma_noise (for residual variance)
sigma_noise_draws <- as.numeric(fit$draws("sigma_noise", format = "matrix"))

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Variance decomposition
# ═══════════════════════════════════════════════════════════════════════════════
cat("Computing variance decomposition...\n")

# Fixed effect = sum of all main + interaction components
mu_fixed_draws <- matrix(0, nrow = S, ncol = N)
for (d in 1:D) mu_fixed_draws <- mu_fixed_draws + mu_main_draws[, , d]
for (m in 1:N_int) mu_fixed_draws <- mu_fixed_draws + mu_int_draws[, , m]

# Verify reconstruction: f_draw = mu_fixed + mu_phylo (exact by construction)
recon_err <- max(abs(mu_fixed_draws[1, ] + mu_phylo_draws[1, ] - f_draw_draws[1, ]))
cat(sprintf("Reconstruction check |mu_fixed + mu_phylo - f_draw|: %.2e (should be ~0)\n",
            recon_err))

# Noise is independent: sampled from prior eps ~ N(0, sigma_noise) in Stan GQ
eps_draws <- matrix(nrow = S, ncol = N)
for (i in 1:N) {
    eps_draws[, i] <- as.numeric(
        fit$draws(paste0("eps[", i, "]"), format = "matrix"))
}

# V_total: signal + noise, computed per sample then averaged
# var(f_draw) + var(eps) since eps is independent of f
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

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Shapley decomposition of fixed effects
# ═══════════════════════════════════════════════════════════════════════════════
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

shap_mat <- do.call(rbind, lapply(results_shapley, function(r) r$shapley))
marg_mat <- do.call(rbind, lapply(results_shapley, function(r) r$marginal))
vfix_vec <- sapply(results_shapley, function(r) r$V_fixed)

term_props <- list()
marg_props <- list()
for (j in 1:n_comps) {
    term_props[[all_comp_names[j]]] <- summarize_posterior(shap_mat[, j])
    marg_props[[all_comp_names[j]]] <- summarize_posterior(marg_mat[, j])
}
vfix_summary <- summarize_posterior(vfix_vec)

cat(sprintf("V_fixed: %.4f (95%% CI: %.4f - %.4f)\n",
            vfix_summary$mean, vfix_summary$q2.5, vfix_summary$q97.5))
cat(sprintf("Shapley sum check: %.6f (should be 1.0)\n", mean(rowSums(shap_mat))))

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Save outputs
# ═══════════════════════════════════════════════════════════════════════════════
file_prefix <- paste0(model_type, "_", tree_name)

# --- 9a. Variance samples CSV ---
samples_df <- data.frame(
    sample_id     = 1:S,
    tree          = tree_name,
    model_type    = model_type,
    R2_full       = R2_full,
    prop_fixed    = prop_fixed,
    prop_linear   = prop_fixed,  # for GP, prop_linear == prop_fixed
    prop_phylo    = prop_phylo,
    prop_residual = prop_residual,
    prop_cov_fix_phy = prop_cov_fix_phy,
    V_total       = V_total,
    V_fixed       = vfix_vec
)

for (term in all_comp_names) {
    term_clean <- gsub(":", "_", term)
    samples_df[[paste0("shapley_", term_clean)]]  <- shap_mat[, term]
    samples_df[[paste0("marginal_", term_clean)]] <- marg_mat[, term]
}

samples_path <- file.path(out_dir, paste0("variance_samples_", file_prefix, ".csv"))
write.csv(samples_df, samples_path, row.names = FALSE)
cat("Posterior samples written to", samples_path, "\n")

# --- 9b. Summary CSV ---
csv_results <- data.frame(
    model             = "cmdstanr_gp",
    model_type        = model_type,
    tree              = tree_name,
    stringsAsFactors  = FALSE
)

csv_results$n_divergent    <- n_div
csv_results$max_rhat       <- max(summ$rhat, na.rm = TRUE)
csv_results$min_bulk_ess   <- min(summ$ess_bulk, na.rm = TRUE)
csv_results$min_tail_ess   <- min(summ$ess_tail, na.rm = TRUE)

csv_results$method                  <- "GP_Variance_Decomposition"
csv_results$total_response_variance <- mean(V_total)

csv_results$R2_full_mean  <- summarize_posterior(R2_full)$mean
csv_results$R2_full_q2_5  <- summarize_posterior(R2_full)$q2.5
csv_results$R2_full_q97_5 <- summarize_posterior(R2_full)$q97.5

csv_results$prop_fixed_mean   <- summarize_posterior(prop_fixed)$mean
csv_results$prop_fixed_q2_5   <- summarize_posterior(prop_fixed)$q2.5
csv_results$prop_fixed_q97_5  <- summarize_posterior(prop_fixed)$q97.5

csv_results$prop_linear_mean  <- csv_results$prop_fixed_mean
csv_results$prop_linear_q2_5  <- csv_results$prop_fixed_q2_5
csv_results$prop_linear_q97_5 <- csv_results$prop_fixed_q97_5

csv_results$prop_phylo_mean   <- summarize_posterior(prop_phylo)$mean
csv_results$prop_phylo_q2_5   <- summarize_posterior(prop_phylo)$q2.5
csv_results$prop_phylo_q97_5  <- summarize_posterior(prop_phylo)$q97.5

csv_results$prop_cov_fix_phy_mean  <- summarize_posterior(prop_cov_fix_phy)$mean
csv_results$prop_cov_fix_phy_q2_5  <- summarize_posterior(prop_cov_fix_phy)$q2.5
csv_results$prop_cov_fix_phy_q97_5 <- summarize_posterior(prop_cov_fix_phy)$q97.5

csv_results$prop_residual_mean  <- summarize_posterior(prop_residual)$mean
csv_results$prop_residual_q2_5  <- summarize_posterior(prop_residual)$q2.5
csv_results$prop_residual_q97_5 <- summarize_posterior(prop_residual)$q97.5

csv_results$V_trace_norm <- mean(diag(Sigma_phylo))
csv_results$n_obs        <- N

csv_results$V_fixed_mean  <- vfix_summary$mean
csv_results$V_fixed_q2_5  <- vfix_summary$q2.5
csv_results$V_fixed_q97_5 <- vfix_summary$q97.5

for (term in names(term_props)) {
    tc <- gsub("\\(|\\)", "", gsub(":", "_", term))
    csv_results[[paste0("shapley_", tc, "_mean")]]  <- term_props[[term]]$mean
    csv_results[[paste0("shapley_", tc, "_q2_5")]]  <- term_props[[term]]$q2.5
    csv_results[[paste0("shapley_", tc, "_q97_5")]] <- term_props[[term]]$q97.5
}

for (term in names(marg_props)) {
    tc <- gsub("\\(|\\)", "", gsub(":", "_", term))
    csv_results[[paste0("marginal_", tc, "_mean")]]  <- marg_props[[term]]$mean
    csv_results[[paste0("marginal_", tc, "_q2_5")]]  <- marg_props[[term]]$q2.5
    csv_results[[paste0("marginal_", tc, "_q97_5")]] <- marg_props[[term]]$q97.5
}

csv_path <- file.path(out_dir, paste0("phylolm_", file_prefix, ".csv"))
write.csv(csv_results, csv_path, row.names = FALSE)
cat("Summary CSV written to", csv_path, "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Print summary
# ═══════════════════════════════════════════════════════════════════════════════
cat("\n======================================================================\n")
cat(sprintf("COMPLETED: %s x %s x %s\n", model_type, tree_name, variant))
cat("======================================================================\n")
cat(sprintf("  R2: %.3f (95%% CI: %.3f - %.3f)\n",
            summarize_posterior(R2_full)$mean,
            summarize_posterior(R2_full)$q2.5,
            summarize_posterior(R2_full)$q97.5))

cat(sprintf("\n  Variance Components (Proportion of V_total, mean = %.4f):\n", mean(V_total)))
cat(sprintf("    Fixed Effects:    %.1f%% (95%% CI: %.1f - %.1f)\n",
            summarize_posterior(prop_fixed)$mean * 100,
            summarize_posterior(prop_fixed)$q2.5 * 100,
            summarize_posterior(prop_fixed)$q97.5 * 100))
cat(sprintf("    Phylogenetic:     %.1f%% (95%% CI: %.1f - %.1f)\n",
            summarize_posterior(prop_phylo)$mean * 100,
            summarize_posterior(prop_phylo)$q2.5 * 100,
            summarize_posterior(prop_phylo)$q97.5 * 100))
cat(sprintf("    Cov(Fixed,Phylo): %.1f%% (95%% CI: %.1f - %.1f)\n",
            summarize_posterior(prop_cov_fix_phy)$mean * 100,
            summarize_posterior(prop_cov_fix_phy)$q2.5 * 100,
            summarize_posterior(prop_cov_fix_phy)$q97.5 * 100))
cat(sprintf("    Residual:         %.1f%% (95%% CI: %.1f - %.1f)\n",
            summarize_posterior(prop_residual)$mean * 100,
            summarize_posterior(prop_residual)$q2.5 * 100,
            summarize_posterior(prop_residual)$q97.5 * 100))

cat(sprintf("\n  V_fixed: %.4f (95%% CI: %.4f - %.4f)\n",
            vfix_summary$mean, vfix_summary$q2.5, vfix_summary$q97.5))

cat("\n  Per-term Shapley (proportion of V_fixed, sum to 1):\n")
for (term in names(term_props)) {
    cat(sprintf("    %-40s: %.1f%% (95%% CI: %.1f - %.1f)\n", term,
                term_props[[term]]$mean * 100,
                term_props[[term]]$q2.5 * 100,
                term_props[[term]]$q97.5 * 100))
}

cat("\n  Per-term Marginal Variance (proportion of V_fixed, do NOT sum to 1):\n")
for (term in names(marg_props)) {
    cat(sprintf("    %-40s: %.1f%% (95%% CI: %.1f - %.1f)\n", term,
                marg_props[[term]]$mean * 100,
                marg_props[[term]]$q2.5 * 100,
                marg_props[[term]]$q97.5 * 100))
}
cat("======================================================================\n")
