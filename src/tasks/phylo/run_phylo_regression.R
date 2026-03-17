#!/usr/bin/env Rscript

# run_phylo_regression.R
#
# Bayesian phylogenetic regression script (linear models only).
#
# USAGE:
#   Rscript final_scripts/run_phylo_regression.R \
#       --model_type <type> --tree <tree_name> \
#       [--variant with_inventory] \
#       [--iter 30000] [--warmup 10000] [--adapt_delta 0.9999] [--thin 10]
#
# MODEL TYPES:
#   linear_geo          : all main effects + lon×lat + {delta,n_speakers}×{lon,lat}

# ═══════════════════════════════════════════════════════════════════════════════
# 0. Libraries
# ═══════════════════════════════════════════════════════════════════════════════
suppressPackageStartupMessages({
    library(ape)
    library(brms)
    library(dplyr)
    library(tibble)
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

parse_args <- function(args = commandArgs(trailingOnly = TRUE)) {
    defaults <- list(
        model_type   = NULL,
        tree         = NULL,
        variant      = "with_inventory",
        iter         = 100000,
        warmup       = 20000,
        adapt_delta  = 0.9999,
        thin         = 10,
        seed         = 20231103L
    )

    valid_models <- c("linear_geo")

    args <- args[args != "--"]
    i <- 1
    while (i <= length(args)) {
        key <- sub("^--", "", args[i])
        if (i + 1 > length(args)) stop(sprintf("Missing value for --%s", key))
        val <- args[i + 1]
        if (key %in% c("iter", "warmup", "thin", "seed")) val <- as.integer(val)
        if (key == "adapt_delta") val <- as.numeric(val)
        defaults[[key]] <- val
        i <- i + 2
    }

    if (is.null(defaults$model_type))
        stop("--model_type is required")
    if (!(defaults$model_type %in% valid_models))
        stop(sprintf("Invalid model_type '%s'. Must be one of: %s",
                      defaults$model_type, paste(valid_models, collapse = ", ")))
    if (is.null(defaults$tree))
        stop("--tree is required")

    defaults
}

apply_coord_scaling <- function(vec) {
    transformed <- vec
    scaled <- scale(transformed)
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

build_brms_formula <- function(model_type, use_segments = TRUE) {
    seg <- if (use_segments) "n_phonemes_norm + " else ""
    seg_interactions <- if (use_segments) paste0(
        "log_n_speakers_norm:n_phonemes_norm + ",
        "n_phonemes_norm:delta_norm + ") else ""
    seg_triple <- if (use_segments) "log_n_speakers_norm:n_phonemes_norm:delta_norm + " else ""

    re <- "(1 | gr(language_factor, cov = V))"

    geo_interactions <- paste0(
        "longitude_norm:latitude_norm + ",
        "longitude_norm:log_n_speakers_norm + ",
        "longitude_norm:delta_norm + ",
        "latitude_norm:log_n_speakers_norm + ",
        "latitude_norm:delta_norm")
    fstr <- paste0(
        "log_rate_median ~ longitude_norm + latitude_norm + ",
        "log_n_speakers_norm + ", seg, "delta_norm + ",
        geo_interactions, " + ", re)

    bf(as.formula(fstr), decomp = "QR")
}

build_design_formula <- function(model_type, use_segments = TRUE) {
    seg <- if (use_segments) " + n_phonemes_norm" else ""
    as.formula(paste0("~ longitude_norm + latitude_norm + log_n_speakers_norm",
                        seg, " + delta_norm + ",
                        "longitude_norm:latitude_norm + ",
                        "longitude_norm:log_n_speakers_norm + ",
                        "longitude_norm:delta_norm + ",
                        "latitude_norm:log_n_speakers_norm + ",
                        "latitude_norm:delta_norm"))
}

run_mcmc_diagnostics <- function(model, brm_args) {
    cat("\n======================================================================\n")
    cat("                    MCMC DIAGNOSTICS\n")
    cat("======================================================================\n")

    np <- nuts_params(model)
    n_div <- sum(np[np$Parameter == "divergent__", "Value"])
    n_samples <- (brm_args$iter - brm_args$warmup) * brm_args$chains
    cat(sprintf("Divergent transitions: %d / %d (%.2f%%)\n",
                n_div, n_samples, 100 * n_div / n_samples))
    if (n_div > 0) cat("  [!] Divergences detected - consider increasing adapt_delta\n")

    max_td <- sum(np[np$Parameter == "treedepth__", "Value"] >= brm_args$control$max_treedepth)
    cat(sprintf("Max treedepth reached: %d times (max=%d)\n",
                max_td, brm_args$control$max_treedepth))

    energy <- np[np$Parameter == "energy__", "Value"]
    if (length(energy) > 0) {
        chain_ids <- np[np$Parameter == "energy__", "Chain"]
        chains <- unique(chain_ids)
        bfmi_vals <- sapply(chains, function(ch) {
            e <- energy[chain_ids == ch]
            var(diff(e)) / var(e)
        })
        cat(sprintf("BFMI per chain: %s\n",
                     paste(sprintf("%.3f", bfmi_vals), collapse = ", ")))
        if (any(bfmi_vals < 0.2))
            cat("  [!] Low BFMI (<0.2) detected - may indicate poor posterior exploration\n")
    }

    summ_fixed  <- summary(model)$fixed
    summ_spec   <- summary(model)$spec_pars
    summ_random <- tryCatch(summary(model)$random$language_factor, error = function(e) NULL)

    all_rhat     <- c(summ_fixed$Rhat, summ_spec$Rhat)
    all_bulk_ess <- c(summ_fixed$Bulk_ESS, summ_spec$Bulk_ESS)
    all_tail_ess <- c(summ_fixed$Tail_ESS, summ_spec$Tail_ESS)

    if (!is.null(summ_random)) {
        all_rhat     <- c(all_rhat, summ_random$Rhat)
        all_bulk_ess <- c(all_bulk_ess, summ_random$Bulk_ESS)
        all_tail_ess <- c(all_tail_ess, summ_random$Tail_ESS)
    }

    cat("\n--- Convergence (all parameters) ---\n")
    cat(sprintf("  Max Rhat:      %.4f %s\n", max(all_rhat, na.rm = TRUE),
                ifelse(max(all_rhat, na.rm = TRUE) < 1.01, "[OK]", "[!] > 1.01")))
    cat(sprintf("  Min Bulk ESS:  %.0f %s\n", min(all_bulk_ess, na.rm = TRUE),
                ifelse(min(all_bulk_ess, na.rm = TRUE) >= 400, "[OK]", "[!] < 400")))
    cat(sprintf("  Min Tail ESS:  %.0f %s\n", min(all_tail_ess, na.rm = TRUE),
                ifelse(min(all_tail_ess, na.rm = TRUE) >= 400, "[OK]", "[!] < 400")))

    cat("\n--- Fixed effects summary ---\n")
    cat(sprintf("  Max Rhat:      %.4f\n", max(summ_fixed$Rhat)))
    cat(sprintf("  Min Bulk ESS:  %.0f\n", min(summ_fixed$Bulk_ESS)))
    cat(sprintf("  Min Tail ESS:  %.0f\n", min(summ_fixed$Tail_ESS)))
    cat("======================================================================\n\n")

    list(
        n_div          = n_div,
        summ_fixed     = summ_fixed,
        all_rhat       = all_rhat,
        all_bulk_ess   = all_bulk_ess,
        all_tail_ess   = all_tail_ess
    )
}

extract_fixef_samples <- function(model) {
    all_fixef_names <- rownames(brms::fixef(model))
    fixef_names <- all_fixef_names[!grepl("^t2|^s\\(|^te\\(|^ti\\(", all_fixef_names)]

    post_colnames <- colnames(as.matrix(model))
    fixef_cols <- sapply(fixef_names, function(fn) {
        candidate <- paste0("b_", fn)
        if (candidate %in% post_colnames) return(candidate)
        candidate_alt <- paste0("b_", gsub(":", ".", fn))
        if (candidate_alt %in% post_colnames) return(candidate_alt)
        stop(sprintf("Could not find posterior column for '%s'. Available b_ cols: %s",
                      fn, paste(grep("^b_", post_colnames, value = TRUE), collapse = ", ")))
    })

    fixef_samples <- as.matrix(model)[, fixef_cols, drop = FALSE]
    colnames(fixef_samples) <- fixef_names
    fixef_samples
}

align_fixef_to_design <- function(fixef_samples, X, S) {
    X_colnames <- colnames(X)
    fixef_reordered <- matrix(NA, nrow = S, ncol = ncol(X))
    colnames(fixef_reordered) <- X_colnames

    for (j in seq_along(X_colnames)) {
        xname <- X_colnames[j]
        if (xname %in% colnames(fixef_samples)) {
            fixef_reordered[, j] <- fixef_samples[, xname]
        } else if (xname == "(Intercept)" && "Intercept" %in% colnames(fixef_samples)) {
            fixef_reordered[, j] <- fixef_samples[, "Intercept"]
        } else {
            xname_alt <- gsub(":", ".", xname)
            if (xname_alt %in% colnames(fixef_samples)) {
                fixef_reordered[, j] <- fixef_samples[, xname_alt]
            } else {
                stop(sprintf("Could not find coefficient for '%s'. Available: %s",
                              xname, paste(colnames(fixef_samples), collapse = ", ")))
            }
        }
    }
    fixef_reordered
}

shapley_one_sample <- function(components) {
    D <- length(components)
    comp_names <- names(components)
    comp_mat <- do.call(cbind, components)  # N x D

    V_fixed <- var(rowSums(comp_mat))

    # v(S) for all 2^D coalitions
    n_coal <- 2^D
    v <- numeric(n_coal)
    for (bits in 1:(n_coal - 1)) {
        members <- which(as.logical(intToBits(bits)[1:D]))
        v[bits + 1] <- var(rowSums(comp_mat[, members, drop = FALSE]))
    }

    # Shapley values
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

    # Marginal variances
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
model_type   <- cfg$model_type
tree_name    <- cfg$tree
variant      <- cfg$variant

if (!variant %in% c("with_inventory", "no_inventory"))
    stop(sprintf("--variant must be 'with_inventory' or 'no_inventory', got '%s'", variant))

cat(sprintf("======================================================================\n"))
cat(sprintf("  model_type:   %s\n", model_type))
cat(sprintf("  tree:         %s\n", tree_name))
cat(sprintf("  variant:      %s\n", variant))
cat(sprintf("  iter:         %d\n", cfg$iter))
cat(sprintf("  warmup:       %d\n", cfg$warmup))
cat(sprintf("  adapt_delta:  %s\n", cfg$adapt_delta))
cat(sprintf("  thin:         %d\n", cfg$thin))
cat(sprintf("  seed:         %d\n", cfg$seed))
cat(sprintf("======================================================================\n\n"))

out_dir <- file.path("data/phyloregression", variant)
if (cfg$seed != 20231103L) {
    out_dir <- file.path(out_dir, paste0("seed_", cfg$seed))
}
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

source("src/tasks/phylo/beast.R")

tree_mcc <- list(
    input_v12_combined_resampled = "data/trees/beast/input_v12_combined_resampled.mcc",
    heggarty2024_raw             = "data/trees/references/raw/iecor.nex"
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
# 3. Normalize predictors
# ═══════════════════════════════════════════════════════════════════════════════
log_n_scaled <- scale(log(df$n_speakers))
df$log_n_speakers_norm <- log_n_scaled[, 1]

lon_stats <- apply_coord_scaling(df$longitude)
lat_stats <- apply_coord_scaling(df$latitude)
df$longitude_norm <- lon_stats$values
df$latitude_norm  <- lat_stats$values

x_center <- lon_stats$center
x_scale  <- lon_stats$scale
y_center <- lat_stats$center
y_scale  <- lat_stats$scale

use_segments <- variant == "with_inventory"
if (use_segments) {
    df$n_phonemes_norm <- as.numeric(scale(df$n_phonemes))
}
df$delta_norm <- as.numeric(scale(df$delta))

model_df <- df
model_df$log_rate_median <- log(model_df$rate_median)
language_levels <- rownames(model_df)
model_df$language_factor <- factor(language_levels, levels = language_levels)

V <- V_raw[language_levels, language_levels]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Build formula & fit brms model
# ═══════════════════════════════════════════════════════════════════════════════
formula_obj <- build_brms_formula(model_type, use_segments)

brm_args <- list(
    formula = formula_obj,
    data    = model_df,
    family  = gaussian(),
    backend = "cmdstanr",
    refresh = 0,
    chains  = 4,
    iter    = cfg$iter,
    warmup  = cfg$warmup,
    seed    = cfg$seed,
    cores   = max(1L, min(4L, parallel::detectCores())),
    control = list(adapt_delta = cfg$adapt_delta, max_treedepth = 15),
    data2   = list(V = V),
    thin    = cfg$thin
)

cat(sprintf("\nFitting model (%s, %s)...\n", model_type, tree_name))
model <- do.call(brm, brm_args)

model_rds_path <- file.path(out_dir,
    paste0("model_", model_type, "_", tree_name, ".rds"))
saveRDS(model, model_rds_path)
cat("Model saved to", model_rds_path, "\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════
diag <- run_mcmc_diagnostics(model, brm_args)

posterior <- as.data.frame(model)
S <- nrow(posterior)
N <- nrow(model_df)

coef_df <- as.data.frame(brms::fixef(model, probs = c(0.025, 0.975)))
colnames(coef_df) <- c("Estimate", "Est.Error", "CI_lower", "CI_upper")
cat("\nPosterior summaries:\n")
print(coef_df)

r2_vals <- as.numeric(brms::bayes_R2(model))
cat(sprintf("\nBayes R2: %.3f (95%% CI: %.3f - %.3f)\n",
            mean(r2_vals), quantile(r2_vals, 0.025), quantile(r2_vals, 0.975)))

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Variance decomposition (posterior_linpred based, Gelman 2019)
# ═══════════════════════════════════════════════════════════════════════════════
cat("\nExtracting variance components via posterior_linpred...\n")

eta_full  <- t(brms::posterior_linpred(model, re_formula = NULL, transform = FALSE))  # N x S
eta_no_re <- t(brms::posterior_linpred(model, re_formula = NA,   transform = FALSE))  # N x S
eta_phylo <- eta_full - eta_no_re  # N x S

# Verify posterior_epred matches posterior_linpred for Gaussian
ypred_epred   <- posterior_epred(model)
ypred_linpred <- t(eta_full)
cat(sprintf("Max diff epred vs linpred: %.6f\n", max(abs(ypred_epred - ypred_linpred))))

# Extract fixed-effect coefficients & build design matrix
fixef_samples    <- extract_fixef_samples(model)
design_formula   <- build_design_formula(model_type, use_segments)
X                <- model.matrix(design_formula, data = model_df)
fixef_reordered  <- align_fixef_to_design(fixef_samples, X, S)
eta_linear       <- X %*% t(fixef_reordered)  # N x S

sigma_samples <- as.matrix(model)[, "sigma"]

# Per-sample variances
var_eta_full   <- apply(eta_full, 2, var)
var_eta_fixed  <- apply(eta_no_re, 2, var)
var_eta_linear <- apply(eta_linear, 2, var)
var_eta_phylo  <- apply(eta_phylo, 2, var)

V_total       <- var_eta_full + sigma_samples^2
prop_fixed    <- var_eta_fixed / V_total
prop_linear   <- var_eta_linear / V_total
prop_phylo    <- var_eta_phylo / V_total
prop_residual <- sigma_samples^2 / V_total
R2_full       <- var_eta_full / V_total

# Covariance: fixed vs phylo
cov_fix_phy      <- sapply(1:S, function(s) cov(eta_no_re[, s], eta_phylo[, s]))
prop_cov_fix_phy <- 2 * cov_fix_phy / V_total

# Decomposition verification
decomp_check <- prop_fixed + prop_phylo + prop_cov_fix_phy + prop_residual
cat(sprintf("\nDecomposition check: %.4f +/- %.4f (ideally = 1)\n",
            mean(decomp_check), sd(decomp_check)))

if (mean(abs(r2_vals - R2_full)) > 0.01) {
    cat("\nWarning: brms::bayes_R2 not close to Gelman R2_full\n")
}

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Shapley values (component-variance, no refit)
# ═══════════════════════════════════════════════════════════════════════════════

X_design <- X[, colnames(X) != "(Intercept)", drop = FALSE]
terms_no_intercept <- colnames(X_design)
p <- length(terms_no_intercept)

cat(sprintf("\nComputing Shapley decomposition (%d terms, %d samples)...\n", p, S))

results_shapley <- parallel::mclapply(1:S, function(s) {
    beta_s <- fixef_reordered[s, ]
    comps <- setNames(
        lapply(terms_no_intercept, function(term) beta_s[term] * X_design[, term]),
        terms_no_intercept
    )
    shapley_one_sample(comps)
}, mc.cores = max(1L, parallel::detectCores() - 1))

shap_mat <- do.call(rbind, lapply(results_shapley, function(r) r$shapley))
marg_mat <- do.call(rbind, lapply(results_shapley, function(r) r$marginal))
vfix_vec <- sapply(results_shapley, function(r) r$V_fixed)

# Per-term summaries
term_props <- list()
marg_props <- list()
for (j in 1:p) {
    term_props[[terms_no_intercept[j]]] <- summarize_posterior(shap_mat[, j])
    marg_props[[terms_no_intercept[j]]] <- summarize_posterior(marg_mat[, j])
}
vfix_summary <- summarize_posterior(vfix_vec)

cat(sprintf("V_fixed: %.4f (95%% CI: %.4f - %.4f)\n",
            vfix_summary$mean, vfix_summary$q2.5, vfix_summary$q97.5))
cat(sprintf("Shapley sum check: %.6f (should be 1.0)\n", mean(rowSums(shap_mat))))

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Save outputs
# ═══════════════════════════════════════════════════════════════════════════════
file_prefix <- paste0(model_type, "_", tree_name)

# --- 8a. Variance samples CSV ---
samples_df <- data.frame(
    sample_id     = 1:S,
    tree          = tree_name,
    model_type    = model_type,
    R2_full       = R2_full,
    prop_fixed    = prop_fixed,
    prop_linear   = prop_linear,
    prop_phylo    = prop_phylo,
    prop_residual = prop_residual,
    prop_cov_fix_phy = prop_cov_fix_phy,
    V_total       = V_total,
    V_fixed       = vfix_vec
)

for (term in terms_no_intercept) {
    term_clean <- gsub(":", "_", term)
    samples_df[[paste0("shapley_", term_clean)]]  <- shap_mat[, term]
    samples_df[[paste0("marginal_", term_clean)]] <- marg_mat[, term]
}

samples_path <- file.path(out_dir, paste0("variance_samples_", file_prefix, ".csv"))
write.csv(samples_df, samples_path, row.names = FALSE)
cat("Posterior samples written to", samples_path, "\n")

# --- 8b. Coefficient samples CSV ---
coef_samples_df <- data.frame(sample_id = 1:S, tree = tree_name, fixef_samples)
colnames(coef_samples_df) <- gsub("^b_", "", colnames(coef_samples_df))

coef_samples_path <- file.path(out_dir, paste0("coef_samples_", file_prefix, ".csv"))
write.csv(coef_samples_df, coef_samples_path, row.names = FALSE)
cat("Coefficient samples written to", coef_samples_path, "\n")

# --- 8c. Summary CSV ---
csv_results <- data.frame(
    model             = "brms_phylo",
    model_type        = model_type,
    tree              = tree_name,
    stringsAsFactors  = FALSE
)

# MCMC diagnostics
csv_results$n_divergent        <- diag$n_div
csv_results$max_rhat           <- max(diag$all_rhat, na.rm = TRUE)
csv_results$min_bulk_ess       <- min(diag$all_bulk_ess, na.rm = TRUE)
csv_results$min_tail_ess       <- min(diag$all_tail_ess, na.rm = TRUE)
csv_results$max_rhat_fixed     <- max(diag$summ_fixed$Rhat, na.rm = TRUE)
csv_results$min_bulk_ess_fixed <- min(diag$summ_fixed$Bulk_ESS, na.rm = TRUE)
csv_results$min_tail_ess_fixed <- min(diag$summ_fixed$Tail_ESS, na.rm = TRUE)

# Coefficient estimates
for (i in seq_len(nrow(coef_df))) {
    tn <- rownames(coef_df)[i]
    tc <- gsub(":", "_", gsub("_norm", "", tn))
    csv_results[[paste0("coef_", tc)]]     <- coef_df[i, "Estimate"]
    csv_results[[paste0("se_", tc)]]       <- coef_df[i, "Est.Error"]
    csv_results[[paste0("ci_lower_", tc)]] <- coef_df[i, "CI_lower"]
    csv_results[[paste0("ci_upper_", tc)]] <- coef_df[i, "CI_upper"]
}

# Variance decomposition
csv_results$method                    <- "Bayesian_Variance_Decomposition_Gelman2019"
csv_results$total_response_variance   <- var(model_df$log_rate_median)

csv_results$R2_full_mean   <- summarize_posterior(R2_full)$mean
csv_results$R2_full_q2_5   <- summarize_posterior(R2_full)$q2.5
csv_results$R2_full_q97_5  <- summarize_posterior(R2_full)$q97.5

csv_results$prop_fixed_mean   <- summarize_posterior(prop_fixed)$mean
csv_results$prop_fixed_q2_5   <- summarize_posterior(prop_fixed)$q2.5
csv_results$prop_fixed_q97_5  <- summarize_posterior(prop_fixed)$q97.5

csv_results$prop_linear_mean  <- summarize_posterior(prop_linear)$mean
csv_results$prop_linear_q2_5  <- summarize_posterior(prop_linear)$q2.5
csv_results$prop_linear_q97_5 <- summarize_posterior(prop_linear)$q97.5

csv_results$prop_phylo_mean   <- summarize_posterior(prop_phylo)$mean
csv_results$prop_phylo_q2_5   <- summarize_posterior(prop_phylo)$q2.5
csv_results$prop_phylo_q97_5  <- summarize_posterior(prop_phylo)$q97.5

csv_results$prop_cov_fix_phy_mean  <- summarize_posterior(prop_cov_fix_phy)$mean
csv_results$prop_cov_fix_phy_q2_5  <- summarize_posterior(prop_cov_fix_phy)$q2.5
csv_results$prop_cov_fix_phy_q97_5 <- summarize_posterior(prop_cov_fix_phy)$q97.5

csv_results$prop_residual_mean  <- summarize_posterior(prop_residual)$mean
csv_results$prop_residual_q2_5  <- summarize_posterior(prop_residual)$q2.5
csv_results$prop_residual_q97_5 <- summarize_posterior(prop_residual)$q97.5

csv_results$V_trace_norm <- mean(diag(V))
csv_results$n_obs        <- N

csv_results$V_fixed_mean  <- vfix_summary$mean
csv_results$V_fixed_q2_5  <- vfix_summary$q2.5
csv_results$V_fixed_q97_5 <- vfix_summary$q97.5

# Shapley per-term (proportions of V_fixed, sum to 1)
for (term in names(term_props)) {
    tc <- gsub("\\(|\\)", "", gsub(":", "_", term))
    csv_results[[paste0("shapley_", tc, "_mean")]]  <- term_props[[term]]$mean
    csv_results[[paste0("shapley_", tc, "_q2_5")]]  <- term_props[[term]]$q2.5
    csv_results[[paste0("shapley_", tc, "_q97_5")]] <- term_props[[term]]$q97.5
}

# Marginal variance per-term (proportions of V_fixed, do NOT sum to 1)
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
# 9. Print summary
# ═══════════════════════════════════════════════════════════════════════════════
cat("\n======================================================================\n")
cat(sprintf("COMPLETED: %s x %s\n", model_type, tree_name))
cat("======================================================================\n")
cat(sprintf("  R2: %.3f (95%% CI: %.3f - %.3f)\n",
            summarize_posterior(R2_full)$mean,
            summarize_posterior(R2_full)$q2.5,
            summarize_posterior(R2_full)$q97.5))

cat("\n  Variance Components (Proportion of V_total):\n")
cat(sprintf("    Fixed Effects:    %.1f%% (95%% CI: %.1f - %.1f)\n",
            summarize_posterior(prop_fixed)$mean * 100,
            summarize_posterior(prop_fixed)$q2.5 * 100,
            summarize_posterior(prop_fixed)$q97.5 * 100))

cat(sprintf("    Phylogenetic RE:  %.1f%% (95%% CI: %.1f - %.1f)\n",
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
