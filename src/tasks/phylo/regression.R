# Phylogenetic regression helpers for brms models with phylogenetic covariance
#
# Functions for preparing data, fitting models, diagnostics, and variance
# decomposition using Shapley values.

library(brms)
library(dplyr)
library(stringr)

# ─── Coordinate Scaling ──────────────────────────────────────────────────────

#' Apply coordinate scaling transformation
#'
#' @param vec Numeric vector to transform
#' @param method One of "standard", "sqrt_plus_10", "log_plus_10"
#' @param shift Shift value for sqrt/log transforms (default 10)
#' @return List with values, center, scale, description, shift
apply_coord_scaling <- function(vec, method, shift = 10) {
  forward_transform <- NULL
  description <- NULL
  shift_value <- NA_real_

  if (method == "standard") {
    forward_transform <- function(x) x
    description <- "scale(x)"
  } else if (method == "sqrt_plus_10") {
    if (any(vec + shift <= 0)) {
      stop(sprintf(
        "sqrt_plus_10 requires all values to be greater than -%s",
        shift
      ))
    }
    forward_transform <- function(x) sqrt(x + shift)
    description <- sprintf("scale(sqrt(x + %s))", shift)
    shift_value <- shift
  } else if (method == "log_plus_10") {
    if (any(vec + shift <= 0)) {
      stop(sprintf(
        "log_plus_10 requires all values to be greater than -%s",
        shift
      ))
    }
    forward_transform <- function(x) log(x + shift)
    description <- sprintf("scale(log(x + %s))", shift)
    shift_value <- shift
  } else {
    stop(sprintf("Unknown coordinate scaling method: %s", method))
  }

  transformed <- forward_transform(vec)
  scaled <- scale(transformed)
  center <- as.numeric(attr(scaled, "scaled:center"))
  scale_val <- as.numeric(attr(scaled, "scaled:scale"))

  list(
    values = as.numeric(scaled),
    center = center,
    scale = scale_val,
    description = description,
    shift = shift_value
  )
}

# ─── Posterior Summarization ─────────────────────────────────────────────────

#' Summarize posterior samples
#'
#' @param x Numeric vector of posterior samples
#' @return List with mean, sd, q2.5, q50, q97.5
summarize_posterior <- function(x) {
  list(
    mean = mean(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE),
    q2.5 = unname(quantile(x, 0.025, na.rm = TRUE)),
    q50 = unname(quantile(x, 0.50, na.rm = TRUE)),
    q97.5 = unname(quantile(x, 0.975, na.rm = TRUE))
  )
}

# ─── Data Preparation ────────────────────────────────────────────────────────

#' Prepare data for brms phylogenetic regression
#'
#' @param base_df Data frame with longitude, latitude, rate_median, n_speakers
#' @param V_raw Raw phylogenetic covariance matrix
#' @param coord_method Coordinate scaling method
#' @return List with model_df and V (reordered covariance matrix)
prepare_brms_data <- function(base_df, V_raw, coord_method) {
  df_loop <- base_df

  lon_stats <- apply_coord_scaling(df_loop$longitude, coord_method)
  lat_stats <- apply_coord_scaling(df_loop$latitude, coord_method)

  df_loop$longitude_norm <- lon_stats$values
  df_loop$latitude_norm <- lat_stats$values

  # Normalize additional covariates if present

  if ("n_segments" %in% colnames(df_loop)) {
    n_segments_scaled <- scale(df_loop$n_segments)
    df_loop$n_segments_norm <- n_segments_scaled[, 1]
  }
  if ("delta" %in% colnames(df_loop)) {
    delta_scaled <- scale(df_loop$delta)
    df_loop$delta_norm <- delta_scaled[, 1]
  }

  model_df <- df_loop
  model_df$log_rate_median <- log(model_df$rate_median)
  language_levels <- rownames(model_df)
  model_df$language_factor <- factor(
    language_levels,
    levels = language_levels
  )

  V <- V_raw[language_levels, language_levels]

  list(
    model_df = model_df,
    V = V,
    lon_stats = lon_stats,
    lat_stats = lat_stats
  )
}

# ─── Model Diagnostics ───────────────────────────────────────────────────────

#' Run comprehensive MCMC diagnostics on a brms model
#'
#' @param model A brms model object
#' @param brm_args The arguments used to fit the model
#' @return List with diagnostic values (invisibly)
diagnose_brms <- function(model, brm_args) {
  cat("\n══════════════════════════════════════════════════════════════════\n")
  cat("                    MCMC DIAGNOSTICS\n")
  cat("══════════════════════════════════════════════════════════════════\n")

  # Divergent transitions

  np <- nuts_params(model)
  n_div <- sum(np[np$Parameter == "divergent__", "Value"])
  n_samples <- (brm_args$iter - brm_args$warmup) * brm_args$chains
  cat(sprintf(
    "Divergent transitions: %d / %d (%.2f%%)\n",
    n_div,
    n_samples,
    100 * n_div / n_samples
  ))
  if (n_div > 0) {
    cat("  Warning: Divergences detected - consider increasing adapt_delta\n")
  }

  # Tree depth

  max_td <- sum(
    np[np$Parameter == "treedepth__", "Value"] >= brm_args$control$max_treedepth
  )
  cat(sprintf(
    "Max treedepth reached: %d times (max=%d)\n",
    max_td,
    brm_args$control$max_treedepth
  ))

  # BFMI (Bayesian Fraction of Missing Information)
  energy <- np[np$Parameter == "energy__", "Value"]
  bfmi_vals <- NULL
  if (length(energy) > 0) {
    chain_ids <- np[np$Parameter == "energy__", "Chain"]
    chains <- unique(chain_ids)
    bfmi_vals <- sapply(chains, function(ch) {
      e <- energy[chain_ids == ch]
      var(diff(e)) / var(e)
    })
    cat(sprintf(
      "BFMI per chain: %s\n",
      paste(sprintf("%.3f", bfmi_vals), collapse = ", ")
    ))
    if (any(bfmi_vals < 0.2)) {
      cat("  Warning: Low BFMI (<0.2) - may indicate poor posterior exploration\n")
    }
  }

  # Rhat and ESS for ALL parameters
  summ_fixed <- summary(model)$fixed
  summ_spec <- summary(model)$spec_pars
  summ_random <- tryCatch(
    summary(model)$random$language_factor,
    error = function(e) NULL
  )

  all_rhat <- c(summ_fixed$Rhat, summ_spec$Rhat)
  all_bulk_ess <- c(summ_fixed$Bulk_ESS, summ_spec$Bulk_ESS)
  all_tail_ess <- c(summ_fixed$Tail_ESS, summ_spec$Tail_ESS)

  if (!is.null(summ_random)) {
    all_rhat <- c(all_rhat, summ_random$Rhat)
    all_bulk_ess <- c(all_bulk_ess, summ_random$Bulk_ESS)
    all_tail_ess <- c(all_tail_ess, summ_random$Tail_ESS)
  }

  cat("\n--- Convergence (all parameters) ---\n")
  max_rhat <- max(all_rhat, na.rm = TRUE)
  min_bulk_ess <- min(all_bulk_ess, na.rm = TRUE)
  min_tail_ess <- min(all_tail_ess, na.rm = TRUE)

  cat(sprintf(
    "  Max Rhat:      %.4f %s\n",
    max_rhat,
    ifelse(max_rhat < 1.01, "[OK]", "[WARNING > 1.01]")
  ))
  cat(sprintf(
    "  Min Bulk ESS:  %.0f %s\n",
    min_bulk_ess,
    ifelse(min_bulk_ess >= 400, "[OK]", "[WARNING < 400]")
  ))
  cat(sprintf(
    "  Min Tail ESS:  %.0f %s\n",
    min_tail_ess,
    ifelse(min_tail_ess >= 400, "[OK]", "[WARNING < 400]")
  ))

  cat("\n--- Fixed effects summary ---\n")
  cat(sprintf("  Max Rhat:      %.4f\n", max(summ_fixed$Rhat)))
  cat(sprintf("  Min Bulk ESS:  %.0f\n", min(summ_fixed$Bulk_ESS)))
  cat(sprintf("  Min Tail ESS:  %.0f\n", min(summ_fixed$Tail_ESS)))
  cat("══════════════════════════════════════════════════════════════════\n\n")

  invisible(list(
    n_divergent = n_div,
    max_rhat = max_rhat,
    min_bulk_ess = min_bulk_ess,
    min_tail_ess = min_tail_ess,
    bfmi = bfmi_vals
  ))
}

# ─── Model Metrics ───────────────────────────────────────────────────────────

#' Compute R², RMSE, MAE for a brms model
#'
#' @param model A brms model object
#' @param model_df The model data frame
#' @return List with r2, rmse, mae and their uncertainty
metrics_brms <- function(model, model_df) {
  r2_vals <- as.numeric(brms::bayes_R2(model))

  fitted_summary <- fitted(model, summary = TRUE, probs = c(0.025, 0.975))
  residuals <- model_df$log_rate_median - fitted_summary[, "Estimate"]
  rmse <- sqrt(mean(residuals^2, na.rm = TRUE))
  mae <- mean(abs(residuals), na.rm = TRUE)


  list(
    r2 = mean(r2_vals),
    r2_sd = sd(r2_vals),
    r2_q2.5 = unname(stats::quantile(r2_vals, 0.025)),
    r2_q97.5 = unname(stats::quantile(r2_vals, 0.975)),
    rmse = rmse,
    mae = mae
  )
}

# ─── Variance Decomposition ──────────────────────────────────────────────────

#' Decompose variance using Shapley values
#'
#' Decomposes the total variance in the response into contributions from
#' each predictor, phylogenetic structure, and residual noise using
#' Shapley values from cooperative game theory.
#'
#' @param model A brms model object
#' @param posterior Posterior samples as data.frame
#' @param V Phylogenetic covariance matrix
#' @param X Design matrix
#' @param beta_samples Posterior samples of fixed effects
#' @return List with variance decomposition results
decompose_variance <- function(model, posterior, V, X, beta_samples) {
  phylo_sd_samples <- posterior$sd_language_factor__Intercept
  sigma_samples <- posterior$sigma
  residual_var_samples <- sigma_samples^2

  # Phylogenetic variance: tr(V)/n * sigma^2_phylo
  V_mean_diag <- mean(diag(V)) # = tr(V)/n
  phylo_var_samples <- (phylo_sd_samples^2) * V_mean_diag

  coef_df <- as.data.frame(brms::fixef(model, probs = c(0.025, 0.975)))
  colnames(coef_df) <- c("Estimate", "Est.Error", "CI_lower", "CI_upper")

  cat("\nPosterior summaries (normalized scale):\n")
  print(coef_df[, c("Estimate", "Est.Error", "CI_lower", "CI_upper")])

  # Compute total variance explained by fixed effects
  # X*beta = predicted values from fixed effects only
  Xbeta_samples <- tcrossprod(X, beta_samples)
  fixed_var_samples <- apply(Xbeta_samples, 2, var)

  # Prepare for Shapley decomposition (exclude intercept)
  term_names <- colnames(X)
  terms_no_intercept <- setdiff(term_names, "(Intercept)")
  p <- length(terms_no_intercept)
  S <- ncol(Xbeta_samples)

  # Center the design matrix and predictions
  X_terms <- X[, terms_no_intercept, drop = FALSE]
  X_terms_centered <- scale(X_terms, center = TRUE, scale = FALSE)
  mu_centered <- sweep(Xbeta_samples, 2, colMeans(Xbeta_samples), "-")

  # Precompute QR decompositions for all 2^p subsets
  Q_list <- vector("list", 2^p)
  Q_list[[1]] <- NULL

  for (mask in 1:(2^p - 1)) {
    idx <- which(as.logical(intToBits(mask))[1:p])
    XS <- X_terms_centered[, idx, drop = FALSE]
    qrS <- qr(XS)
    Q_list[[mask + 1]] <- qr.Q(qrS)
  }

  # Compute value function v(S) = variance explained by subset S
  v <- vector("list", 2^p)
  v[[1]] <- rep(0, S)

  for (mask in 1:(2^p - 1)) {
    Q <- Q_list[[mask + 1]]
    Qt_mu <- crossprod(Q, mu_centered)
    mu_hat <- Q %*% Qt_mu
    v[[mask + 1]] <- apply(mu_hat, 2, var)
  }

  # Compute Shapley values
  fact <- factorial(0:p)
  den <- fact[p + 1]

  phi <- matrix(0, nrow = S, ncol = p)
  colnames(phi) <- terms_no_intercept

  for (j in 1:p) {
    bitj <- bitwShiftL(1L, j - 1L)
    for (mask in 0:(2^p - 1)) {
      if (bitwAnd(mask, bitj) != 0L) next
      k <- sum(as.logical(intToBits(mask))[1:p])
      w <- (fact[k + 1] * fact[p - k]) / den
      phi[, j] <- phi[, j] + w * (v[[bitwOr(mask, bitj) + 1]] - v[[mask + 1]])
    }
  }

  # Sanity check
  fixed_from_phi <- rowSums(phi)
  cat(sprintf(
    "Shapley check: max|sum(phi)-fixed| = %.3e\n",
    max(abs(fixed_from_phi - fixed_var_samples))
  ))

  # Combine into full variance decomposition
  total_var_samples <- fixed_var_samples + phylo_var_samples + residual_var_samples

  term_prop_samples <- phi / total_var_samples
  prop_phylo_samples <- phylo_var_samples / total_var_samples
  prop_residual_samples <- residual_var_samples / total_var_samples

  # Summarize posterior distributions
  term_variances <- list()
  term_proportions <- list()
  for (j in 1:p) {
    term_variances[[terms_no_intercept[j]]] <- summarize_posterior(phi[, j])
    term_proportions[[terms_no_intercept[j]]] <- summarize_posterior(term_prop_samples[, j])
  }

  list(
    coef_df = coef_df,
    fixed_variance = summarize_posterior(fixed_var_samples),
    term_variances = term_variances,
    term_proportions = term_proportions,
    phylo_variance = summarize_posterior(phylo_var_samples),
    phylo_prop = summarize_posterior(prop_phylo_samples),
    residual_variance = summarize_posterior(residual_var_samples),
    residual_prop = summarize_posterior(prop_residual_samples),
    total_variance = summarize_posterior(total_var_samples),
    V_trace_normalized = V_mean_diag,
    n_taxa = nrow(V)
  )
}

# ─── Results Saving ──────────────────────────────────────────────────────────

#' Save phylogenetic regression results to CSV
#'
#' @param output_path Path to save CSV
#' @param model_type Model type string
#' @param tree_name Tree identifier
#' @param coord_method Coordinate method used
#' @param coef_df Coefficient data frame
#' @param variance_decomposition Variance decomposition list
#' @param diagnostics Diagnostics list (optional)
save_phylolm_results <- function(
  output_path,
  model_type,
  tree_name,
  coord_method,
  coef_df,
  variance_decomposition,
  diagnostics = NULL
) {
  csv_results <- data.frame(
    model = model_type,
    tree = tree_name,
    coordinate_method = coord_method,
    stringsAsFactors = FALSE
  )

  # Add diagnostics if provided
  if (!is.null(diagnostics)) {
    csv_results$n_divergent <- diagnostics$n_divergent
    csv_results$max_rhat <- diagnostics$max_rhat
    csv_results$min_bulk_ess <- diagnostics$min_bulk_ess
    csv_results$min_tail_ess <- diagnostics$min_tail_ess
  }

  # Add coefficients
  for (i in seq_len(nrow(coef_df))) {
    term_name <- rownames(coef_df)[i]
    term_name_clean <- gsub("_norm", "", term_name)
    term_name_clean <- gsub(":", "_", term_name_clean)

    csv_results[[paste0("coef_", term_name_clean)]] <- coef_df[i, "Estimate"]
    csv_results[[paste0("se_", term_name_clean)]] <- coef_df[i, "Est.Error"]
    csv_results[[paste0("ci_lower_", term_name_clean)]] <- coef_df[i, "CI_lower"]
    csv_results[[paste0("ci_upper_", term_name_clean)]] <- coef_df[i, "CI_upper"]
  }

  # Add variance decomposition
  term_variances <- variance_decomposition$term_variances
  term_proportions <- variance_decomposition$term_proportions

  for (term in names(term_variances)) {
    term_clean <- gsub(":", "_", term)
    term_clean <- gsub("\\(|\\)", "", term_clean)
    csv_results[[paste0("var_", term_clean)]] <- term_variances[[term]]$mean
    csv_results[[paste0("prop_", term_clean)]] <- term_proportions[[term]]$mean
    csv_results[[paste0("prop_", term_clean, "_q2_5")]] <- term_proportions[[term]]$q2.5
    csv_results[[paste0("prop_", term_clean, "_q97_5")]] <- term_proportions[[term]]$q97.5
  }

  # Phylogenetic variance
  csv_results$phylo_var_mean <- variance_decomposition$phylo_variance$mean
  csv_results$phylo_prop_mean <- variance_decomposition$phylo_prop$mean
  csv_results$phylo_prop_q2_5 <- variance_decomposition$phylo_prop$q2.5
  csv_results$phylo_prop_q97_5 <- variance_decomposition$phylo_prop$q97.5

  # Residual
  csv_results$residual_var_mean <- variance_decomposition$residual_variance$mean
  csv_results$residual_prop_mean <- variance_decomposition$residual_prop$mean
  csv_results$residual_prop_q2_5 <- variance_decomposition$residual_prop$q2.5
  csv_results$residual_prop_q97_5 <- variance_decomposition$residual_prop$q97.5

  # V matrix info
  csv_results$V_trace_norm <- variance_decomposition$V_trace_normalized

  write.csv(csv_results, output_path, row.names = FALSE)
  cat("Results saved to", output_path, "\n")

  invisible(csv_results)
}

#' Print variance decomposition summary
#'
#' @param variance_decomposition Variance decomposition list
print_variance_decomposition <- function(variance_decomposition) {
  cat("\nVariance Decomposition:\n")
  cat(sprintf(
    "  V scaling factor (tr(V)/n): %.4f\n",
    variance_decomposition$V_trace_normalized
  ))
  cat(sprintf(
    "  Phylogenetic:    %.4f (%.1f%%, 95%% CI: %.1f%% - %.1f%%)\n",
    variance_decomposition$phylo_variance$mean,
    variance_decomposition$phylo_prop$mean * 100,
    variance_decomposition$phylo_prop$q2.5 * 100,
    variance_decomposition$phylo_prop$q97.5 * 100
  ))
  cat(sprintf(
    "  Residual:        %.4f (%.1f%%, 95%% CI: %.1f%% - %.1f%%)\n",
    variance_decomposition$residual_variance$mean,
    variance_decomposition$residual_prop$mean * 100,
    variance_decomposition$residual_prop$q2.5 * 100,
    variance_decomposition$residual_prop$q97.5 * 100
  ))

  cat("\n  Per-term proportions:\n")
  term_variances <- variance_decomposition$term_variances
  term_proportions <- variance_decomposition$term_proportions
  for (term in names(term_proportions)) {
    cat(sprintf(
      "    %s: %.4f (%.1f%%, 95%% CI: %.1f%% - %.1f%%)\n",
      term,
      term_variances[[term]]$mean,
      term_proportions[[term]]$mean * 100,
      term_proportions[[term]]$q2.5 * 100,
      term_proportions[[term]]$q97.5 * 100
    ))
  }
}
