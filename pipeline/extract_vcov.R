# ─── 0. Setup ──────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(ape)
  library(brms)
  library(dplyr)
  library(tibble)
  library(jsonlite)
  library(purrr)
})

beast_dir <- "data/trees/beast"

tree_names <- c(
  "iecor/raw_CCD.nex",
  "eab44e7f-54cc-4469-87d1-282cc81e02c2/0.25/long_v3_44_CCD.nex"
)

dataset <- "fleurs-r"
min_speakers <- 1.0 # in millions

coordinate_shift <- 10
coord_methods <- c("standard", "sqrt_plus_10", "log_plus_10")

apply_coord_scaling <- function(vec, method, shift = coordinate_shift) {
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

  return(list(
    values = as.numeric(scaled),
    center = center,
    scale = scale_val,
    description = description,
    shift = shift_value
  ))
}

# Helper function for summarizing posterior samples
summarize_posterior <- function(x) {
  list(
    mean = mean(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE),
    q2.5 = unname(quantile(x, 0.025, na.rm = TRUE)),
    q50 = unname(quantile(x, 0.50, na.rm = TRUE)),
    q97.5 = unname(quantile(x, 0.975, na.rm = TRUE))
  )
}

# ─── 1. Load BEAST helpers ─────────────────────────────────
# must define read.annot.beast() & extract_beast_metadata()
source("src/tasks/phylo/beast.R")
stopifnot(exists("read.annot.beast"))
stopifnot(exists("extract_beast_metadata"))

# ─── 2. Load language metadata metadata ─────────────────────────────────
meta_dir <- "data/metadata"

json_path <- file.path(meta_dir, dataset, "languages.json")
mapping <- read_json(json_path)
# Make a full to iecor mapping from `mapping``
# for each value, v["full"] -> v["iecor"]
# the column `name` in lang_metadata corresponds to v["full"]
glottocode_to_iecor <- tibble(
  glottocode = map_chr(mapping, "glottolog"),
  iecor = map_chr(mapping, ~ .x$iecor %||% NA_character_)
)

glotto_path <- file.path(meta_dir, dataset, "glottolog.csv")
# Drop rows with NULL language mapping
# Group by `language` & keep first (identical rates)
# Should only happen for Serbian & Croatian
# which are both mapped to Serbo-Croatian
# For FLEURS-R and min_speakers = 1.0, dim should be 36 x N
# Remove columns H0, ..., HN (not needed here) (using a regex)
lang_metadata <- read.csv(glotto_path, stringsAsFactors = FALSE) %>%
  left_join(glottocode_to_iecor, by = "glottocode") %>%
  rename(language = iecor) %>%
  filter(!is.na(language)) %>%
  filter(n_speakers >= min_speakers) %>%
  group_by(language) %>%
  slice(1L) %>%
  ungroup() %>%
  select(-matches("^H\\d+$"))

for (tree_name in tree_names) {
  # ─── 3. Read tree ────────────────────────────────────────────────────────
  tree_path <- paste0(beast_dir, "/", tree_name) # BEAST .trees or .nexus file

  tr <- read.annot.beast(tree_path)

  tree_dir <- dirname(tree_path)
  stem <- tools::file_path_sans_ext(basename(tree_path))

  # ─── 4. Extract & merge metadata ─────────────────────────────────────────
  beast_metadata <- extract_beast_metadata(tree = tr)
  df <- inner_join(beast_metadata, lang_metadata, by = "language") %>%
    column_to_rownames(var = "language")

  # ─── 5. write BEAST metadata with annotations ────────────────────────────
  beast_metadata_path <- file.path(
    tree_dir,
    paste0(stem, "_metadata.csv")
  )
  write.csv(
    df %>% rownames_to_column("language"),
    file = beast_metadata_path,
    row.names = FALSE,
    quote = TRUE
  )

  # ─── 6. Prune tree & compute cophenetic distances ────────────────────────
  to_drop <- setdiff(tr$tip.label, rownames(df))
  tr <- drop.tip(tr, to_drop)
  coph_df <- as.data.frame(cophenetic(unroot(tr)))

  # ─── 7. compute raw phylogenetic covariance (shared branch lengths) ──────
  V_raw <- vcv(unroot(tr)) # n×n matrix

  # ─── 8. write V_raw.csv (with row & col names)
  beast_vcv_path <- file.path(
    tree_dir,
    paste0(stem, "_vcv.csv")
  )
  write.csv(
    V_raw,
    file = beast_vcv_path,
    row.names = TRUE,
    quote = FALSE
  )

  cat(
    "Wrote:\n",
    " • metadata →",
    beast_metadata_path,
    "\n",
    " • raw phylo covariance →",
    beast_vcv_path,
    "\n"
  )

  # ─── 9. Speaker count transform (fixed: log + scale) ─────────────
  log_n_scaled <- scale(log(df$n_speakers))
  df$log_n_speakers_norm <- log_n_scaled[, 1]
  log_n_center <- as.numeric(attr(log_n_scaled, "scaled:center"))
  log_n_scale <- as.numeric(attr(log_n_scaled, "scaled:scale"))

  base_df <- df

  # ─── 10. Modelling ────────────────────────────────────────────────
  for (coord_method in coord_methods) {
    df_loop <- base_df

    lon_stats <- apply_coord_scaling(df_loop$longitude, coord_method)
    lat_stats <- apply_coord_scaling(df_loop$latitude, coord_method)

    df_loop$longitude_norm <- lon_stats$values
    df_loop$latitude_norm <- lat_stats$values

    x_center <- lon_stats$center
    x_scale <- lon_stats$scale
    y_center <- lat_stats$center
    y_scale <- lat_stats$scale

    model_df <- df_loop
    model_df$log_rate_median <- log(model_df$rate_median)
    language_levels <- rownames(model_df)
    model_df$language_factor <- factor(
      language_levels,
      levels = language_levels
    )

    V <- V_raw[language_levels, language_levels]

    formula_str <- paste(
      "log_rate_median ~",
      "longitude_norm + latitude_norm + log_n_speakers_norm +",
      "longitude_norm:latitude_norm +",
      "latitude_norm:log_n_speakers_norm +",
      "longitude_norm:log_n_speakers_norm +",
      "(1|gr(language_factor, cov = V))"
    )
    formula_obj <- as.formula(formula_str)

    brm_args <- list(
      formula = formula_obj,
      data = model_df,
      family = gaussian(),
      refresh = 0,
      chains = 4,
      iter = 4000,
      warmup = 1000,
      seed = 20231103,
      cores = max(1L, min(4L, parallel::detectCores())),
      control = list(adapt_delta = 0.99, max_treedepth = 15),
      data2 = list(V = V)
    )

    cat(sprintf(
      "\nFitting phylogenetic spatial regression (%s)...\n",
      coord_method
    ))
    model <- do.call(brm, brm_args)
    model_type <- "brms_phylo"

    # Save model
    model_path <- file.path(
      tree_dir,
      paste0(stem, "_phylolm_", coord_method, ".rds")
    )
    saveRDS(model, file = model_path)
    cat("✔️ Model saved to", model_path, "\n")

    # Extract posterior samples
    posterior <- as.data.frame(model)

    # Extract variance components from posterior
    phylo_sd_samples <- posterior$sd_language_factor__Intercept
    sigma_samples <- posterior$sigma
    residual_var_samples <- sigma_samples^2

    # Phylogenetic variance: tr(V)/n * σ²_phylo
    V_mean_diag <- mean(diag(V)) # = tr(V)/n
    phylo_var_samples <- (phylo_sd_samples^2) * V_mean_diag

    coef_df <- as.data.frame(brms::fixef(model, probs = c(0.025, 0.975)))
    colnames(coef_df) <- c("Estimate", "Est.Error", "CI_lower", "CI_upper")

    cat("\nPosterior summaries (normalized / transformed scale):\n")
    print(coef_df[, c("Estimate", "Est.Error", "CI_lower", "CI_upper")])

    # ─── 11. R² and fitted values ─────────────────────────────────────
    r2_vals <- as.numeric(brms::bayes_R2(model))
    r_squared_summary <- list(
      mean = mean(r2_vals),
      sd = stats::sd(r2_vals),
      q2.5 = unname(stats::quantile(r2_vals, 0.025)),
      q97.5 = unname(stats::quantile(r2_vals, 0.975))
    )

    fitted_summary <- fitted(model, summary = TRUE, probs = c(0.025, 0.975))
    preds_df <- data.frame(
      language = language_levels,
      fitted_mean = fitted_summary[, "Estimate"],
      fitted_sd = fitted_summary[, "Est.Error"],
      fitted_lower = fitted_summary[, "Q2.5"],
      fitted_upper = fitted_summary[, "Q97.5"],
      stringsAsFactors = FALSE
    )

    residuals <- model_df$log_rate_median - fitted_summary[, "Estimate"]
    rmse <- sqrt(mean(residuals^2, na.rm = TRUE))
    mae <- mean(abs(residuals), na.rm = TRUE)

    # ─── 12. Save results as CSV (transformed scale only) ────────────────
    csv_results <- data.frame(
      model = model_type,
      tree = tree_name,
      coordinate_method = coord_method,
      stringsAsFactors = FALSE
    )

    # ─── 13. Fixed Effects Variance Decomposition ─────────────────────────
    # Get design matrix
    X <- model.matrix(
      ~ longitude_norm +
        latitude_norm +
        log_n_speakers_norm +
        longitude_norm:latitude_norm +
        latitude_norm:log_n_speakers_norm +
        longitude_norm:log_n_speakers_norm,
      data = model_df
    )

    # brms names fixed effects with "b_" prefix
    brms_names <- c(
      "(Intercept)" = "b_Intercept",
      "longitude_norm" = "b_longitude_norm",
      "latitude_norm" = "b_latitude_norm",
      "log_n_speakers_norm" = "b_log_n_speakers_norm",
      "longitude_norm:latitude_norm" = "b_longitude_norm:latitude_norm",
      "latitude_norm:log_n_speakers_norm" = "b_latitude_norm:log_n_speakers_norm",
      "longitude_norm:log_n_speakers_norm" = "b_longitude_norm:log_n_speakers_norm"
    )

    # Get posterior samples of fixed effects using brms names
    beta_samples <- as.matrix(posterior[, brms_names[colnames(X)]])
    colnames(beta_samples) <- colnames(X)

    # ─────────────────────────────────────────────────────────────────────────────
    # VARIANCE DECOMPOSITION USING SHAPLEY VALUES
    # ─────────────────────────────────────────────────────────────────────────────
    #
    # Goal: Partition the total variance in the response (log speech rate) into
    # contributions from each predictor, phylogenetic structure, and residual noise.
    #
    # The model decomposes the response as:
    #   y = Xβ (fixed effects) + u (phylogenetic random effect) + ε (residual)
    #
    # We want to know: "How much variance does each predictor explain?"
    #
    # Problem: When predictors are correlated, there's no unique way to assign
    # variance. The order in which you add predictors matters (Type I vs Type III SS).
    #
    # Solution: Shapley values from cooperative game theory provide a fair,
    # order-independent allocation. Each predictor's contribution is its average
    # marginal contribution across ALL possible orderings of predictors.
    #
    # Interpretation: A predictor with Shapley value φ_j explains φ_j units of
    # variance on average, accounting for its correlations with other predictors.
    # ─────────────────────────────────────────────────────────────────────────────

    # Step 1: Compute total variance explained by fixed effects
    # Xβ = predicted values from fixed effects only (before adding phylo/residual)
    Xbeta_samples <- tcrossprod(X, beta_samples) # predictions for each posterior draw
    fixed_var_samples <- apply(Xbeta_samples, 2, var) # variance across languages

    # Step 2: Prepare for Shapley decomposition
    # We exclude the intercept (it shifts predictions but doesn't explain variance)
    term_names <- colnames(X)
    terms_no_intercept <- setdiff(term_names, "(Intercept)")
    p <- length(terms_no_intercept) # number of predictors to decompose
    S <- ncol(Xbeta_samples) # number of posterior samples

    # Center the design matrix and predictions (variance is computed on centered data)
    X_terms <- X[, terms_no_intercept, drop = FALSE]
    X_terms_centered <- scale(X_terms, center = TRUE, scale = FALSE)
    mu_centered <- sweep(Xbeta_samples, 2, colMeans(Xbeta_samples), "-")

    # Step 3: Precompute orthonormal bases for all 2^p subsets of predictors
    # For each subset S of predictors, we need to project μ onto span(X_S).
    # Using QR decomposition gives us an orthonormal basis Q for efficient projection.
    # We use bitmasks to enumerate all 2^p subsets (mask=0 is empty, mask=2^p-1 is full).
    Q_list <- vector("list", 2^p)
    Q_list[[1]] <- NULL # empty subset has no basis

    for (mask in 1:(2^p - 1)) {
      # Convert bitmask to column indices (e.g., mask=5=101 → columns 1 and 3)
      idx <- which(as.logical(intToBits(mask))[1:p])
      XS <- X_terms_centered[, idx, drop = FALSE]
      qrS <- qr(XS)
      Q_list[[mask + 1]] <- qr.Q(qrS)
    }

    # Step 4: Compute value function v(S) = variance explained by subset S
    # v(S) = Var(projection of μ onto span(X_S))
    # This measures how much of the prediction variance is captured by subset S.
    v <- vector("list", 2^p)
    v[[1]] <- rep(0, S) # empty set explains nothing

    for (mask in 1:(2^p - 1)) {
      Q <- Q_list[[mask + 1]]
      Qt_mu <- crossprod(Q, mu_centered) # project onto orthonormal basis
      mu_hat <- Q %*% Qt_mu # reconstruction
      v[[mask + 1]] <- apply(mu_hat, 2, var) # variance of projection
    }

    # Step 5: Compute Shapley values using the classic formula
    # φ_j = Σ_{S⊆N\{j}} [|S|!(p-|S|-1)!/p!] × [v(S∪{j}) - v(S)]
    #
    # This averages the marginal contribution of predictor j over all orderings.
    # The weight |S|!(p-|S|-1)!/p! is the probability of S being the set of
    # predictors that come before j in a random ordering.
    fact <- factorial(0:p)
    den <- fact[p + 1] # p!

    phi <- matrix(0, nrow = S, ncol = p)
    colnames(phi) <- terms_no_intercept

    for (j in 1:p) {
      bitj <- bitwShiftL(1L, j - 1L) # bit for predictor j
      for (mask in 0:(2^p - 1)) {
        # Skip if j is already in the subset
        if (bitwAnd(mask, bitj) != 0L) {
          next
        }
        k <- sum(as.logical(intToBits(mask))[1:p]) # |S|
        w <- (fact[k + 1] * fact[p - k]) / den # Shapley weight
        # Marginal contribution: v(S ∪ {j}) - v(S)
        phi[, j] <- phi[, j] +
          w * (v[[bitwOr(mask, bitj) + 1]] - v[[mask + 1]])
      }
    }

    # Sanity check: Shapley values should sum to total fixed variance (efficiency axiom)
    fixed_from_phi <- rowSums(phi)
    cat(sprintf(
      "Shapley check: max|sum(phi)-fixed| = %.3e\n",
      max(abs(fixed_from_phi - fixed_var_samples))
    ))

    # ─────────────────────────────────────────────────────────────────────────────
    # COMBINE INTO FULL VARIANCE DECOMPOSITION
    # ─────────────────────────────────────────────────────────────────────────────
    # Total variance = Fixed effects + Phylogenetic + Residual
    #                = Σ_j φ_j       + σ²_phylo      + σ²_resid
    #
    # Each component's proportion tells us what fraction of the total variance
    # is attributable to that source. All proportions sum to 1.
    # ─────────────────────────────────────────────────────────────────────────────

    total_var_samples <- fixed_var_samples +
      phylo_var_samples +
      residual_var_samples

    # Convert Shapley values to proportions of total variance
    term_prop_samples <- phi / total_var_samples
    prop_phylo_samples <- phylo_var_samples / total_var_samples
    prop_residual_samples <- residual_var_samples / total_var_samples

    # Summarize posterior distributions for each term
    term_variances <- list()
    term_proportions <- list()
    for (j in 1:p) {
      tnj <- terms_no_intercept[j]
      term_variances[[tnj]] <- summarize_posterior(phi[, j])
      term_proportions[[tnj]] <- summarize_posterior(term_prop_samples[, j])
    }

    # Full variance decomposition
    variance_decomposition <- list(
      # Fixed effects (total)
      fixed_variance = summarize_posterior(fixed_var_samples),

      # Per-term breakdown
      term_variances = term_variances,
      term_proportions = term_proportions,

      # Phylogenetic: tr(V)/n * σ²_phylo
      phylo_variance = summarize_posterior(phylo_var_samples),
      phylo_prop = summarize_posterior(prop_phylo_samples),

      # Residual
      residual_variance = summarize_posterior(residual_var_samples),
      residual_prop = summarize_posterior(prop_residual_samples),

      # Total
      total_variance = summarize_posterior(total_var_samples),

      # Covariance matrix info
      V_trace_normalized = V_mean_diag, # i.e., tr(V) / n
      n_taxa = nrow(V)
    )

    cat("\nVariance Decomposition:\n")
    cat(sprintf("  V scaling factor (tr(V)/n): %.4f\n", V_mean_diag))
    cat(sprintf(
      "  Phylogenetic:          %.4f (%.1f%%, 95%% CI: %.1f%% – %.1f%%)\n",
      variance_decomposition$phylo_variance$mean,
      variance_decomposition$phylo_prop$mean * 100,
      variance_decomposition$phylo_prop$q2.5 * 100,
      variance_decomposition$phylo_prop$q97.5 * 100
    ))
    cat(sprintf(
      "  Residual:              %.4f (%.1f%%, 95%% CI: %.1f%% – %.1f%%)\n",
      variance_decomposition$residual_variance$mean,
      variance_decomposition$residual_prop$mean * 100,
      variance_decomposition$residual_prop$q2.5 * 100,
      variance_decomposition$residual_prop$q97.5 * 100
    ))
    cat("\n  Per-term proportions (sum with phylo + residual = 1):\n")
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
    # Add coefficient information (normalized / transformed scale)
    for (i in seq_len(nrow(coef_df))) {
      term_name <- rownames(coef_df)[i]
      term_name_clean <- gsub("_norm", "", term_name) # Remove _norm suffix
      term_name_clean <- gsub(":", "_", term_name_clean) # Replace : with _

      csv_results[paste0("coef_", term_name_clean)] <- coef_df[i, "Estimate"]
      csv_results[paste0("se_", term_name_clean)] <- coef_df[i, "Est.Error"]
      csv_results[paste0("ci_lower_", term_name_clean)] <- coef_df[
        i,
        "CI_lower"
      ]
      csv_results[paste0("ci_upper_", term_name_clean)] <- coef_df[
        i,
        "CI_upper"
      ]
    }

    ## Variance decomposition
    # Per-term variances and proportions
    for (term in names(term_variances)) {
      term_clean <- gsub(":", "_", term)
      term_clean <- gsub("\\(|\\)", "", term_clean)
      csv_results[[paste0("var_", term_clean)]] <- term_variances[[term]]$mean
      csv_results[[paste0("prop_", term_clean)]] <- term_proportions[[
        term
      ]]$mean
      # save proportions too
      csv_results[[paste0(
        "prop_",
        term_clean,
        "_q2_5"
      )]] <- term_proportions[[term]]$q2.5
      csv_results[[paste0(
        "prop_",
        term_clean,
        "_q97_5"
      )]] <- term_proportions[[term]]$q97.5
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

    # Save CSV
    csv_path <- file.path(
      tree_dir,
      paste0(stem, "_phylolm_", coord_method, ".csv")
    )

    write.csv(csv_results, csv_path, row.names = FALSE)
    cat("✔️ CSV results written to", csv_path, "\n")
  }
}
