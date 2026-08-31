# beast_phylolm.R - Phylogenetic regression pipeline dispatcher
#
# Runs linear (brms) and GP (cmdstanr) regression for speech and cognate trees.
# By default runs all 4 combinations; use --model_type to restrict to one model.
#
# Usage:
#   pixi run -e regression beast_phylolm <run_id> <subdir> [options]
#
# Arguments:
#   run_id    BEAST run UUID, prefix, or full path (speech tree)
#   subdir    Subdirectory name or prefix within the run
#
# Options:
#   --model_type <type>        Run only this model (linear_geo or gp_geo; default: both)
#   --tree_file <path>         Explicit tree file for speech (when multiple .mcc exist)
#   --cognate_beast_dir <path> (default: data/trees/beast/iecor)
#   --variant with_inventory|no_inventory  (default: with_inventory)
#   ... plus model-specific options forwarded to the regression scripts
#
# Examples:
#   pixi run -e regression beast_phylolm ba9 0.05
#   pixi run -e regression beast_phylolm ba9 0.05 --model_type linear_geo
#   pixi run -e regression beast_phylolm ba9 0.05 --tree_file data/trees/beast/ba9f2d2a-.../0.05_.../input_v2.mcc

BEAST_DIR <- "data/trees/beast"
COGNATE_BEAST_DIR_DEFAULT <- "data/trees/beast/iecor"

SCRIPTS <- list(
    linear_geo = "src/tasks/phylo/run_phylo_regression.R",
    gp_geo = "src/tasks/phylo/run_phylo_regression_nonlinear.R"
)

all_args <- commandArgs(trailingOnly = TRUE)
args <- all_args[all_args != "--"]

if (length(args) > 0 && args[1] %in% c("-h", "--help")) {
    cat("Usage: pixi run -e regression beast_phylolm <run_id> <subdir> [options]\n")
    cat("\nSee header of this script for full documentation.\n")
    quit(status = 0)
}

if (length(args) < 2) {
    stop(
        "Usage: pixi run -e regression beast_phylolm <run_id> <subdir> [options]",
        call. = FALSE
    )
}

run_id <- args[1]
subdir <- args[2]

# Parse remaining named options
cognate_beast_dir <- COGNATE_BEAST_DIR_DEFAULT
model_type <- NULL
tree_file <- NULL
forward_args <- character(0)

i <- 3
while (i <= length(args)) {
    key <- sub("^--", "", args[i])
    if (key == "cognate_beast_dir" && i + 1 <= length(args)) {
        cognate_beast_dir <- args[i + 1]
        i <- i + 2
    } else if (key == "model_type" && i + 1 <= length(args)) {
        model_type <- args[i + 1]
        i <- i + 2
    } else if (key == "tree_file" && i + 1 <= length(args)) {
        tree_file <- args[i + 1]
        i <- i + 2
    } else {
        forward_args <- c(forward_args, args[i])
        i <- i + 1
    }
}

if (!is.null(model_type) && !model_type %in% names(SCRIPTS)) {
    stop(sprintf(
        "Unknown model_type '%s'. Must be one of: %s",
        model_type,
        paste(names(SCRIPTS), collapse = ", ")
    ))
}

# Resolve run_id to BEAST directory
if (dir.exists(run_id)) {
    beast_root <- run_id
} else {
    matches <- Sys.glob(file.path(BEAST_DIR, paste0(run_id, "*")))
    matches <- matches[dir.exists(matches)]
    if (length(matches) == 0) {
        stop(
            sprintf("No BEAST run matching '%s' in %s/", run_id, BEAST_DIR),
            call. = FALSE
        )
    }
    if (length(matches) > 1) {
        stop(
            sprintf(
                "Ambiguous run_id '%s': matches %s",
                run_id,
                paste(matches, collapse = ", ")
            ),
            call. = FALSE
        )
    }
    beast_root <- matches[1]
}

# Resolve subdir within the run
subdir_matches <- Sys.glob(file.path(beast_root, paste0(subdir, "*")))
subdir_matches <- subdir_matches[dir.exists(subdir_matches)]
if (length(subdir_matches) == 0) {
    stop(
        sprintf("No subdirectory matching '%s' in %s/", subdir, beast_root),
        call. = FALSE
    )
}
if (length(subdir_matches) > 1) {
    stop(
        sprintf(
            "Ambiguous subdir '%s': matches %s",
            subdir,
            paste(subdir_matches, collapse = ", ")
        ),
        call. = FALSE
    )
}

speech_beast_dir <- subdir_matches[1]

# Determine runs
model_types <- if (is.null(model_type)) names(SCRIPTS) else model_type
beast_dirs <- list(speech = speech_beast_dir, cognate = cognate_beast_dir)

n_runs <- length(model_types) * length(beast_dirs)
cat(sprintf(
    "Running %d regression(s): %s x {%s}\n",
    n_runs,
    paste(model_types, collapse = ", "),
    paste(names(beast_dirs), collapse = ", ")
))
cat(sprintf("  speech:  %s\n", speech_beast_dir))
cat(sprintf("  cognate: %s\n\n", cognate_beast_dir))

failures <- character(0)

for (mt in model_types) {
    for (label in names(beast_dirs)) {
        bd <- beast_dirs[[label]]
        run_label <- sprintf("%s x %s", mt, label)

        cat(sprintf("\n========================================\n"))
        cat(sprintf("  %s\n  %s\n", run_label, bd))
        cat(sprintf("========================================\n\n"))

        tree_arg <- ""
        if (label == "speech" && !is.null(tree_file)) {
            tree_arg <- paste("--tree_file", shQuote(tree_file))
        }
        cmd <- paste(
            "Rscript",
            shQuote(SCRIPTS[[mt]]),
            "--beast_dir",
            shQuote(bd),
            tree_arg,
            paste(shQuote(forward_args), collapse = " ")
        )
        cat(sprintf("Running: %s\n\n", cmd))
        status <- system(cmd)
        if (status != 0) {
            warning(sprintf("FAILED: %s (exit %d)", run_label, status))
            failures <- c(failures, run_label)
        }
    }
}

if (length(failures) > 0) {
    cat(sprintf(
        "\n%d/%d runs failed: %s\n",
        length(failures),
        n_runs,
        paste(failures, collapse = ", ")
    ))
    quit(status = 1)
} else {
    cat(sprintf("\nAll %d runs completed successfully.\n", n_runs))
}
