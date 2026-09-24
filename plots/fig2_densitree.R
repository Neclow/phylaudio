# Extended Data Fig 8: DensiTree sensitivity comparison (1% vs 5% of sentences).
#
# Caption: DensiTree comparison of posterior tree samples from the 1% and 5%
# runs. The two posteriors are broadly concordant (nRF = 0.021, quartet
# similarity 0.999); MCC node heights for all major clades differ by 0.8-6.3%.

library(ape)
library(phangorn)

source("src/tasks/phylo/beast.R")

# --- Config ---
TREE_DIR <- file.path(
  "data/trees/beast",
  "ba9f2d2a-27f3-4100-a1c0-43f8fe1c39fc",
  "0.05_brsupport_dev_test/combined_v2"
)
TREES_FILE <- file.path(TREE_DIR, "input_v2_resampled.trees")
MCC_FILE <- file.path(TREE_DIR, "input_v2_resampled.mcc")
OUT_DIR <- "img_v2/fig2"
FONT_FAMILY <- "Arial"
FONT_SIZE_PT <- 9
N_SAMPLE <- 500
EDGE_ALPHA <- 0.08
CONSENSUS_ALPHA <- 0.6
CONSENSUS_LWD <- 1.5
CLOUD_DPI <- 300
# Shared x-frame (inches / ka BP): keep in sync with FIG2_* in plots/_config.py
FIG_W <- 7.2
AX_LEFT <- 0.55
AX_W <- 5.6
X_MAX <- 8
# Tall enough for 50 tip labels at FONT_SIZE_PT (~1.07 x font size per tip)
FIG_H <- 7.8

TIP_ORDER <- c(
  "Assamese",
  "Bengali",
  "Oriya",
  "Nepali",
  "Marathi",
  "Gujarati",
  "Punjabi",
  "Hindi",
  "Urdu",
  "Sindhi",
  "Pashto",
  "Sorani-Kurdish",
  "Tajik",
  "Persian",
  "Armenian",
  "Lithuanian",
  "Latvian",
  "Bulgarian",
  "Macedonian",
  "Slovenian",
  "Croatian",
  "Bosnian",
  "Serbian",
  "Ukrainian",
  "Belarusian",
  "Russian",
  "Polish",
  "Czech",
  "Slovak",
  "Icelandic",
  "Norwegian",
  "Swedish",
  "Danish",
  "German",
  "Luxembourgish",
  "Dutch",
  "English",
  "Irish",
  "Welsh",
  "Greek",
  "Romanian",
  "Italian",
  "French",
  "Occitan",
  "Catalan",
  "Spanish",
  "Asturian",
  "Galician",
  "Portuguese",
  "Kabuverdianu"
)

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# --- Load trees ---
cat("Reading trees from", TREES_FILE, "...\n")
all_trees <- read.annot.beast(TREES_FILE, stride = 18)
trees <- all_trees[sapply(all_trees, function(tr) !is.null(tr$metadata))]
rm(all_trees)
invisible(gc())
cat("Loaded", length(trees), "trees with rate metadata\n")

set.seed(42)
if (length(trees) > N_SAMPLE) {
  idx <- sort(sample.int(length(trees), N_SAMPLE))
  trees <- trees[idx]
  cat("Subsampled to", length(trees), "trees\n")
}

consensus_tree <- read.annot.beast(MCC_FILE)

# --- Rate color scale (viridis) ---
all_rates <- unlist(lapply(trees, function(tr) as.numeric(tr$metadata$rate)))
all_rates <- all_rates[!is.na(all_rates) & all_rates > 0]
log_rate_range <- range(log(all_rates))

viridis_hex <- c(
  "#440154",
  "#482173",
  "#433E85",
  "#38588C",
  "#2D708E",
  "#25858E",
  "#1E9B8A",
  "#2BB07F",
  "#51C56A",
  "#85D54A",
  "#C2DF23",
  "#FDE725"
)
RATE_PALETTE <- colorRampPalette(viridis_hex)(256)
RATE_PALETTE_A <- adjustcolor(RATE_PALETTE, alpha.f = EDGE_ALPHA)
RATE_PALETTE_C <- adjustcolor(RATE_PALETTE, alpha.f = CONSENSUS_ALPHA)
GREY_A <- adjustcolor("grey50", alpha.f = EDGE_ALPHA)

rate_idx <- function(rates) {
  idx <- round((log(rates) - log_rate_range[1]) / diff(log_rate_range) * 255) +
    1
  idx[is.na(idx) | !is.finite(idx)] <- 128
  pmax(1L, pmin(256L, idx))
}

# --- Layout ---
n_tips <- length(TIP_ORDER)
tip_y <- setNames(seq(n_tips, 1), TIP_ORDER)

heights <- sapply(trees, function(t) max(node.depth.edgelength(t)))
max_height <- max(heights, max(node.depth.edgelength(consensus_tree)))
if (max_height > X_MAX) {
  cat("Note: trees up to", round(max_height, 2), "ka clipped at X_MAX =", X_MAX, "\n")
}

tree_layout <- function(tr, palette) {
  n <- Ntip(tr)
  total <- n + tr$Nnode

  depths <- node.depth.edgelength(tr)
  tree_h <- max(depths)
  x <- max_height - tree_h + depths

  y <- numeric(total)
  for (i in seq_len(n)) {
    y[i] <- tip_y[tr$tip.label[i]]
  }

  tr_po <- reorder(tr, "postorder")
  child_y <- vector("list", total)
  for (i in seq_len(nrow(tr_po$edge))) {
    p <- tr_po$edge[i, 1]
    child_y[[p]] <- c(child_y[[p]], y[tr_po$edge[i, 2]])
    y[p] <- mean(child_y[[p]])
  }

  edge <- tr$edge
  rates_vec <- setNames(as.numeric(tr$metadata$rate), tr$metadata$node)
  cols <- palette[rate_idx(rates_vec[as.character(edge[, 2])])]
  cols[is.na(cols)] <- GREY_A

  list(x = x, y = y, edge = edge, cols = cols)
}

# Pre-compute all segments for posterior trees
cat("Computing layouts...\n")
n_edges <- nrow(trees[[1]]$edge)
n_total <- n_edges * length(trees)

hx0 <- numeric(n_total)
hy0 <- numeric(n_total)
hx1 <- numeric(n_total)
hy1 <- numeric(n_total)
vx0 <- numeric(n_total)
vy0 <- numeric(n_total)
vx1 <- numeric(n_total)
vy1 <- numeric(n_total)
seg_col <- character(n_total)

off <- 0L
for (k in seq_along(trees)) {
  lay <- tree_layout(trees[[k]], RATE_PALETTE_A)
  e <- lay$edge
  ne <- nrow(e)
  ii <- off + seq_len(ne)

  hx0[ii] <- lay$x[e[, 1]]
  hy0[ii] <- lay$y[e[, 2]]
  hx1[ii] <- lay$x[e[, 2]]
  hy1[ii] <- lay$y[e[, 2]]
  vx0[ii] <- lay$x[e[, 1]]
  vy0[ii] <- lay$y[e[, 1]]
  vx1[ii] <- lay$x[e[, 1]]
  vy1[ii] <- lay$y[e[, 2]]
  seg_col[ii] <- lay$cols

  off <- off + ne
}
cat("Pre-computed", off, "edge segments\n")

con_lay <- tree_layout(consensus_tree, RATE_PALETTE_C)

rm(trees)
invisible(gc())

# --- Plot ---
# Draw the posterior cloud into a transparent PNG covering exactly the current
# plot region, then place it back as a single image (keeps the SVG small).
draw_cloud_raster <- function() {
  usr <- par("usr")
  pin <- par("pin")
  vec_dev <- dev.cur()
  tmp_png <- tempfile(fileext = ".png")
  png(
    tmp_png,
    width = pin[1],
    height = pin[2],
    units = "in",
    res = CLOUD_DPI,
    type = "cairo",
    bg = "transparent"
  )
  par(mai = c(0, 0, 0, 0), xaxs = "i", yaxs = "i")
  plot.new()
  plot.window(xlim = usr[1:2], ylim = usr[3:4])
  segments(hx0, hy0, hx1, hy1, col = seg_col, lwd = 0.5)
  segments(vx0, vy0, vx1, vy1, col = seg_col, lwd = 0.5)
  dev.off()
  dev.set(vec_dev)
  rasterImage(png::readPNG(tmp_png), usr[1], usr[3], usr[2], usr[4])
  unlink(tmp_png)
}

plot_fig2a <- function(outfile, fmt = "pdf") {
  # pointsize sets every text element to FONT_SIZE_PT (no per-element cex)
  if (fmt == "pdf") {
    cairo_pdf(
      outfile,
      width = FIG_W,
      height = FIG_H,
      family = FONT_FAMILY,
      pointsize = FONT_SIZE_PT
    )
  } else {
    svg(
      outfile,
      width = FIG_W,
      height = FIG_H,
      family = FONT_FAMILY,
      pointsize = FONT_SIZE_PT
    )
  }

  par(
    family = FONT_FAMILY,
    mai = c(0.5, AX_LEFT, 0.05, FIG_W - AX_LEFT - AX_W),
    xaxs = "i"
  )

  plot(
    NULL,
    xlim = c(max_height - X_MAX, max_height),
    ylim = c(0.5, n_tips + 0.5),
    xlab = "",
    ylab = "",
    axes = FALSE
  )

  ax_vals <- sort(unique(c(pretty(c(X_MAX, 0)), X_MAX)))
  ax_pos <- max_height - ax_vals
  vis <- ax_pos >= (max_height - X_MAX) & ax_pos <= max_height
  abline(v = ax_pos[vis], col = "grey80", lty = "dashed", lwd = 0.4)

  draw_cloud_raster()

  ce <- con_lay$edge
  segments(
    con_lay$x[ce[, 1]],
    con_lay$y[ce[, 2]],
    con_lay$x[ce[, 2]],
    con_lay$y[ce[, 2]],
    col = con_lay$cols,
    lwd = CONSENSUS_LWD
  )
  segments(
    con_lay$x[ce[, 1]],
    con_lay$y[ce[, 1]],
    con_lay$x[ce[, 1]],
    con_lay$y[ce[, 2]],
    col = con_lay$cols,
    lwd = CONSENSUS_LWD
  )

  text(
    max_height,
    tip_y,
    names(tip_y),
    pos = 4,
    xpd = TRUE
  )

  axis(
    1,
    at = ax_pos[vis],
    labels = ax_vals[vis],
    family = FONT_FAMILY
  )
  mtext(
    "Age (ka BP)",
    side = 1,
    line = 2,
    family = FONT_FAMILY
  )

  # --- Inset color bar (horizontal, top-left) ---
  cb_x0 <- max_height - 5.5
  cb_x1 <- max_height - 4.5
  cb_y0 <- n_tips - 1.5
  cb_y1 <- n_tips - 0.5
  n_cb <- length(RATE_PALETTE)
  cb_breaks <- seq(cb_x0, cb_x1, length.out = n_cb + 1)
  rect(
    cb_breaks[1:n_cb],
    cb_y0,
    cb_breaks[2:(n_cb + 1)],
    cb_y1,
    col = RATE_PALETTE,
    border = NA
  )
  rect(cb_x0, cb_y0, cb_x1, cb_y1, col = NA, border = "black", lwd = 0.5)

  lr_ticks <- seq(log_rate_range[1], log_rate_range[2], length.out = 3)
  tick_x <- cb_x0 +
    (lr_ticks - log_rate_range[1]) /
      diff(log_rate_range) *
      (cb_x1 - cb_x0)
  segments(tick_x, cb_y0, tick_x, cb_y0 - 0.3, lwd = 0.5)
  text(
    tick_x,
    cb_y0 - 0.5,
    sprintf("%.1f", lr_ticks),
    family = FONT_FAMILY
  )
  text(
    mean(c(cb_x0, cb_x1)),
    cb_y1 + 0.4,
    "Log evol. rate",
    family = FONT_FAMILY
  )

  dev.off()
  cat("Saved", outfile, "\n")
}

# --- Export ---
plot_fig2a(file.path(OUT_DIR, "fig2a_densitree.pdf"), "pdf")
plot_fig2a(file.path(OUT_DIR, "fig2a_densitree.svg"), "svg")
