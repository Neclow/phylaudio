library(dplyr)
library(phangorn)
library(stringr)

extract_delta <- function(filepath) {
  content <- scan(filepath, "rb", sep = "\n")
  do_save <- FALSE
  delta_content <- NULL
  for (line in content) {
    if (grepl("Delta scores for individual taxa:", line)) {
      do_save <- TRUE
      next
    }

    if (do_save && str_trim(line) == "''") {
      do_save <- FALSE
      break
    }

    if (do_save) {
      delta_content <- c(
        delta_content,
        gsub(line, pattern = "'", replacement = "")
      )
    }
  }

  df <- read.table(
    text = delta_content,
    sep = "\t",
    header = TRUE,
    stringsAsFactors = FALSE,
    quote = "'",
    row.names = 2
  ) |>
    select_if(~ !any(is.na(.)))

  df
}
