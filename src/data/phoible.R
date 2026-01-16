library(dplyr)

extract_n_segments <- function(filepath) {
  # From https://phoible.org/faq
  # SPA: The Stanford Phonology Archive (Crothers et al., 1979);
  # UPSID: The UCLA Phonological Segment Inventory Database
  #   (Maddieson, 1984; Maddieson & Precoda, 1990);
  # UZ: Data drawn from journal articles, theses, and published grammars,
  #   added by the phoible developers while at the Department of Comparative
  #   Linguistics at the University of Zurich;
  # EA: represents languages of Eurasia;
  # RA: represents languages of India;

  # Belarusian is missing. Its closest neighbour is accepted to be Ukrainian.
  # Source: https://doi.org/10.1515/9783110542431-007; https://doi.org/10.1017/CBO9780511486807
  read.csv(filepath) |>
    filter(Source %in% c("spa", "upsid", "uz", "ea", "ra")) |>
    group_by(LanguageName, Glottocode) |>
    summarise(n_segments = n_distinct(Phoneme)) |>
    ungroup()
}
