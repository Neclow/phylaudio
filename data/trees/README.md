# Phylogenetic trees

The `trees` directory should contain the following subdirectories:

- `beast`: [BEAST 2](https://www.beast2.org/) runs. Contains `template.xml` used
  by `beast_generate_xml` to generate run configurations.
- `per_sentence`: sentence_wise trees
  - `discrete`: trees obtained via `pipeline/sentence_discrete_trees`
  - `pdist`: trees obtained via `pipeline/sentence_distance_trees`
- `references`: reference trees obtained via `pipeline/download_reference_trees`

Each of these tree directories should be formatted as follows:

- `xxxxxxxx-xxxx-4xxx-Nxxx-xxxxxxxxxxxx`: run ID
  - `cfg.json`: run parameters
  - various outputs (depend on the task)
