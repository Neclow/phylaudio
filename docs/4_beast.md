# Bayesian and network-based phylogenetics

Bayesian phylogenetic inference via BEAST2 with CoupledMCMC (MC3). The pipeline
goes: select top sentence trees, generate XML, run MCMC chains, combine runs,
summarize the posterior tree, and infer a phylogenetic network using SplitsTree.

## XML generation

- **Command:** `pixi run beast_generate_xml <run_id> -p <fraction>`
- **Requires:** `sentence_stats`
- **Inputs:** `_stats.csv` and FASTA files in `data/trees/per_sentence/`,
  `languages.json` (committed), XML templates in `data/trees/beast/templates/`
- **Outputs:** `data/trees/beast/<run_id>/<p>_<by>/` (`merged.fasta`,
  `mapped.fasta`, `input_v1.xml`, `prior_v1.xml`, `ns/input_ns_*.xml`)

Selects the top-p% per-sentence trees (ranked by branch support, stemminess, or
clock-likeness via `--by`), merges their FASTA sequences, maps taxon IDs to
reference names, and fills BEAST2 XML templates for main, prior-only, and
nested-sampling runs. Templates live in `data/trees/beast/templates/` and define
the clock model, tree prior, substitution model, and calibration priors; this
script only injects the alignment and taxon set.

Selection criteria (`--by`):

- `brsupport` (branch support, default)
- `stemmy` (stemminess)
- `clock` (clock-likeness)

```bash
pixi run beast_generate_xml ba9f2d2a -p 0.05
```

## MCMC run

- **Command:** `pixi run beazt [-r] [-o] <uuid> <size> <version> <run_number>`
- **Requires:** BEAST2 (`extern/beast2/`), BEAGLE GPU library
- **Inputs:** `data/trees/beast/<uuid>/<size>_brsupport/input_v<version>.xml`
- **Outputs:** `input_v<version>_<seed>.log`, `.trees`, `.xml.state` in the same
  directory

Runs a single BEAST2 CoupledMCMC analysis with BEAGLE GPU acceleration.

Flags:

- `-r`: resume from a previous state
- `-o`: overwrite existing output
- `-v`: validate XML only (no sampling)

## Combine runs

- **Command:**
  `pixi run beazt_combine [-o] [-b burnin_pct] [-n samples_per_run] <uuid> <size> <version>`
- **Requires:** BEAST2 LogCombiner, TreeAnnotator
- **Inputs:** `input_v<version>_*.log` and `.trees` files from multiple MCMC
  runs
- **Outputs:** `combined_v<version>/` directory containing
  `input_v<version>_combined.log`, `.trees`, optionally `_resampled.log`,
  `.trees`, and an MCC summary tree (`.mcc`)

Combines independent MCMC runs using LogCombiner (discarding the first
`burnin_pct`% of each), optionally resamples to a fixed number of states per
run, and produces an MCC (maximum clade credibility) tree via TreeAnnotator.

Flags:

- `-b <pct>`: burn-in percentage (default: 10)
- `-n <count>`: resample to this many states per run (optional)
- `-o`: overwrite existing combined output

```bash
pixi run beazt_combine ba9 0.05_brsupport_dev_test 2 -b 10 -n 2500
```

## Nested sampling (model selection)

- **Command:** `bash pipeline/run_beast_ns.sh <uuid> <size> [seed]`
- **Requires:** BEAST2 with NS package, BEAGLE GPU
- **Inputs:** `data/trees/beast/<uuid>/<size>/ns/input_ns_*.xml`
  (auto-discovered)
- **Outputs:** per-model stdout logs, `ns/results/ns_summary.tsv` (marginal
  likelihoods, SDs, ln Bayes factors)

Batch-runs all nested-sampling XML files in an `ns/` subdirectory and produces a
sorted summary with marginal likelihoods and ln Bayes factors relative to the
baseline model.

## Phylogenetic network

- **Command:**
  `pixi run network_analysis -- <run_id> <subdir> [--version <int>]`
- **Requires:** SplitsTree6 (`extern/splitstree/`)
- **Inputs:** BEAST2 input XML (extracts the FASTA alignment)
- **Outputs:** `mapped.fasta`, `splitstree.fasta`, `splitstree.stree6` in the
  BEAST run directory

Extracts the alignment from a BEAST2 input XML and runs a SplitsTree6
NeighborNet workflow to produce a phylogenetic network. Note: SplitsTree doesn't
work on a headless display; on Linux, we used MobaXTerm.

```bash
pixi run network_analysis -- ba9f2d2a 0.05_brsupport_dev_test
```

## BEAST2 utilities

Thin wrappers around BEAST2 and SplitsTree binaries in `extern/`, exposed as
pixi tasks for convenience:

| Task             | Binary                             |
| ---------------- | ---------------------------------- |
| `beast2`         | `extern/beast2/bin/beast`          |
| `loganalyser`    | `extern/beast2/bin/loganalyser`    |
| `logcombiner`    | `extern/beast2/bin/logcombiner`    |
| `treeannotator`  | `extern/beast2/bin/treeannotator`  |
| `packagemanager` | `extern/beast2/bin/packagemanager` |
| `splitstree`     | `extern/splitstree/SplitsTree`     |

```bash
pixi run loganalyser input_v2_101.log
pixi run treeannotator combined.trees combined.mcc
```
