# Phylaudio

Repository for "Scheidwasser, Zhu, Duchêne & Bhatt. Indo-European language
acoustics through space and time. Under review, 2026."

## Installation

### Prerequisites

Install [pixi](https://pixi.sh), a conda-based package manager:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Dependencies

> [!NOTE] > **GPU vs. CPU.** Embedding extraction and LID training were run on a
> CUDA 13.2 GPU. On CPU-only machines, set the CUDA override so pixi can resolve
> the dependency tree, then pass `--device cpu` to any script that accepts a
> device flag:
>
> ```bash
> CONDA_OVERRIDE_CUDA=13.2 pixi install
> ```
>
> To use a different CUDA version, edit the `cuda` key under
> `[system-requirements]` in `pixi.toml`.

```bash
pixi install
pixi run post_install
```

Additional environments are needed for specific pipeline stages:

```bash
pixi install -e regression  # phylogenetic regression (R + brms)
pixi install -e gp          # phylogenetic regression with Gaussian processes (tensorflow, GPflow)
pixi install -e viz         # publication figures (plotly, seaborn)
```

## Usage

Data download and pipeline steps are documented in
[docs/pipeline.md](docs/pipeline.md).

## License

[MIT](LICENSE)
