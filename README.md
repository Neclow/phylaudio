# Phylaudio

Repository for _Indo-European language acoustics through space and time_, Under
review, 2026.

## Installation

### Prerequisites

Install [pixi](https://pixi.sh), a conda-based package manager:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Dependencies

> [!IMPORTANT] > **GPU vs. CPU.** Compute-intensive steps were run on a CUDA
> 13.2-powered GPU. On CPU-only machines, set the CUDA override so `pixi` can
> resolve the dependency tree, then pass `--device cpu` to any script that
> accepts a device flag:
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
[docs/README.md](docs/README.md).

## Citation

If you use this code, please cite:

```bibtex
@article{scheidwasser2026phylaudio,
  title   = {Indo-European language acoustics through space and time},
  author  = {Scheidwasser, Neil and Zhu, Harrison Bo Hua and Fosse, Samuel and Huang, Hengguan and Greenhill, Simon J. and Bhatt, Samir and Duch{\^e}ne, David A.},
  year    = {2026},
  note    = {Under review}
}
```

## License

[MIT](LICENSE)
