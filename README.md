# Phylaudio

Repository for _Indo-European language acoustics through space and time_, Under
review, 2026.

Phylaudio investigates the phylogenetic signal in speech acoustics. Pre-trained
audio models (XLS-R, MMS, Whisper, ECAPA-TDNN, openSMILE) extract embeddings
from multilingual read speech (FLEURS-R), which are discretised into
phylogenetic characters. Maximum-likelihood (IQ-TREE), distance-based (FastME),
and Bayesian (BEAST2) methods infer trees over 50 Indo-European languages and
compare them against established linguistic phylogenies.

## Installation

### Prerequisites

All code was developed and tested on Ubuntu 24.04 (linux-64) with CUDA 12+ (used
version: 13.2).

Install [pixi](https://pixi.sh), a conda-based package manager:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

### Dependencies

To install the main dependencies, run:

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

See [docs/](docs) for detailed instructions. The pipeline runs in six stages:

1. **Download** — audio, metadata, pre-trained models, reference trees
2. **Language identification** — embedding extraction, MLP-based LID training,
   speech quality estimation
3. **Sentence trees** — per-sentence phylogenetic inference (IQ-TREE or FastME)
   and supertree construction (ASTRAL)
4. **BEAST** — Bayesian divergence-time estimation with BEAST2
5. **Post-BEAST** — NMF population structure, phylogenetic regression,
   geographic rate surfaces
6. **Plots** — publication figures

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
