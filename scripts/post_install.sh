#!/bin/bash

# Install missing python packages
## Install fairseq manually to make it compatible with xls-r weigts
uv pip install fairseq --no-deps
uv pip install bitarray sacrebleu omegaconf hydra-core
## Flash attention
uv pip install flash-attn --no-build-isolation

# Install missing R packages
Rscript -e 'install.packages("Quartet", repos="https://cloud.r-project.org")'

# Install BEAST 2.7
wget -O extern/beast2.tgz https://github.com/CompEvol/beast2/releases/download/v2.7.7/BEAST.v2.7.7.Linux.x86.tgz
tar -xvzf extern/beast2.tgz -C extern/
mv extern/beast extern/beast2

