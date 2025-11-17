#!/bin/bash

# Install missing python packages
## Install fairseq manually to make it compatible with xls-r weigts
uv pip install fairseq --no-deps
uv pip install bitarray sacrebleu omegaconf hydra-core
## Flash attention
uv pip install flash-attn --no-build-isolation

# Install missing R packages
Rscript -e 'install.packages("Quartet", repos="https://cloud.r-project.org")'

# Install BEAST 2.7 and packages
BEAST_TARGET=extern/beast2.tgz
BEAST_PKG_DIR=.beast
wget -O $BEAST_TARGET https://github.com/CompEvol/beast2/releases/download/v2.7.7/BEAST.v2.7.7.Linux.x86.tgz
tar -xvzf $BEAST_TARGET -C extern/
mv extern/beast extern/beast2
mkdir -p $BEAST_PKG_DIR
pixi run packagemanager -add BEASTLabs -version 2.0.2 -dir $BEAST_PKG_DIR
pixi run packagemanager -add BDSKY -version 1.5.1 -dir $BEAST_PKG_DIR
pixi run packagemanager -add SA -version 2.1.1 -dir $BEAST_PKG_DIR
pixi run packagemanager -add CCD -version 1.0.2 -dir $BEAST_PKG_DIR

# Install DensiTree
wget -O extern/DensiTree.jar https://github.com/rbouckaert/DensiTree/releases/download/v3.0.0/DensiTree.v3.1.0.jar