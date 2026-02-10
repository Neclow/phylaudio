#!/bin/bash

# Install missing python packages
## Install fairseq manually to make it compatible with xls-r weights
uv pip install fairseq --no-deps
uv pip install bitarray sacrebleu omegaconf hydra-core
## Flash attention
uv pip install flash-attn --no-build-isolation

# Install missing R packages
Rscript -e 'install.packages(c("Quartet", "brms"), repos="https://cloud.r-project.org")'

# Install BEAST 2.7 and packages
BEAST_TARGET=extern/beast2.tgz
BEAST_PKG_DIR=.beast
wget --secure-protocol=auto --max-redirect=3 -O $BEAST_TARGET https://github.com/CompEvol/beast2/releases/download/v2.7.7/BEAST.v2.7.7.Linux.x86.tgz
tar -xvzf $BEAST_TARGET -C extern/
mv extern/beast extern/beast2
mkdir -p $BEAST_PKG_DIR
pixi run packagemanager -add BEASTLabs -version 2.0.2 -dir $BEAST_PKG_DIR
pixi run packagemanager -add BDSKY -version 1.5.1 -dir $BEAST_PKG_DIR
pixi run packagemanager -add SA -version 2.1.1 -dir $BEAST_PKG_DIR
pixi run packagemanager -add CCD -version 1.0.2 -dir $BEAST_PKG_DIR
rm $BEAST_TARGET

# Install DensiTree
wget -O extern/DensiTree.jar https://github.com/rbouckaert/DensiTree/releases/download/v3.0.0/DensiTree.v3.1.0.jar

# Install SplitsTree6
SPLITSTREE_VERSION=6_3_20
wget -O extern/SplitsTree_unix_$SPLITSTREE_VERSION.sh https://software-ab.cs.uni-tuebingen.de/download/splitstree6/SplitsTree_unix_$SPLITSTREE_VERSION.sh

# Make SplitsTree executable
chmod +x extern/SplitsTree_unix_$SPLITSTREE_VERSION.sh

# Install SplitsTree to extern/splitstree
./extern/SplitsTree_unix_$SPLITSTREE_VERSION.sh -dir splitstree

# Install Fine-tuned XLS-R weights
wget -O data/models/xlsr_300m_voxlingua107_ft.pt https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr_300m_voxlingua107_ft.pt
