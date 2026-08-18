#!/usr/bin/env bash
#SBATCH -J dl_ncbi
#SBATCH -p cpu-single
#SBATCH -N 1 -c 4 --mem=24G
#SBATCH -t 02:00:00
#SBATCH --export=NONE
#SBATCH -o logs/01_download_ncbi.out
#SBATCH -e logs/01_download_ncbi.err

set -euo pipefail
export PATH="$HOME/bin:/usr/bin:/bin"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate evoplm-paper

mkdir -p data/raw/ncbi_virus

datasets download virus genome taxon 2697049 \
  --host "Homo sapiens" \
  --include genome,annotation \
  --filename data/raw/ncbi_virus/ncbi_all.zip

unzip -o data/raw/ncbi_virus/ncbi_all.zip -d data/raw/ncbi_virus/all/
