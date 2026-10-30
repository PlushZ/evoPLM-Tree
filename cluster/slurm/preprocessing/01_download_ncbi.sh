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
[[ -f cluster/env.local.sh ]] && source cluster/env.local.sh
conda activate "${CONDA_ENV:-evoplm}"
# pandas/scipy load C++ extensions built against the environment's libstdc++;
# compute nodes can ship an older /lib64 one (GLIBCXX_3.4.29 not found).
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

mkdir -p data/raw/ncbi_virus

datasets download virus genome taxon 2697049 \
  --host "Homo sapiens" \
  --include genome,annotation \
  --filename data/raw/ncbi_virus/ncbi_all.zip

unzip -o data/raw/ncbi_virus/ncbi_all.zip -d data/raw/ncbi_virus/all/

# 02_prefilter_split.sh expects a single flat multi-FASTA at this path
cp data/raw/ncbi_virus/all/ncbi_dataset/data/genomic.fna data/raw/ncbi_virus/all_genomes.fasta
