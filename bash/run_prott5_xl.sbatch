#!/usr/bin/env bash
#SBATCH -J esm2_650m
#SBATCH -p batch
#SBATCH -A marlowe-m000226-free
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail

source ~/.bashrc
conda activate plm

export HF_HOME=/projects/m000226/hf
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SCRIPT_DIR=/projects/m000226/seqfm_geometry/src/emb
OUT_DIR=/projects/m000226/seqfm_geometry/outputs

python $SCRIPT_DIR/emb_prott5.py \
  --model Rostlab/prot_t5_xl_uniref50 \
  --fasta seqs.fasta \
  --node_ids node_ids.txt \
  --out $OUT_DIR/prott5_xl.npy \
  --max_len 1024 \
  --stride 768 \
  --bucket 128 \
  --win_bs 16 \
  --l2norm