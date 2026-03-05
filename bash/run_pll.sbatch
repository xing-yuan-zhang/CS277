#!/usr/bin/env bash
#SBATCH -J fmppi
#SBATCH -p batch
#SBATCH -A marlowe-m000226-free
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail

mkdir -p logs

source /projects/m000226/miniconda3/etc/profile.d/conda.sh
conda activate plm

which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

SCRIPT_DIR=/projects/m000226/seqfm-geometry/src/graph
OUT_DIR=/projects/m000226/seqfm-geometry/graph
mkdir -p "$OUT_DIR"

python $SCRIPT_DIR/attention_proxy.py \
  --emb_npz esm2_t33_650m.npz \
  --fasta seqs.fasta \
  --nodes node_ids.txt \
  --model_name facebook/esm2_t33_650M_UR50D \
  --method pll \
  --pll_sample_k 64 \
  --cand_M 200 \
  --topm 20 \
  --outdir graphs