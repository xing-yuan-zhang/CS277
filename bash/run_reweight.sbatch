#!/usr/bin/env bash
#SBATCH -J pll
#SBATCH -p batch
#SBATCH -A marlowe-m000226-free
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH -t 48:00:00
#SBATCH --array=0-9
#SBATCH -o logs/%x.%A_%a.out
#SBATCH -e logs/%x.%A_%a.err

set -euo pipefail

mkdir -p logs

source /projects/m000226/miniconda3/etc/profile.d/conda.sh
conda activate plm

which python
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

SCRIPT_DIR=/projects/m000226/seqfm-geometry/src/graph
EDGE_DIR=/projects/m000226/seqfm-geometry/parent_graph/
DATA_DIR=/projects/m000226/seqfm-geometry/graph/
OUT_DIR=/projects/m000226/seqfm-geometry/graph
mkdir -p "$OUT_DIR"

python $SCRIPT_DIR/reweight.py \
  --string_edges "$EDGE_DIR/string_edges.tsv" \
  --fmppi_edges "$DATA_DIR/fmppi.tsv" \
  --beta 1.0 \
  --f exp \
  --outdir "$OUT_DIR"