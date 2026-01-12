#!/bin/bash
#SBATCH --job-name=cika_step1_seed
#SBATCH --partition=normal
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/step1_seed_%j.out
#SBATCH --error=logs/step1_seed_%j.err

set -euo pipefail

echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-local}"

PROJECT_ROOT=${PROJECT_ROOT:-$PWD}
cd "$PROJECT_ROOT"

# module purge
# module load python/3.10

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "Using active conda env: $CONDA_PREFIX"
else
    echo "WARNING: No conda environment detected."
    echo "Make sure python >=3.9 and requests/pyyaml are available."
fi

CONFIG_DIR="configs"
DATA_DIR="data/seeds"
LOG_DIR="logs"

SCOPE_YAML="${CONFIG_DIR}/scope.yaml"
SEEDS_YAML="${CONFIG_DIR}/seeds.yaml"

UNIPROT_CACHE="${DATA_DIR}/uniprot_cache.json"
OUT_TSV="${DATA_DIR}/seed_normalized.tsv"
OUT_JSON="${DATA_DIR}/seed_normalized.json"

STEP_LOG="${LOG_DIR}/step1_seed_normalize.log"

echo "Checking required files..."

for f in "$SCOPE_YAML" "$SEEDS_YAML"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: Missing required file: $f"
        exit 1
    fi
done

mkdir -p "$DATA_DIR" "$LOG_DIR"

python - <<'EOF'
import sys
missing = []
try:
    import requests
except ImportError:
    missing.append("requests")
try:
    import yaml
except ImportError:
    missing.append("pyyaml")

if missing:
    sys.stderr.write(
        "ERROR: Missing python packages: " + ", ".join(missing) + "\n"
        "Install them via:\n"
        "  pip install " + " ".join(missing) + "\n"
    )
    sys.exit(1)
EOF

echo "Running seed normalization..."

python -m src.seed_normalize \
  --scope "$SCOPE_YAML" \
  --seeds "$SEEDS_YAML" \
  --cache "$UNIPROT_CACHE" \
  --out_tsv "$OUT_TSV" \
  --out_json "$OUT_JSON" \
  --log "$STEP_LOG"

echo "---------- Summary ----------"
echo "Normalized seeds written to:"
echo "  - $OUT_TSV"
echo "  - $OUT_JSON"
echo "UniProt cache:"
echo "  - $UNIPROT_CACHE"
echo "Log file:"
echo "  - $STEP_LOG"

echo "End time: $(date)"
echo "=============================================="
