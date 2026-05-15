#!/bin/bash
#SBATCH --account=project_465002754
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4          # 4 GCDs
#SBATCH --cpus-per-task=28         # 4 x 7 cores
#SBATCH --mem=240G                 # 4 x 60 GB
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project_465002754/kingdom_stress/logs/train_%j.out
#SBATCH --error=/scratch/project_465002754/kingdom_stress/logs/train_%j.err

# =============================================================================
# PlantRNA-FM fine-tuning: 12 models (6 stresses x 2 directions)
# 4 models run in parallel (one per GCD), 3 batches.
# =============================================================================

CODE_DIR=/scratch/project_465002754/kingdom_stress/code
DATA_DIR=/scratch/project_465002754/kingdom_stress/data
SIF=/scratch/project_465002754/kingdom_stress/containers/plantrna.sif
RESULTS_DIR=/scratch/project_465002754/kingdom_stress/results

export PYTHONPATH=${CODE_DIR}
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

REAL_SCRATCH=$(realpath /scratch/project_465002754)
BIND="${REAL_SCRATCH}:/scratch/project_465002754"

mkdir -p "${RESULTS_DIR}" "/scratch/project_465002754/kingdom_stress/logs"

if [[ ! -f "${SIF}" ]]; then
    echo "ERROR: container not found at ${SIF}"
    exit 1
fi

echo "========================================================"
echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     ${SLURM_NODELIST}"
echo "Started:  $(date)"
echo "========================================================"

# All 12 stress x direction combinations
MODELS=(
    "Heat UP"
    "Heat DOWN"
    "Cold UP"
    "Cold DOWN"
    "Drought UP"
    "Drought DOWN"
    "Salt UP"
    "Salt DOWN"
    "Pathogen UP"
    "Pathogen DOWN"
    "Flooding UP"
    "Flooding DOWN"
)

# Run 4 at a time
GCD=0
RUNNING=0

for MODEL_SPEC in "${MODELS[@]}"; do
    STRESS=$(echo ${MODEL_SPEC} | cut -d' ' -f1)
    DIR=$(echo ${MODEL_SPEC} | cut -d' ' -f2)
    TAG="${STRESS}_${DIR}"

    # Skip if already done
    if [[ -f "${RESULTS_DIR}/${TAG}_result.json" ]]; then
        echo "SKIP: ${TAG} (already done)"
        continue
    fi

    echo "START: ${TAG} on GCD ${GCD}"

    ROCR_VISIBLE_DEVICES=${GCD} PYTHONUNBUFFERED=1 \
        singularity exec --rocm --bind "${BIND}" "${SIF}" \
        stdbuf -oL python "${CODE_DIR}/train_rna.py" \
        --stress "${STRESS}" \
        --direction "${DIR}" \
        --data_dir "${DATA_DIR}" \
        --output_dir "${RESULTS_DIR}" \
        --batch_size 32 \
        --epochs 5 \
        > "/scratch/project_465002754/kingdom_stress/logs/${TAG}_${SLURM_JOB_ID}.log" 2>&1 &

    GCD=$(( (GCD + 1) % 4 ))
    RUNNING=$((RUNNING + 1))

    # Wait after every 4 launches
    if [[ ${RUNNING} -ge 4 ]]; then
        echo "Waiting for batch to finish..."
        wait
        RUNNING=0
        echo "Batch complete: $(date)"
    fi
done

# Wait for any remaining
wait

echo "========================================================"
echo "All training complete: $(date)"
echo "Results: ${RESULTS_DIR}/*_result.json"
echo "========================================================"

# Print summary
echo ""
echo "=== Results Summary ==="
for f in ${RESULTS_DIR}/*_result.json; do
    if [[ -f "$f" ]]; then
        TAG=$(basename "$f" _result.json)
        AUROC=$(python3 -c "import json; print(json.load(open('$f'))['test_auROC'])" 2>/dev/null)
        echo "  ${TAG}: auROC=${AUROC}"
    fi
done
