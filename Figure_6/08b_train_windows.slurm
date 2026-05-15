#!/bin/bash
#SBATCH --account=project_465002754
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/project_465002754/kingdom_stress/logs/train_windows_%j.out
#SBATCH --error=/scratch/project_465002754/kingdom_stress/logs/train_windows_%j.err

# =============================================================================
# Train PlantRNA-FM on different transcript windows.
# Runs 4 models in parallel (one per GCD), iterates through batches.
#
# Usage:
#   sbatch train_windows.slurm w05_15kb
#   sbatch train_windows.slurm w1_2kb
#   sbatch train_windows.slurm w3end
# =============================================================================

WINDOW=${1:?"Usage: sbatch train_windows.slurm <window_name>"}

CODE_DIR=/scratch/project_465002754/kingdom_stress/code
DATA_DIR=/scratch/project_465002754/kingdom_stress/data
RESULTS_DIR=/scratch/project_465002754/kingdom_stress/results_${WINDOW}
SIF=/scratch/project_465002754/kingdom_stress/containers/plantrna.sif

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

REAL_SCRATCH=$(realpath /scratch/project_465002754)
BIND="${REAL_SCRATCH}:/scratch/project_465002754"

mkdir -p "${RESULTS_DIR}"
mkdir -p "/scratch/project_465002754/kingdom_stress/logs"

echo "========================================================"
echo "Job:      ${SLURM_JOB_ID}"
echo "Node:     ${SLURM_NODELIST}"
echo "Window:   ${WINDOW}"
echo "Started:  $(date)"
echo "========================================================"

# Step 1: Preprocess data for this window (builds ready/*.pt files)
echo "Preprocessing data for window ${WINDOW}..."
singularity exec --bind "${BIND}" "${SIF}" \
    python "${CODE_DIR}/preprocess_data.py" \
    --data_dir "${DATA_DIR}" \
    --output_dir "${DATA_DIR}/ready_${WINDOW}" \
    --tok_cache "${DATA_DIR}/rna_tokens_windows/${WINDOW}"

echo "Preprocessing done: $(date)"

# Step 2: Train models (4 at a time)
STRESSES=("Heat" "Cold" "Drought" "Salt" "Pathogen" "Flooding")
DIRECTIONS=("UP" "DOWN")

# Build list of all stress_direction combos
TAGS=()
for S in "${STRESSES[@]}"; do
    for D in "${DIRECTIONS[@]}"; do
        TAGS+=("${S}_${D}")
    done
done

# Run in batches of 4
for ((i=0; i<${#TAGS[@]}; i+=4)); do
    GCD=0
    for ((j=i; j<i+4 && j<${#TAGS[@]}; j++)); do
        TAG="${TAGS[$j]}"
        STRESS="${TAG%_*}"
        DIR="${TAG##*_}"

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
            --ready_dir "${DATA_DIR}/ready_${WINDOW}" \
            > "/scratch/project_465002754/kingdom_stress/logs/${TAG}_${WINDOW}_${SLURM_JOB_ID}.log" 2>&1 &

        GCD=$((GCD + 1))
    done

    echo "Waiting for batch to finish..."
    wait
    echo "Batch complete: $(date)"
done

echo "========================================================"
echo "All training complete for window ${WINDOW}: $(date)"
echo "Results: ${RESULTS_DIR}/*_result.json"
echo "========================================================"
