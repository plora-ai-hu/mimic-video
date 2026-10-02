#!/bin/bash
#SBATCH --account=nk_plora
#SBATCH --job-name=mimic_video_train
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=4000
#SBATCH --output=/project/nk_plora/logs/%x-%j.out

# Train a mimic-video experiment (video backbone or action decoder) in the container, from the repository root:
#   EXPERIMENT=v2w_so101_1arm_cabling_lora_rank256_lr1.778e-04_bsz4 sbatch container/train.sbatch [override ...]
# Extra arguments are passed to scripts.train as config overrides, e.g. trainer.max_iter=1000.
# Use more GPUs with `sbatch --gres=gpu:4 ...`; GPUS defaults to the allocated count.
#
# Weights & Biases: put WANDB_API_KEY (and optionally WANDB_ENTITY / WANDB_PROJECT / WANDB_MODE) into secrets.env
# at the repository root (cp secrets.env.example secrets.env && chmod 600 secrets.env). It is sourced here and
# passed into the container; values are never printed. WANDB_RUN_ID / WANDB_RESUME from the submitting environment
# continue one WandB run across resumed jobs (see container/v2w_autoresume.sh).
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/project/nk_plora}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/nk_plora}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
SIF=${SIF:-$PROJECT_ROOT/mimic-video.sif}
EXPERIMENT=${EXPERIMENT:?set EXPERIMENT to the experiment name}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/mimic-video}
GPUS=${GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}

if [[ ! -f "$REPO/model/pyproject.toml" ]]; then
    echo "REPO=$REPO is not the mimic-video repository root; submit from the repo root or set REPO." >&2
    exit 1
fi
[[ -f "$SIF" ]] || { echo "Container image missing: $SIF (run: sbatch container/build.sbatch)" >&2; exit 1; }
[[ "$GPUS" =~ ^[1-9][0-9]*$ ]] || { echo "GPUS must be a positive integer, got: $GPUS" >&2; exit 1; }

# Optional secrets, kept out of the tracked scripts.
# shellcheck disable=SC1091
[[ -f "$REPO/secrets.env" ]] && source "$REPO/secrets.env"
# Singularity forwards the host environment, and wandb rejects empty values (e.g. WANDB_MODE=""), so drop unset ones.
for var in WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_RUN_ID WANDB_RESUME; do
    [[ -n "${!var:-}" ]] || unset "$var"
done

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "experiment: $EXPERIMENT"
echo "overrides:  $*"
echo "output:     $OUTPUT_ROOT"
echo "wandb:      key $([[ -n "${WANDB_API_KEY:-}" ]] && echo set || echo unset), mode ${WANDB_MODE:-online}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

module load singularity/4.0
mkdir -p "$OUTPUT_ROOT" "$PROJECT_ROOT/logs"

# Images built before container/mimic-video.def added the cuDNN full-version symlinks load two copies of cuDNN
# (see the comment there) and every convolution fails with "ptrDesc->finalize()". Provide the symlinks in a job-local
# directory on LD_LIBRARY_PATH instead; this is a no-op for newer images.
COMPAT_DIR=${TMPDIR:-/tmp}/mimic_cudnn_compat_${SLURM_JOB_ID:-$$}
CUDNN_COMPAT='
CUDNN_LIB=$(python -c "import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], \"lib\"))")
CUDNN_VER=$(python -c "from importlib.metadata import version; print(\".\".join(version(\"nvidia-cudnn-cu12\").split(\".\")[:3]))")
if [[ ! -e "$CUDNN_LIB/libcudnn_graph.so.$CUDNN_VER" ]]; then
    mkdir -p '"$COMPAT_DIR"'
    for lib in "$CUDNN_LIB"/libcudnn*.so.9; do ln -sfn "$lib" '"$COMPAT_DIR"'/"$(basename "${lib%.so.9}").so.$CUDNN_VER"; done
    export LD_LIBRARY_PATH='"$COMPAT_DIR"':$LD_LIBRARY_PATH
fi
'

cd "$REPO/model"
singularity exec --nv \
    -B "$PROJECT_ROOT" \
    -B "$SCRATCH_ROOT" \
    -B "$REPO" \
    --env IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" \
    --env WANDB_DIR="${WANDB_DIR:-$OUTPUT_ROOT}" \
    --env PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    ${WANDB_API_KEY:+--env WANDB_API_KEY="$WANDB_API_KEY"} \
    ${WANDB_ENTITY:+--env WANDB_ENTITY="$WANDB_ENTITY"} \
    ${WANDB_PROJECT:+--env WANDB_PROJECT="$WANDB_PROJECT"} \
    ${WANDB_MODE:+--env WANDB_MODE="$WANDB_MODE"} \
    ${WANDB_RUN_ID:+--env WANDB_RUN_ID="$WANDB_RUN_ID"} \
    ${WANDB_RESUME:+--env WANDB_RESUME="$WANDB_RESUME"} \
    "$SIF" \
    bash -c "$CUDNN_COMPAT"' exec torchrun --standalone --nproc_per_node="$0" -m scripts.train \
        --config=cosmos_predict2/configs/config.py -- experiment="$1" "${@:2}"' \
    "$GPUS" "$EXPERIMENT" "$@"
