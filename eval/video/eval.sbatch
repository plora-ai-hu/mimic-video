#!/bin/bash
#SBATCH --account=nk_plora
#SBATCH --job-name=v2w_eval
#SBATCH --partition=test
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000
#SBATCH --output=/project/nk_plora/logs/%x-%j.out

# Video backbone rollouts: condition on the first 5 frames of dataset clips, generate 61 frames with each checkpoint
# and compare them against the real clips (eval/video/compare.py). Submit from the repository root:
#   CKPTS="ft=/path/to/iter_000010000_fused.pt" sbatch eval/video/eval.sbatch
#
# Environment variables:
#   CKPTS     required; space-separated name=path pairs (fused DiT checkpoints; paths relative to the repo root work)
#   BASE      1 to also roll out the base Cosmos backbone as the first column "base" (default: 0, ground truth only)
#   DATASET   video dataset with video/ and metas/ (default: SO-101 cabling)
#   SPLIT     val | train | all, clips taken from the dataset in sorted order (default: val)
#   NUM       number of clips (default: 5)
#   EPISODES  explicit clip names or indices, e.g. "0 1 2"; overrides SPLIT/NUM
#   OUT       output directory (default: $SCRATCH_ROOT/mimic/v2w_eval/<job id>)
#   GUIDANCE  classifier-free guidance (default: 7), SEED (default: 0)
# Each rollout takes about 1 min on an A100, plus about 2 min per checkpoint to load the pipeline.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/project/nk_plora}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/nk_plora}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
SIF=${SIF:-$PROJECT_ROOT/mimic-video.sif}
CKPTS=${CKPTS:?set CKPTS to space-separated name=checkpoint pairs}
BASE=${BASE:-0}
BASE_CKPT=model/checkpoints/video_backbone/v2w_pretrained_cosmos.pt
DATASET=${DATASET:-$SCRATCH_ROOT/mimic/so101_cabling/video_dataset}
SPLIT=${SPLIT:-val}
NUM=${NUM:-5}
EPISODES=${EPISODES:-}
OUT=${OUT:-$SCRATCH_ROOT/mimic/v2w_eval/${SLURM_JOB_ID:-local}}
GUIDANCE=${GUIDANCE:-7}
SEED=${SEED:-0}

if [[ ! -f "$REPO/model/pyproject.toml" ]]; then
    echo "REPO=$REPO is not the mimic-video repository root; submit from the repo root or set REPO." >&2
    exit 1
fi
[[ -f "$SIF" ]] || { echo "Container image missing: $SIF (run: sbatch container/build.sbatch)" >&2; exit 1; }

[[ "$BASE" == 1 ]] && CKPTS="base=$BASE_CKPT $CKPTS"
NAMES=()
PATHS=()
for pair in $CKPTS; do
    name=${pair%%=*}
    path=${pair#*=}
    [[ "$name" =~ ^[A-Za-z0-9_-]+$ && "$pair" == *=* ]] || { echo "Bad CKPTS entry: $pair (want name=path)" >&2; exit 1; }
    [[ "$path" == /* ]] || path=$REPO/$path
    [[ -f "$path" ]] || { echo "Checkpoint missing: $path" >&2; exit 1; }
    [[ " ${NAMES[*]} " != *" $name "* ]] || { echo "Duplicate CKPTS name: $name" >&2; exit 1; }
    NAMES+=("$name")
    PATHS+=("$path")
done

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "dataset: $DATASET"
echo "output:  $OUT"
for i in "${!NAMES[@]}"; do echo "model:   ${NAMES[$i]} = ${PATHS[$i]}"; done

module load singularity/4.0
mkdir -p "$OUT"

# Images built with the first cuDNN symlink fix in container/mimic-video.def have libcudnn.so.<version> next to
# libcudnn.so.9, and transformer_engine asserts a single nvidia/cudnn/lib/libcudnn.so.* match. For those, bind a
# job-local copy of the lib dir without that link. Images built from the current def need nothing.
CUDNN_LIB=/opt/mimic-video/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
BINDS=(-B "$PROJECT_ROOT" -B "$SCRATCH_ROOT" -B "$REPO")
if singularity exec "$SIF" sh -c "ls $CUDNN_LIB/libcudnn.so.9.* >/dev/null 2>&1"; then
    FIX_LIB=${TMPDIR:-/tmp}/cudnn_lib_${SLURM_JOB_ID:-$$}
    rm -rf "$FIX_LIB"
    trap 'rm -rf "$FIX_LIB"' EXIT
    singularity exec "$SIF" cp -a "$CUDNN_LIB" "$FIX_LIB"
    rm -f "$FIX_LIB"/libcudnn.so.9.[0-9]*
    BINDS+=(-B "$FIX_LIB:$CUDNN_LIB")
fi

EPISODE_ARGS=(--split "$SPLIT" --num "$NUM")
# shellcheck disable=SC2206
[[ -n "$EPISODES" ]] && EPISODE_ARGS=(--episodes $EPISODES)

cd "$REPO/model"
run() { singularity exec --nv "${BINDS[@]}" --env PYTHONPATH=. --env TOKENIZERS_PARALLELISM=false "$SIF" "$@"; }

run python ../eval/video/prepare.py --dataset "$DATASET" --out "$OUT" --models "${NAMES[@]}" "${EPISODE_ARGS[@]}"
for i in "${!NAMES[@]}"; do
    # Drop rollouts from an earlier job that used the same OUT, so compare.py only sees videos generated now.
    rm -rf "${OUT:?}/${NAMES[$i]}"
    # --save_path only places the config yaml; the batch JSON sets the video paths.
    run python scripts/run_video2world.py --dit_path "${PATHS[$i]}" --batch_input_json "$OUT/batch_${NAMES[$i]}.json" \
        --save_path "$OUT/${NAMES[$i]}/config.mp4" --num_conditional_frames 5 --disable_guardrail \
        --guidance "$GUIDANCE" --seed "$SEED"
done
run python ../eval/video/compare.py --dataset "$DATASET" --out "$OUT" --models "${NAMES[@]}"
echo "results: $OUT"
