#!/bin/bash
#SBATCH --account=nk_plora
#SBATCH --job-name=so101_policy_server
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8000
#SBATCH --output=/project/nk_plora/logs/%x-%j.out

# Serve an SO-101 action decoder to so100/so101_robot_client.py. Submit from the repository root:
#   EXPERIMENT=w2a_so101_1arm_cabling_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz4 \
#   VIDEO_DIT=/path/to/iter_XXXXXXXXX_fused.pt DECODER=/path/to/decoder/iter_XXXXXXXXX.pt \
#     sbatch so100/so101_serve.sbatch [server option ...]
# Extra arguments go to so100/so101_policy_server.py (e.g. --stop-step 5 --debug-dir /scratch/...).
# The server binds 127.0.0.1:$PORT on the compute node; the log prints the SSH tunnel command for the laptop.
# MIMIC_POLICY_TOKEN comes from secrets.env at the repository root and is never printed.
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/project/nk_plora}
SCRATCH_ROOT=${SCRATCH_ROOT:-/scratch/nk_plora}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
SIF=${SIF:-$PROJECT_ROOT/mimic-video.sif}
EXPERIMENT=${EXPERIMENT:?set EXPERIMENT to the decoder experiment name}
VIDEO_DIT=${VIDEO_DIT:?set VIDEO_DIT to the fused video backbone checkpoint}
DECODER=${DECODER:?set DECODER to the action decoder checkpoint}
PORT=${PORT:-8766}

if [[ ! -f "$REPO/model/pyproject.toml" ]]; then
    echo "REPO=$REPO is not the mimic-video repository root; submit from the repo root or set REPO." >&2
    exit 1
fi
[[ -f "$SIF" ]] || { echo "Container image missing: $SIF (run: sbatch container/build.sbatch)" >&2; exit 1; }
for path in "$VIDEO_DIT" "$DECODER"; do
    [[ -f "$path" ]] || { echo "Checkpoint missing: $path" >&2; exit 1; }
done

# shellcheck disable=SC1091
[[ -f "$REPO/secrets.env" ]] && source "$REPO/secrets.env"
[[ -n "${MIMIC_POLICY_TOKEN:-}" ]] || { echo "Set MIMIC_POLICY_TOKEN in $REPO/secrets.env" >&2; exit 1; }
export MIMIC_POLICY_TOKEN

echo "=== job ${SLURM_JOB_ID:-local} on $(hostname) ==="
echo "experiment: $EXPERIMENT"
echo "backbone:   $VIDEO_DIT"
echo "decoder:    $DECODER"
echo "tunnel:     ssh -N -L $PORT:localhost:$PORT -J <user>@<login node> $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

module load singularity/4.0

# cuDNN workarounds for older images, see container/train.sbatch and eval/video/eval.sbatch.
BINDS=(-B "$PROJECT_ROOT" -B "$SCRATCH_ROOT" -B "$REPO")
CUDNN_LIB=/opt/mimic-video/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib
if singularity exec "$SIF" sh -c "ls $CUDNN_LIB/libcudnn.so.9.* >/dev/null 2>&1"; then
    FIX_LIB=${TMPDIR:-/tmp}/cudnn_lib_${SLURM_JOB_ID:-$$}
    rm -rf "$FIX_LIB"
    trap 'rm -rf "$FIX_LIB"' EXIT
    singularity exec "$SIF" cp -a "$CUDNN_LIB" "$FIX_LIB"
    rm -f "$FIX_LIB"/libcudnn.so.9.[0-9]*
    BINDS+=(-B "$FIX_LIB:$CUDNN_LIB")
fi
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
singularity exec --nv "${BINDS[@]}" \
    --env PYTHONPATH=. \
    --env TOKENIZERS_PARALLELISM=false \
    "$SIF" \
    bash -c "$CUDNN_COMPAT"' exec python ../so100/so101_policy_server.py "$@"' \
    _ --experiment "$EXPERIMENT" --video-dit "$VIDEO_DIT" --decoder "$DECODER" --port "$PORT" "$@"
