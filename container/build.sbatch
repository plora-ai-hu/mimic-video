#!/bin/bash
#SBATCH --account=nk_plora
#SBATCH --job-name=build_mimic_video
#SBATCH --partition=test
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000
#SBATCH --output=/project/nk_plora/logs/%x-%j.out

# Build the container on a compute node, from the repository root:
#   sbatch container/build.sbatch
#
# Blackwell (cu129) variant. It compiles apex and transformer-engine from source,
# so it needs more than the 1h limit of the test partition:
#   BUILD_ARGS='--build-arg CUDA_IMAGE=12.9.1-cudnn-devel-ubuntu22.04 --build-arg CUDA_EXTRA=cu129 --build-arg TORCH_CUDA_ARCH_LIST=10.0;12.0' \
#       SIF=/project/nk_plora/mimic-video-cu129.sif sbatch --partition=cpu --time=04:00:00 container/build.sbatch
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/project/nk_plora}
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}
DEF=${DEF:-$REPO/container/mimic-video.def}
SIF=${SIF:-$PROJECT_ROOT/mimic-video.sif}
BUILD_ARGS=${BUILD_ARGS:-}

if [[ ! -f "$REPO/model/pyproject.toml" ]]; then
    echo "REPO=$REPO is not the mimic-video repository root; submit from the repo root or set REPO." >&2
    exit 1
fi

module load singularity/4.0

# --fakeroot builds inside a user namespace that maps only your PRIMARY uid/gid;
# supplementary groups are dropped. Access to /project/nk_plora and
# /scratch/nk_plora comes from the nk_plora *supplementary* group, so both are
# unreadable inside the build. Stage inputs and scratch on node-local /tmp
# (world-writable), then copy the result back afterwards as your normal self.
BUILD_DIR=${SLURM_TMPDIR:-/tmp}/mimic-video-build-${SLURM_JOB_ID:-$$}
mkdir -p "$BUILD_DIR"
trap 'rm -rf "$BUILD_DIR"' EXIT

export SINGULARITY_TMPDIR="$BUILD_DIR/tmp"
export SINGULARITY_CACHEDIR="$BUILD_DIR/cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

cp "$DEF" "$BUILD_DIR/build.def"

# Stage the %files inputs. Their paths in the definition file are relative to
# the build directory, which mirrors the repository layout.
CONTEXT="$BUILD_DIR/context"
mkdir -p "$CONTEXT/model" "$CONTEXT/eval/bridge" "$CONTEXT/eval/libero/LIBERO"
cp "$REPO/model/pyproject.toml" "$CONTEXT/model/"
cp -r "$REPO/eval/bridge/SimplerEnv" "$CONTEXT/eval/bridge/"
cp "$REPO/eval/libero/LIBERO/requirements.txt" "$CONTEXT/eval/libero/LIBERO/"
chmod -R a+rX "$CONTEXT"

echo "Building $DEF -> $SIF"
echo "build args: ${BUILD_ARGS:-<none>}"
echo "staging in $BUILD_DIR on $(hostname)"
df -h "$BUILD_DIR" | tail -1

cd "$CONTEXT"
# shellcheck disable=SC2086
singularity build --fakeroot --fix-perms $BUILD_ARGS "$BUILD_DIR/out.sif" "$BUILD_DIR/build.def"

# Outside the namespace again, so nk_plora group access works normally.
# Keep one generation of rollback, and move into place atomically so a reader
# never sees a partially-copied image.
if [[ -f "$SIF" ]]; then
    mv -f "$SIF" "$SIF.prev"
fi
cp "$BUILD_DIR/out.sif" "$SIF.incoming"
mv -f "$SIF.incoming" "$SIF"

echo "Built: $SIF"
ls -lh "$SIF"
