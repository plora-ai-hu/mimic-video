#!/bin/bash
# Keep an action decoder (W2A) training run going on a partition with a short time limit (e.g. `test`, 1 hour) by
# submitting one job after another, each resuming from the previous one's last checkpoint. The W2A counterpart of
# container/v2w_autoresume.sh. Run it on the login node from the repository root, and keep it alive with nohup:
#
#   nohup container/w2a_autoresume.sh > /project/nk_plora/logs/w2a_autoresume.log 2>&1 &
#
# Each job runs container/train.sbatch with trainer.max_wall_time_min=WALL_MIN: the trainer stops itself after that
# many minutes, saves a checkpoint and exits before the Slurm limit, so no steps are lost between jobs.
#
# Every INTERVAL seconds, the watcher checks squeue for the run's job:
#   - a job pending or running -> nothing to do
#   - none                     -> prune old optimizer state, then submit the next job
# It stops when the latest checkpoint reaches MAX_ITER, or after MAX_STRIKES submissions in a row that made no
# checkpoint progress (crash loop).
#
# Settings are environment variables (defaults below); extra arguments are passed to scripts.train as overrides.
# A new RUN_NAME starts a new run on the VIDEO_DIT_PATH backbone; an existing one resumes.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-/project/nk_plora}
OUTPUT_ROOT=${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/mimic-video}

EXPERIMENT=${EXPERIMENT:-w2a_so101_2arm_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1}
# Decoder net of the experiment (job.group), part of the output path.
DECODER_NET=${DECODER_NET:-so101_2arm}
RUN_NAME=${RUN_NAME:-w2a_so101_2arm_v2w2arm_iter13000}
# Frozen backbone: a LoRA-fused checkpoint of the two-arm cabling V2W run.
VIDEO_DIT_PATH=${VIDEO_DIT_PATH:-$OUTPUT_ROOT/posttraining/video2world/v2w_so101_2arm_from_1arm10k_bsz1/checkpoints/model/iter_000013000_fused.pt}
MAX_ITER=${MAX_ITER:-30000}
SAVE_ITER=${SAVE_ITER:-2000}
# Number of complete checkpoints kept for resuming; older optimizer/scheduler/trainer states are deleted before each
# submission. Decoder weights (model/) are kept at every multiple of SAVE_ITER.
KEEP=${KEEP:-2}

PARTITION=${PARTITION:-test}
TIME=${TIME:-01:00:00}
# Counted from trainer start; leaves room in the hour for startup (backbone load, normalization statistics on the first
# run), the final save and shutdown.
WALL_MIN=${WALL_MIN:-45}
GPUS=${GPUS:-1}
CPUS=${CPUS:-16}
MEM_PER_CPU=${MEM_PER_CPU:-4000}

INTERVAL=${INTERVAL:-120}
MAX_STRIKES=${MAX_STRIKES:-2}

RUN_DIR=$OUTPUT_ROOT/vam/$DECODER_NET/$RUN_NAME
CKPT_DIR=$RUN_DIR/checkpoints
EXTRA_OVERRIDES=("$@")

log() { echo "[$(date '+%F %T')] $*"; }

# Iteration of latest_checkpoint.txt, 0 if there is none.
ckpt_iter() {
    local file
    file=$(cat "$CKPT_DIR/latest_checkpoint.txt" 2>/dev/null) || { echo 0; return; }
    [[ "$file" =~ ^iter_([0-9]+)\.pt$ ]] && echo $((10#${BASH_REMATCH[1]})) || echo 0
}

# Delete all but the KEEP newest complete checkpoints, never the one latest_checkpoint.txt names. Decoder weights at
# multiples of SAVE_ITER are kept. Only called while no job of the run is running.
prune() {
    [[ -d "$CKPT_DIR/trainer" ]] || return 0
    local latest keep=() f it n dir
    latest=$(cat "$CKPT_DIR/latest_checkpoint.txt" 2>/dev/null)
    mapfile -t keep < <(cd "$CKPT_DIR/trainer" && ls iter_[0-9]*.pt 2>/dev/null | sort | tail -n "$KEEP")
    keep+=("$latest")
    for dir in model optim scheduler trainer; do
        for f in "$CKPT_DIR/$dir"/iter_*.pt "$CKPT_DIR/$dir"/*.tmp; do
            [[ -e "$f" ]] || continue
            it=$(basename "$f")
            if [[ "$dir" == model && "$it" =~ ^iter_([0-9]+)\.pt$ ]]; then
                n=$((10#${BASH_REMATCH[1]}))
                [[ $((n % SAVE_ITER)) -eq 0 ]] && continue
            fi
            [[ "$it" != *.tmp ]] && printf '%s\n' "${keep[@]}" | grep -qxF "$it" && continue
            log "pruning $f"
            rm -f "$f"
        done
    done
}

submit() {
    local it
    it=$(ckpt_iter)
    if [[ -n "$LAST_SUBMIT_ITER" && "$it" -le "$LAST_SUBMIT_ITER" ]]; then
        STRIKES=$((STRIKES + 1))
        log "no checkpoint progress since last submit (still iteration $it), strike $STRIKES/$MAX_STRIKES"
        if [[ $STRIKES -ge $MAX_STRIKES ]]; then
            log "giving up - check $PROJECT_ROOT/logs/$RUN_NAME-*.out"
            exit 1
        fi
    else
        STRIKES=0
    fi
    LAST_SUBMIT_ITER=$it
    prune
    log "submitting $RUN_NAME (checkpoint iteration $it)"
    (cd "$REPO" && EXPERIMENT=$EXPERIMENT WANDB_RUN_ID=$WANDB_RUN_ID WANDB_RESUME=allow sbatch --parsable \
        --job-name="$RUN_NAME" --partition="$PARTITION" --time="$TIME" --gres=gpu:"$GPUS" \
        --cpus-per-task="$CPUS" --mem-per-cpu="$MEM_PER_CPU" \
        container/train.sbatch \
        job.name="$RUN_NAME" \
        model.config.video_dit_path="$VIDEO_DIT_PATH" \
        trainer.max_iter="$MAX_ITER" \
        trainer.max_wall_time_min="$WALL_MIN" \
        checkpoint.save_iter="$SAVE_ITER" \
        trainer.epoch_checkpoint_throttling_min_period_minutes=1000000 \
        trainer.logging_iter=100 \
        trainer.validation_iter="$SAVE_ITER" \
        trainer.max_val_iter=1 \
        trainer.run_validation_at_start=False \
        dataloader_train.num_workers=6 \
        dataloader_train.prefetch_factor=2 \
        "${EXTRA_OVERRIDES[@]}") \
        | sed "s/^/submitted job /" || log "sbatch failed (exit $?), will retry next check"
}

[[ -f "$VIDEO_DIT_PATH" ]] || { echo "VIDEO_DIT_PATH not found: $VIDEO_DIT_PATH" >&2; exit 1; }
[[ "$MAX_ITER" =~ ^[0-9]+$ ]] || { echo "MAX_ITER must be an integer, got: $MAX_ITER" >&2; exit 1; }
[[ "$SAVE_ITER" =~ ^[1-9][0-9]*$ ]] || { echo "SAVE_ITER must be a positive integer, got: $SAVE_ITER" >&2; exit 1; }

# One WandB run for the whole chain of jobs; the id is kept next to the checkpoints.
mkdir -p "$RUN_DIR"
[[ -s "$RUN_DIR/wandb_run_id" ]] || tr -dc 'a-z0-9' < /dev/urandom | head -c 8 > "$RUN_DIR/wandb_run_id"
WANDB_RUN_ID=$(cat "$RUN_DIR/wandb_run_id")

LAST_SUBMIT_ITER=""
STRIKES=0
log "watching $RUN_NAME: every ${INTERVAL}s, $PARTITION partition, ${WALL_MIN} min per job, target iteration $MAX_ITER"
log "backbone: $VIDEO_DIT_PATH"
log "checkpoints: $CKPT_DIR, wandb run id $WANDB_RUN_ID"

while true; do
    it=$(ckpt_iter)
    if [[ "$it" -ge "$MAX_ITER" ]]; then
        log "final checkpoint at iteration $it reached - done"
        exit 0
    fi

    if ! states=$(squeue -h -u "$USER" -n "$RUN_NAME" -o '%T'); then
        log "squeue failed, retrying next check"
        sleep "$INTERVAL"; continue
    fi

    if [[ -n "$states" ]]; then
        log "job $(echo "$states" | tr '\n' ' ')- checkpoint iteration $it"
    else
        submit
    fi

    sleep "$INTERVAL"
done
