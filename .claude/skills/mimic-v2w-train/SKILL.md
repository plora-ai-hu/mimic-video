---
name: mimic-v2w-train
description: Launch, monitor and resume video2world (V2W) video backbone fine-tuning of Cosmos-Predict2 2B on the Slurm cluster. Use when asked to fine-tune / train / post-train the video backbone on a video dataset (e.g. SO-101 cabling), run a V2W smoke test, resume a V2W run, or find its WandB link and checkpoints.
---

# Fine-tuning the video backbone (video2world)

The video backbone is a LoRA (rank 256) finetune of Cosmos-Predict2 2B. The action decoder is trained afterwards on top of the frozen, fused backbone with the `mimic-w2a-train` skill. For evaluating a finished or intermediate checkpoint, use the `mimic-eval` skill.

## 1. Preconditions

Check these before submitting, from the repository root:

- `secrets.env` exists (mode 600) with `WANDB_API_KEY`. Check with `grep -o '^[A-Z_]*=' secrets.env`; never print the values.
- `model/checkpoints` is a symlink to `/project/nk_plora/mimic-video-checkpoints` and contains `video_backbone/v2w_pretrained_cosmos.pt`, the T5 encoder and the VAE tokenizer.
- The container image `/project/nk_plora/mimic-video.sif` exists (otherwise `sbatch container/build.sbatch`).
- The dataset directory from `model/cosmos_predict2/configs/defaults/data_video.py` exists and has `video/*.mp4`, `metas/*.txt` and `language_embeddings/`. For SO-101, build it as described in `SO101.md`.
- `squeue -u $USER` shows no other job with the same job name. Two jobs writing to one output directory corrupt the checkpoints.

## 2. Pick the experiment

Experiments are generated in `model/cosmos_predict2/configs/experiment/video2world.py` as `v2w_<dataset>_lora_rank256_lr1.778e-04_bsz<global batch>`, for every dataset in `train_datasets` and every batch size in `bszs`. The global batch must be divisible by the GPU count: batch size 1 per GPU is what fits on an A100-40GB, so use `bsz<N>` with `--gres=gpu:<N>`.

A new dataset needs only a new entry in `train_datasets` (`data_video.py`); the experiment names follow automatically.

## 3. Submit

Always pass `model.config.pipe_config.net.sac_config.mode=block_wise`; the default activation checkpointing runs out of memory on 40 GB at 61x480x640.

**Smoke test** (1 GPU, `test` partition, about 5 min of training):

```bash
EXPERIMENT=v2w_so101_1arm_cabling_lora_rank256_lr1.778e-04_bsz1 sbatch --partition=test --gres=gpu:1 --cpus-per-task=16 --time=01:00:00 \
  container/train.sbatch trainer.max_iter=20 trainer.logging_iter=5 checkpoint.save_iter=20 trainer.validation_iter=10 trainer.max_val_iter=2 \
  job.name=smoke_v2w_so101_1arm_cabling model.config.pipe_config.net.sac_config.mode=block_wise
```

**Full run** (4 GPUs, global batch 4, 10k iterations):

```bash
EXPERIMENT=v2w_so101_1arm_cabling_lora_rank256_lr1.778e-04_bsz4 \
  sbatch --gres=gpu:4 --time=20:00:00 --job-name=v2w_so101_1arm_cabling container/train.sbatch \
  trainer.max_iter=10000 checkpoint.save_iter=1000 trainer.logging_iter=50 \
  trainer.validation_iter=500 trainer.max_val_iter=16 \
  model.config.pipe_config.net.sac_config.mode=block_wise
```

- A step takes about 4.8 s on one A100 at batch size 1 per GPU, a little more with DDP. 10k steps are about 14 h, so raise `--time` above the 12 h default of `train.sbatch` (the `gpu` partition allows up to 7 days).
- `trainer.validation_iter` and `checkpoint.save_iter` default to never. Without them there is no val loss and no checkpoint.
- `trainer.logging_iter` defaults to 1000 in the experiment; lower it to see loss curves early.
- The LR schedule is constant (1.778e-4) without warmup. Short runs at this LR produce noise; judge the run from the 1000-iteration checkpoint onwards.
- The job name (and therefore the output directory and WandB run name) defaults to the experiment name. Override it with `job.name=<name>` for variants, so that they do not resume from each other's checkpoints.

## 4. Monitor and share the WandB link

The log is `/project/nk_plora/logs/<slurm job name>-<job id>.out`. The WandB run URL appears once the model has loaded (a few minutes after start), in a line `wandb: 🚀 View run at https://wandb.ai/...`. Wait for it with a Monitor, then give the URL to the user:

```bash
until grep -m1 -o 'View run at https://[^ ]*' LOG 2>/dev/null || grep -m1 -E 'Traceback|Error' LOG 2>/dev/null \
  || sacct -j JOBID -X -n -o State | grep -qE 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|COMPLETED'; do sleep 30; done
```

A pending job has no log yet; `squeue -j JOBID -o '%T %r %S'` shows the reason and the estimated start time.

Healthy signs in the log: `Iteration N: ... Loss:` lines with a loss of a few units (single-step spikes to about 20 are normal early on), `iter_speed` around 5 s, and `Saved checkpoint` every `save_iter`. The WandB project is `mimic_video_v2w`; the entity comes from `WANDB_ENTITY` or the API key's default.

## 5. Outputs and resuming

- Checkpoints: `/project/nk_plora/outputs/mimic-video/posttraining/video2world/<job name>/checkpoints/model/iter_XXXXXXXXX.pt`, each with a LoRA-fused copy `iter_XXXXXXXXX_fused.pt` (written by `VideoEvalCallback`). The fused copy is what `mimic-eval` and the action decoder's `model.config.video_dit_path` take.
- Resuming: resubmit the exact same command (same job name). The checkpointer reads `checkpoints/latest_checkpoint.txt` and restores model, optimizer and iteration. Do this after a `TIMEOUT` or a node failure.
- Once the first checkpoints exist, evaluate them against the base model with the `mimic-eval` skill, rather than trusting the train loss.

## Known issues

- The smoke job may keep running for several minutes after the final `Saved checkpoint` line while it writes the fused copy and shuts down; check that `iter_XXXXXXXXX_fused.pt` exists before cancelling it.
- `ptrDesc->finalize()` or `Found 2 libcudnn.so.x` comes from an old image; `train.sbatch` works around the first. Rebuild the image to fix both.
- `wandb: WARNING Tried to log to step N that is less than the current step` is harmless.
