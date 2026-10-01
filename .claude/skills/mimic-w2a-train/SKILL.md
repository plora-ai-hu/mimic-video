---
name: mimic-w2a-train
description: Launch, monitor and resume world2action (W2A) action decoder training on top of a frozen, fine-tuned mimic-video backbone on the Slurm cluster. Use when asked to train the action decoder / action head / policy (e.g. for SO-101 cabling), to pick which fused video backbone checkpoint to train it on, to run a W2A smoke test, to resume a W2A run, or to find its WandB link, checkpoints and normalization statistics. Use it even when the user only says "train the policy" or "next step after the video finetune".
---

# Training the action decoder (world2action)

The action decoder is a small DiT that cross-attends to latent features of one layer (default 20) of the frozen video backbone. Video and action use decoupled flow times. Only the decoder is trained, so each step is cheaper than a backbone step. The backbone itself is trained with the `mimic-v2w-train` skill and evaluated with the `mimic-eval` skill.

## 1. Pick the backbone

The decoder learns from the backbone's features, so a weak backbone limits the policy. Before training on a fine-tuned backbone, check it with `mimic-eval`: the tiled layout is kept (`blank` near 0), and the arm moves plausibly in the grids. A backbone that keeps the scene but predicts an almost still arm is not ready yet. Tell the user and suggest a later checkpoint, but let them decide.

- Use the LoRA-fused file `.../video2world/<v2w job name>/checkpoints/model/iter_XXXXXXXXX_fused.pt`. Only the `model/` subdirectory holds backbone weights. The `_fused.pt` files in `optim/`, `scheduler/` and `trainer/` are not backbones.
- List the candidates with `find /project/nk_plora/outputs/mimic-video/posttraining/video2world -path '*/checkpoints/model/*_fused.pt'` and ask the user which one to use.
- If `model.config.video_dit_path` is left out, the decoder trains on the base Cosmos backbone (`model/checkpoints/video_backbone/v2w_pretrained_cosmos.pt`). That is a valid baseline, but it is rarely what the user means after a finetune.

## 2. Preconditions

Check these from the repository root before submitting:

- `secrets.env` exists (mode 600) with `WANDB_API_KEY`. Check with `grep -o '^[A-Z_]*=' secrets.env`, and never print the values.
- The container image `/project/nk_plora/mimic-video.sif` exists, and `model/checkpoints` links to `/project/nk_plora/mimic-video-checkpoints`.
- The episode directory named in the dataset yaml (`model/cosmos_predict2/configs/dataloading/dataset/<dataset>.yaml`, key `data_dir`) holds `episode_*.safetensors` with a `language_embedding` key. For SO-101 this is `/scratch/nk_plora/mimic/so101_cabling/episodes`, built by steps 1 and 3 of `SO101.md`. To check the keys without the container, read the safetensors header:
  ```bash
  python3 -c "import json,struct,sys; b=open(sys.argv[1],'rb'); n=struct.unpack('<Q',b.read(8))[0]; print(sorted(json.loads(b.read(n))))" <episode>.safetensors
  ```
- The video embeddings of the video dataset are not needed. The action dataloader reads the raw frames from the episodes and tiles them itself (`TileViews`, `model/cosmos_predict2/data/action/tiling.py`).
- `squeue -u $USER` shows no other job with the same job name, because two jobs that write to one output directory corrupt the checkpoints.

## 3. Pick the experiment

Experiments are generated in `model/cosmos_predict2/configs/experiment/world2action.py` as

```
w2a_<data config>_<video ckpt name>_lr<lr>_layer<xattn layer>_bsz<global batch>
```

- `<data config>` is a top-level yaml in `configs/dataloading/` (for example `so101_1arm_cabling`). Its prefix selects the decoder net in `configs/defaults/world2action_pipe.py`: `so101_1arm*` uses a 6-in/6-out net, and `so101_2arm*` uses a 12-in/12-out net.
- `<video ckpt name>` only selects the default `video_dit_path`. Keep `v2w_pretrained_cosmos` and override `model.config.video_dit_path=<fused ckpt>`. The experiment name then does not show which backbone was used, so set `job.name` (see below).
- The generated values are `lr1.000e-04`, `layer20` and `bsz` 1, 4, 8, 32, 64, 128 or 256. Other values need an edit of the lists at the top of `world2action.py`. The global batch must be divisible by the GPU count.

## 4. Submit

**Always** set `trainer.max_iter`. The default is 500,000 iterations, and the LR schedule (`lambdalinear`: 1,000 warm-up steps, then linear decay towards 0.2x over 500,000 steps) is sized for that default. `checkpoint.save_iter`, `trainer.validation_iter` and `trainer.logging_iter` default to 1,000.

Put the backbone into the job name, for example `job.name=w2a_so101_1arm_cabling_v2w1arm_iter2041`, so that runs on different backbones do not resume from each other's checkpoints.

**Smoke test** (1 GPU, `test` partition, 20 iterations):

```bash
EXPERIMENT=w2a_so101_1arm_cabling_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1 \
  sbatch --partition=test --gres=gpu:1 --cpus-per-task=16 --time=01:00:00 container/train.sbatch \
  trainer.max_iter=20 trainer.logging_iter=5 checkpoint.save_iter=20 trainer.validation_iter=10 trainer.max_val_iter=2 \
  job.name=smoke_w2a_so101_1arm_cabling \
  model.config.video_dit_path=<fused backbone ckpt>
```

**Full run** (4 GPUs, global batch 4):

```bash
EXPERIMENT=w2a_so101_1arm_cabling_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz4 \
  sbatch --gres=gpu:4 --time=24:00:00 --job-name=w2a_so101_1arm_cabling container/train.sbatch \
  trainer.max_iter=30000 checkpoint.save_iter=2000 trainer.logging_iter=100 \
  trainer.validation_iter=2000 trainer.max_val_iter=16 \
  job.name=<name that includes the backbone> \
  model.config.video_dit_path=<fused backbone ckpt>
```

- A step takes about 1.8 s on one A100 at batch size 1 per GPU, a little more with DDP. 30k steps are about 15 h, so raise `--time` above the 12 h default of `train.sbatch`.
- Each validation pass samples the backbone and takes a few minutes. Keep `validation_iter` sparse and `max_val_iter` small.
- The block-wise activation checkpointing override of the backbone run is not needed here, because the decoder fits on 40 GB without it.
- The number of iterations above is a starting point, not a tuned value. Ask the user if they have a budget, and judge the run by the val loss curve.

## 5. Monitor and share the WandB link

The log is `/project/nk_plora/logs/<slurm job name>-<job id>.out`. On the first run on a dataset, the log shows `Iterating dataset to get normalization` before training starts. Later runs read the cached statistics.

The WandB URL appears in a line `wandb: 🚀 View run at https://wandb.ai/...` after the model has loaded. Wait for it with a Monitor, then give the URL to the user:

```bash
until grep -m1 -o 'View run at https://[^ ]*' LOG 2>/dev/null || grep -m1 -E 'Traceback|Error' LOG 2>/dev/null \
  || sacct -j JOBID -X -n -o State | grep -qE 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|COMPLETED'; do sleep 30; done
```

A pending job has no log yet. `squeue -j JOBID -o '%T %r %S'` shows the reason and the estimated start time. The WandB project is `mimic_video_w2a`.

## 6. Outputs and resuming

- Output directory: `/project/nk_plora/outputs/mimic-video/vam/<decoder net>/<job name>/`, for example `vam/so101_1arm/...`. The path is `<job.project>/<job.group>/<job.name>`: `job.project` is `vam`, and `job.group` is the decoder net. This is not the `posttraining/` tree of the backbone runs.
- Checkpoints: `checkpoints/model/iter_XXXXXXXXX.pt`. They hold only the decoder weights (`net.*`), not the backbone, and they need no fusing. Inference must load the same backbone through `video_dit_path`, and the same normalization statistics.
- Normalization statistics: `<data_dir>/.statistics_cache/<hash>`, a JSON file that is computed on the first run and reused afterwards. The policy server (`mimic-policy-serve` skill) finds it on its own when it is the only file there, else it needs `--stats <path>`. The hash changes when the data components or transforms change.
- Resuming: resubmit the exact same command (same `job.name`). The checkpointer reads `checkpoints/latest_checkpoint.txt` and restores the model, the optimizer and the iteration.

## 7. Judging the result

No SO-101 simulator exists. Judge SO-101 decoders by the action val loss in WandB (and its trend across checkpoints), and then on the real robot. For Bridge and LIBERO decoders, run the simulator evals of the `mimic-eval` skill.

When you report back, give the user the job id, the WandB link, the output directory, the backbone that was used and the statistics path. They need these values for deployment.

## Known issues

- `ptrDesc->finalize()` or `Found 2 libcudnn.so.x` comes from an old image. `train.sbatch` works around the first error. Rebuild the image to fix both.
- `wandb: WARNING Tried to log to step N that is less than the current step` is harmless.
