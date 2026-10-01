---
name: mimic-eval
description: Run and interpret mimic-video evaluations on the Slurm cluster. Use when asked to test or evaluate a video backbone checkpoint (did the finetune work, compare checkpoints, generate rollouts or videos from a checkpoint), or to run the SIMPLER-Bridge / LIBERO simulator evals of an action decoder.
---

# Running mimic-video evaluations

Two kinds of evaluation exist:

1. **Video backbone rollouts** (`eval/video/`): the main check for a video2world finetune. Every checkpoint gets the first 5 frames of real dataset clips, generates 61 frames, and is compared against the real clip (the ground truth). Works for any video dataset (`video/*.mp4` + `metas/*.txt`), including the tiled SO-101 data.
2. **Simulator evals of the full policy** (`eval/bridge/`, `eval/libero/`): the success rate of video backbone + action decoder in SIMPLER-Bridge or LIBERO. See the "Evaluation" section of `README.md`.

## Video backbone rollouts

### 1. Find the checkpoints

- The inference pipeline loads full DiT weights only. For a LoRA run, use `iter_XXXXXXXXX_fused.pt`, never the raw `iter_XXXXXXXXX.pt`. Training writes the fused copy when it saves a checkpoint. If it is missing, create it with `python model/scripts/fuse_lora_ckpt.py <iter_XXXXXXXXX.pt>` inside the container.
- Video2world runs are under `/project/nk_plora/outputs/mimic-video/posttraining/video2world/<job name>/checkpoints/model/`.
- Evaluate only the finetuned checkpoints against the ground truth by default. Add the base Cosmos backbone (`model/checkpoints/video_backbone/v2w_pretrained_cosmos.pt`) only when the user asks for it, with `BASE=1`. Never reuse base rollouts from an earlier job.
- List the candidates with `find /project/nk_plora/outputs/mimic-video -name "*_fused.pt"` before asking the user which one to use.

### 2. Submit

Submit from the repository root. `CKPTS` holds space-separated `name=path` pairs, and each name becomes a column in the results:

```bash
CKPTS="ft=/project/nk_plora/outputs/.../iter_000010000_fused.pt" sbatch eval/video/eval.sbatch
```

The options are environment variables, documented at the top of `eval/video/eval.sbatch`:
- `BASE=1`: also roll out the base backbone, as the first column `base`. Off by default.
- `DATASET`: default is the SO-101 cabling video dataset.
- `SPLIT` and `NUM`: default `val` and `5`. The val clips use the same md5 split as training (`val_ratio=0.05`), so they are held out. Prefer them over `EPISODES`, because the model may have trained on other clips.
- `EPISODES="0 1 2"` or clip names: pick explicit clips.
- `OUT`: default `/scratch/nk_plora/mimic/v2w_eval/<job id>`. When an existing `OUT` is reused, the rollouts of every model in this job are deleted and regenerated first, so results never mix in videos from an earlier job.
- `GUIDANCE`, `SEED`.

The job runs on the `test` partition (1 GPU, 1 h limit). Each rollout takes about 1 min, and each checkpoint takes about 2 min to load. Budget accordingly: 1 checkpoint x 5 clips is about 8 min, and 2 checkpoints x 5 clips about 15 min. For more clips, pass `sbatch --partition=gpu --time=...`.

Watch the job with a Monitor that polls `sacct -j <id> -X -n -o State`. Count `Successfully saved` lines in `/project/nk_plora/logs/v2w_eval-<id>.out` to track progress. The filter must also match the terminal states `FAILED`, `CANCELLED`, `TIMEOUT` and `OUT_OF_MEMORY`, and `Traceback` in the log.

### 3. Read the results

`compare.py` prints a table and writes the following files to `$OUT`:
- `metrics.csv`: per clip, `psnr_<name>` and `blank_<name>`, over the predicted frames 5 to 60 only.
- `grid_<episode>.jpg`: frames 0, 4, 20 and 60 as rows. The columns are ground truth first, then `base` if `BASE=1`, then each checkpoint in `CKPTS` order. **Always look at a few of these with the Read tool**, because the numbers alone are misleading.
- `side_by_side/<episode>.mp4`: the full videos in the same column order, for the user to watch.

How to interpret the results:
- `psnr` against the ground truth is weak: valid futures diverge from the recorded one. Use it only to compare checkpoints with each other, or a checkpoint with the base model when `BASE=1`. About 9 dB means the prediction has little to do with the scene.
- `blank` is the mean pixel value (0-255) of the tile cells that are black in the ground truth. For single-arm SO-101 data these are the top-right and bottom-left cells. A model that has learned the tiled layout keeps them near 0. The base Cosmos model fills them with unrelated scenes, at about 80-90, so `blank` judges a single finetune on its own, without a base run.
- Frames 0 to 4 are the model's copy of the conditioning frames and should be about 40 dB against the ground truth. If they are much lower, the input or alignment is broken. Suspect the eval, not the model.
- In the grids, a good finetune keeps the 2x2 grid, the static background and consistent views, and it moves the arm plausibly for the task. Uniform noise or texture after a few frames means the checkpoint is broken (seen with a 20-iteration smoke run at a constant LR of 1.778e-4 without warmup).

Report the table, what the grids show, and the output paths to the user.

### Known issues

- `Found 2 libcudnn.so.x in nvidia-cudnn-cuXX` on `import transformer_engine` means the image was built from an old `container/mimic-video.def`, one that also linked the base `libcudnn.so.9`. `eval.sbatch` works around this, but training does not. Rebuild with `sbatch container/build.sbatch`.
- imageio's `pyav` plugin fails with the `av` version in the image, on both write and read. Use `imageio.v2.mimread` / `mimwrite`, which use the ffmpeg backend.
- `run_video2world.py` writes a config yaml next to `--save_path`. The default for that path is inside the checkout (`model/output/`), so always pass `--save_path` explicitly.

## Simulator evals (Bridge, LIBERO)

These evals run a full policy (video backbone + action decoder + dataset statistics) and report task success. Setup and the checkpoint downloads are in `README.md`, section "Evaluation".

- The scripts are `eval/bridge/eval.sh` (vanilla), `eval/bridge/eval_hil.sh` (human-in-the-loop: ground-truth future video, an oracle study) and `eval/libero/eval.sh`.
- Before running, edit `GPUS` and `checkpoint_dir` at the top of each script. Each script declares one associative array per model (`action_model`, `video_model`, `stats`, and `suite` for LIBERO) and runs 2 evals per GPU in parallel.
- They are plain bash scripts, not Slurm jobs. Run them inside an `salloc`/`srun` GPU allocation, in the container (`singularity exec --nv`), never on the login node. LIBERO needs `PYTHONPATH=LIBERO`, which `eval/libero/eval.sh` sets.
- No SO-101 simulator exists. For SO-101 action decoders, use the action val loss of the training run, or test on the real robot.
