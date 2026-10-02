---
name: mimic-preprocess
description: Convert a recorded LeRobot v3.0 dataset (SO-100 / SO-101, one or two arms, several cameras) into the mimic-video formats - safetensors episodes for the action decoder and a tiled video dataset with T5 embeddings for the video backbone. Use when asked to preprocess, convert or import a dataset / recording / dataroot for mimic-video, or before V2W or W2A training on new robot data.
---

# Preprocessing a LeRobot dataset for mimic-video

The pipeline is described in `SO101.md` ("Pipeline"). It has three steps, all run from the repository root on compute nodes, never on the login node:

1. LeRobot v3.0 -> `episodes/episode_XXXXXX.safetensors` with `so100/process_so100_v3.py` (lerobot image, CPU).
2. Episodes -> tiled `video_dataset/{video,metas}` with `data_preprocessing/video/process_so101_video.py` (mimic-video image, CPU).
3. T5 embeddings with `data_preprocessing/action/precompute_t5.py` (adds `language_embedding` to the episodes) and `data_preprocessing/video/precompute_t5_embeddings.py` (writes `video_dataset/language_embeddings/`) (mimic-video image, 1 GPU on the `test` partition; the `gpu` partition is down).

## 1. Get the dataroot

The dataroot is the local LeRobot v3.0 dataset directory (it holds `meta/info.json`, `data/`, `videos/`).

- If the user did not give one, ask for it with AskUserQuestion before doing anything else. Offer the dataset directories found by `ls -d /scratch/nk_plora/*/meta/info.json` (strip `/meta/info.json`) as options. Do not guess.
- Check that `<dataroot>/meta/info.json` exists and that `codebase_version` is `v3.0`. The converter does not read v2.x datasets.

## 2. Inspect the dataset

Read `meta/info.json` (plain JSON, fine on the login node) and note:

- `fps`, `total_episodes`, `robot_type`;
- the `observation.images.*` video keys and their resolution;
- the joint names of `observation.state` / `action` and their count D (6 = one arm, 12 = two arms).

The task string is in `meta/tasks.parquet`. The login node has no pyarrow, so read it in the lerobot image on a CPU node:

```bash
srun --account=nk_plora --partition=cpu --ntasks=1 --cpus-per-task=2 --time=00:10:00 bash -c \
  "module load singularity/4.0 && singularity exec -B /scratch/nk_plora /project/nk_plora/lerobot_train.sif python -c \
  \"import pyarrow.parquet as pq; print(pq.read_table('<dataroot>/meta/tasks.parquet').to_pandas())\""
```

`singularity` is not on the default PATH; every container call needs `module load singularity/4.0` first.

## 3. Decide the arguments

- **Views.** Map each camera to a tile of the fixed 2x2 layout (`model/cosmos_predict2/data/action/tiling.py`): the workspace / top camera to `scene_rgb`, the left wrist to `left_wrist_rgb`, the right wrist to `right_wrist_rgb`. A single-arm wrist camera goes to `right_wrist_rgb`. If the camera names do not make the mapping obvious, ask the user.
- **Prompt.** `--prompt` is used only when the dataset task string is empty, and it is baked into the episodes and the video metas. If the task string is empty, ask the user for the instruction; do not fall back to the converter default ("route the cable") unless the data is the cabling task.
- **Output directory.** The configs read fixed paths: two-arm data from `/scratch/nk_plora/mimic/so101_2arm/{episodes,video_dataset}` (`dataloading/dataset/so101_2arm.yaml`, `so101_2arm` in `configs/defaults/data_video.py`), the one-arm cabling data from `/scratch/nk_plora/mimic/so101_cabling/`. Use the matching path when it is free. If it already holds data from another recording, ask whether to overwrite it or to write to a new `/scratch/nk_plora/mimic/<name>/` (new data then needs a copied dataset yaml, top-level yaml and `data_video.py` entry, see `SO101.md` "Configs").
- **Image rate.** Keep `--image-fps 5` (the model's rate). The dataset fps should be a multiple of 5.

Summarize the chosen arguments to the user before starting step 1.

## 4. Run

Run steps 1 and 2 with `srun --ntasks=1` (so each starts once), in the background with Bash `run_in_background`, and step 3 with `sbatch`. Check the result of each step before starting the next. Replace `<dataroot>`, `<out>` and the views.

```bash
# Step 1: about 4 min per 165 episodes at 15 fps on 32 CPUs; longer for 30 fps and three cameras.
srun --account=nk_plora --partition=cpu --ntasks=1 --cpus-per-task=32 --mem-per-cpu=4000 --time=02:00:00 bash -c \
  "module load singularity/4.0 && singularity exec -B /project -B /scratch/nk_plora /project/nk_plora/lerobot_train.sif \
  python so100/process_so100_v3.py --root <dataroot> --out <out>/episodes \
    --view observation.images.top=scene_rgb \
    --view observation.images.left_wrist=left_wrist_rgb \
    --view observation.images.right_wrist=right_wrist_rgb \
    --prompt '<instruction>' --num-workers 24"

# Step 2
srun --account=nk_plora --partition=cpu --ntasks=1 --cpus-per-task=16 --mem-per-cpu=4000 --time=01:00:00 bash -c \
  "module load singularity/4.0 && singularity exec -B /project -B /scratch/nk_plora /project/nk_plora/mimic-video.sif bash -c \
  'cd model && PYTHONPATH=. python ../data_preprocessing/video/process_so101_video.py \
    --episodes <out>/episodes --out <out>/video_dataset --num-workers 16'"

# Step 3 (needs model/checkpoints -> /project/nk_plora/mimic-video-checkpoints with text_encoder/)
sbatch --account=nk_plora --partition=test --ntasks=1 --gres=gpu:1 --cpus-per-task=24 --mem-per-cpu=4000 --time=00:30:00 \
  --job-name=mimic_t5 --output=/project/nk_plora/logs/mimic_t5-%j.out --chdir=$PWD \
  --wrap "module load singularity/4.0 && singularity exec --nv -B /project -B /scratch/nk_plora /project/nk_plora/mimic-video.sif bash -c \
  'cd model && PYTHONPATH=. python ../data_preprocessing/action/precompute_t5.py --dataset-path <out>/episodes && \
   PYTHONPATH=. python ../data_preprocessing/video/precompute_t5_embeddings.py --dataset-path <out>/video_dataset'"
```

Steps 1 and 2 skip existing outputs; pass `--overwrite` to redo them.

Step 3 notes:
- Submit it with `sbatch`, not a background `srun`. A background `srun` is killed when the Claude session ends, and the job is dropped without a log.
- It needs about 50 GB of host RAM: `text_encoder/t5-11b/pytorch_model.bin` is the full fp32 T5-11B (45 GB), and `torch.load` reads it whole before keeping the encoder. The partition default of 1000 MB per CPU gets OOM-killed (`oom_kill event` in the log), and `test` caps memory at 4000 MB per CPU, so ask for 24 CPUs at 4000 MB (96 GB). The encoder then fits on one A100-40GB.
- Both datasets need the embeddings: the video dataset loader reads `video_dataset/language_embeddings/<episode>.safetensors` with no fallback, and the action dataloader reads `language_embedding` from the episodes.

## 5. Verify

- Step 1 ends with `DONE: N episodes in <out>/episodes`; N must equal `total_episodes`. Check one episode's keys and shapes by reading the safetensors header (works on the login node):
  ```bash
  python3 -c "import json,struct,sys; b=open(sys.argv[1],'rb'); n=struct.unpack('<Q',b.read(8))[0]; h=json.loads(b.read(n)); print(h.pop('__metadata__',None)); [print(k,v['dtype'],v['shape']) for k,v in sorted(h.items())]" <out>/episodes/episode_000000.safetensors
  ```
  Expect one `(T, 240, 320, 3)` uint8 array per view, `joint_state_lowdim` / `joint_action_lowdim` `(T, D)` float32, `language_instruction`, and after step 3 `language_embedding`.
- Step 2: one `.mp4` and one `.txt` per episode in `video_dataset/video` and `video_dataset/metas`. Step 3: one file per episode in `video_dataset/language_embeddings`.
- Report the output paths, episode count and the matching configs to the user. Training is next: `mimic-v2w-train` for the video backbone, then `mimic-w2a-train` for the action decoder.
