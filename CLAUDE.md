# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Communication style

Always use caveman mode (ultra level) for replies in this repository: terse, no filler, all technical substance kept. Written artifacts (code, comments, commits, docs) stay normal prose.

## Fork change log

This repository is a fork of the original mimic-video release. Every change made on top of the original must be recorded in the "Changes From Original Repository" table at the top of `README.md` (date, change, files; newest first).

## Environment

- Python 3.10 only. Dependencies are managed with `uv` from `model/pyproject.toml` (no committed lock file; `uv.lock` is gitignored).
- Extras are mutually exclusive: `cu126` (Hopper, torch 2.6.0, x86 + aarch64) and `cu129` (Blackwell, torch 2.8.0, x86 only).
  - `cd model && uv sync --extra cu126 && source .venv/bin/activate`
  - `uv sync` builds a universal lock and needs metadata from the git sources of every extra (for example apex for `cu129`). That fails when the local nvcc does not match. The container uses `uv pip install -r pyproject.toml --extra <cu126|cu129>` instead, which resolves only for the current platform.
- Container: `container/mimic-video.def` (Singularity/Apptainer), built on a compute node by `sbatch container/build.sbatch` from the repo root (never build on the login node). Output: `/project/nk_plora/mimic-video.sif` (override with `SIF=`, extra `--build-arg`s via `BUILD_ARGS=`). Run with `singularity exec --nv /project/nk_plora/mimic-video.sif ...`. The venv lives in the image at `/opt/mimic-video/.venv`; `cosmos_predict2`/`imaginaire` are imported from the host checkout by running from `model/`. SimplerEnv is copied into the image at build time; LIBERO is used from the host via `PYTHONPATH=LIBERO`.
- Environment sanity check: `cd model && python scripts/test_environment.py`.
- Checkpoints: `cd model && python scripts/download_checkpoints.py` (needs `hf auth login`), stored in `model/checkpoints/` (gitignored).

## Common commands

All training runs from `model/` through torchrun with the Hydra-style experiment override:

```bash
cd model
torchrun -m scripts.train --config=cosmos_predict2/configs/config.py -- experiment=<name>
```

- Video model finetuning experiments: `model/cosmos_predict2/configs/experiment/video2world.py`, datasets in `configs/defaults/data_video.py`.
- Action decoder experiments: `model/cosmos_predict2/configs/experiment/world2action.py`, dataset yaml in `configs/dataloading/dataset/`.
- Wandb entity: `model/cosmos_predict2/configs/defaults/callbacks.py`.
- Eval: `bash eval/bridge/eval.sh`, `bash eval/bridge/eval_hil.sh`, `bash eval/libero/eval.sh` (edit `GPUS` and `checkpoint_dir` at the top of each script first).
- Lint: `ruff` with `model/ruff.toml` (line length 120, py310).

There is no test suite beyond `scripts/test_environment.py`.

## Architecture

- `model/imaginaire/`: generic training framework from NVIDIA (trainer, lazy config, callbacks, utils).
- `model/cosmos_predict2/`: Cosmos-Predict2 2B video model plus the mimic-video additions.
  - `configs/`: `config.py` is the entry point; `config_video2world.py` and `config_world2action.py` define the two model families; `experiment/` holds named experiments; `dataloading/` holds dataset yamls (see `DATA.md`).
  - `models/`, `networks/`, `pipelines/`, `conditioner.py`: video backbone (video2world) and action decoder (world2action). The action decoder cross-attends to latent features from a chosen layer of the frozen video backbone; video and action use decoupled flow times, so inference needs one video forward pass per action chunk.
  - `data/action/`: action dataloading (safetensors episodes, dataset statistics).
- `data_preprocessing/video/`: builds video finetuning datasets (`video/*.mp4`, `metas/*.txt`, T5 `language_embeddings/`, optional `video_embeddings/`).
- `data_preprocessing/action/`: converts raw Bridge / LIBERO data to safetensors, precomputes or transfers T5 and video embeddings.
- `eval/bridge/SimplerEnv/` and `eval/libero/LIBERO/`: vendored simulators with integrated mimic-video policies (`simpler_env/main_inference.py`, `main_inference_hil.py`, `eval/libero/run.py`).

See `README.md`, `MODEL.md` and `DATA.md` for details.
