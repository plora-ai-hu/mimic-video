# LIBERO inference on one 24 GB GPU, with a native MuJoCo viewer

Run the released LIBERO checkpoints on a single **NVIDIA L4 (24 GB)** and watch the policy act
live in **MuJoCo's own viewer** on a laptop. The policy runs on the GPU box; the LIBERO sim and
the viewer run locally and talk to it through an SSH tunnel.

```
laptop (Linux)                                   GPU box (L4)
viewer_client.py                                 policy_server.py
  LIBERO env + mujoco.viewer   --- SSH tunnel -->  VAMInference.step (same call as run.py)
  sends agent-view PNG + eef state every step     returns one action; queries the model
  executes the returned action                     every 5 steps (~3.5 s on an L4)
```

Tested with `libero_spatial_full` on an AWS g6.4xlarge (L4, driver 595, host CUDA 13.2) and an
Ubuntu laptop.

## GPU box

```bash
cd model && uv sync --extra cu126 && source .venv/bin/activate
cd ../eval/libero && uv pip install -r LIBERO/requirements.txt && uv pip install -e LIBERO

# Checkpoints for one suite (~5.5 GB). Not scripts/download_checkpoints.py: its LIBERO
# video-backbone pattern formats a Python list and matches nothing. No T5 weights needed.
hf download jonpai/mimic-video --local-dir ../../model/checkpoints \
  --include "text_encoder/t5-11b/*.json" "video_backbone/tokenizer/*" \
            "video_backbone/v2w_libero_spatial_*" "action_decoder/w2a_libero_spatial_full_*" \
            "dataset_statistics/libero_spatial_full.json"

# Once: T5 embeddings of all 40 LIBERO instructions -> libero_prompt_embeddings.pt.
# CPU only; reads ~20 GB of encoder weights into RAM, ~15 min on 16 vCPUs.
PYTHONPATH=LIBERO python precompute_prompt_embeddings.py

./policy_server_l4.sh            # serves libero_spatial_full on 127.0.0.1:8766
```

The first query after startup also captures CUDA graphs and takes about a minute.

## Laptop (Linux, Python 3.10)

```bash
git clone -b libero-l4-native-viewer https://github.com/plora-ai-hu/mimic-video
cd mimic-video/eval/libero
uv venv -p 3.10 .venv-viewer && source .venv-viewer/bin/activate
uv pip install -r requirements-viewer.txt     # ~1.2 GB, no torch

ssh -L 8766:localhost:8766 <gpu-host>         # separate terminal
python viewer_client.py --task 0 --episode 0  # LIBERO asks for a dataset path once: answer N
```

**Viewer:** drag to orbit, right-drag to pan, scroll to zoom; `[` / `]` cycle cameras
(`agentview` is the policy's camera, un-rotated). Double-click a body, then Ctrl+right-drag to
push it; the policy sees the result on its next query.
**Terminal** (type + Enter): `p` pause, `r` restart, `n` next init state, `t` next task.

The arm moves in bursts: it holds still while the model runs (~3.5 s), then plays the 5 predicted
actions at 20 Hz.

## What the model sees

One camera: the fixed `agentview`, 480×640, rotated 180° (as in the training data). Each query
gets the last 17 frames at 20 Hz subsampled to **5 frames at 5 fps** (0.8 s of history; the first
frame is repeated at episode start), plus a 10-D state (eef position, 6-D rotation, one gripper
finger) and the instruction's T5 embedding.

## Modifications

| File | Change | Why |
|---|---|---|
| [eval/libero/run.py](eval/libero/run.py) | Optional `prompt_embeddings_path` / `--vam_prompt_embeddings_path`; loads the pipeline without T5 and passes `prompt_embedding` (+14 lines) | T5-11B is loaded onto the GPU in fp32 (~45 GB) by default |
| [eval/libero/precompute_prompt_embeddings.py](eval/libero/precompute_prompt_embeddings.py) | New. Range-reads only T5-11B's encoder from `google-t5/t5-11b` (refs/pr/6 safetensors, same weights) and encodes on CPU in fp32, mirroring `CosmosT5TextEncoder.encode_prompts` | Keeps T5 off the GPU and off the disk |
| [eval/libero/policy_server.py](eval/libero/policy_server.py), [policy_server_l4.sh](eval/libero/policy_server_l4.sh) | New. WebSocket server around `VAMInference.step`; launcher drops `/usr/local/cuda*` from `LD_LIBRARY_PATH` | A host cuDNN 9.20 loaded next to the venv's 9.5 makes the video tokenizer's conv3d fail with `ptrDesc->finalize()` |
| [eval/libero/viewer_client.py](eval/libero/viewer_client.py), [requirements-viewer.txt](eval/libero/requirements-viewer.txt) | New. `run.py`'s episode loop in `mujoco.viewer`; env built with `hard_reset=False` so the viewer keeps one model | robosuite otherwise recompiles the model on every reset |
| [model/pyproject.toml](model/pyproject.toml) | Static apex metadata (`[[tool.uv.dependency-metadata]]`) | Without a lock file, uv builds the cu129/aarch64 apex source variants just to resolve, which fails against CUDA 13.2 |

## Notes

- **Pin MuJoCo for `run.py` batch evals:** `uv pip install mujoco==3.1.6` in the model venv.
  robosuite 1.4.0 fails on newer MuJoCo with an assertion in `get_joint_qpos_addr`. The policy
  server doesn't need it; `requirements-viewer.txt` pins it for the laptop.
- **Batch eval on the L4:** `eval.sh`'s `run.py` command plus
  `--vam_prompt_embeddings_path libero_prompt_embeddings.pt` and the `LD_LIBRARY_PATH` line from
  `policy_server_l4.sh`. On the L4: 3.5 s per query, 14.3 GB peak GPU memory.
- **Outcomes are not reproducible run to run.** Physics is deterministic and so is the policy
  (identical inputs give bit-identical actions), but LIBERO's EGL rendering differs by ~1 intensity
  level between runs. That is enough to change a bf16 action by one ULP, which contact physics then
  amplifies. The same init state can succeed in one run and time out in the next.
