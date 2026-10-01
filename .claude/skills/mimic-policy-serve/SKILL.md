---
name: mimic-policy-serve
description: Start, check and stop the SO-101 policy server (so100/so101_policy_server.py) on a Slurm GPU node, and give the user the SSH tunnel and robot client commands. Use when asked to serve / deploy / start the policy or action decoder, to run the robot or the SO-101 arm with a trained decoder, to get the tunnel command, to replay an episode against the server, or when the user says "start the server" or "I want to test on the robot".
---

# Serving an SO-101 action decoder

The policy server runs in the mimic-video container on one GPU and listens on `127.0.0.1:8766` of the compute node. The robot client (`so100/so101_robot_client.py`) runs on the user's laptop in a LeRobot environment and connects through an SSH tunnel. Deployment is described in the Deployment section of `SO101.md`, and the wire format in `so100/so101_protocol.py`. Decoders are trained with the `mimic-w2a-train` skill.

## 1. Pick the checkpoints

The server needs three values that must belong together:

- `DECODER`: a decoder checkpoint `/project/nk_plora/outputs/mimic-video/vam/<decoder net>/<job name>/checkpoints/model/iter_XXXXXXXXX.pt`. List them with `find /project/nk_plora/outputs/mimic-video/vam -path '*/checkpoints/model/iter_*.pt'`. Ask the user which run and iteration to use when more than one fits. Runs named `smoke_*` are plumbing tests only; their actions are meaningless.
- `VIDEO_DIT`: the fused backbone the decoder was trained on. Read it from the run, never guess it: `grep -m1 video_dit_path <run dir>/config.yaml`. A different backbone gives wrong features and wrong actions without any error.
- `EXPERIMENT`: `w2a_<data config>_v2w_pretrained_cosmos_lr1.000e-04_layer<layer>_bsz<N>`. The server uses it only for the data config (views, joints, statistics, tile layout) and the decoder net, so the data config and the layer must match the run; the `bsz` part does not matter. The data config is `config_name` in `<run dir>/config.yaml`. The exact name is in the training log `/project/nk_plora/logs/<slurm job name>-<job id>.out` (line `experiment:`).

The normalization statistics default to the single file in `<data_dir>/.statistics_cache/`. If there are several, find the one the run used and pass `--stats <path>`.

## 2. Preconditions

Check from the repository root:

- `secrets.env` has a non-empty `MIMIC_POLICY_TOKEN`: `grep -c '^export MIMIC_POLICY_TOKEN=.' secrets.env`. Never print the value. If it is missing, ask the user to add one (for example from `openssl rand -hex 32`) with `secrets.env` at mode 600. Do not write the token yourself, because the user must also set it on the laptop.
- `/project/nk_plora/mimic-video.sif` exists.
- `squeue -u $USER -n so101_policy_server` shows no other server. Two servers on the same node collide on the port; give a second one `PORT=<other>`.

## 3. Submit

```bash
EXPERIMENT=<experiment> VIDEO_DIT=<fused backbone> DECODER=<decoder iter_XXXXXXXXX.pt> \
  sbatch so100/so101_serve.sbatch [--stop-step N] [--debug-dir /scratch/nk_plora/mimic/so101_serve_debug]
```

- The default time limit is 8 h on the `gpu` partition. Pass `sbatch --time=...` for a longer robot session. The `test` partition (1 h) is enough for a quick check.
- `--stop-step` (default 0) is the video denoising step whose features the decoder reads; 0 is a single backbone pass. Leave it at the default unless the user asks.
- `--debug-dir` writes every tiled input as PNG. Only pass it when the user wants to check the camera input, because it writes one file per query.

## 4. Wait until it listens and hand over the commands

The log is `/project/nk_plora/logs/so101_policy_server-<job id>.out`. Loading takes about 1 min. Wait with a Monitor that also catches failures:

```bash
until grep -m1 'listening on' LOG 2>/dev/null || grep -m1 -E 'Traceback|Error|SystemExit' LOG 2>/dev/null \
  || sacct -j JOBID -X -n -o State | grep -qE 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|COMPLETED'; do sleep 10; done
```

Then give the user:

- the job id and the node (the `tunnel:` line of the log names it);
- the `>>> ready:` line: joint names, views, prompt;
- the tunnel command for the laptop, with the login node filled in if known: `ssh -N -L 8766:localhost:8766 -J <user>@<login node> <node>`;
- the client commands, in this order:
  ```bash
  export MIMIC_POLICY_TOKEN=...   # same value as in secrets.env
  python so100/so101_robot_client.py --replay <episode>.safetensors --cycles 10     # no robot, checks the tunnel
  python so100/so101_robot_client.py --port /dev/ttyACM0 --cal-id <id> \
    --camera top=/dev/video0 --camera wrist=/dev/video2 \
    --view top=scene_rgb --view wrist=right_wrist_rgb --dry-run
  python so100/so101_robot_client.py ... --exec-steps 3 --max-rel 5              # real motion
  ```

The `--view` values must be the view names from the `ready` line. Remind the user that real motion needs a clear workspace and a hand near the power switch. Only the user runs the robot; never suggest skipping `--dry-run` on a new checkpoint.

## 5. Checking without the laptop

To test the server alone, run the replay client in the container on the same node while the job runs:

```bash
srun --jobid=JOBID --overlap --ntasks=1 bash -c 'module load singularity/4.0 && \
  singularity exec -B /project -B /scratch/nk_plora -B $PWD /project/nk_plora/mimic-video.sif \
  python so100/so101_robot_client.py --replay <episode>.safetensors --cycles 4'
```

`MIMIC_POLICY_TOKEN` must be set in that environment (for example by sourcing `secrets.env` inside the `bash -c`). The output shows the latency per query (about 2 s on an A100) and the mean absolute error against the recorded actions in degrees. Use a validation episode for a fair number; training episodes look better than they are.

## 6. Stop

`scancel JOBID` when the user is done. The server holds a GPU for the whole time limit otherwise.

## Known issues

- `bad token`: the laptop's `MIMIC_POLICY_TOKEN` differs from `secrets.env`.
- `Connection refused` on the laptop: the tunnel is not open, it points at the wrong node, or the server is still loading.
- `... has no joint names ...`: the episodes were written by an older `precompute_t5.py` that dropped the converter metadata. For 6 joints the server assumes the SO follower order; for other joint counts, reconvert the episodes.
- `robot joints [...] differ from the trained joints [...]`: the LeRobot robot's motor names or order differ from the training data. Do not work around this by reordering blindly; check the dataset's joint names first.
- `ptrDesc->finalize()` or `Found 2 libcudnn.so.x`: old image. `so101_serve.sbatch` applies the same workarounds as the training and eval scripts; rebuild the image to fix it for good.
