"""Convert a local LeRobot v3.0 dataset (SO-100 / SO-101, one or two arms) to mimic-video safetensors episodes.

Runs in an environment with a v3.0-capable ``lerobot`` (e.g. /project/nk_plora/lerobot_train.sif); the output is
read by the mimic-video action dataloader and by data_preprocessing/video/process_so101_video.py. One
``episode_XXXXXX.safetensors`` per episode with, each next to a ``<key>_timestamps`` array (uint64 ns, relative to
the episode start):

- ``<view name>``: (T_img, H, W, 3) uint8 RGB per camera view, subsampled to ``--image-fps`` and resized to
  ``--cell-size`` (one cell of the tiled model input, see model/cosmos_predict2/data/action/tiling.py),
- ``joint_state_lowdim`` / ``joint_action_lowdim``: (T, D) float32 ``observation.state`` / ``action`` at the full
  dataset rate, any D (6 for one SO-101, 12 for two),
- ``language_instruction``: (1, L) uint8 utf-8 bytes (``--prompt`` if the dataset task string is empty).

The joint names, views and source fps are stored in the safetensors metadata.

Example (single SO-101 cabling demo):

    singularity exec -B /scratch/nk_plora /project/nk_plora/lerobot_train.sif \
        python so100/process_so100_v3.py \
        --root /scratch/nk_plora/cabling_demo_dataset \
        --out /scratch/nk_plora/mimic/so101_cabling/episodes \
        --view observation.images.top=scene_rgb \
        --view observation.images.wrist=right_wrist_rgb \
        --prompt "route the cable"
"""

import argparse
import json
import multiprocessing
import os
import pathlib

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import av
import numpy as np
import PIL.Image
import safetensors.numpy as st
from lerobot.datasets.lerobot_dataset import LeRobotDataset

S_TO_NS = 1_000_000_000

# Set in main() before the worker pool forks so workers share the loaded dataset.
_DS: LeRobotDataset | None = None
_ARGS: argparse.Namespace | None = None


def parse_view(spec: str) -> tuple[str, str]:
    key, sep, name = spec.partition("=")
    if not sep or not key or not name:
        msg = f"--view expects LEROBOT_KEY=STORED_NAME, got {spec!r}"
        raise argparse.ArgumentTypeError(msg)
    return key, name


def to_array(column) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float32) for value in column]).reshape(len(column), -1)


def decode_view(
    video_path: pathlib.Path, from_ts: float, to_ts: float, fps: float, keep_step: int, cell_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Decode the ``[from_ts, to_ts)`` span of a (possibly multi-episode) video file.

    Keeps every ``keep_step``-th frame, resized to ``cell_size`` = (height, width). Returns frames and their
    timestamps in seconds relative to ``from_ts``.
    """
    height, width = cell_size
    half_frame = 0.5 / fps
    frames, timestamps = [], []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        container.seek(int(from_ts / stream.time_base), backward=True, any_frame=False, stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            t = float(frame.pts * stream.time_base)
            if t < from_ts - half_frame:
                continue
            if t >= to_ts - half_frame:
                break
            if round((t - from_ts) * fps) % keep_step:
                continue
            image = frame.to_image()
            if image.size != (width, height):
                image = image.resize((width, height), PIL.Image.BILINEAR)
            frames.append(np.asarray(image, dtype=np.uint8))
            timestamps.append(t - from_ts)
    return np.stack(frames), np.asarray(timestamps)


def convert_episode(ep: int) -> str:
    ds, args = _DS, _ARGS
    out_path = args.out / f"episode_{ep:06d}.safetensors"
    if out_path.exists() and not args.overwrite:
        return f"ep {ep}: skip (exists)"

    meta = ds.meta.episodes[ep]
    f0, f1 = int(meta["dataset_from_index"]), int(meta["dataset_to_index"])
    n = f1 - f0
    if n < 3:
        return f"ep {ep}: too short ({n} frames), skip"

    rows = ds.hf_dataset[f0:f1]
    state = to_array(rows["observation.state"])
    action = to_array(rows["action"])
    ts = np.asarray([float(np.asarray(t).reshape(-1)[0]) for t in rows["timestamp"]], dtype=np.float64)
    lowdim_ts = ((ts - ts[0]) * S_TO_NS).round().astype(np.uint64)

    fps = float(ds.fps)
    keep_step = max(1, round(fps / args.image_fps))
    res: dict[str, np.ndarray] = {
        "joint_state_lowdim": state,
        "joint_state_lowdim_timestamps": lowdim_ts,
        "joint_action_lowdim": action,
        "joint_action_lowdim_timestamps": lowdim_ts.copy(),
    }
    for key, name in args.view:
        frames, frame_ts = decode_view(
            ds.root / ds.meta.get_video_file_path(ep, key),
            from_ts=float(meta[f"videos/{key}/from_timestamp"]),
            to_ts=float(meta[f"videos/{key}/to_timestamp"]),
            fps=fps,
            keep_step=keep_step,
            cell_size=tuple(args.cell_size),
        )
        res[name] = frames
        res[f"{name}_timestamps"] = (frame_ts * S_TO_NS).round().astype(np.uint64)

    tasks = [task for task in (meta.get("tasks") or []) if task]
    prompt = tasks[0] if tasks else args.prompt
    res["language_instruction"] = np.frombuffer(prompt.encode("utf-8"), dtype=np.uint8)[None]
    res["language_instruction_timestamps"] = np.array([0], dtype=np.uint64)

    metadata = {
        "repo_id": ds.repo_id,
        "episode_index": str(ep),
        "source_fps": str(fps),
        "image_fps": str(fps / keep_step),
        "views": json.dumps({name: key for key, name in args.view}),
        "joint_state_names": json.dumps(ds.meta.features["observation.state"].get("names")),
        "joint_action_names": json.dumps(ds.meta.features["action"].get("names")),
    }
    tmp_path = out_path.with_suffix(".tmp")
    st.save_file(res, tmp_path, metadata=metadata)
    tmp_path.rename(out_path)
    view_frames = ", ".join(f"{name}={len(res[name])}" for _, name in args.view)
    return f"ep {ep}: {n} rows, D={state.shape[1]}, frames {view_frames}, prompt={prompt!r}"


def main() -> None:
    global _DS, _ARGS

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=pathlib.Path, required=True, help="local LeRobot v3.0 dataset directory")
    p.add_argument("--repo-id", default=None, help="dataset repo id (default: name of --root)")
    p.add_argument("--out", type=pathlib.Path, required=True, help="output directory for the episodes")
    p.add_argument(
        "--view",
        type=parse_view,
        action="append",
        required=True,
        help="LEROBOT_KEY=STORED_NAME, repeat per camera (e.g. observation.images.top=scene_rgb)",
    )
    p.add_argument("--cell-size", type=int, nargs=2, default=[240, 320], metavar=("H", "W"))
    p.add_argument("--image-fps", type=float, default=5.0, help="stored image rate (lowdim keeps the full rate)")
    p.add_argument("--prompt", default="route the cable", help="language instruction if the task string is empty")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    names = [name for _, name in args.view]
    if len(set(names)) != len(names):
        p.error(f"stored view names must be unique, got {names}")

    ds = LeRobotDataset(args.repo_id or args.root.name, root=args.root)
    missing = [key for key, _ in args.view if key not in ds.meta.video_keys]
    if missing:
        p.error(f"views {missing} not in dataset video keys {ds.meta.video_keys}")
    if abs(ds.fps / max(1, round(ds.fps / args.image_fps)) - args.image_fps) > 1e-6:
        print(f"warning: {ds.fps=} is not a multiple of {args.image_fps=}, storing at the nearest divisor", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    episodes = list(range(ds.meta.total_episodes))[: args.max_episodes]
    print(f"{args.root}: fps={ds.fps} episodes={len(episodes)} views={dict(args.view)} -> {args.out}", flush=True)

    _DS, _ARGS = ds, args
    if args.num_workers > 1:
        with multiprocessing.get_context("fork").Pool(args.num_workers) as pool:
            for msg in pool.imap_unordered(convert_episode, episodes):
                print(msg, flush=True)
    else:
        for ep in episodes:
            print(convert_episode(ep), flush=True)
    print(f"DONE: {len(list(args.out.glob('episode_*.safetensors')))} episodes in {args.out}", flush=True)
    # Release the dataset before interpreter shutdown, lerobot's metadata __del__ fails once modules are torn down.
    _DS = None
    del ds


if __name__ == "__main__":
    main()
