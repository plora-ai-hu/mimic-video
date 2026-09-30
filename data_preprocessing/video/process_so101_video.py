"""Build a video finetuning dataset from SO-100 / SO-101 safetensors episodes (so100/process_so100_v3.py).

The camera views of each episode are tiled into one frame with the same layout the action dataloader uses
(model/cosmos_predict2/data/action/tiling.py), so the video backbone is finetuned on exactly what the action decoder
feeds it. Writes ``<out>/video/<episode>.mp4`` at the stored image rate and ``<out>/metas/<episode>.txt`` with the
language instruction. Afterwards run precompute_t5_embeddings.py on ``<out>`` to create ``language_embeddings/``.

Run from ``model/`` so ``cosmos_predict2`` is importable:

    cd model && PYTHONPATH=. python ../data_preprocessing/video/process_so101_video.py \
        --episodes /scratch/nk_plora/mimic/so101_cabling/episodes \
        --out /scratch/nk_plora/mimic/so101_cabling/video_dataset
"""

import argparse
import json
import multiprocessing
import pathlib
from fractions import Fraction
from functools import partial

import av
import numpy as np
import safetensors
import tqdm

from cosmos_predict2.data.action.tiling import CELL_SIZE, DUAL_ARM_LAYOUT, tile_views


def parse_layout(spec: str) -> list[list[str | None]]:
    """``scene_rgb,-;left_wrist_rgb,right_wrist_rgb`` -> rows separated by ``;``, ``-`` is an empty slot."""
    return [[None if name == "-" else name for name in row.split(",")] for row in spec.split(";")]


def write_mp4(frames: np.ndarray, path: pathlib.Path, fps: float, crf: int) -> None:
    tmp_path = path.with_suffix(".tmp.mp4")
    with av.open(str(tmp_path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000))
        stream.width, stream.height = frames.shape[2], frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}
        for frame in frames:
            container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
        container.mux(stream.encode())
    tmp_path.rename(path)


def convert(episode_path: pathlib.Path, args: argparse.Namespace) -> str:
    video_path = args.out / "video" / f"{episode_path.stem}.mp4"
    meta_path = args.out / "metas" / f"{episode_path.stem}.txt"
    if video_path.exists() and meta_path.exists() and not args.overwrite:
        return "skip"

    placed = [name for row in args.layout for name in row if name is not None]
    with safetensors.safe_open(episode_path, "np") as f:
        metadata = f.metadata() or {}
        keys = set(f.keys())
        views = {name: f.get_tensor(name) for name in placed if name in keys}
        prompt = bytes(f.get_tensor("language_instruction")[0]).decode("utf-8")
    if not views:
        return f"{episode_path.name}: none of the layout views {placed} present, skip"

    # Views are recorded at the same rate but can differ by a frame at the episode end.
    n = min(len(frames) for frames in views.values())
    tiled = tile_views({name: frames[:n] for name, frames in views.items()}, args.layout, args.cell_size)

    fps = args.fps or float(metadata.get("image_fps", 0)) or None
    if fps is None:
        return f"{episode_path.name}: no image_fps in metadata, pass --fps"
    write_mp4(tiled, video_path, fps=fps, crf=args.crf)
    meta_path.write_text(prompt + "\n")
    return "ok"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=pathlib.Path, required=True, help="directory with episode_*.safetensors")
    p.add_argument("--out", type=pathlib.Path, required=True, help="video dataset directory")
    p.add_argument(
        "--layout",
        type=parse_layout,
        default=DUAL_ARM_LAYOUT,
        help=f"tile layout, rows separated by ';', slots by ',', '-' for empty (default: {json.dumps(DUAL_ARM_LAYOUT)})",
    )
    p.add_argument("--cell-size", type=int, nargs=2, default=list(CELL_SIZE), metavar=("H", "W"))
    p.add_argument("--fps", type=float, default=None, help="output fps (default: image_fps from the episode metadata)")
    p.add_argument("--crf", type=int, default=18, help="x264 CRF (lower = better quality)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    (args.out / "video").mkdir(parents=True, exist_ok=True)
    (args.out / "metas").mkdir(parents=True, exist_ok=True)
    episode_paths = sorted(args.episodes.glob("**/*.safetensors"))
    if not episode_paths:
        raise SystemExit(f"No safetensors episodes found in {args.episodes}")

    with multiprocessing.Pool(args.num_workers) as pool:
        for msg in tqdm.tqdm(
            pool.imap_unordered(partial(convert, args=args), episode_paths),
            total=len(episode_paths),
            desc="Writing tiled videos",
        ):
            if msg not in ("ok", "skip"):
                print(msg, flush=True)


if __name__ == "__main__":
    main()
