"""Select episodes of a video dataset and write the inputs for video backbone rollouts.

For every selected clip, the first 5 frames are written to ``<out>/cond/<episode>.mp4``. ``run_video2world.py``
conditions on the last 5 frames of its input video, so the model predicts the frames that follow these in the real
clip. One ``batch_<model>.json`` per model lists input, prompt (from ``metas/<episode>.txt``) and output path.
"""

import argparse
import hashlib
import json
from pathlib import Path

import imageio
import numpy as np

NUM_COND_FRAMES = 5


def read_video(path: Path) -> np.ndarray:
    return np.stack(imageio.v2.mimread(path, memtest=False))


def is_val(stem: str, val_ratio: float) -> bool:
    # Same split as cosmos_predict2/data/dataset_video.py (stable md5 hash of the clip name).
    denom = 10_000
    return int(hashlib.md5(stem.encode("utf-8")).hexdigest(), 16) % denom < round(val_ratio * denom)


def select_episodes(videos: list[Path], episodes: list[str], split: str, num: int, val_ratio: float) -> list[str]:
    stems = [p.stem for p in videos]
    if episodes:
        by_index = {str(i): s for i, s in enumerate(stems)}
        chosen = [by_index.get(e, e) for e in episodes]
        missing = [e for e in chosen if e not in stems]
        if missing:
            raise SystemExit(f"Episodes not in the dataset: {missing}")
        return chosen
    if split == "val":
        stems = [s for s in stems if is_val(s, val_ratio)]
    elif split == "train":
        stems = [s for s in stems if not is_val(s, val_ratio)]
    return stems[:num]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="video dataset with video/ and metas/")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--models", nargs="+", required=True, help="model names, one batch JSON each")
    parser.add_argument("--episodes", nargs="*", default=[], help="clip names or indices; overrides --split")
    parser.add_argument("--split", choices=["all", "val", "train"], default="val", help="which clips to take")
    parser.add_argument("--num", type=int, default=5, help="number of clips taken from --split")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="val_ratio of the training dataset config")
    args = parser.parse_args()

    videos = sorted((args.dataset / "video").glob("*.mp4"))
    episodes = select_episodes(videos, args.episodes, args.split, args.num, args.val_ratio)
    if not episodes:
        raise SystemExit("No episodes selected.")
    print(f"Episodes: {' '.join(episodes)}")

    (args.out / "cond").mkdir(parents=True, exist_ok=True)
    for ep in episodes:
        cond = args.out / "cond" / f"{ep}.mp4"
        if not cond.exists():
            frames = read_video(args.dataset / "video" / f"{ep}.mp4")
            imageio.v2.mimwrite(cond, frames[:NUM_COND_FRAMES], fps=5, macro_block_size=1)
    (args.out / "episodes.txt").write_text("\n".join(episodes) + "\n")

    for model in args.models:
        items = [
            {
                "input_video": str(args.out / "cond" / f"{ep}.mp4"),
                "prompt": (args.dataset / "metas" / f"{ep}.txt").read_text().strip(),
                "output_video": str(args.out / model / f"{ep}.mp4"),
            }
            for ep in episodes
        ]
        (args.out / f"batch_{model}.json").write_text(json.dumps(items, indent=2))


if __name__ == "__main__":
    main()
