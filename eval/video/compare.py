"""Compare video backbone rollouts against the ground-truth clips.

Reads ``<out>/episodes.txt`` and ``<out>/<model>/<episode>.mp4`` (written by prepare.py and run_video2world.py) and
writes:
- ``side_by_side/<episode>.mp4``: ground truth, then each model, left to right;
- ``grid_<episode>.jpg``: frames 0, 4, 20 and 60 as rows (half resolution), same column order;
- ``metrics.csv`` and a printed table.

Metrics, over the predicted frames only (after the 5 conditioning frames):
- ``psnr``: PSNR against the ground truth. Futures can validly diverge, so this is a weak, relative signal.
- ``blank``: mean pixel value (0-255) of the tile cells that are black in the ground truth, e.g. the empty top-right
  cell of the SO-101 layout. It should stay near 0 when the model has learned the tiled layout.
"""

import argparse
import csv
from pathlib import Path

import imageio
import numpy as np

NUM_COND_FRAMES = 5
CELL = (240, 320)  # matches cosmos_predict2/data/action/tiling.py CELL_SIZE
GRID_FRAMES = (0, 4, 20, 60)


def read_video(path: Path) -> np.ndarray:
    return np.stack(imageio.v2.mimread(path, memtest=False))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0**2 / mse)) if mse > 0 else float("inf")


def black_cells(gt: np.ndarray) -> list[tuple[slice, slice]]:
    cells = []
    for y in range(0, gt.shape[1], CELL[0]):
        for x in range(0, gt.shape[2], CELL[1]):
            cell = (slice(y, y + CELL[0]), slice(x, x + CELL[1]))
            if gt[:, cell[0], cell[1]].mean() < 2:
                cells.append(cell)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="video dataset with video/")
    parser.add_argument("--out", type=Path, required=True, help="directory written by prepare.py")
    parser.add_argument("--models", nargs="+", required=True, help="model names, in column order")
    args = parser.parse_args()

    episodes = (args.out / "episodes.txt").read_text().split()
    (args.out / "side_by_side").mkdir(exist_ok=True)

    rows = []
    for ep in episodes:
        gt = read_video(args.dataset / "video" / f"{ep}.mp4")
        preds = {m: read_video(args.out / m / f"{ep}.mp4") for m in args.models}
        n = min(len(gt), *(len(p) for p in preds.values()))
        gt = gt[:n]
        videos = [gt] + [p[:n] for p in preds.values()]

        imageio.v2.mimwrite(
            args.out / "side_by_side" / f"{ep}.mp4", np.concatenate(videos, axis=2), fps=5, macro_block_size=1
        )
        grid = np.concatenate([np.concatenate([v[t] for v in videos], axis=1) for t in GRID_FRAMES if t < n], axis=0)
        imageio.imwrite(args.out / f"grid_{ep}.jpg", grid[::2, ::2])

        cells = black_cells(gt)
        row = {"episode": ep, "frames": n}
        for m, p in preds.items():
            pred = p[NUM_COND_FRAMES:n]
            row[f"psnr_{m}"] = psnr(gt[NUM_COND_FRAMES:], pred)
            row[f"blank_{m}"] = float(np.mean([pred[:, ys, xs].mean() for ys, xs in cells])) if cells else float("nan")
        rows.append(row)

    with open(args.out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    cols = list(rows[0])
    print("".join(f"{c:>16}" for c in cols))
    for row in rows:
        print("".join(f"{row[c]:>16.2f}" if isinstance(row[c], float) else f"{row[c]:>16}" for c in cols))
    means = {c: np.mean([r[c] for r in rows]) for c in cols[2:]}
    print("".join(f"{v:>16}" for v in ("mean", "")) + "".join(f"{means[c]:>16.2f}" for c in cols[2:]))


if __name__ == "__main__":
    main()
