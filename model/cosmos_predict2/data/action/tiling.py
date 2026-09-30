"""Tile several camera views into one frame.

The video backbone and the action decoder consume a single 480x640 ``workspace_rgb`` stream. Multi-camera setups
are supported by placing each view (resized to ``CELL_SIZE``) into a fixed grid. Slots that are ``None`` or whose
view is missing stay black, so datasets with fewer cameras (e.g. a single-arm recording) share the layout of the
full setup and a backbone finetuned on one transfers to the other.

This module only depends on numpy and PIL so it can be imported from data conversion scripts and deployment code.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import PIL.Image

# (height, width) of one grid cell. A 2x2 grid of these gives the 480x640 frame the video backbone expects.
CELL_SIZE: tuple[int, int] = (240, 320)

# Dual SO-101 setup: workspace camera top left, top right empty, one wrist camera per arm in the bottom row.
DUAL_ARM_LAYOUT: list[list[str | None]] = [["scene_rgb", None], ["left_wrist_rgb", "right_wrist_rgb"]]


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize ``(T, h, w, 3)`` uint8 frames to ``size`` = (height, width). No-op if they already match."""
    height, width = size
    if frames.shape[1:3] == (height, width):
        return frames
    return np.stack(
        [np.asarray(PIL.Image.fromarray(frame).resize((width, height), PIL.Image.BILINEAR)) for frame in frames]
    )


def tile_views(
    views: Mapping[str, np.ndarray],
    layout: Sequence[Sequence[str | None]] = DUAL_ARM_LAYOUT,
    cell_size: Sequence[int] = CELL_SIZE,
) -> np.ndarray:
    """Tile ``(T, h, w, 3)`` uint8 views into ``(T, rows * cell_h, cols * cell_w, 3)`` following ``layout``.

    ``views`` maps a view name to its frames. All given views must have the same number of frames. Layout slots
    that are ``None`` or name a view not present in ``views`` are filled with black.
    """
    cell_h, cell_w = cell_size
    placed = {name for row in layout for name in row if name is not None}
    unknown = set(views) - placed
    if unknown:
        msg = f"Views {sorted(unknown)} are not part of the tile layout {layout}."
        raise ValueError(msg)
    if not views:
        msg = "Need at least one view to tile."
        raise ValueError(msg)

    lengths = {name: len(frames) for name, frames in views.items()}
    if len(set(lengths.values())) != 1:
        msg = f"All views need the same number of frames, got {lengths}."
        raise ValueError(msg)
    num_frames = next(iter(lengths.values()))

    n_rows, n_cols = len(layout), max(len(row) for row in layout)
    out = np.zeros((num_frames, n_rows * cell_h, n_cols * cell_w, 3), dtype=np.uint8)
    for r, row in enumerate(layout):
        for c, name in enumerate(row):
            if name is None or name not in views:
                continue
            out[:, r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w] = resize_frames(
                np.asarray(views[name], dtype=np.uint8), (cell_h, cell_w)
            )
    return out
