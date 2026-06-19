"""Convert SO100 LeRobot datasets (v2.0/v2.1) -> mimic-video action zarr.

Design notes
------------
- SO100 is joint-controlled: action and observation.state are both 6-D
  (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper).
  We store them as JOINT_POS lowdim. No EEF/pose/IK machinery.
- Video frames are stored as JPEG-encoded bytes in an object-dtype zarr array
  (numcodecs.VLenBytes). mimic-video's ChunkReader already decodes this path:
  `if values.dtype == object and obs_type in COLOR_VISUAL: cv2.imdecode(...)`.
  This keeps disk ~mp4-scale instead of the ~200x blowup of raw uint8 frames.
- Frames are subsampled to --store-fps (the policy only consumes ~5-10 Hz; the
  ChunkReader interpolates/picks-nearest from whatever rate we store).
- Lowdim (joints) are kept at the native rate (tiny) with their own timestamps.
- Frames are stored RGB (decoded BGR->RGB before imencode) to match the bridge
  converter's PIL-RGB convention; cv2 encode->decode round-trips the array.

Output layout (one .zarr group per episode), matching process_bridge keys:
    <out>/<safe_repo>/episode_<idx>.zarr/
        workspace_rgb                 (n_kept,)  object  JPEG bytes  [RGB]
        workspace_rgb_timestamps      (n_kept,)  uint64  ns
        joint_state_lowdim            (T,6)      f32
        joint_state_lowdim_timestamps (T,)       uint64  ns
        joint_action_lowdim           (T,6)      f32
        joint_action_lowdim_timestamps(T,)       uint64  ns
        language_instruction          (1,)       bytes
        language_instruction_timestamps (1,)     uint64
"""

import argparse
import json
import logging
import pathlib
import re
import sys

import av
import cv2
import numcodecs
import numpy as np
import pyarrow.parquet as pq
import tqdm
import zarr
from numcodecs import Blosc

S_TO_NS = 1_000_000_000
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("process_so100")

EP_RE = re.compile(r"episode_(\d+)\.(?:parquet|mp4)$")


def _index_by_episode(paths) -> dict[int, pathlib.Path]:
    out = {}
    for p in paths:
        m = EP_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def _read_parquet(fp: pathlib.Path):
    df = pq.read_table(str(fp)).to_pandas()
    state = np.asarray(df["observation.state"].tolist(), dtype=np.float32)
    action = np.asarray(df["action"].tolist(), dtype=np.float32)
    ts = df["timestamp"].to_numpy(dtype=np.float64)
    return state, action, ts


def _decode_jpeg_frames(mp4: pathlib.Path, keep_step: int, quality: int):
    """Sequentially decode mp4, keep every keep_step-th frame as RGB JPEG bytes.

    Returns (list_of_jpeg_bytes, list_of_kept_frame_indices, total_frame_count).
    Only kept frames are held in memory.
    """
    jpegs, kept_idx = [], []
    i = 0
    enc_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    # PyAV decodes both h264 and av1 (its wheels bundle a dav1d AV1 decoder),
    # unlike the cv2/ffmpeg build available here. Frames come in presentation order.
    with av.open(str(mp4)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            if i % keep_step == 0:
                rgb = frame.to_ndarray(format="rgb24")
                ok2, buf = cv2.imencode(".jpg", rgb, enc_params)
                if not ok2:
                    raise RuntimeError(f"jpeg encode failed on frame {i} of {mp4}")
                jpegs.append(buf.tobytes())
                kept_idx.append(i)
            i += 1
    return jpegs, np.array(kept_idx, dtype=np.int64), i


def _ts_ns(ts_s: np.ndarray) -> np.ndarray:
    rel = ts_s - ts_s[0]
    return np.round(rel * S_TO_NS).astype(np.uint64)


def convert_dataset(raw_dir: pathlib.Path, cam: str, repo: str, out_root: pathlib.Path,
                    store_fps: float, quality: int, max_episodes: int | None) -> tuple[int, int]:
    info = json.loads((raw_dir / "meta/info.json").read_text())
    fps = float(info["fps"])
    keep_step = max(1, round(fps / store_fps))

    # task_index -> task string; some datasets store episodes.jsonl "tasks" as
    # integer indices into tasks.jsonl rather than the strings themselves.
    task_map = {}
    tj = raw_dir / "meta/tasks.jsonl"
    if tj.exists():
        for line in tj.open():
            t = json.loads(line)
            task_map[int(t["task_index"])] = str(t.get("task", "")).strip()

    def _resolve(tasks_field):
        if not tasks_field:
            return ""
        t0 = tasks_field[0]
        return t0.strip() if isinstance(t0, str) else task_map.get(int(t0), "").strip()

    # episode_index -> instruction
    ep_lang = {}
    with (raw_dir / "meta/episodes.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            ep_lang[int(d["episode_index"])] = _resolve(d.get("tasks"))

    parquet_by_ep = _index_by_episode((raw_dir / "data").rglob("episode_*.parquet"))
    vid_dir_glob = list((raw_dir / "videos").rglob(f"{cam}/episode_*.mp4"))
    mp4_by_ep = _index_by_episode(vid_dir_glob)

    common = sorted(set(parquet_by_ep) & set(mp4_by_ep))
    if max_episodes:
        common = common[:max_episodes]
    if not common:
        log.warning(f"[{repo}] no matching episodes for cam={cam} "
                    f"(parquet={len(parquet_by_ep)}, mp4={len(mp4_by_ep)}) -- SKIP")
        return 0, 0

    safe = repo.replace("/", "__")
    out_ds = out_root / safe
    out_ds.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    for ep in common:
        try:
            state, action, ts = _read_parquet(parquet_by_ep[ep])
            jpegs, kept_idx, n_frames = _decode_jpeg_frames(mp4_by_ep[ep], keep_step, quality)

            # align lowdim rows to decoded frame count
            n = min(len(ts), n_frames)
            if abs(len(ts) - n_frames) > 2:
                log.warning(f"[{repo} ep{ep}] frame/row mismatch frames={n_frames} rows={len(ts)} -> trunc {n}")
            state, action, ts = state[:n], action[:n], ts[:n]
            kmask = kept_idx < n
            jpegs = [j for j, k in zip(jpegs, kmask) if k]
            kept_idx = kept_idx[kmask]
            if n < 3 or len(jpegs) < 3:
                log.warning(f"[{repo} ep{ep}] too short (n={n}, frames={len(jpegs)}) -- skip ep")
                continue

            ts_ns = _ts_ns(ts)
            img_ts = ts_ns[kept_idx]
            lang = (ep_lang.get(ep, "") or "no_instruction").encode("utf-8")

            out_path = out_ds / f"episode_{ep:06d}.zarr"
            comp = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
            root: zarr.Group
            with zarr.open(str(out_path), "w") as root:
                rgb = root.create_dataset("workspace_rgb", shape=(len(jpegs),), chunks=(len(jpegs),),
                                          dtype=object, object_codec=numcodecs.VLenBytes())
                rgb[:] = np.array(jpegs, dtype=object)
                root.create_dataset("workspace_rgb_timestamps", shape=(len(img_ts),),
                                    chunks=(len(img_ts),), dtype="uint64", compressor=comp)[:] = img_ts

                for name, arr in [("joint_state_lowdim", state), ("joint_action_lowdim", action)]:
                    root.create_dataset(name, shape=arr.shape, chunks=(min(1024, n), arr.shape[1]),
                                        dtype=np.float32, compressor=comp)[:] = arr
                    root.create_dataset(f"{name}_timestamps", shape=(n,), chunks=(n,),
                                        dtype="uint64", compressor=comp)[:] = ts_ns

                la = np.array([lang])  # dtype |S{len}
                root.create_dataset("language_instruction", shape=(1,), chunks=(1,),
                                    dtype=la.dtype, compressor=comp)[:] = la
                root.create_dataset("language_instruction_timestamps", shape=(1,), chunks=(1,),
                                    dtype="uint64", compressor=comp)[:] = np.array([0], dtype=np.uint64)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            log.warning(f"[{repo} ep{ep}] FAILED: {type(e).__name__}: {e}")
    log.info(f"[{repo}] wrote {n_ok}/{len(common)} episodes  (cam={cam}, keep_step={keep_step}) -> {out_ds}")
    return n_ok, len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=pathlib.Path, help="TSV: raw_dir<TAB>cam<TAB>repo per line")
    ap.add_argument("--raw-dir", type=pathlib.Path)
    ap.add_argument("--cam", type=str)
    ap.add_argument("--repo", type=str)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--store-fps", type=float, default=10.0)
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--max-datasets", type=int, default=None)
    ap.add_argument("--max-episodes", type=int, default=None, help="per-dataset cap (for quick tests)")
    args = ap.parse_args()

    jobs = []
    if args.list:
        for line in args.list.read_text().splitlines():
            if not line.strip():
                continue
            d, cam, repo = line.split("\t")
            jobs.append((pathlib.Path(d), cam, repo))
    else:
        jobs.append((args.raw_dir, args.cam, args.repo))
    if args.max_datasets:
        jobs = jobs[: args.max_datasets]

    tot_ep = 0
    for raw_dir, cam, repo in tqdm.tqdm(jobs, desc="datasets"):
        n, _ = convert_dataset(raw_dir, cam, repo, args.out, args.store_fps, args.jpeg_quality, args.max_episodes)
        tot_ep += n
    log.info(f"DONE: {tot_ep} episodes from {len(jobs)} datasets -> {args.out}")


if __name__ == "__main__":
    main()
