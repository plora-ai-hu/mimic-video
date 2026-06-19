"""Convert a LeRobot v3.0 dataset -> mimic zarr (same format as process_so100.py).

Runs in the lerobot5.0 env (has v3.0-capable LeRobotDataset + cv2 + zarr).
Reads via LeRobotDataset (handles v3.0 layout + video decoding), writes one .zarr
per episode: workspace_rgb (JPEG@store-fps), joint_state/joint_action lowdim (full rate),
language_instruction — all with ns timestamps.
"""
import pathlib
import numpy as np
import cv2
import zarr
import numcodecs
from numcodecs import Blosc
from lerobot.datasets.lerobot_dataset import LeRobotDataset

REPO = "pranavsaroha/MimicFinetunerealdataset_20260618_172416"
CAM = "observation.images.front"
OUT = pathlib.Path("/home/pranavsaroha/so100_finetune3_zarr")
SAFE = "pranavsaroha__MimicFinetunereal_0618"
STORE_FPS = 10.0
JPEG_Q = 95
S_TO_NS = 1_000_000_000

ds = LeRobotDataset(REPO)
fps = ds.fps
keep_step = max(1, round(fps / STORE_FPS))
efrom = list(ds.meta.episodes["dataset_from_index"])
eto = list(ds.meta.episodes["dataset_to_index"])
hf = ds.hf_dataset
out_ds = OUT / SAFE
out_ds.mkdir(parents=True, exist_ok=True)
comp = Blosc(cname="lz4", clevel=1, shuffle=Blosc.BITSHUFFLE)
print(f"fps={fps} keep_step={keep_step} episodes={len(efrom)} cam={CAM}", flush=True)

n_ok = 0
for ep, (f0, f1) in enumerate(zip(efrom, eto)):
    n = int(f1 - f0)
    if n < 3:
        print(f"ep {ep}: too short ({n}) skip", flush=True); continue
    rows = hf[f0:f1]
    state = np.asarray(rows["observation.state"], dtype=np.float32)
    action = np.asarray(rows["action"], dtype=np.float32)
    ts_ns = (np.arange(n, dtype=np.float64) / fps * S_TO_NS).astype(np.uint64)

    keep = list(range(0, n, keep_step))
    jpegs, img_ts, lang = [], [], None
    for k in keep:
        item = ds[f0 + k]
        if lang is None:
            lang = item.get("task", "")
        img = item[CAM]                                   # (3,H,W) float[0,1] RGB
        img = (img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
        jpegs.append(buf.tobytes()); img_ts.append(ts_ns[k])
    img_ts = np.asarray(img_ts, dtype=np.uint64)
    lang_b = (lang or "no_instruction").encode("utf-8")

    out_path = out_ds / f"episode_{ep:06d}.zarr"
    with zarr.open(str(out_path), "w") as root:
        rgb = root.create_dataset("workspace_rgb", shape=(len(jpegs),), chunks=(len(jpegs),),
                                  dtype=object, object_codec=numcodecs.VLenBytes())
        rgb[:] = np.array(jpegs, dtype=object)
        root.create_dataset("workspace_rgb_timestamps", shape=(len(img_ts),), chunks=(len(img_ts),),
                            dtype="uint64", compressor=comp)[:] = img_ts
        for name, arr in [("joint_state_lowdim", state), ("joint_action_lowdim", action)]:
            root.create_dataset(name, shape=arr.shape, chunks=(min(1024, n), arr.shape[1]),
                                dtype=np.float32, compressor=comp)[:] = arr
            root.create_dataset(f"{name}_timestamps", shape=(n,), chunks=(n,),
                                dtype="uint64", compressor=comp)[:] = ts_ns
        la = np.array([lang_b])
        root.create_dataset("language_instruction", shape=(1,), chunks=(1,),
                            dtype=la.dtype, compressor=comp)[:] = la
        root.create_dataset("language_instruction_timestamps", shape=(1,), chunks=(1,),
                            dtype="uint64", compressor=comp)[:] = np.array([0], dtype=np.uint64)
    n_ok += 1
    if ep % 10 == 0 or ep == len(efrom) - 1:
        print(f"ep {ep}: {len(jpegs)} frames / {n} rows  (lang={lang!r})", flush=True)
print(f"DONE: {n_ok}/{len(efrom)} episodes -> {out_ds}", flush=True)
