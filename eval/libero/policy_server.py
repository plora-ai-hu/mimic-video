"""Serve the mimic-video LIBERO policy over a WebSocket, for a sim running elsewhere.

The client (viewer_client.py, e.g. on a laptop with MuJoCo's native viewer) runs the
LIBERO env and sends one observation per control step; this server feeds it to
VAMInference.step -- the same call run.py makes -- and returns the action. Steps that
land on an empty action buffer trigger a model query (~3.5 s on an L4); the rest return
immediately from the buffered chunk.

Listens on 127.0.0.1 only; reach it through an SSH tunnel.

Protocol (one client at a time):
  text   {"type": "tasks"}                    -> {"suite", "tasks": [{language, prompt, bddl, num_init_states}]}
  text   {"type": "init_state", task, episode} -> {"state": [...]}
  text   {"type": "reset", task}               -> {"ok": true}       (clears policy history)
  binary <u32 header_len><json header><png>    -> {"action": [7], "queried": bool, "latency": s}
         header = {eef_pos: [3], eef_quat: [4], gripper_qpos: [2]}; png = get_libero_image(obs), lossless
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import struct
import time

import cv2
import numpy as np
import tyro
from libero.libero import benchmark
from websockets.asyncio.server import serve

from run import VAMInference, set_seed_everywhere


def main(
    vam_experiment_name: str,
    vam_video_model_path: str,
    vam_action_model_path: pathlib.Path,
    vam_dataset_statistics_path: pathlib.Path,
    vam_prompt_embeddings_path: pathlib.Path,
    task_suite_name: str = "libero_spatial",
    vam_img_horizon: int = 5,
    vam_lowdim_horizon: int = 1,
    vam_stop_video_denoising_step: int = 0,
    vam_num_execute_actions: int = 5,
    port: int = 8766,
    seed: int = 0,
) -> None:
    set_seed_everywhere(seed)
    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    tasks = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        tasks.append({
            "language": task.language,
            "prompt": task.language.replace("black bowl", "bowl"),  # as run.py's get_libero_env
            "bddl": f"{task.problem_folder}/{task.bddl_file}",
            "num_init_states": len(suite.get_task_init_states(i)),
        })

    print("Loading policy ...", flush=True)
    policy = VAMInference(
        vam_experiment_name, vam_video_model_path, str(vam_action_model_path), vam_dataset_statistics_path,
        vam_img_horizon, vam_lowdim_horizon, vam_stop_video_denoising_step, vam_num_execute_actions,
        rollout_dir=pathlib.Path("."), prompt_embeddings_path=vam_prompt_embeddings_path,
    )
    state = {"prompt": None}

    def act(message: bytes) -> dict:
        (header_len,) = struct.unpack_from("<I", message)
        header = json.loads(message[4 : 4 + header_len])
        png = np.frombuffer(message[4 + header_len :], np.uint8)
        image = cv2.cvtColor(cv2.imdecode(png, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        obs = {
            "robot0_eef_pos": np.asarray(header["eef_pos"], np.float64),
            "robot0_eef_quat": np.asarray(header["eef_quat"], np.float64),
            "robot0_gripper_qpos": np.asarray(header["gripper_qpos"], np.float64),
        }
        queried = policy.action_buffer is None
        t0 = time.time()
        action = policy.step(image, state["prompt"], obs)
        return {"action": action.tolist(), "queried": queried, "latency": time.time() - t0}

    async def handler(ws) -> None:
        print(f"client connected: {ws.remote_address}", flush=True)
        async for message in ws:
            if isinstance(message, bytes):
                reply = await asyncio.to_thread(act, message)
            else:
                req = json.loads(message)
                if req["type"] == "tasks":
                    reply = {"suite": task_suite_name, "tasks": tasks}
                elif req["type"] == "init_state":
                    init_states = suite.get_task_init_states(int(req["task"]))
                    reply = {"state": init_states[int(req["episode"]) % len(init_states)].tolist()}
                elif req["type"] == "reset":
                    state["prompt"] = tasks[int(req["task"])]["prompt"]
                    policy.reset(state["prompt"])
                    reply = {"ok": True}
                else:
                    reply = {"error": f"unknown request {req['type']}"}
            await ws.send(json.dumps(reply))
        print("client disconnected", flush=True)

    async def serve_forever() -> None:
        async with serve(handler, "127.0.0.1", port, max_size=2**24):
            print(f"Policy server on ws://127.0.0.1:{port}  (from a laptop: ssh -L {port}:localhost:{port} <this host>)",
                  flush=True)
            await asyncio.Future()

    asyncio.run(serve_forever())


if __name__ == "__main__":
    tyro.cli(main)
