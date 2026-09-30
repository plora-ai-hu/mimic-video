"""Run a LIBERO task locally in MuJoCo's native viewer, driven by a remote mimic-video policy.

Runs on a Linux laptop. The GPU box runs policy_server.py; reach it with
    ssh -L 8766:localhost:8766 <gpu-host>
then
    python viewer_client.py --task 0

The control loop mirrors run.py's run_episode: settle for 10 no-op steps, then send the
agent-view image and proprioception every step and execute the action the server returns.
The only deliberate difference: the env is built with hard_reset=False so the viewer can
keep one model across episodes (robosuite otherwise recompiles the model on every reset).

Viewer (MuJoCo's own; its keyboard shortcuts are left untouched):
  mouse             left-drag orbit, right-drag pan, scroll zoom, double-click select a body
  Ctrl+right-drag   push the selected body   (Ctrl+left-drag rotates it)
  [ / ]             cycle cameras (agentview is what the policy sees)
Episode control is typed in the terminal (then Enter):
  p  pause / resume     r  restart this init state
  n  next init state    t  next task
"""

from __future__ import annotations

import json
import os
import pathlib
import queue
import struct
import sys
import threading
import time

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from websockets.sync.client import connect

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "LIBERO"))  # as eval.sh's PYTHONPATH=LIBERO
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

CAMERA_HEIGHT, CAMERA_WIDTH = 480, 640  # as run.py
DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]  # as run.py
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300}  # as run.py
CONTROL_HZ = 20
COMMANDS = {"p": "pause", "r": "restart", "n": "next_init", "t": "next_task"}


class PolicyClient:
    def __init__(self, url: str) -> None:
        self.ws = connect(url, max_size=2**24, open_timeout=30, ping_timeout=None)

    def request(self, **msg) -> dict:
        self.ws.send(json.dumps(msg))
        return json.loads(self.ws.recv())

    def act(self, obs: dict) -> dict:
        image = obs["agentview_image"][::-1, ::-1]  # as run.py's get_libero_image
        ok, png = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR))
        assert ok
        header = json.dumps({
            "eef_pos": obs["robot0_eef_pos"].tolist(),
            "eef_quat": obs["robot0_eef_quat"].tolist(),
            "gripper_qpos": obs["robot0_gripper_qpos"].tolist(),
        }).encode()
        self.ws.send(struct.pack("<I", len(header)) + header + png.tobytes())
        return json.loads(self.ws.recv())


def read_commands(commands: queue.Queue[str]) -> None:
    """Terminal input, since MuJoCo's viewer binds most letter keys to render toggles."""
    for line in sys.stdin:
        if line.strip().lower() in COMMANDS:
            commands.put(COMMANDS[line.strip().lower()])


def run_episode(env, viewer, client: PolicyClient, obs: dict, commands: queue.Queue[str],
                max_steps: int, num_steps_wait: int) -> str:
    """Step until success or timeout, then idle; returns the command that ends the episode."""
    step, paused, result, t_next = 0, False, None, time.time()
    while viewer.is_running():
        while not commands.empty():
            command = commands.get()
            if command != "pause":
                return command
            paused = not paused
            print("paused" if paused else "resumed")
        if paused or result:
            viewer.sync()
            time.sleep(1 / CONTROL_HZ)
            continue

        if step < num_steps_wait:
            action = DUMMY_ACTION
        else:
            reply = client.act(obs)
            action = reply["action"]
            if reply["queried"]:
                print(f"  step {step - num_steps_wait:3d}: queried policy ({reply['latency']:.2f} s)")
        with viewer.lock():
            obs, _, done, _ = env.step(action)
        viewer.sync()  # also applies Ctrl+drag perturbation forces
        step += 1

        if done:
            result = "SUCCESS"
        elif step >= max_steps + num_steps_wait:
            result = "timeout"
        if result:
            print(f"  -> {result} after {step - num_steps_wait} steps.  r: retry  n: next init  t: next task")

        t_next = max(t_next + 1 / CONTROL_HZ, time.time() - 0.2)
        time.sleep(max(0.0, t_next - time.time()))
    return "closed"


def main(server: str = "ws://localhost:8766", task: int = 0, episode: int = 0, num_steps_wait: int = 10) -> None:
    client = PolicyClient(server)
    info = client.request(type="tasks")
    tasks, max_steps = info["tasks"], MAX_STEPS[info["suite"]]
    print(f"Connected. Suite {info['suite']}, {len(tasks)} tasks.")
    print("Commands (type + Enter): p pause, r restart, n next init state, t next task")

    commands: queue.Queue[str] = queue.Queue()
    threading.Thread(target=read_commands, args=(commands,), daemon=True).start()

    command = "next_task"
    while command == "next_task":
        spec = tasks[task]
        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(get_libero_path("bddl_files"), spec["bddl"]),
            camera_heights=CAMERA_HEIGHT,
            camera_widths=CAMERA_WIDTH,
            hard_reset=False,
        )
        env.seed(0)
        env.reset()  # builds the sim; with hard_reset=False later resets keep this model
        viewer = mujoco.viewer.launch_passive(env.sim.model._model, env.sim.data._data)

        command = "restart"
        while command in ("restart", "next_init"):
            state = np.asarray(client.request(type="init_state", task=task, episode=episode)["state"])
            client.request(type="reset", task=task)
            with viewer.lock():
                env.reset()
                obs = env.set_init_state(state)
            print(f"\nTask {task}: {spec['prompt']}  |  init state {episode}/{spec['num_init_states']}")
            command = run_episode(env, viewer, client, obs, commands, max_steps, num_steps_wait)
            if command == "next_init":
                episode = (episode + 1) % spec["num_init_states"]

        viewer.close()
        env.close()
        if command == "next_task":
            task, episode = (task + 1) % len(tasks), 0


if __name__ == "__main__":
    tyro.cli(main)
