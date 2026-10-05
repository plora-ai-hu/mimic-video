"""SO-101 robot client: drives one or two LeRobot SO followers with action chunks from ``so100/so101_policy_server.py``.

Runs on the machine with the robot, in a LeRobot environment (only ``so100/so101_protocol.py`` is needed from this
repository). The server is reached through the login node: a reverse tunnel from the compute node
(``srun --jobid=<job id> --overlap --pty ssh -N -R 8766:localhost:8766 <user>@<login node IP>``, see SO101.md) and
a local tunnel from this machine, e.g.

    ssh -N -L 8766:localhost:8766 <user>@komondor.hpc.dkf.hu
    export MIMIC_POLICY_TOKEN=...   # same value as in the server's secrets.env
    python so100/so101_robot_client.py --port /dev/ttyACM0 \
        --camera top=/dev/video0 --camera wrist=/dev/video2 \
        --view top=scene_rgb --view wrist=right_wrist_rgb --rotate top=180 --dry-run

Two arms (12 joints, LeRobot's bimanual SO follower with joints ``left_*`` / ``right_*`` and calibrations
``<cal id>_left`` / ``<cal id>_right``): pass ``--left-port`` and ``--right-port`` instead of ``--port``, e.g.

    python so100/so101_robot_client.py --left-port /dev/ttyACM0 --right-port /dev/ttyACM1 --cal-id bi_so_follower \
        --camera top=/dev/video0 --camera left=/dev/video2 --camera right=/dev/video4 \
        --view top=scene_rgb --view left=left_wrist_rgb --view right=right_wrist_rgb --dry-run

Control loop, synchronous at the policy rate (5 Hz): every tick reads the joints and all cameras into an ``n_obs``
frame history. When no targets are pending, a query is sent in the background while the loop keeps ticking with the
arm holding still; then the first ``--exec-steps`` targets of the chunk are sent, one per tick.

``--replay EPISODE.safetensors`` needs no robot: it feeds a converted episode (``so100/process_so100_v3.py``) to the
server and prints the error of each chunk against the recorded actions.

``--replay-robot EPISODE.safetensors`` needs no server: it plays the recorded actions of a converted episode on the
arms at their recorded timing (no cameras needed), after easing the arms into the first recorded pose, and prints how
far the measured joints were from the recorded joint states. Use it to check that the robot's calibration matches the
one used for recording; with ``--dry-run`` it only prints the distance to the first pose.
"""

import argparse
import concurrent.futures
import json
import os
import socket
import sys
import time
from collections import deque

import cv2
import numpy as np
import so101_protocol as proto


def parse_mapping(spec: str) -> tuple[str, str]:
    key, sep, value = spec.partition("=")
    if not sep or not key or not value:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {spec!r}")
    return key, value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=proto.DEFAULT_PORT)
    p.add_argument("--token", help=f"server token (default: ${proto.TOKEN_ENV})")
    p.add_argument("--port", default="/dev/ttyACM0", help="follower arm serial port (one arm)")
    p.add_argument("--left-port", help="left follower serial port (two arms, with --right-port)")
    p.add_argument("--right-port", help="right follower serial port (two arms, with --left-port)")
    p.add_argument("--cal-id", help="LeRobot calibration id (default: so_follower, or bi_so_follower for two arms)")
    p.add_argument("--camera", type=parse_mapping, action="append", default=[], help="NAME=/dev/videoN, repeatable")
    p.add_argument("--view", type=parse_mapping, action="append", default=[], help="camera NAME=trained view name")
    p.add_argument(
        "--rotate", type=parse_mapping, action="append", default=[], help="camera NAME=DEGREES (90, 180, -90)"
    )
    p.add_argument("--cam-width", type=int, default=640)
    p.add_argument("--cam-height", type=int, default=480)
    p.add_argument("--cam-fps", type=int, default=30)
    p.add_argument("--exec-steps", type=int, default=5, help="targets executed per chunk")
    p.add_argument("--max-rel", type=float, default=5.0, help="max joint change per command (degrees)")
    p.add_argument("--stop-step", type=int, help="video denoising step, overrides the server default")
    p.add_argument("--cycles", type=int, default=0, help="number of chunks, 0 = until Ctrl-C")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--dry-run", action="store_true", help="query and print targets without moving the arm")
    p.add_argument("--replay", help="episode .safetensors to feed instead of the robot")
    p.add_argument("--replay-robot", help="episode .safetensors whose recorded actions are played on the arms")
    args = p.parse_args()
    if args.replay and args.replay_robot:
        p.error("--replay and --replay-robot are exclusive")
    if (args.left_port is None) != (args.right_port is None):
        p.error("--left-port and --right-port go together")
    args.bimanual = args.left_port is not None
    if args.cal_id is None:
        args.cal_id = "bi_so_follower" if args.bimanual else "so_follower"
    return args


class PolicyClient:
    def __init__(self, host: str, port: int, token: str, jpeg_quality: int):
        self.sock = socket.create_connection((host, port), timeout=30)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.info = proto.call(self.sock, {"cmd": "hello", "token": token})
        self.jpeg = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    def predict(self, joints: np.ndarray, history: dict[str, list[np.ndarray]], stop_step: int | None) -> dict:
        blobs = []
        for frames in history.values():
            for rgb in frames:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), self.jpeg)
                if not ok:
                    raise RuntimeError("JPEG encoding failed")
                blobs.append(buf.tobytes())
        header = {
            "cmd": "predict",
            "joints": [float(j) for j in joints],
            "views": {name: len(frames) for name, frames in history.items()},
            "stop_step": stop_step,
        }
        reply = proto.call(self.sock, header, blobs)
        reply["chunk"] = np.asarray(reply["chunk"], dtype=np.float32)
        return reply

    def close(self) -> None:
        self.sock.close()


def replay(args: argparse.Namespace, client: PolicyClient) -> None:
    from safetensors.numpy import load_file

    ep = load_file(args.replay)
    info = client.info
    n_obs, hz, chunk_len = info["n_obs"], info["hz"], info["chunk_len"]
    views = [v for v in info["views"] if v in ep]
    t_img = ep[f"{views[0]}_timestamps"].astype(np.float64) / 1e9
    t_low = ep["joint_state_lowdim_timestamps"].astype(np.float64) / 1e9
    state, action = ep["joint_state_lowdim"], ep["joint_action_lowdim"]

    def interp(values: np.ndarray, t: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(t, t_low, values[:, d]) for d in range(values.shape[1])], axis=-1)

    horizon = np.arange(1, chunk_len + 1) / hz  # target i is for (i + 1) / hz after the newest frame
    errors = []
    for k in range(0, len(t_img), args.exec_steps):
        if args.cycles and len(errors) >= args.cycles:
            break
        idx = np.clip(np.arange(k - n_obs + 1, k + 1), 0, None)  # left-pad by repeating the first frame
        history = {v: list(ep[v][idx]) for v in views}
        joints = interp(state, np.array([t_img[k]]))[0]
        reply = client.predict(joints, history, args.stop_step)
        target = interp(action, np.minimum(t_img[k] + horizon, t_low[-1]))
        err = np.abs(reply["chunk"] - target)
        errors.append(err)
        print(
            f"[t={t_img[k]:6.1f}s] {reply['latency']:.2f} s  mean |err| {err.mean():5.2f} deg"
            f"  first {np.round(err[0], 1)}  last {np.round(err[-1], 1)}",
            flush=True,
        )
    errors = np.stack(errors)
    print(
        f"mean |err| per joint (deg): {dict(zip(info['joint_names'], np.round(errors.mean((0, 1)), 2), strict=True))}"
    )
    print(f"mean |err| per step (deg):  {np.round(errors.mean((0, 2)), 2)}")


def replay_robot(args: argparse.Namespace) -> None:
    from safetensors import safe_open
    from safetensors.numpy import load_file

    ep = load_file(args.replay_robot)
    with safe_open(args.replay_robot, "np") as f:
        meta = f.metadata() or {}
    state, action = ep["joint_state_lowdim"], ep["joint_action_lowdim"]
    t_low = ep["joint_state_lowdim_timestamps"].astype(np.float64) / 1e9
    t_low -= t_low[0]
    names = json.loads(meta.get("joint_action_names", "null"))

    robot = make_robot(args)
    ports = f"left {args.left_port}, right {args.right_port}" if args.bimanual else args.port
    print(f">>> connecting robot ({ports}, calibration {args.cal_id}) ...", flush=True)
    robot.connect()
    try:
        motors = [key.removesuffix(".pos") for key in robot.action_features]
        if names is None:
            if len(motors) != action.shape[1]:
                raise SystemExit(f"episode has no joint names, D={action.shape[1]}, robot has {len(motors)} joints")
            print(f">>> episode has no joint names, assuming the robot order {motors}", flush=True)
            names = motors
        names = [n.removesuffix(".pos") for n in names]
        if motors != names:
            raise SystemExit(f"robot joints {motors} differ from the recorded joints {names}")

        def read_joints() -> np.ndarray:
            obs = robot.get_observation()
            return np.array([obs[f"{name}.pos"] for name in names], dtype=np.float32)

        joints = read_joints()
        print(f">>> episode {args.replay_robot}: {len(action)} actions over {t_low[-1]:.1f} s", flush=True)
        print(f"   state       {np.round(joints, 1)}\n   first pose  {np.round(action[0], 1)}", flush=True)
        print(f"   delta       {np.round(action[0] - joints, 1)}", flush=True)
        if args.dry_run:
            return
        input(">>> Clear workspace, hand near the power switch. ENTER to move to the first pose, Ctrl-C to stop.")

        # Ease into the first pose at about 30 deg/s, slower than --max-rel alone would allow.
        deadline = time.monotonic() + 20.0
        while np.abs(action[0] - joints).max() > 2.0 and time.monotonic() < deadline:
            send_action(robot, names, joints + np.clip(action[0] - joints, -1.0, 1.0))
            time.sleep(1 / 30)
            joints = read_joints()
        print(f">>> at first pose, max delta {np.abs(action[0] - joints).max():.1f} deg", flush=True)
        input(">>> ENTER to play the episode, Ctrl-C to stop.")

        errors = []
        start = time.monotonic()
        for i in range(len(action)):
            delay = start + t_low[i] - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            send_action(robot, names, action[i])
            errors.append(np.abs(read_joints() - state[i]))
            if i % 30 == 0:
                print(f"[t={t_low[i]:6.1f}s] |measured - recorded state| {np.round(errors[-1], 1)}", flush=True)
        per_joint = dict(zip(names, np.round(np.stack(errors).mean(0), 2), strict=True))
        print(f"mean |measured - recorded state| per joint (deg): {per_joint}")
    except KeyboardInterrupt:
        print("\n>>> stopped", flush=True)
    finally:
        robot.disconnect()
        print(">>> robot disconnected", flush=True)


def make_robot(args: argparse.Namespace):
    from lerobot.cameras.configs import ColorMode, Cv2Backends, Cv2Rotation
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig, SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    rotation_of = {name: Cv2Rotation(int(degrees)) for name, degrees in args.rotate}
    cameras = {
        name: OpenCVCameraConfig(
            index_or_path=path,
            fps=args.cam_fps,
            width=args.cam_width,
            height=args.cam_height,
            color_mode=ColorMode.RGB,
            backend=Cv2Backends.V4L2,
            rotation=rotation_of.get(name, Cv2Rotation.NO_ROTATION),
        )
        for name, path in args.camera
    }
    max_rel = None if args.dry_run else args.max_rel
    if not args.bimanual:
        config = SOFollowerRobotConfig(port=args.port, id=args.cal_id, cameras=cameras, max_relative_target=max_rel)
        return SOFollower(config)

    from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig

    # Cameras are top-level, so their observation keys stay unprefixed (no left_/right_).
    config = BiSOFollowerConfig(
        id=args.cal_id,
        left_arm_config=SOFollowerConfig(port=args.left_port, max_relative_target=max_rel),
        right_arm_config=SOFollowerConfig(port=args.right_port, max_relative_target=max_rel),
        cameras=cameras,
    )
    return BiSOFollower(config)


def send_action(robot, joint_names: list[str], target: np.ndarray) -> None:
    action = {f"{name}.pos": float(value) for name, value in zip(joint_names, target, strict=True)}
    for attempt in range(3):  # retry transient Feetech bus errors
        try:
            robot.send_action(action)
            return
        except Exception as e:
            if attempt == 2:
                raise
            print(f"      bus error, retry {attempt + 1}/2: {str(e)[:60]}", flush=True)
            time.sleep(0.05)


def control(args: argparse.Namespace, client: PolicyClient) -> None:
    info = client.info
    joint_names, n_obs, dt = info["joint_names"], info["n_obs"], 1.0 / info["hz"]
    view_of = dict(args.view)
    if set(view_of) != {name for name, _ in args.camera}:
        raise SystemExit("--view must map every --camera name")
    if set(dict(args.rotate)) - set(view_of):
        raise SystemExit("--rotate names must be --camera names")
    if set(view_of.values()) - set(info["views"]):
        raise SystemExit(f"views {sorted(view_of.values())} not among the trained views {info['views']}")

    robot = make_robot(args)
    ports = f"left {args.left_port}, right {args.right_port}" if args.bimanual else args.port
    print(f">>> connecting robot ({ports}, calibration {args.cal_id}) and cameras ...", flush=True)
    robot.connect()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        motors = [key.removesuffix(".pos") for key in robot.action_features]
        if motors != joint_names:
            raise SystemExit(f"robot joints {motors} differ from the trained joints {joint_names}")
        proto.call(client.sock, {"cmd": "reset"})
        mode = "DRY-RUN (no motion)" if args.dry_run else "EXECUTE"
        print(f">>> ready, mode {mode}, prompt {info['prompt']!r}", flush=True)
        if not args.dry_run:
            input(">>> Arms in a safe pose, keep a hand near the power switch. ENTER to start, Ctrl-C to stop.")

        history = {view: deque(maxlen=n_obs) for view in view_of.values()}
        pending: deque[np.ndarray] = deque()
        query = None
        chunks = 0
        next_tick = time.monotonic()
        while not args.cycles or chunks < args.cycles or pending or query:
            obs = robot.get_observation()
            joints = np.array([obs[f"{name}.pos"] for name in joint_names], dtype=np.float32)
            for camera, view in view_of.items():
                frames = history[view]
                frames.append(obs[camera])
                while len(frames) < n_obs:  # left-pad at start by repeating the first frame
                    frames.appendleft(frames[0])

            if pending:
                target = pending.popleft()
                print(f"   target {np.round(target, 1)}  delta {np.round(target - joints, 1)}", flush=True)
                if not args.dry_run:
                    send_action(robot, joint_names, target)
            elif query is None:
                if not args.cycles or chunks < args.cycles:
                    snapshot = {view: list(frames) for view, frames in history.items()}
                    query = executor.submit(client.predict, joints.copy(), snapshot, args.stop_step)
            elif query.done():
                reply = query.result()
                query = None
                chunks += 1
                print(f"[chunk {chunks}] {reply['latency']:.2f} s  state {np.round(joints, 1)}", flush=True)
                pending.extend(reply["chunk"][: args.exec_steps])

            next_tick += dt
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\n>>> stopped", flush=True)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        robot.disconnect()
        print(">>> robot disconnected", flush=True)


def main() -> None:
    args = parse_args()
    if args.replay_robot:
        replay_robot(args)
        return
    token = args.token or os.environ.get(proto.TOKEN_ENV, "")
    if not token:
        sys.exit(f"Set {proto.TOKEN_ENV} or pass --token.")
    client = PolicyClient(args.server, args.server_port, token, args.jpeg_quality)
    print(f">>> policy server: {client.info}", flush=True)
    try:
        if args.replay:
            replay(args, client)
        else:
            control(args, client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
