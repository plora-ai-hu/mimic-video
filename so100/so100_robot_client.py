"""SO100 closed-loop client (run in Mimiclerobot env). Reads laptop camera + joints,
calls the policy server, executes the first few predicted joint targets, re-observes.

Dry-run (no motion):  ~/miniforge3/envs/Mimiclerobot/bin/python /tmp/so100_robot_client.py --dry-run
Execute:              ... (drop --dry-run; add --exec-steps/--max-rel)
"""
import argparse, socket, struct, pickle, time, itertools
import numpy as np
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode

MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ap = argparse.ArgumentParser()
ap.add_argument("--port", default="/dev/ttyACM0")
ap.add_argument("--cam", default="/dev/video0")
ap.add_argument("--cam-w", type=int, default=1920)
ap.add_argument("--cam-h", type=int, default=1080)
ap.add_argument("--cal-id", default="so_follower")
ap.add_argument("--server", default="127.0.0.1"); ap.add_argument("--server-port", type=int, default=9999)
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--n-steps", type=int, default=4)
ap.add_argument("--exec-steps", type=int, default=2)
ap.add_argument("--max-rel", type=float, default=5.0)
ap.add_argument("--cycles", type=int, default=0)     # 0 = run forever
ap.add_argument("--hz", type=float, default=5.0)
ap.add_argument("--nstep", type=int, default=4)   # video denoising steps (lower=faster; 4~5s, 2~3.7s)
args = ap.parse_args()

def _recvall(c, n):
    d = b""
    while len(d) < n:
        p = c.recv(n - len(d))
        if not p: raise ConnectionError
        d += p
    return d
def call(c, o):
    b = pickle.dumps(o); c.sendall(struct.pack(">I", len(b)) + b)
    return pickle.loads(_recvall(c, struct.unpack(">I", _recvall(c, 4))[0]))

cam_cfg = OpenCVCameraConfig(index_or_path=args.cam, fps=30, width=args.cam_w, height=args.cam_h,
                            color_mode=ColorMode.RGB, warmup_s=6)
robot_cfg = SOFollowerRobotConfig(port=args.port, id=args.cal_id, cameras={"laptop": cam_cfg},
                                  max_relative_target=(None if args.dry_run else args.max_rel))
robot = SOFollower(robot_cfg)
print(f">>> connecting robot (port={args.port}, cal id={args.cal_id}) + camera ({args.cam}) ...", flush=True)
robot.connect()
print(">>> robot connected.", flush=True)
sock = socket.create_connection((args.server, args.server_port), timeout=120); sock.settimeout(None)
call(sock, {"cmd": "reset"})
print(f">>> policy server connected. MODE = {'DRY-RUN (no motion)' if args.dry_run else 'EXECUTE'}", flush=True)
if not args.dry_run:
    input(">>> EXECUTE mode. Put arm at a safe rest pose, hand near power. Press ENTER to begin (Ctrl-C to stop)...")

dt = 1.0 / args.hz
cyc = itertools.count() if args.cycles <= 0 else range(args.cycles)
try:
    for c in cyc:
        try:
            obs = robot.get_observation()
            joints = np.array([obs[f"{m}.pos"] for m in MOTORS], dtype=np.float32)
            frame = obs["laptop"]
            t = time.time()
            chunk = np.asarray(call(sock, {"frame": frame, "joints": joints, "nstep": args.nstep})["chunk"])
            print(f"[cycle {c}] predict {time.time()-t:.1f}s  cur={np.round(joints,1)}", flush=True)
            for s in range(min(args.exec_steps, len(chunk))):
                tgt = chunk[s]
                print(f"   step {s}: pred={np.round(tgt,1)}  d={np.round(tgt-joints,1)}", flush=True)
                if not args.dry_run:
                    for attempt in range(3):           # retry transient Feetech bus glitches
                        try:
                            robot.send_action({f"{m}.pos": float(tgt[j]) for j, m in enumerate(MOTORS)}); break
                        except Exception as se:
                            if attempt == 2: raise
                            print(f"      bus glitch, retry {attempt+1}/2: {str(se)[:50]}", flush=True); time.sleep(0.15)
                    time.sleep(dt)
        except Exception as e:
            print(f"[cycle {c}] hiccup, continuing next cycle: {str(e)[:90]}", flush=True); time.sleep(0.3); continue
except KeyboardInterrupt:
    print("\n>>> stopped by user", flush=True)
finally:
    robot.disconnect()
    try: sock.close()
    except Exception: pass
    print(">>> disconnected.", flush=True)
