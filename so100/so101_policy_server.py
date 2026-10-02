"""SO-101 policy server: serves action chunks of a mimic-video action decoder (W2A) over a local TCP port.

Runs in the mimic-video container on a GPU node, from ``model/`` (see ``so100/so101_serve.sbatch``):

    PYTHONPATH=. python ../so100/so101_policy_server.py \
        --experiment w2a_so101_1arm_cabling_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz4 \
        --video-dit <fused backbone .pt, the video_dit_path of the decoder run> \
        --decoder <decoder run>/checkpoints/model/iter_XXXXXXXXX.pt

The robot client (``so100/so101_robot_client.py``) reaches it through an SSH tunnel. Each ``predict`` request carries
the last ``n_obs`` frames of every camera view at 5 Hz plus the current joint positions. The server tiles the views
exactly like training (``cosmos_predict2/data/action/tiling.py``), runs the video backbone up to ``--stop-step`` and
returns the decoder's chunk of absolute joint targets (degrees), the first one 0.2 s after the newest frame.
Protocol: ``so100/so101_protocol.py``. Connections must present the token from ``$MIMIC_POLICY_TOKEN``.
"""

import argparse
import hmac
import json
import os
import pathlib
import socket
import time

import cv2
import numpy as np
import safetensors
import safetensors.torch
import so101_protocol as proto
import torch
from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.tiling import CELL_SIZE, tile_views
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override

DTYPE = torch.bfloat16
# Joint order of LeRobot's SO-100/SO-101 follower, used when the episodes carry no joint names.
SO_FOLLOWER_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, help="W2A experiment the decoder was trained with")
    p.add_argument("--video-dit", required=True, help="fused video backbone checkpoint (the run's video_dit_path)")
    p.add_argument("--decoder", required=True, help="action decoder checkpoint iter_XXXXXXXXX.pt")
    p.add_argument("--stats", help="normalization statistics JSON (default: the one in <data_dir>/.statistics_cache)")
    p.add_argument("--episode", help="episode .safetensors for the prompt embedding and joint names (default: first)")
    p.add_argument("--num-sampling-step", type=int, default=35, help="video denoising schedule length")
    p.add_argument("--stop-step", type=int, default=0, help="video denoising step whose features the decoder reads")
    p.add_argument("--cuda-graphs", action="store_true", help="capture CUDA graphs (slow first query)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=proto.DEFAULT_PORT)
    p.add_argument("--debug-dir", type=pathlib.Path, help="save the tiled input of every query as PNG here")
    return p.parse_args()


def find_stats(data_dir: pathlib.Path) -> pathlib.Path:
    files = sorted((data_dir / ".statistics_cache").glob("*"))
    if len(files) != 1:
        raise SystemExit(f"Found {len(files)} statistics files in {data_dir}/.statistics_cache; pass --stats.")
    return files[0]


def joint_names_of(episode: pathlib.Path, dim: int) -> list[str]:
    """Joint names from the converter metadata, stripped of LeRobot's ``.pos`` suffix."""
    with safetensors.safe_open(str(episode), "np") as f:
        meta = f.metadata() or {}
    names = json.loads(meta.get("joint_state_names", "null"))
    if names is None:
        if dim != len(SO_FOLLOWER_JOINTS):
            raise SystemExit(f"{episode} has no joint names and D={dim}; reconvert with so100/process_so100_v3.py.")
        print(f"    {episode.name} has no joint names, assuming the SO follower order", flush=True)
        names = SO_FOLLOWER_JOINTS
    names = [n.removesuffix(".pos") for n in names]
    if len(names) != dim:
        raise SystemExit(f"{episode}: {len(names)} joint names for D={dim}")
    return names


class Policy:
    def __init__(self, args: argparse.Namespace):
        config = override(make_config(), ["--", f"experiment={args.experiment}"])
        config.model.config.video_pipe_config.guardrail_config.enabled = False
        data_config = instantiate(config.data_config)
        policy_io = data_config.policy_io.policy_io

        tile = next(t for t in data_config.dataset.data_transforms if t.name == "TileViews")
        self.layout = [list(row) for row in tile.layout]
        if list(tile.cell_size) != list(CELL_SIZE):
            raise SystemExit(f"TileViews cell_size {tile.cell_size} differs from tiling.CELL_SIZE {CELL_SIZE}")
        placed = {name for row in self.layout for name in row if name is not None}
        self.views = [k for k in policy_io.obs if k in placed]
        self.n_obs = int(policy_io.obs[self.views[0]].horizon)
        self.hz = float(policy_io.obs[self.views[0]].target_frequency)
        self.chunk_len = int(policy_io.action.joint_action_lowdim.horizon)

        data_dir = pathlib.Path(data_config.dataset.dataset.data_dir)
        stats_path = pathlib.Path(args.stats) if args.stats else find_stats(data_dir)
        episode = pathlib.Path(args.episode) if args.episode else sorted(data_dir.glob("*.safetensors"))[0]
        stats = json.loads(stats_path.read_text())
        self.dim = int(np.asarray(stats["obs/joint_state_lowdim"]["mean"]).shape[-1])
        self.joint_names = joint_names_of(episode, self.dim)
        ep = safetensors.torch.load_file(str(episode))
        self.prompt = ep["language_instruction"][0].numpy().tobytes().decode("utf-8")
        self.prompt_embedding = ep["language_embedding"].to("cuda", DTYPE)
        print(f"    stats {stats_path}\n    prompt {self.prompt!r} from {episode}", flush=True)

        # T5 stays off the GPU: the prompt embedding is precomputed in the episodes.
        self.video = Video2WorldPipeline.from_config(
            config.model.config.video_pipe_config, dit_path=args.video_dit, use_text_encoder=False
        )
        self.action = World2ActionPipeline.from_config(
            config.model.config.pipe_config, dit_path=args.decoder, device="cuda", dtype=DTYPE
        )
        on_meta = [n for n, t in self.action.dit.state_dict().items() if t.is_meta]
        if on_meta:
            raise SystemExit(f"{args.decoder} misses decoder weights, e.g. {on_meta[:3]}")
        # Checkpoint weights are loaded with assign=True and may still sit on the CPU.
        Video2World2ActionPipeline(self.video, self.action).cuda()
        self.action.normalizer.build_from_stats(
            stats,
            normalization_types=extract_normalization_types(policy_io),
            concat_groups=data_config.policy_io.concat_groups,
            device="cuda",
            dtype=DTYPE,
        )
        self.xattn_layer_idx = int(self.action.config.xattn_layer_idx)
        self.num_sampling_step = args.num_sampling_step
        self.default_stop_step = args.stop_step
        self.cuda_graphs = args.cuda_graphs
        self.debug_dir = args.debug_dir
        self.queries = 0

    def info(self) -> dict:
        return {
            "joint_names": self.joint_names,
            "views": self.views,
            "n_obs": self.n_obs,
            "hz": self.hz,
            "chunk_len": self.chunk_len,
            "prompt": self.prompt,
        }

    def decode_views(self, counts: dict, blobs: list[bytes]) -> dict[str, np.ndarray]:
        if set(counts) - set(self.views) or not counts:
            raise ValueError(f"views {sorted(counts)} do not match the trained views {self.views}")
        if sum(counts.values()) != len(blobs) or any(n != self.n_obs for n in counts.values()):
            raise ValueError(f"need {self.n_obs} frames per view")
        views, i = {}, 0
        for name, n in counts.items():
            frames = []
            for blob in blobs[i : i + n]:
                bgr = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"cannot decode a frame of {name}")
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            views[name] = np.stack(frames)
            i += n
        return views

    @torch.no_grad()
    def predict(self, joints: list, views: dict[str, np.ndarray], stop_step: int | None) -> np.ndarray:
        if len(joints) != self.dim:
            raise ValueError(f"expected {self.dim} joints, got {len(joints)}")
        stop_step = self.default_stop_step if stop_step is None else int(stop_step)
        if not 0 <= stop_step <= self.num_sampling_step:
            raise ValueError(f"stop_step must be in [0, {self.num_sampling_step}]")

        tiled = tile_views(views, self.layout, CELL_SIZE)  # (T, 480, 640, 3) uint8, same as TileViews
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            strip = np.concatenate(list(tiled), axis=1)
            cv2.imwrite(str(self.debug_dir / f"query_{self.queries:06d}.png"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
        self.queries += 1

        # CosmosProcessImage: c t h w in [-1, 1]; the backbone needs the full 61-frame clip, zero-padded.
        obs = torch.from_numpy(tiled).permute(3, 0, 1, 2).to("cuda", DTYPE) / 255.0 * 2.0 - 1.0
        vid = torch.zeros((1, 3, 61, *obs.shape[-2:]), device="cuda", dtype=DTYPE)
        vid[:, :, : self.n_obs] = obs
        cross, sigma = self.video.generate_video(
            vid_input=vid,
            is_video_embedding=False,
            num_latent_conditional_frames=2,
            prompt_embedding=self.prompt_embedding,
            guidance=0.0,
            num_sampling_step=self.num_sampling_step,
            seed=0,
            use_cuda_graphs=self.cuda_graphs,
            return_context_at_step=stop_step,
            hidden_state_layer_idx=self.xattn_layer_idx,
        )
        cross = cross.reshape(cross.shape[0], -1, cross.shape[-1])
        state = torch.tensor(joints, dtype=torch.float32).reshape(1, 1, -1).to("cuda", DTYPE)
        chunk = self.action(state, cross, sigma.unsqueeze(1), seed=0, use_cuda_graphs=self.cuda_graphs)
        return chunk.float().cpu().numpy()[0]


def serve_client(conn: socket.socket, policy: Policy, token: str) -> None:
    header, _ = proto.recv(conn)
    if header.get("cmd") != "hello" or not hmac.compare_digest(str(header.get("token", "")), token):
        proto.send(conn, {"ok": False, "error": "bad token"})
        return
    proto.send(conn, {"ok": True, **policy.info()})
    while True:
        header, blobs = proto.recv(conn)
        try:
            if header.get("cmd") == "reset":
                proto.send(conn, {"ok": True})
            elif header.get("cmd") == "predict":
                t0 = time.time()
                views = policy.decode_views(header["views"], blobs)
                chunk = policy.predict(header["joints"], views, header.get("stop_step"))
                latency = time.time() - t0
                print(f"    query {policy.queries}: {latency:.2f} s", flush=True)
                proto.send(conn, {"ok": True, "chunk": chunk.tolist(), "latency": latency})
            else:
                proto.send(conn, {"ok": False, "error": f"unknown cmd {header.get('cmd')!r}"})
        except (ValueError, KeyError, TypeError) as e:
            proto.send(conn, {"ok": False, "error": f"{type(e).__name__}: {e}"})


def main() -> None:
    args = parse_args()
    token = os.environ.get(proto.TOKEN_ENV, "")
    if not token:
        raise SystemExit(f"Set {proto.TOKEN_ENV} (secrets.env); clients must present it.")

    print(">>> loading policy ...", flush=True)
    policy = Policy(args)
    print(f">>> ready: {json.dumps(policy.info())}", flush=True)
    print(f"    GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f">>> listening on {args.host}:{args.port}", flush=True)
    print(f"    on the cluster: srun --jobid=<job id> --overlap --pty ssh -N -R {args.port}:localhost:{args.port} <user>@<login node IP>")
    print(f"    on the laptop:  ssh -N -L {args.port}:localhost:{args.port} <user>@komondor.hpc.dkf.hu")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f">>> client {addr} connected", flush=True)
        try:
            serve_client(conn, policy, token)
        except (ConnectionError, ValueError, json.JSONDecodeError) as e:
            print(f">>> client {addr} gone: {e}", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
