"""SO100 policy server (run in mimic venv). Loads the fine-tuned VAM and serves
joint-chunk predictions over a localhost socket. No robot I/O here.

  ~/mimic-video/model/.venv/bin/python /tmp/so100_policy_server.py
"""
import socket, struct, pickle, json, glob
import numpy as np, torch, PIL.Image
from torchvision import transforms
from cosmos_predict2.configs.config import make_config
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override
from imaginaire.utils import distributed
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.models.utils import load_state_dict

EXP   = "w2a_so100_ft2_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz1"
CKPT  = "checkpoints/vam/so100/so100_finetune2/checkpoints/model/iter_000000150.pt"
STATS = "/home/pranavsaroha/so100_finetune2_zarr/.statistics_cache/75b2c2ad8c4c8692a574c8bfa8a82cf2c4fcb9bf4e3e88f537e84738c72c8021"
ZARR_GLOB = "/home/pranavsaroha/so100_finetune2_zarr/*/episode_*.zarr"
PORT = 9999
DTYPE = torch.bfloat16
RESIZE = (480, 640)   # (H, W) — matches policy_io.img_resize_sizes
N_OBS = 5

distributed.init()
print(">>> building model + loading fine-tune #2 ...", flush=True)
config = make_config(); config = override(config, ["--", f"experiment={EXP}"])
config.model.config.video_pipe_config.guardrail_config.enabled = False
model = instantiate(config.model).cuda().eval()
model.tensor_kwargs = {"device": "cuda", "dtype": DTYPE}
xattn_idx = model.pipe.config.xattn_layer_idx
sd = load_state_dict(CKPT); sd = {(k[4:] if k.startswith("net.") else k): v.cuda() for k, v in sd.items()}
m_, u_ = model.pipe.dit.load_state_dict(sd, strict=False, assign=True)
model.cuda()  # assign=True can leave params on CPU -> force everything to GPU
print(f"    decoder loaded (missing={len(m_)} unexpected={len(u_)})", flush=True)
data_config = instantiate(config.data_config)
model.pipe.normalizer.build_from_stats(
    json.load(open(STATS)),
    normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
    concat_groups=data_config.policy_io.concat_groups, device="cuda", dtype=DTYPE)
import zarr
lang_emb = torch.from_numpy(zarr.open(sorted(glob.glob(ZARR_GLOB))[0], "r")["language_embedding"][:]).to("cuda", DTYPE)
print(">>> model ready.", flush=True)

_buf = []
def _fmt(img):  # HWC uint8 RGB -> HWC float [-1,1], resized
    im = np.asarray(transforms.Resize(RESIZE)(PIL.Image.fromarray(img, "RGB"))).astype(np.float32)
    return 2.0 * (im / 255.0 - 0.5)

def predict(frame, joints, nstep=4):
    _buf.append(_fmt(frame))
    while len(_buf) > N_OBS: _buf.pop(0)
    frames = _buf[:]
    while len(frames) < N_OBS: frames.insert(0, frames[0])     # left-pad by repeating oldest
    obs = np.transpose(np.stack(frames, 0), (3, 0, 1, 2))       # (3,5,480,640)
    input_vid = torch.from_numpy(obs).unsqueeze(0).to("cuda", DTYPE)
    B, C, T, H, W = input_vid.shape
    vid = torch.zeros((B, C, 61, H, W), device="cuda", dtype=DTYPE); vid[:, :, :T] = input_vid
    state = torch.from_numpy(np.asarray(joints, np.float32)).reshape(1, 1, 6).to("cuda", DTYPE)
    with torch.no_grad():
        ctx = model.video2world_pipe.generate_video(
            vid_input=vid, num_latent_conditional_frames=2, prompt_embedding=lang_emb,
            guidance=0.0, num_sampling_step=nstep, seed=0, use_cuda_graphs=False,
            return_all_context=True, hidden_state_layer_idx=xattn_idx)
        vsig, cross = ctx[-1]; cross = cross.reshape(cross.shape[0], -1, cross.shape[-1])
        chunk = model.pipe(state, cross, vsig.repeat(B).unsqueeze(1)).float().cpu().numpy()[0]
    return chunk  # (15,6) absolute joint targets, denormalized (degrees)

def _recvall(c, n):
    d = b""
    while len(d) < n:
        p = c.recv(n - len(d))
        if not p: raise ConnectionError
        d += p
    return d
def _recv(c): return pickle.loads(_recvall(c, struct.unpack(">I", _recvall(c, 4))[0]))
def _send(c, o): b = pickle.dumps(o); c.sendall(struct.pack(">I", len(b)) + b)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", PORT)); srv.listen(1)
print(f">>> policy server listening on 127.0.0.1:{PORT}", flush=True)
while True:
    c, _ = srv.accept(); print(">>> client connected", flush=True)
    try:
        while True:
            msg = _recv(c)
            if msg.get("cmd") == "reset": _buf.clear(); _send(c, {"ok": True}); continue
            try:
                _send(c, {"chunk": predict(msg["frame"], msg["joints"], msg.get("nstep", 4))})
            except Exception as e:
                import traceback; traceback.print_exc()
                _send(c, {"error": str(e)[:200]})
    except (ConnectionError, EOFError):
        print(">>> client disconnected", flush=True)
    finally:
        c.close()
