"""Round-trip tiled clips through the frozen Cosmos VAE and report per-view reconstruction quality.

Run from model/ so cosmos_predict2 / imaginaire import from the checkout:
    cd model && python ../eval/video/vae_roundtrip.py --videos <clip.mp4> [...] --out <dir>

For every clip it writes <name>_roundtrip.mp4 (original | reconstruction | 4x abs error) and a frame-0 PNG,
and prints PSNR per 240x320 tile of the 2x2 view grid (see SO101.md).
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch

from cosmos_predict2.tokenizers.tokenizer import TokenizerInterface
from imaginaire.constants import get_cosmos_predict2_video2world_tokenizer

TILES = {"scene": (0, 0), "top_right": (0, 1), "left_wrist": (1, 0), "right_wrist": (1, 1)}


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=61, help="pixel frames to encode, rounded down to 1 + 4k")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    vae = TokenizerInterface(
        chunk_duration=81, temporal_window=16, vae_pth=get_cosmos_predict2_video2world_tokenizer(model_size="2B")
    )

    for path in args.videos:
        frames = np.stack(imageio.v2.mimread(path, memtest=False))[: args.frames]  # T, H, W, 3 uint8
        frames = frames[: 1 + (len(frames) - 1) // 4 * 4]
        T, H, W, _ = frames.shape
        x = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].cuda().to(torch.bfloat16) / 127.5 - 1.0

        latent = vae.encode(x)
        recon = vae.decode(latent)
        recon = ((recon[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 3, 0).cpu().numpy()

        th, tw = H // 2, W // 2
        report = [
            f"{path.name}: {T} frames {H}x{W} -> latent {tuple(latent.shape)}",
            f"  full: {psnr(frames, recon):.2f} dB",
        ]
        for name, (r, c) in TILES.items():
            crop = np.s_[:, r * th : (r + 1) * th, c * tw : (c + 1) * tw]
            if frames[crop].max() == 0:
                report.append(f"  {name}: empty")
            else:
                report.append(f"  {name}: {psnr(frames[crop], recon[crop]):.2f} dB")
        print("\n".join(report), flush=True)

        err = np.clip(np.abs(frames.astype(np.int16) - recon.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        side = np.concatenate([frames, recon, err], axis=2)
        imageio.v2.mimwrite(args.out / f"{path.stem}_roundtrip.mp4", side, fps=5, macro_block_size=1)
        imageio.imwrite(args.out / f"{path.stem}_frame0.png", side[0])


if __name__ == "__main__":
    main()
