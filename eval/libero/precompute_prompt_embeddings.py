"""Precompute T5-11B prompt embeddings for every LIBERO task instruction.

The eval normally loads T5-11B onto the GPU in fp32 (~45 GB), which does not fit a 24 GB
card. LIBERO has a small, fixed set of instructions, so we encode them once here and
run.py loads them via --vam_prompt_embeddings_path.

Only the encoder half of T5-11B is needed. It is range-read straight from the Hub's
safetensors conversion (google-t5/t5-11b refs/pr/6, same weights as the pytorch_model.bin
that mimic-video ships) into CPU RAM, so nothing large touches disk. Encoding runs on CPU
in fp32, exactly matching CosmosT5TextEncoder with its default torch_dtype=None.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch
import tyro
from huggingface_hub import HfFileSystem
from transformers import T5Config, T5EncoderModel, T5TokenizerFast

from libero.libero import benchmark

REPO = "google-t5/t5-11b"
REVISION = "refs/pr/6"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
NUM_TOKENS = 512  # CosmosT5TextEncoderConfig.NUM_TOKENS

_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}


def libero_prompts() -> list[str]:
    """Prompts exactly as run.py's get_libero_env derives them."""
    prompts = []
    for name in SUITES:
        suite = benchmark.get_benchmark_dict()[name]()
        for i in range(suite.n_tasks):
            prompts.append(suite.get_task(i).language.replace("black bowl", "bowl"))
    return sorted(set(prompts))


def load_encoder_state_dict() -> dict[str, torch.Tensor]:
    fs = HfFileSystem()
    path = f"{REPO}@{REVISION.replace('/', '%2F')}/model.safetensors"
    with fs.open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        wanted = sorted(
            (k for k in header if k != "__metadata__" and (k.startswith("encoder.") or k == "shared.weight")),
            key=lambda k: header[k]["data_offsets"][0],
        )
        state = {}
        total = sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0] for k in wanted)
        done = 0
        for k in wanted:
            start, end = header[k]["data_offsets"]
            f.seek(base + start)
            buf = bytearray(f.read(end - start))
            state[k] = torch.frombuffer(buf, dtype=_DTYPES[header[k]["dtype"]]).reshape(header[k]["shape"])
            done += end - start
            print(f"\r  read {done / 1e9:6.2f} / {total / 1e9:.2f} GB", end="", flush=True)
        print()
    return state


def main(
    t5_config_dir: Path = Path("../../model/checkpoints/text_encoder/t5-11b"),
    out_path: Path = Path("libero_prompt_embeddings.pt"),
) -> None:
    prompts = libero_prompts()
    print(f"{len(prompts)} unique LIBERO prompts")

    assert (t5_config_dir / "config.json").is_file(), f"missing {t5_config_dir / 'config.json'}"
    config = T5Config.from_pretrained(t5_config_dir)
    assert config.d_model == 1024 and config.num_layers == 24, f"not the T5-11B config: {config}"
    tokenizer = T5TokenizerFast.from_pretrained(t5_config_dir)
    with torch.device("meta"):
        model = T5EncoderModel(config)

    print(f"Range-reading encoder weights from {REPO}@{REVISION}")
    state = load_encoder_state_dict()
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    # encoder.embed_tokens is tied to shared; it may be absent from the file.
    missing = [k for k in missing if k != "encoder.embed_tokens.weight"]
    assert not missing and not unexpected, (missing, unexpected)
    model.tie_weights()
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Encoder loaded: {n_params / 1e9:.2f}B params, dtype {next(model.parameters()).dtype}")

    embeddings = {}
    with torch.inference_mode():
        for i, prompt in enumerate(prompts):
            # Mirrors CosmosT5TextEncoder.encode_prompts.
            enc = tokenizer.batch_encode_plus(
                [prompt], return_tensors="pt", truncation=True, padding="max_length",
                max_length=NUM_TOKENS, return_length=True, return_offsets_mapping=False,
            )
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
            length = int(enc.attention_mask.sum())
            out[0, length:] = 0
            embeddings[prompt] = out.clone()
            print(f"  [{i + 1}/{len(prompts)}] {prompt}  ({length} tokens)")

    torch.save(embeddings, out_path)
    print(f"Saved {len(embeddings)} embeddings to {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
