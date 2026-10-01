import argparse
import pathlib

import safetensors
import safetensors.torch as st
import torch
import tqdm
from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig
from imaginaire.constants import T5_MODEL_DIR


def add_t5(path: pathlib.Path, encoder: CosmosT5TextEncoder, embedding: torch.Tensor | None):
    data = st.load_file(path)
    with safetensors.safe_open(path, "pt") as f:
        metadata = f.metadata()  # keep the converter's joint names and view mapping

    if embedding is None:
        prompt = data["language_instruction"][0].numpy().tobytes().decode("utf-8")
        embedding: torch.Tensor = encoder.encode_prompts(prompt, max_length=512, return_mask=False)

    data["language_embedding"] = embedding.float()  # needs to be a dtype numpy has (fix with ml_dtypes?)
    data["language_embedding_timestamps"] = torch.tensor([0], dtype=torch.uint64)

    st.save_file(data, path, metadata=metadata)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", nargs="+", type=pathlib.Path, required=True)
    p.add_argument("--prompt", required=False)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world-size", type=int, default=1)

    args = p.parse_args()

    encoder_config = CosmosT5TextEncoderConfig(ckpt_path=T5_MODEL_DIR)
    encoder = CosmosT5TextEncoder(config=encoder_config)

    embedding = args.prompt and encoder.encode_prompts(args.prompt, max_length=512, return_mask=False)

    for dataset in args.dataset_path:
        paths = sorted(map(str, pathlib.Path(dataset).glob("**/*.safetensors")))[args.rank :: args.world_size]

        for path in tqdm.tqdm(paths, desc="Precomputing language embeddings."):
            add_t5(path, encoder=encoder, embedding=embedding)


if __name__ == "__main__":
    main()
