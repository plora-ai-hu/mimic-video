# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import pathlib
from typing import Any

import numpy as np
import safetensors as st
import torch
from torch.utils.data import Dataset

from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
from imaginaire.utils import log


def _stable_hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


class LatentDataset(Dataset):
    def __init__(
        self,
        dataset_dir: pathlib.Path,
        num_frames: int,
        obs_history: int,
        video_height: int,
        video_width: int,
        target_fps: float,
        is_val: bool,
        include_only_with_substrings: list[str] | None = None,
        exclude_with_substring: str | None = None,
        is_multi_img: bool = False,
        val_ratio: float = 0.0,
    ) -> None:
        """Dataset class for loading image-text-to-video generation data with precomputed video embeddings."""

        super().__init__()

        self._rng = np.random.default_rng()
        self._dataset_dir = pathlib.Path(dataset_dir)

        if (num_frames - 1) % 4 != 0:
            msg = "Number of frames must be 1 + 4n."
            raise ValueError(msg)

        self._sequence_length = num_frames

        include_only_with_substrings = include_only_with_substrings or []

        video_dir = self._dataset_dir / "video"
        self._t5_dir = self._dataset_dir / "language_embeddings"
        self._video_embeddings_dir = self._dataset_dir / "video_embeddings"

        video_paths = sorted([
            video
            for video in video_dir.iterdir()
            if video.suffix == ".mp4"
            and all(substring in str(video) for substring in include_only_with_substrings)
            and (exclude_with_substring is None or exclude_with_substring not in str(video))
        ])

        for video_path in video_paths:
            assert video_path.exists(), video_path
            ep_name = video_path.stem[:-1] if is_multi_img else video_path.stem
            assert (self._t5_dir / f"{ep_name}.safetensors").exists(), video_path

        log.info(f"{len(video_paths)} videos in total")

        # make val set deterministic even if list of paths changes minorly
        denom = 10_000
        thresh = round(val_ratio * denom)
        if is_val:
            self._video_paths = [p for p in video_paths if _stable_hash_int(p.stem) % denom < thresh]
        else:
            self._video_paths = [p for p in video_paths if _stable_hash_int(p.stem) % denom >= thresh]

        log.info(f"{len(self._video_paths)} videos in {'val' if is_val else 'train'}.")

        self._video_height = video_height
        self._video_width = video_width
        self._target_fps = target_fps
        self._num_cond_frames = 1 + (obs_history - 1) / 4
        self._is_multi_img = is_multi_img

    def __str__(self) -> str:
        return f"{len(self._video_paths)} samples from {self._dataset_dir}"

    def __len__(self) -> int:
        return len(self._video_paths)

    def _get_embedding(self, video_path: pathlib.Path) -> tuple[torch.Tensor, int]:
        embeddings_path = self._video_embeddings_dir / f"{video_path.stem}.safetensors"

        with st.safe_open(embeddings_path, framework="pt") as f:
            if f.get_tensor("fps").item() != self._target_fps:
                msg = f"Precomputed video embeddings ({f.get_tensor('fps').item()}) don't match the target fps."
                raise ValueError(msg)
            if f.get_tensor("num_conditional_frames").item() != self._num_cond_frames:
                msg = f"Precomputed video embeddings ({f.get_tensor('num_conditional_frames').item()}) don't match the target obs history."
                raise ValueError(msg)

            embeddings = f.get_slice("video_embeddings")
            num_samples, _c, T, h, w = embeddings.get_shape()

            if (h, w) != (self._video_height / 8, self._video_width / 8):
                msg = f"Precomputed video embeddings ({(h, w)}) don't match the target shape."
                raise ValueError(msg)

            if T - 1 < (self._sequence_length - 1) / 4:
                msg = f"Precomputed video embeddings are too short ({T})."
                raise ValueError(msg)

            idx = self._rng.choice(num_samples)
            return embeddings[idx, :, : (self._sequence_length // 4 + 1)], f.get_tensor("num_conditional_frames")

    def __getitem__(self, index) -> dict | Any:
        data = dict()
        (video_embedding, num_conditional_frames) = self._get_embedding(self._video_paths[index])
        video_path = self._video_paths[index]
        ep_name = video_path.stem[:-1] if self._is_multi_img else video_path.stem
        t5_embedding_path = self._t5_dir / f"{ep_name}.safetensors"
        data["video_embedding"] = video_embedding

        with st.safe_open(t5_embedding_path, "torch") as f:
            t5_embedding = f.get_tensor("encoded_text")
        assert len(t5_embedding.shape) == 2
        n_tokens = t5_embedding.shape[0]
        if n_tokens < CosmosTextEncoderConfig.NUM_TOKENS:
            t5_embedding = torch.cat(
                [
                    t5_embedding,
                    torch.zeros(
                        (CosmosTextEncoderConfig.NUM_TOKENS - n_tokens, CosmosTextEncoderConfig.EMBED_DIM),
                        dtype=t5_embedding.dtype,
                        device=t5_embedding.device,
                    ),
                ],
                dim=0,
            )
        t5_text_mask = torch.zeros(CosmosTextEncoderConfig.NUM_TOKENS, dtype=torch.int64)
        t5_text_mask[:n_tokens] = 1

        data["obs/language_embedding"] = t5_embedding
        data["t5_text_mask"] = t5_text_mask
        data["fps"] = self._target_fps
        data["padding_mask"] = torch.zeros(1, self._video_height, self._video_width)
        data["num_conditional_frames"] = num_conditional_frames

        return data
