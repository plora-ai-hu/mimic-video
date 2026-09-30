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
import torch.nn.functional as F
from decord import VideoReader, cpu
from torch.utils.data import Dataset as _Dataset

from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig
from imaginaire.utils import log


def _stable_hash_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


class Dataset(_Dataset):
    def __init__(
        self,
        dataset_dir: pathlib.Path,
        num_frames: int,
        video_height: int,
        video_width: int,
        target_fps: float,
        is_val: bool,
        include_only_with_substrings: list[str] | None = None,
        exclude_with_substring: str | None = None,
        data_fps: int | None = None,
        obs_history: int = 5,
        is_multi_img: bool = False,
        val_ratio: float = 0.0,
    ) -> None:
        """Dataset class for loading image-text-to-video generation data."""

        super().__init__()

        # this dataset impl is very sloppy.
        # an index corresponds to a video and then we sample a chunk.
        # this first of all means chunks don't have uniform probability :/
        # we sample by keeping this rng in the dataset. this is also sloppy and will have different behavior
        # depending on persistent_workers (if not kept persistent, we reset the rng state every epoch).
        # but since every epoch every dataset object copy will probably access different videos in a different order
        # thanks to the sampler shuffling, i don't think there is any actual big issue with this.
        # in my opinion dataset getitem should be deterministic and the index should reflect the chunk.
        self._rng = np.random.default_rng()
        self._dataset_dir = pathlib.Path(dataset_dir)
        self._sequence_length = num_frames

        include_only_with_substrings = include_only_with_substrings or []

        video_dir = self._dataset_dir / "video"
        self._t5_dir = self._dataset_dir / "language_embeddings"

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
            self.video_paths = [p for p in video_paths if _stable_hash_int(p.stem) % denom < thresh]
        else:
            self.video_paths = [p for p in video_paths if _stable_hash_int(p.stem) % denom >= thresh]

        log.info(f"{len(self.video_paths)} videos in {'val' if is_val else 'train'}.")

        self._video_height = video_height
        self._video_width = video_width
        self._target_fps = target_fps
        self._is_multi_img = is_multi_img
        self._obs_history = obs_history
        self._data_fps = data_fps

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self._dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _get_frames(self, video_path: str) -> torch.Tensor:
        vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=0)
        n = len(vr)
        if n == 0:
            raise ValueError(f"Empty video: {video_path}")

        fps = self._data_fps or vr.get_avg_fps()
        step = fps / self._target_fps

        i = self._rng.integers(int(n + step * (self._obs_history - 1)))

        T = self._sequence_length
        k = np.arange(T, dtype=np.float64) - (self._obs_history - 1)
        idx = np.rint(i + k * step).astype(np.int64)
        idx = np.clip(idx, 0, n - 1)

        uniq, inverse = np.unique(idx, return_inverse=True)

        frames_np = vr.get_batch(uniq).asnumpy()  # [Tv,H,W,3]

        del vr

        x = torch.from_numpy(frames_np).permute(3, 0, 1, 2).contiguous()  # [C,Tv,H,W]
        C, _Tv, H, W = x.shape
        T = self._sequence_length

        x = F.interpolate(
            x.float().view(-1, 1, H, W),
            size=(self._video_height, self._video_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(C, -1, self._video_height, self._video_width)
        x = x.round().clamp_(0, 255).to(torch.uint8)

        j = torch.from_numpy(inverse).to(torch.long)
        x = x.index_select(1, j)  # [C,T,H,W]

        return x

    def __getitem__(self, index) -> dict | Any:
        data = dict()
        video = self._get_frames(self.video_paths[index])
        video_path = self.video_paths[index]
        ep_name = video_path.stem[:-1] if self._is_multi_img else video_path.stem
        t5_embedding_path = self._t5_dir / f"{ep_name}.safetensors"
        data["video"] = video

        with st.safe_open(t5_embedding_path, "torch") as f:
            t5_embedding = f.get_tensor("encoded_text")
        assert len(t5_embedding.shape) == 2
        n_tokens = t5_embedding.shape[0]
        if n_tokens < CosmosTextEncoderConfig.NUM_TOKENS:
            t5_embedding = np.concatenate(
                [
                    t5_embedding,
                    np.zeros(
                        (CosmosTextEncoderConfig.NUM_TOKENS - n_tokens, CosmosTextEncoderConfig.EMBED_DIM),
                        dtype=np.float32,
                    ),
                ],
                axis=0,
            )
        t5_text_mask = torch.zeros(CosmosTextEncoderConfig.NUM_TOKENS, dtype=torch.int64)
        t5_text_mask[:n_tokens] = 1

        data["obs/language_embedding"] = t5_embedding
        data["t5_text_mask"] = t5_text_mask
        data["fps"] = self._target_fps
        data["padding_mask"] = torch.zeros(1, self._video_height, self._video_width)
        data["num_conditional_frames"] = torch.tensor(1 + (self._obs_history - 1) // 4)

        return data


class MultiDataset(_Dataset):
    def __init__(self, **datasets: Dataset) -> None:
        self._datasets = list(datasets.values())
        self._lens = [len(ds) for ds in self._datasets]
        log.info(f"Dataset mix has {len(self)} videos.")

    def __len__(self):
        return sum(self._lens)

    def __getitem__(self, index):
        for i, len_ in enumerate(self._lens):
            if index < len_:
                return self._datasets[i][index]
            index -= len_
