from __future__ import annotations

import os

from imaginaire.utils import distributed
from imaginaire.utils.callback import Callback
from scripts.fuse_lora_ckpt import fuse_ckpt

# import subprocess as sp


class VideoEvalCallback(Callback):
    def __init__(self, fuse_lora: bool, fuse_iter: int = 1):
        """Write a LoRA-fused copy of the model checkpoint after each save at a multiple of fuse_iter."""
        self._fuse_lora = fuse_lora
        self._fuse_iter = fuse_iter

    @distributed.rank0_only
    def on_save_checkpoint_success(
        self, iteration: int = 0, elapsed_time: float = 0, checkpoint_path: str | None = None
    ) -> None:
        # Called once per saved folder (model, optim, scheduler, trainer); only the model weights are fused.
        if checkpoint_path is None or os.path.basename(os.path.dirname(checkpoint_path)) != "model":
            return
        if self._fuse_lora and iteration % self._fuse_iter == 0:
            checkpoint_path = fuse_ckpt(checkpoint_path)

        try:
            # sp.run(
            #     [submit job to cluster management that generates videos from snippets],
            #     timeout=5,
            # )
            pass
        except Exception:
            pass
