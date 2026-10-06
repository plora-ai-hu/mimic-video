from __future__ import annotations

import gc
import os
from collections import defaultdict
from typing import Literal

import attrs
import numpy as np
import torch
import wandb

from imaginaire.utils import distributed, log
from imaginaire.utils.callback import Callback


def sanitize_for_wandb(val):
    """Recursively traverses a config and stringifies non-primitive leaf nodes."""
    if isinstance(val, dict):
        return {k: sanitize_for_wandb(v) for k, v in val.items()}
    elif isinstance(val, (list, tuple, set)):
        return [sanitize_for_wandb(v) for v in val]
    elif isinstance(val, (int, float, str, bool, type(None))):
        return val
    else:
        return str(val)


class WandBCallback(Callback):
    """Log to Weights & Biases.

    ``WANDB_ENTITY``, ``WANDB_PROJECT`` and ``WANDB_MODE`` from the environment (e.g. from ``secrets.env``, see
    ``container/train.sbatch``) override the configured entity, project and mode. ``WANDB_API_KEY`` is read by wandb.
    """

    def __init__(
        self,
        mode: Literal["online", "offline", "disabled"],
        entity_name: str | None,
        run_name: str,
        project_name: str,
    ):
        self._mode = os.environ.get("WANDB_MODE") or mode
        self._entity_name = os.environ.get("WANDB_ENTITY") or entity_name
        self._run_name = run_name
        self._project_name = os.environ.get("WANDB_PROJECT") or project_name

        self._val_mse_sum = defaultdict(lambda: defaultdict(float))
        self._val_mse_step_joint_sum = defaultdict(dict)  # mode -> sigma -> (HA, A) tensor
        self._val_loss_sum = 0.0
        self._val_step_count = 0

    def _log(self, output_batch, step, prefix):
        cpu_batch = {
            k: v.detach().to("cpu", non_blocking=True, copy=True) if torch.is_tensor(v) else v
            for k, v in output_batch.items()
        }
        del output_batch
        gc.collect(0)
        batch = {f"{prefix}/{k}": v for k, v in cpu_batch.items()}
        wandb.log(batch, step=step)

    @distributed.rank0_only
    def on_train_start(self, model, iteration: int = 0):
        run = wandb.init(
            entity=self._entity_name,
            project=self._project_name,
            mode=self._mode,
            name=self._run_name,
            config=sanitize_for_wandb(attrs.asdict(self.config, recurse=True)),
        )

    @distributed.rank0_only
    def on_train_end(self, model, iteration: int = 0):
        wandb.finish()

    @distributed.rank0_only
    def on_before_optimizer_step(
        self, model_ddp, optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, grad_scaler, iteration
    ):
        del model_ddp, optimizer, grad_scaler

        output_batch = {"lr": scheduler.get_last_lr()[0].item()}
        self._log(output_batch, iteration, "train")

    @distributed.rank0_only
    def on_training_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0):
        del model, data_batch, loss
        self._log(output_batch, iteration - 1, "train")

    @distributed.rank0_only
    def on_validation_step_end(self, model, data_batch, output_batch, loss, iteration: int = 0):
        del model, data_batch, loss

        loss = output_batch.pop("loss")
        self._val_loss_sum += loss
        self._val_step_count += 1

        if "mses" in output_batch:
            mses = output_batch.pop("mses")
            for category, values in mses.items():
                for sigma, mse in values:
                    self._val_mse_sum[category][sigma] += mse

        if "mses_per_step_joint" in output_batch:
            for mode, values in output_batch.pop("mses_per_step_joint").items():
                sums = self._val_mse_step_joint_sum[mode]
                for sigma, mse_HA_A in values:
                    sums[sigma] = sums[sigma] + mse_HA_A if sigma in sums else mse_HA_A.clone()

        self._log(output_batch, iteration, "val")

    @distributed.rank0_only
    def on_validation_end(self, model, iteration):
        del model

        output_batch = {"loss": self._val_loss_sum / self._val_step_count}
        charts = {}

        for category, mses in self._val_mse_sum.items():
            sigmas = sorted(mses.keys())
            vals = [mses[s] / self._val_step_count for s in sigmas]

            output_batch.update(
                {f"{category}/sigma_{sigma:.5g}": mean_mse for sigma, mean_mse in zip(sigmas, vals, strict=True)}
            )

        self._log(output_batch, iteration, "val")
        self._save_val_mse_step_joint(iteration)

        self._val_mse_sum.clear()
        self._val_mse_step_joint_sum.clear()
        self._val_loss_sum = 0.0
        self._val_step_count = 0

    def _save_val_mse_step_joint(self, iteration: int) -> None:
        """Save the validation MSE per mode, video sigma, horizon step and joint to ``val_mse/iter_XXXXXXXXX.npz``.

        Arrays: ``mse`` (mode, sigma, step, joint) in squared raw action units, averaged over the validation batches;
        ``sigmas`` (mode, sigma), ascending; ``modes``; ``iteration``; ``num_val_batches``.
        """
        if not self._val_mse_step_joint_sum:
            return
        modes = list(self._val_mse_step_joint_sum.keys())
        sigmas = [sorted(self._val_mse_step_joint_sum[mode].keys()) for mode in modes]
        mse = torch.stack(
            [
                torch.stack([self._val_mse_step_joint_sum[mode][s] for s in mode_sigmas])
                for mode, mode_sigmas in zip(modes, sigmas, strict=True)
            ]
        )
        mse = mse / self._val_step_count

        out_dir = os.path.join(self.config.job.path_local, "val_mse")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"iter_{iteration:09d}.npz")
        np.savez(
            path,
            mse=mse.numpy().astype(np.float32),
            sigmas=np.array(sigmas, dtype=np.float64),
            modes=np.array(modes),
            iteration=iteration,
            num_val_batches=self._val_step_count,
        )
        log.info(f"Saved per-step, per-joint validation MSE to {path}")
