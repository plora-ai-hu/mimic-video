from __future__ import annotations

import gc
import os
from collections import defaultdict
from typing import Literal

import attrs
import torch
import wandb

from imaginaire.utils import distributed
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

        self._val_mse_sum.clear()
        self._val_loss_sum = 0.0
        self._val_step_count = 0
