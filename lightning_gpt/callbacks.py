import time
from copy import copy
from statistics import mean

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_info


class CPUMetricsCallback(Callback):

    @property
    def _torch_runner(self):
        return torch.cpu

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if hasattr(self._torch_runner, 'reset_peak_memory_stats'):
            # Reset the memory use counter for CUDA and XPU
            self._torch_runner.reset_peak_memory_stats(self.root_device(trainer))
            self._torch_runner.synchronize(self.root_device(trainer))
        self.start_time = time.time()
        self._epoch_losses: list[list[float]] = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._torch_runner.synchronize(self.root_device(trainer))
        if hasattr(self._torch_runner, "max_memory_allocated"):
            # CUDA and XPU have 'max_memory_allocated', cpu doesn't
            max_memory = self._torch_runner.max_memory_allocated(self.root_device(trainer)) / 2**20
        else:
            max_memory = float("nan")
        epoch_time: float = time.time() - self.start_time

        # For debugging uncomment the following
        # from pprint import pprint
        # pprint(pl_module._losses)

        self._epoch_losses.append(copy(pl_module._losses))
        pl_module._losses.clear()
        loss_mean: float = mean(self._epoch_losses[-1])

        epoch: int = pl_module.current_epoch

        max_memory = trainer.strategy.reduce(max_memory)
        epoch_time = trainer.strategy.reduce(epoch_time)
        loss_mean_means: float = trainer.strategy.reduce(loss_mean)

        rank_zero_info(f"Epoch: {epoch}")
        rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
        rank_zero_info(f"Average Peak memory: {max_memory:.2f}MiB")
        rank_zero_info(f"Loss Mean: {loss_mean_means:.2f}")

    def root_device(self, trainer: Trainer) -> int:
        return trainer.strategy.root_device.index


class CUDAMetricsCallback(CPUMetricsCallback):

    @property
    def _torch_runner(self):
        return torch.cuda

class XPUMetricsCallback(CPUMetricsCallback):

    @property
    def _torch_runner(self):
        return torch.xpu
