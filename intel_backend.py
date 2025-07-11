# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License
from __future__ import annotations

from contextlib import nullcontext
from datetime import timedelta
from logging import getLogger
from typing import Any

# import intel_extension_for_pytorch as ipex
# import oneccl_bindings_for_pytorch
import torch
from lightning.fabric.utilities.distributed import (
    _init_dist_connection,
)
from lightning.fabric.utilities.seed import reset_seed
from lightning.pytorch.accelerators import Accelerator
from lightning.pytorch.overrides.distributed import _register_ddp_comm_hook
from lightning.pytorch.strategies import (
    DDPStrategy,
    FSDPStrategy,
    SingleDeviceStrategy,
    StrategyRegistry,
)
from torch import distributed as dist
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from typing_extensions import override

default_pg_timeout = timedelta(seconds=1800)

log = getLogger(__file__)


class XPUAccelerator(Accelerator):
    """Implements a Lightning Accelerator class for Intel GPU usage.

    Depends on Intel Extension for PyTorch to be installed.
    """

    @staticmethod
    def parse_devices(devices: int | list[int]) -> list[int]:
        """Parse the `trainer` input for devices and homogenize them.

        Parameters
        ----------
        devices : Union[int, List[int]]
            Single or list of device numbers to use
        Returns
        -------
        List[int]
            List of device numbers to use
        """
        if isinstance(devices, int):
            # assume that this is the number of devices to use
            devices = list(range(devices))
        return devices

    def setup_device(self, device: torch.device) -> None:
        """Configure the current process to use a specified device.

        Perhaps unreliably and misguiding, the IPEX implementation of this method
        tries to mirror the CUDA version but `ipex.xpu.set_device` actually refuses
        to accept anything other than an index. I've tried to work around this
        by grabbing the index from the device if possible, and just setting
        it to the first device if not using a distributed/multitile setup.
        """
        # first try and see if we can grab the index from the device
        index = getattr(device, "index", None)
        if index is None and not dist.is_initialized():
            index = 0
        torch.xpu.set_device(index)

    def teardown(self) -> None:
        # as it suggests, this is run on cleanup
        torch.xpu.empty_cache()

    def get_device_stats(self, device) -> dict[str, Any]:
        return torch.xpu.memory_stats(device)

    @staticmethod
    def get_parallel_devices(devices: list[int]) -> list[torch.device]:
        """Return a list of torch devices corresponding to what is available. Essentially maps indices to `torch.device`
        objects.

        Parameters
        ----------
        devices : List[int]
            List of integers corresponding to device numbers
        Returns
        -------
        List[torch.device]
            List of `torch.device` objects for each device
        """
        return [torch.device("xpu", i) for i in devices]

    @staticmethod
    def auto_device_count() -> int:
        # by default, PVC has two tiles per GPU
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        """Determines if an XPU is actually available.

        Returns
        -------
        bool
            True if devices are detected, otherwise False
        """
        try:
            return torch.xpu.device_count() != 0
        except (AttributeError, NameError):
            return False

    @classmethod
    def register_accelerators(cls, accelerator_registry) -> None:
        accelerator_registry.register(
            "xpu",
            cls,
            description="Intel Data Center GPU Max - codename Ponte Vecchio",
        )


# add PVC to the registry
# AcceleratorRegistry.register("xpu", XPUAccelerator)


class DDPXPUStrategy(DDPStrategy):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            # device=device,
            # accelerator=XPUAccelerator(),
            # checkpoint_io=checkpoint_io,
            # precision_plugin=precision_plugin,
            process_group_backend="ccl",
            **kwargs,
        )
        # super(process_group_backend="ccl", **kwargs)

    @override
    def _setup_model(self, model: Module) -> DistributedDataParallel:
        """Wraps the model into a `DistributedDataParallel` module."""
        device_ids = self.determine_ddp_device_ids()
        log.debug(f"setting up DDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
        if self.root_device.type == "xpu":
            # https://pytorch.org/docs/stable/notes/cuda.html#id5
            ctx = torch.xpu.stream(torch.xpu.Stream()) if device_ids is not None else nullcontext()
        # elif self.root_device.type == "cuda":
        #     ctx = torch.cuda.stream(torch.cuda.Stream()) if device_ids is not None else nullcontext()
        else:
            raise ValueError("Only 'xpu' are supported")
        with ctx:
            return DistributedDataParallel(module=model, device_ids=device_ids, **self._ddp_kwargs)

    def _register_ddp_hooks(self) -> None:
        log.debug(f"{self.__class__.__name__}: registering ddp hooks")
        # currently, DDP communication hooks only work with NCCL backend and SPSD (single process single device) mode
        # https://github.com/pytorch/pytorch/blob/v1.8.0/torch/nn/parallel/distributed.py#L1080-L1084
        # if self.root_device.type in ("cuda", "xpu"):
        if self.root_device.type == "xpu":
            assert isinstance(self.model, DistributedDataParallel)
            _register_ddp_comm_hook(
                model=self.model,
                ddp_comm_state=self._ddp_comm_state,
                ddp_comm_hook=self._ddp_comm_hook,
                ddp_comm_wrapper=self._ddp_comm_wrapper,
            )
        else:
            raise ValueError("Only 'cuda' and 'xpu' are supported")


class FSDPXPUStrategy(FSDPStrategy):

    def __init__(self, **kwargs) -> None:
        super().__init__(
            # device=device,
            # accelerator=XPUAccelerator(),
            # checkpoint_io=checkpoint_io,
            # precision_plugin=precision_plugin,
            process_group_backend="ccl",
            **kwargs,
        )
        # super(process_group_backend="ccl", **kwargs)

    @override
    def setup_environment(self) -> None:
        super().setup_environment()
        log.debug(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend, timeout=self._timeout)

        # if 'device_mesh' in the `kwargs` is provided as a tuple, update it into the `DeviceMesh` object here
        if isinstance(self.kwargs.get("device_mesh"), tuple):
            from torch.distributed.device_mesh import init_device_mesh

            self.kwargs["device_mesh"] = init_device_mesh("xpu", self.kwargs["device_mesh"])

    # @override
    # def _setup_model(self, model: Module) -> DistributedDataParallel:
    #     """Wraps the model into a `DistributedDataParallel` module."""
    #     device_ids = self.determine_ddp_device_ids()
    #     log.debug(f"setting up FSDP model with device ids: {device_ids}, kwargs: {self._ddp_kwargs}")
    #     if self.root_device.type == "xpu":
    #         # https://pytorch.org/docs/stable/notes/cuda.html#id5
    #         ctx = torch.xpu.stream(torch.xpu.Stream()) if device_ids is not None else nullcontext()
    #     # elif self.root_device.type == "cuda":
    #     #     ctx = torch.cuda.stream(torch.cuda.Stream()) if device_ids is not None else nullcontext()
    #     else:
    #        raise ValueError("Only 'xpu' are supported")
    #     with ctx:
    #         return DistributedDataParallel(module=model, device_ids=device_ids, **self._ddp_kwargs)
    #
    #
    # def _register_ddp_hooks(self) -> None:
    #     log.debug(f"{self.__class__.__name__}: registering fsdp hooks")
    #     # currently, DDP communication hooks only work with NCCL backend and SPSD (single process single device) mode
    #     # https://github.com/pytorch/pytorch/blob/v1.8.0/torch/nn/parallel/distributed.py#L1080-L1084
    #     # if self.root_device.type in ("cuda", "xpu"):
    #     if self.root_device.type == "xpu":
    #         assert isinstance(self.model, DistributedDataParallel)
    #         _register_ddp_comm_hook(
    #             model=self.model,
    #             ddp_comm_state=self._ddp_comm_state,
    #             ddp_comm_hook=self._ddp_comm_hook,
    #             ddp_comm_wrapper=self._ddp_comm_wrapper,
    #         )
    #     else:
    #         raise ValueError("Only 'cuda' and 'xpu' are supported")


class SingleXPUStrategy(SingleDeviceStrategy):
    """This class implements the strategy for using a single PVC tile."""

    strategy_name = "pvc_single"

    def __init__(
        self,
        device: str | None = "xpu",
        checkpoint_io=None,
        precision_plugin=None,
    ):
        super().__init__(
            device=device,
            accelerator=XPUAccelerator(),
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

    # @property
    # def is_distributed(self) -> bool:
    #     return False

    # def setup(self, trainer) -> None:
    #     self.model_to_device()
    #     super().setup(trainer)

    # def setup_optimizers(self, trainer) -> None:
    #     super().setup_optimizers(trainer)

    # def model_to_device(self) -> None:
    #     self.model.to(self.root_device)

    @classmethod
    def register_strategies(cls, strategy_registry) -> None:
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__} - uses a single XPU tile for compute.",
        )


StrategyRegistry.register(
    "single_xpu",
    SingleXPUStrategy,
    description="Strategy utilizing a single Intel GPU device or tile.",
)

StrategyRegistry.register(
    "ddp_xpu",
    DDPXPUStrategy,
    description="XPU ddp utilizing a multiple Intel or CUDA GPU device or tile.",
)

StrategyRegistry.register(
    "fsdp_xpu",
    DDPXPUStrategy,
    description="XPU fsdp utilizing a multiple Intel or CUDA GPU device or tile.",
)
