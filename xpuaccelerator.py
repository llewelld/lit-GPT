import os

# Sets up environment variables needed for MPI with XPU devices
def xpu_setup_environment():
    # MPI_LOCALRANKID
    # Local sequential index of the process on the node
    # See nowhere
    try:
        local_rank = int(os.environ["MPI_LOCALRANKID"])
    except KeyError:
        local_rank = int(os.environ["SLURM_PROCID"])

    # PMI_RANK
    # The rank of this process within the program (zero-origin)
    # See https://flux-framework.readthedocs.io/projects/flux-rfc/en/latest/spec_13.html#environment
    # See https://github.com/intel/torch-ccl?tab=readme-ov-file#usage
    global_rank = int(os.environ["PMI_RANK"])

    # PMI_SIZE
    # The size of the program (number of ranks)
    # See https://flux-framework.readthedocs.io/projects/flux-rfc/en/latest/spec_13.html#environment
    # See https://github.com/intel/torch-ccl?tab=readme-ov-file#usage
    world_size = int(os.environ["PMI_SIZE"])

    os.environ["RANK"] = str(global_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # ZE_FLAT_DEVICE_HIERARCHY
    # Hierarchy model with which the underlying hardware is exposed
    # ZE_AFFINITY_MASK
    # Restrict which devices are visible to the process
    # See https://spec.oneapi.io/level-zero/latest/core/PROG.html#environment-variables
    # See https://www.intel.com/content/www/us/en/developer/articles/technical/flattening-gpu-tile-hierarchy.html
    os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "COMPOSITE"
    os.environ["ZE_AFFINITY_MASK"] = str(local_rank // 2) + "." + str(local_rank % 2)

# We must set up the environment before importing any PyTorch code
xpu_setup_environment()

import torch
import intel_extension_for_pytorch as ipex
from typing import Any, MutableSequence, Union, Dict, Optional, List

import lightning as L
from lightning.pytorch.accelerators.accelerator import Accelerator
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.fabric.utilities.device_parser import _check_data_type
from lightning.pytorch.utilities import rank_zero_info
from lightning.pytorch.strategies import DDPStrategy, DDPFullyShardedNativeStrategy
from lightning.pytorch.plugins import MixedPrecisionPlugin

# Custom XPU Trainer class
class Trainer(L.Trainer):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)

    @staticmethod
    def from_argparse_args(*args, **kwargs):
        precision = kwargs.get("precision") or vars(args[0]).get("precision")
        if precision == "bf16":
            # Lightning will force the device to "cuda" if we set precision to bf16
            # So we need to set up our own custom plugin in this case
            precision = MixedPrecisionPlugin("bf16", "xpu")
            plugins = kwargs.get("plugins", []) + [precision]
            kwargs["plugins"] = plugins
            ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF32, device='xpu')
        else:
            ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.FP32, device='xpu')

        strategy = kwargs.get("strategy") or vars(args[0]).get("strategy")
        accelerator = XPUAccelerator()
        if strategy == "ddp":
            # For ddp we need to configure Lightning to use the XPU and CCL backed
            # accelerator = XPUAccelerator()
            ddp = DDPStrategy(accelerator=accelerator, process_group_backend="ccl")
            kwargs['strategy'] = ddp
        # Return a standard Lightning Trainer but using our adjusted configuration
        elif strategy == "fsdp_native":
            fsdp = DDPFullyShardedNativeStrategy(accelerator=accelerator, process_group_backend="ccl")
            kwargs['strategy'] = fsdp
        else:
            raise MisconfigurationException(f"'{strategy}' is not valid, only 'ddp' and 'fsdp_native' are supported")
        return L.Trainer.from_argparse_args(*args, **kwargs)

# Custom XPU Accelerator class
class XPUAccelerator(Accelerator):
    """Accelerator for Intel XPU devices."""

    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise ValueError(f"Device should be XPU, got {device} instead.")
        if torch.xpu.get_fp32_math_mode() != torch.xpu.FP32MathMode.FP32:
            rank_zero_info(
                f"You are using an XPU device ({torch.xpu.get_device_name(device)!r}). To properly utilize computation "
                "power, you can set `torch.xpu.set_fp32_math_mode(mode=torch.xpu.FP32MathMode.FP32, device='xpu')` "
                "which will trade-off precision for performance. For more details, read https://intel.github.io/"
                "intel-extension-for-pytorch/xpu/latest/tutorials/api_doc.html#torch.xpu.set_fp32_math_mode"
            )

        torch.xpu.set_device(device)

    def teardown(self) -> None:
        torch.xpu.empty_cache()

    @staticmethod
    def parse_devices(devices: Union[int, str, List[int]]) -> Optional[List[int]]:
        # Put parsing logic here how devices can be passed into the Trainer
        print(f'devices passed: {devices}')
        # via the `devices` argument
        xpus = devices
        _check_data_type(xpus)

        # Convert strings to ints
        if not isinstance(xpus, str):
            xpus = xpus
        elif xpus == "-1":
            xpus =  -1
        elif "," in xpus:
            xpus = [int(x.strip()) for x in xpus.split(",") if len(x) > 0]
        else:
            xpus = int(xpus.strip())

        available_xpus = list(range(torch.xpu.device_count()))
        if isinstance(xpus, (MutableSequence, tuple)):
            xpus = list(xpus)
        elif not xpus:
            xpus = None
        elif xpus == -1:
            xpus = available_xpus
        else:
            xpus = list(range(xpus))

        print(f'xpus set: {xpus}')
        print(f'xpus available: {available_xpus}')

        if not xpus:
            raise MisconfigurationException("xpus requested but none are available.")

        
        for gpu in xpus:
            if gpu not in available_xpus:
                print(
                        f"You requested gpu: {xpus}\n But your machine only has: {available_xpus}"
                        )
                # raise MisconfigurationException(
                #     f"You requested gpu: {xpus}\n But your machine only has: {available_xpus}"
                # )

        return xpus


    @staticmethod
    def get_parallel_devices(devices: Any) -> Any:
        # Here, convert the device indices to actual device objects
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        # Return a value for auto-device selection when `Trainer(devices="auto")`
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

    def get_device_stats(self, device: Union[str, torch.device]) -> Dict[str, Any]:
        # Return optional device statistics for loggers
        return torch.xpu.memory_stats(device)

    @classmethod
    def register_accelerators(cls, accelerator_registry: Dict) -> None:
        accelerator_registry.register(
            "xpu",
            cls,
            description=cls.__class__.__name__,
        )
