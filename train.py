"""
Trainer for gpt2 benchmarking.

Notes:
    - Check the `MASTER_ADDR` and `MASTER_PORT` variables are set correctly
    - They may need to be set prior to certain imports (especialy `lightning`)
    - Also be aware they are set in `bin/dawn.bash`
    - Consider importing the `MASTER_ADDR` and `MASTER_PORT` from env
"""
from argparse import ArgumentParser, BooleanOptionalAction
from logging import getLogger
from typing import Union
from urllib.error import URLError
from pathlib import Path
from urllib.request import urlopen
import os

def xpu_setup_environment(composite: bool = False):
    # MPI_LOCALRANKID
    # Local sequential index of the process on the node
    # See nowhere
    local_rank = int(os.environ["MPI_LOCALRANKID"])

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
    if composite:
        os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "COMPOSITE"
    else:
        os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
    os.environ["ZE_AFFINITY_MASK"] = str(local_rank // 2) + "." + str(local_rank % 2)

# # # xpu_setup_environment()

import lightning as L
import torch
from lightning.pytorch.plugins.environments import (
    TorchElasticEnvironment, LightningEnvironment,  SLURMEnvironment, MPIEnvironment
)
from lightning.pytorch.accelerators import AcceleratorRegistry
from torch.utils.data import DataLoader

from lightning_gpt import callbacks, data, models

log = getLogger(__file__)

try:
    # import intel_extension_for_pytorch as ipex
    # import oneccl_bindings_for_pytorch  # noqa: F401

    from intel_backend import XPUAccelerator
    AcceleratorRegistry.register("xpu", XPUAccelerator)
except ImportError:
    log.info("Torch 2.8.0+xpu required to use XPUAccelerator")



LOCAL_SHAKESPEARE_PATH: Path = Path("shakespeare_input.txt")


def none_or_str(value: str) -> Union[str, None]:
    if value == "None":
        return None
    else:
        return value


def main(args):
    try:
        with urlopen("https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt") as f:
            text = f.read()
    except URLError:
        # Open in binary mode for compatibility
        with args.local_shakespeare_path.open(mode="rb") as f:
            text = f.read()
    print(args)

    train_dataset = data.CharDataset(text, args.block_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, persistent_workers=True
    )

    GPT_class = None
    extra_kwargs = {}

    if args.implementation == "mingpt":
        GPT_class = models.MinGPT
        extra_kwargs.update(
            dict(
                embd_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
            )
        )

    elif args.implementation == "nanogpt":
        GPT_class = models.NanoGPT
        extra_kwargs["dropout"] = 0.1

    else:
        raise ValueError(f"Unsupported implementation {args.implementation}")

    if args.strategy == "deepspeed":
        if GPT_class == models.MinGPT:
            GPT_class = models.DeepSpeedMinGPT
        elif GPT_class == models.NanoGPT:
            GPT_class = models.DeepSpeedNanoGPT
        else:
            raise ValueError(f"Implementation {args.implementation} not supported with DeepSpeed")
        extra_kwargs["offload"] = False

    elif args.strategy == "fsdp":
        if GPT_class == models.MinGPT:
            GPT_class = models.FSDPMinGPT
        elif GPT_class == models.NanoGPT:
            GPT_class = models.FSDPNanoGPT
        else:
            raise ValueError(f"Implementation {args.implementation} not supported with FSDP")

    model = GPT_class(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        model_type=args.model_type,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        weight_decay=0.1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        **extra_kwargs,
    )

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"The current torch version ({torch.__version__}) does not have support for compile."
                "Please install torch >= 1.14 or disable compile."
            )
        model = torch.compile(model)

    callback_list = []
    trainer_plugins = []

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        callback_list.append(callbacks.CUDAMetricsCallback())
    elif args.accelerator == "xpu":
        # # Sourced from
        # # https://github.com/alan-turing-institute/aurora-hpc/blob/f9365754c75ebcc82f19b70c294ee0af2f6cd707/train/scripts/train.py#L59

        # # PMI_SIZE set by mpirun
        # import os
        # import re
        # WORLD_SIZE = int(os.environ["PMI_SIZE"])
        # os.environ["WORLD_SIZE"] = str(WORLD_SIZE)

        # # # PMI_RANK set by mpirun
        # RANK = os.environ["PMI_RANK"]
        # os.environ["RANK"] = RANK

        # # # MPI_LOCALRANKID provenance unknown
        # LOCAL_RANK = int(os.environ["MPI_LOCALRANKID"])
        # print(f"LOCAL_RANK: {LOCAL_RANK}")

        # # # get the master address
        # numbers = re.compile("\d+")
        # nodelist_env = os.getenv("SLURM_JOB_NODELIST")

        # # e.g. "pvc-s-[24-25]"
        # try:
        #     # If we're running on >1 node, we should set the MASTER_ADDR
        #     # to the hostname of rank 0.
        #     prefix = nodelist_env[0 : nodelist_env.index("[")]
        #     nodelist = tuple(prefix + x for x in numbers.findall(nodelist_env))
        #     master_addr = nodelist[0]
        # except ValueError:
        #     # We must be running on a single node.
        #     master_addr = "0.0.0.0"

        # os.environ["MASTER_ADDR"] = master_addr
        # os.environ["MASTER_PORT"] = "29876"
        # # End copied section
        trainer_plugins.append(MPIEnvironment())
        if torch.xpu.is_available():
            # torch.set_float32_matmul_precision("high")
            # ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.FP32, device="xpu")
            torch.set_float32_matmul_precision("high")
            callback_list.append(callbacks.XPUMetricsCallback())
        if args.strategy == "ddp":
            args.strategy = "ddp_xpu"
        elif args.strategy == "fsdp":
            print("Applying 'fsdp_xpu' stragey")
            args.strategy = "fsdp_xpu"
        else:
            raise ValueError(f"{args.strategy} is not supported for xpu")
    else:
        callback_list.append(callbacks.CPUMetricsCallback())

    trainer = L.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        callbacks=callback_list,
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        enable_progress_bar=args.progress_bar,
        # plugins=TorchElasticEnvironment(),
        # plugins=MPIEnvironment(),
        plugins=trainer_plugins,
    )

    trainer.fit(model, train_loader)

    context = "Friends of my soul"  # Prime with something
    x = train_dataset.to_tokens(context, model.device)
    y = model.generate(x, max_new_tokens=1000, temperature=1.0, do_sample=True, top_k=10)
    # y is a list of length 1. That sole element is a tensor, hence y[0].
    print(train_dataset.from_tokens(y[0]))


if __name__ == "__main__":
    L.seed_everything(42)

    parser = ArgumentParser()
    # parser = L.Trainer.add_argparse_args(parser)

    parser.add_argument("--model-type", default="gpt2", type=none_or_str)
    parser.add_argument("--n-layer", type=int)
    parser.add_argument("--n-head", type=int)
    parser.add_argument("--n-embd", type=int)
    parser.add_argument("--learning-rate", default=3e-4, type=float)
    parser.add_argument("--block-size", default=128, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--compile", default=None, choices=[None, "dynamo"])
    parser.add_argument("--implementation", default="mingpt", choices=["mingpt", "nanogpt"])
    parser.add_argument("--strategy", default="ddp", choices=["fsdp", "ddp"])
    parser.add_argument("--max-epochs", default=10, type=int)
    parser.add_argument("--gradient-clip-val", default=1.0, type=float)
    parser.add_argument("--gradient-clip-algorithm", default="norm", choices=("norm", "value"))
    parser.add_argument("--devices", default=1, type=int)
    parser.add_argument("--precision", default="bf16-mixed", type=str)
    parser.add_argument("--num-nodes", default=1, type=int)
    parser.add_argument("--local-shakespeare-path", default=LOCAL_SHAKESPEARE_PATH, type=Path)
    parser.add_argument("--progress-bar", action=BooleanOptionalAction)
    parser.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "xpu"))
    args = parser.parse_args()

    main(args)
