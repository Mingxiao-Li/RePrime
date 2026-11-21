"""Torchrun initialization, parameter broadcast, and checkpoint-on-USR1."""
from datetime import timedelta
import os
import signal

import torch
import torch.distributed as dist

_stop_requested = False


def initialize(stage="Pretraining"):
    if not torch.cuda.is_available():
        raise RuntimeError(f"{stage} requires CUDA; launch with torchrun on eight GPUs")
    # H200 test session adaptation: allow a 2-GPU smoke run. The production
    # recipe trajectory uses exactly eight ranks; 2-rank runs follow a different
    # (world-size-equivalent-math) sample stream and are for pipeline testing only.
    if int(os.environ.get("WORLD_SIZE", "0")) not in (1, 2, 8):
        raise ValueError(f"The production {stage.lower()} recipes require eight torchrun ranks (2 allowed for smoke tests)")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    dist.barrier(device_ids=[int(os.environ["LOCAL_RANK"])])
    signal.signal(signal.SIGUSR1, _request_stop)
    return dist.get_rank(), dist.get_world_size()


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True


def stopping(device):
    flag = os.environ.get("PREEMPT_FLAG_FILE")
    value = torch.tensor(int(_stop_requested or bool(flag and os.path.exists(flag))), device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return bool(value.item())


def broadcast_model(model):
    for value in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(value.data, src=0)
