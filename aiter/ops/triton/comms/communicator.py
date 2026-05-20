# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from contextlib import contextmanager
from typing import Optional, Union

import torch
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SIZE = 8 * 1024 * 1024


def _iris_available() -> bool:
    try:
        import iris  # noqa: F401
        return True
    except ImportError:
        return False


def _is_weak_contiguous(inp: torch.Tensor) -> bool:
    return inp.is_contiguous() or (
        inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
        == inp.numel() * inp.element_size()
    )


def _rocm_arch_available() -> bool:
    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        return any(gfx in gcn_arch for gfx in ["gfx94", "gfx95"])
    except Exception:
        return False


class AiterCommunicator:
    """Aiter communicator using Iris CCL GPU-initiated all-reduce (v10h).

    Uses mutates_args custom_op with pre-allocated heap buffers for
    CUDA graph replay compatibility. Output buffer from _buf_cache has
    stable address across graph capture and replay.

    Requires both cpu_group (gloo PG for iris init) and device_group
    (NCCL PG for fallback on oversized tensors).
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _HEAP_SIZE = 2**30  # 1GB

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size: int = _DEFAULT_MAX_SIZE,
        device_group: Optional[ProcessGroup] = None,
    ) -> None:
        self.disabled = True
        self.group = group
        self.max_size = max_size
        self._IS_CAPTURING = False
        self._iris_ctx = None
        self._buf_cache: dict = {}

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device

        if not _rocm_arch_available():
            logger.debug("AiterCommunicator disabled: unsupported ROCm arch")
            return

        if not _iris_available():
            logger.warning("Iris library not available. Allreduce disabled.")
            return

        import iris
        import iris.host.memory.symmetric_heap as sh_mod
        import iris.host.distributed.helpers as helpers_mod
        import torch.distributed as dist
        import numpy as np

        # Dedicated gloo PG for iris init collectives only.
        # Prevents iris dist.barrier/allgather from corrupting the TP NCCL PG.
        gloo_pg = dist.new_group(backend="gloo")

        orig_barrier = dist.barrier
        orig_allgather = helpers_mod.distributed_allgather

        def gloo_barrier(*args, **kwargs):
            orig_barrier(group=gloo_pg)

        def gloo_allgather(local_arr, *args, **kwargs):
            world_size = dist.get_world_size()
            local_tensor = torch.tensor(local_arr, dtype=torch.int64, device="cpu")
            gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
            dist.all_gather(gathered, local_tensor, group=gloo_pg)
            return np.concatenate([t.numpy() for t in gathered])

        dist.barrier = gloo_barrier
        helpers_mod.distributed_allgather = gloo_allgather
        sh_mod.distributed_allgather = gloo_allgather

        try:
            self._iris_ctx = iris.iris(heap_size=self._HEAP_SIZE)
        finally:
            dist.barrier = orig_barrier
            helpers_mod.distributed_allgather = orig_allgather
            sh_mod.distributed_allgather = orig_allgather

        world_size = dist.get_world_size()
        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.warning("Unsupported world size %d", world_size)
            return

        max_elems = self.max_size // 2  # max elements for bf16
        self._inp_buf = self._iris_ctx.zeros(max_elems, dtype=torch.bfloat16)
        self._out_buf = self._iris_ctx.zeros(max_elems, dtype=torch.bfloat16)

        from iris.ccl.all_reduce import all_reduce as iris_ccl_all_reduce

        iris_ctx = self._iris_ctx
        inp_buf = self._inp_buf
        out_buf = self._out_buf
        max_size_bytes = self.max_size
        # NCCL device_group for fallback — NOT gloo cpu_group
        nccl_fallback_group = device_group if device_group is not None else group

        @torch.library.custom_op("aiter::iris_all_reduce", mutates_args=("output",))
        def iris_all_reduce(inp: torch.Tensor, output: torch.Tensor) -> None:
            nbytes = inp.numel() * inp.element_size()
            if inp.is_contiguous() and 0 < nbytes <= max_size_bytes:
                if inp.dtype == torch.bfloat16 or inp.dtype == torch.float16:
                    n = inp.numel()
                    inp_view = inp_buf[:n].view(inp.shape)
                    inp_view.copy_(inp)
                    out_view = out_buf[:n].view(inp.shape)
                    iris_ccl_all_reduce(out_view, inp_view, iris_ctx)
                    output.copy_(out_view)
                elif inp.dtype == torch.float32:
                    n = inp.numel()
                    inp_bf16 = inp_buf[:n].view(inp.numel())
                    inp_bf16.copy_(inp.view(-1).to(torch.bfloat16))
                    out_bf16 = out_buf[:n].view(inp.numel())
                    iris_ccl_all_reduce(out_bf16, inp_bf16, iris_ctx)
                    output.copy_(out_bf16.view(inp.shape).to(torch.float32))
                else:
                    output.copy_(inp)
                    torch.distributed.all_reduce(output, group=nccl_fallback_group)
            else:
                output.copy_(inp)
                torch.distributed.all_reduce(output, group=nccl_fallback_group)

        @iris_all_reduce.register_fake
        def _(inp: torch.Tensor, output: torch.Tensor) -> None:
            pass

        logger.info(
            "AiterCommunicator v10h ready: world_size=%d heap=%dGB max_size=%dMB "
            "fallback=%s",
            self._iris_ctx.get_num_ranks(),
            self._HEAP_SIZE >> 30,
            self.max_size >> 20,
            "nccl" if device_group is not None else "cpu_group",
        )
        self.disabled = False

    def _get_buf(self, shape, dtype):
        key = (shape, dtype)
        if key not in self._buf_cache:
            n = 1
            for s in shape:
                n *= s
            nbytes = n * torch._utils._element_size(dtype)
            if nbytes <= self.max_size:
                self._buf_cache[key] = self._out_buf[:n].view(shape)
            else:
                self._buf_cache[key] = torch.empty(
                    shape, dtype=dtype, device=self.device
                )
        return self._buf_cache[key]

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled:
            return False
        return True

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        out = self._get_buf(inp.shape, inp.dtype)
        torch.ops.aiter.iris_all_reduce(inp, out)
        return out

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
