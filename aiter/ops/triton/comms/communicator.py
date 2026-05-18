# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from contextlib import contextmanager
from typing import Union

import torch
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# Match CustomAllreduce default (8 MB).
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
    """Aiter communicator using Iris CCL GPU-initiated communication.

    API mirrors CustomAllreduce: __init__(group, device, max_size),
    should_allreduce, all_reduce (out-of-place), capture, plus disabled.
    """

    _SUPPORTED_WORLD_SIZES = [2, 4, 8]
    _SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
    _HEAP_SIZE = 2**30  # 1 GB

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size: int = _DEFAULT_MAX_SIZE,
    ) -> None:
        self.disabled = True
        self.group = group
        self.max_size = max_size
        self._IS_CAPTURING = False
        self._shmem = None
        self._buf_cache = {}
        self._ws_cache = {}

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

        try:
            import iris
            from iris.ccl.config import Config

            self._shmem = iris.iris(heap_size=self._HEAP_SIZE)
            self._gluon_config = Config(use_gluon=True)
        except Exception as e:
            logger.warning("Failed to initialize Allreduce: %s", e)
            return

        world_size = self._shmem.num_ranks
        if world_size not in self._SUPPORTED_WORLD_SIZES:
            logger.debug(
                "AiterCommunicator disabled: world_size=%d not in %s",
                world_size,
                self._SUPPORTED_WORLD_SIZES,
            )
            return

        self.disabled = False
        logger.info(
            "AiterCommunicator ready: world_size=%d heap=%dGB max_size=%dMB",
            world_size,
            self._HEAP_SIZE >> 30,
            self.max_size >> 20,
        )

    def should_allreduce(self, inp: torch.Tensor) -> bool:
        if self.disabled or self._shmem is None:
            return False
        if not _is_weak_contiguous(inp):
            return False
        inp_size = inp.numel() * inp.element_size()
        if inp_size % 16 != 0:
            return False
        if inp_size >= self.max_size:
            return False
        if inp.dtype not in self._SUPPORTED_DTYPES:
            return False
        if inp_size * 2 > self._HEAP_SIZE:
            return False
        return True

    def _get_buffers(self, shape, dtype):
        key = (shape, dtype)
        if key not in self._buf_cache:
            assert self._shmem is not None
            self._buf_cache[key] = (
                self._shmem.empty(shape, dtype=dtype),
                torch.empty(shape, dtype=dtype, device=self.device),
            )
        return self._buf_cache[key]

    def all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        assert self._shmem is not None
        try:
            input_buf, out = self._get_buffers(inp.shape, inp.dtype)
            input_buf.copy_(inp)

            key = (inp.shape, inp.dtype)
            ws = self._ws_cache.get(key)
            if ws is None:
                ws = self._shmem.ccl.all_reduce_preamble(
                    out, input_buf, config=self._gluon_config
                )
                self._ws_cache[key] = ws
            ws = self._shmem.ccl.all_reduce(
                out,
                input_buf,
                workspace=ws,
                config=self._gluon_config,
                async_op=True,
            )
            self._ws_cache[key] = ws

            return out
        except Exception as e:
            logger.error(
                "AiterCommunicator.all_reduce failed: shape=%s dtype=%s "
                "capturing=%s err=%s",
                tuple(inp.shape),
                inp.dtype,
                torch.cuda.is_current_stream_capturing(),
                e,
            )
            raise

    @contextmanager
    def capture(self):
        from contextlib import ExitStack
        try:
            self._IS_CAPTURING = True
            with ExitStack() as stack:
                for ws in self._ws_cache.values():
                    if hasattr(ws, 'capture'):
                        stack.enter_context(ws.capture())
                yield
        finally:
            self._IS_CAPTURING = False
