"""Optional NVTX ranges for paper-grade Nsight measurements."""

from __future__ import annotations

import ctypes
from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar

_enabled = False
_UNSET = object()
_cached_nvtx: object | None = _UNSET


class _NvtxMessage(ctypes.Union):
    _fields_: ClassVar = [
        ("ascii", ctypes.c_char_p),
        ("unicode", ctypes.c_void_p),
        ("registered", ctypes.c_void_p),
    ]


class _NvtxPayload(ctypes.Union):
    _fields_: ClassVar = [
        ("ull_value", ctypes.c_uint64),
        ("ll_value", ctypes.c_int64),
        ("d_value", ctypes.c_double),
    ]


class _NvtxEventAttributes(ctypes.Structure):
    _fields_: ClassVar = [
        ("version", ctypes.c_uint16),
        ("size", ctypes.c_uint16),
        ("category", ctypes.c_uint32),
        ("color_type", ctypes.c_int32),
        ("color", ctypes.c_uint32),
        ("payload_type", ctypes.c_int32),
        ("reserved0", ctypes.c_int32),
        ("payload", _NvtxPayload),
        ("message_type", ctypes.c_int32),
        ("reserved1", ctypes.c_int32),
        ("message", _NvtxMessage),
    ]


class _CtypesNvtx:
    _NVTX_VERSION = 2
    _NVTX_MESSAGE_TYPE_REGISTERED = 3

    def __init__(self, library: ctypes.CDLL) -> None:
        self.library = library
        self.registered_strings: dict[str, ctypes.c_void_p] = {}
        self.library.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        self.library.nvtxRangePushA.restype = ctypes.c_int
        self.library.nvtxRangePushEx.argtypes = [ctypes.POINTER(_NvtxEventAttributes)]
        self.library.nvtxRangePushEx.restype = ctypes.c_int
        self.library.nvtxRangePop.argtypes = []
        self.library.nvtxRangePop.restype = ctypes.c_int
        self.library.nvtxRangeStartA.argtypes = [ctypes.c_char_p]
        self.library.nvtxRangeStartA.restype = ctypes.c_uint64
        self.library.nvtxRangeStartEx.argtypes = [ctypes.POINTER(_NvtxEventAttributes)]
        self.library.nvtxRangeStartEx.restype = ctypes.c_uint64
        self.library.nvtxRangeEnd.argtypes = [ctypes.c_uint64]
        self.library.nvtxRangeEnd.restype = None
        self.library.nvtxDomainRegisterStringA.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.library.nvtxDomainRegisterStringA.restype = ctypes.c_void_p

    def range_push(self, name: str) -> None:
        self.library.nvtxRangePushA(name.encode("utf-8"))

    def registered_range_push(self, name: str) -> None:
        attributes = self._registered_attributes(name)
        self.library.nvtxRangePushEx(ctypes.byref(attributes))

    def range_pop(self) -> None:
        self.library.nvtxRangePop()

    def range_start(self, name: str) -> int:
        return int(self.library.nvtxRangeStartA(name.encode("utf-8")))

    def registered_range_start(self, name: str) -> int:
        attributes = self._registered_attributes(name)
        return int(self.library.nvtxRangeStartEx(ctypes.byref(attributes)))

    def range_end(self, handle: int) -> None:
        self.library.nvtxRangeEnd(ctypes.c_uint64(handle))

    def _registered_attributes(self, name: str) -> _NvtxEventAttributes:
        attributes = _NvtxEventAttributes()
        attributes.version = self._NVTX_VERSION
        attributes.size = ctypes.sizeof(_NvtxEventAttributes)
        attributes.message_type = self._NVTX_MESSAGE_TYPE_REGISTERED
        attributes.message.registered = self._registered_string(name)
        return attributes

    def _registered_string(self, name: str) -> ctypes.c_void_p:
        handle = self.registered_strings.get(name)
        if handle is None:
            handle = self.library.nvtxDomainRegisterStringA(None, name.encode("utf-8"))
            self.registered_strings[name] = handle
        return handle


def set_enabled(enabled: bool) -> None:
    """Enable or disable benchmark NVTX ranges process-wide."""

    global _enabled
    _enabled = enabled


def is_enabled() -> bool:
    """Return whether benchmark NVTX ranges are currently enabled."""

    return _enabled


@contextmanager
def range(name: str, *, registered: bool = False) -> Iterator[None]:
    """Emit an NVTX range when enabled and CUDA is available."""

    if not _enabled:
        yield
        return

    nvtx = _cuda_nvtx()
    if nvtx is None:
        yield
        return

    if registered and hasattr(nvtx, "registered_range_push"):
        nvtx.registered_range_push(name)
    else:
        nvtx.range_push(name)
    try:
        yield
    finally:
        nvtx.range_pop()


def range_start(name: str, *, registered: bool = False) -> int | None:
    """Open an NVTX start/end range and return its handle, or ``None`` when no range was opened.

    Unlike push/pop ranges, which form a per-thread stack, a start/end range is closed by handle and
    may span any other range. Callers end a range only when this returned a handle.
    """

    if not _enabled:
        return None
    nvtx = _cuda_nvtx()
    if nvtx is None:
        return None
    if registered and hasattr(nvtx, "registered_range_start"):
        return int(nvtx.registered_range_start(name))
    return int(nvtx.range_start(name))


def range_end(handle: int) -> None:
    """Close the NVTX start/end range :func:`range_start` returned ``handle`` for."""

    nvtx = _cuda_nvtx()
    if nvtx is not None:
        nvtx.range_end(handle)


def _cuda_nvtx() -> object | None:
    global _cached_nvtx
    if _cached_nvtx is not _UNSET:
        return _cached_nvtx

    for library_name in ("libnvToolsExt.so.1", "libnvToolsExt.so"):
        try:
            _cached_nvtx = _CtypesNvtx(ctypes.CDLL(library_name))
            return _cached_nvtx
        except OSError:
            continue

    try:
        import torch
    except Exception:
        _cached_nvtx = None
        return None
    if not torch.cuda.is_available():
        _cached_nvtx = None
        return None
    _cached_nvtx = torch.cuda.nvtx
    return _cached_nvtx
