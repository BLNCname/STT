from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from speech_to_text_app.runtime import configure_cuda_dll_search_paths


@dataclass(frozen=True)
class CudaStatus:
    available: bool
    device_count: int
    supported_compute_types: list[str]
    error: str | None = None


DeviceCountGetter = Callable[[], int]
ComputeTypesGetter = Callable[[str], Iterable[str]]


def _default_device_count_getter() -> int:
    configure_cuda_dll_search_paths()
    import ctranslate2

    return int(ctranslate2.get_cuda_device_count())


def _default_compute_types_getter(device: str) -> Iterable[str]:
    configure_cuda_dll_search_paths()
    import ctranslate2

    return ctranslate2.get_supported_compute_types(device)


def probe_cuda_status(
    device_count_getter: DeviceCountGetter = _default_device_count_getter,
    compute_types_getter: ComputeTypesGetter = _default_compute_types_getter,
) -> CudaStatus:
    try:
        device_count = int(device_count_getter())
        if device_count <= 0:
            return CudaStatus(
                available=False,
                device_count=0,
                supported_compute_types=[],
                error="No CUDA devices detected",
            )
        compute_types = sorted(str(value) for value in compute_types_getter("cuda"))
        if "float16" not in compute_types:
            return CudaStatus(
                available=False,
                device_count=device_count,
                supported_compute_types=compute_types,
                error="CUDA is available, but float16 is not supported",
            )
        return CudaStatus(
            available=True,
            device_count=device_count,
            supported_compute_types=compute_types,
            error=None,
        )
    except Exception as exc:
        return CudaStatus(
            available=False,
            device_count=0,
            supported_compute_types=[],
            error=str(exc),
        )
