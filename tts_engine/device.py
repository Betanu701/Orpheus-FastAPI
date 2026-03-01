"""
Unified device detection layer for Orpheus TTS engine.

Provides hardware-agnostic GPU detection supporting NVIDIA CUDA, AMD ROCm,
Intel Arc (XPU), and Apple Silicon (MPS), with a clean fallback to CPU.
"""

import torch


def get_device() -> str:
    """Return the best available device string.

    Priority: cuda → xpu → mps → cpu
    """
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA CUDA or AMD ROCm
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"  # Intel Arc
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def get_device_info(device: str) -> dict:
    """Return a dict with device name, vendor, memory_gb, and is_high_end flag.

    For CUDA devices, distinguishes NVIDIA from AMD ROCm by inspecting
    ``torch.version.hip``.
    """
    info: dict = {
        "device": device,
        "vendor": "CPU",
        "name": "CPU",
        "memory_gb": 0,
        "is_high_end": False,
    }

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        mem_gb = props.total_memory / (1024**3)
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        info.update({
            "vendor": "AMD (ROCm)" if is_rocm else "NVIDIA (CUDA)",
            "name": props.name,
            "memory_gb": mem_gb,
            "compute_capability": f"{props.major}.{props.minor}",
            "is_high_end": (
                mem_gb >= 16.0
                or props.major >= 8
                or (mem_gb >= 12.0 and props.major >= 7)
            ),
        })
    elif device == "xpu":
        try:
            props = torch.xpu.get_device_properties(0)
            mem_gb = getattr(props, "total_memory", 0) / (1024**3)
            info.update({
                "vendor": "Intel (XPU)",
                "name": getattr(props, "name", "Intel XPU"),
                "memory_gb": mem_gb,
                "is_high_end": mem_gb >= 12.0,
            })
        except Exception:
            info.update({
                "vendor": "Intel (XPU)",
                "name": "Intel XPU",
                "memory_gb": 0,
                "is_high_end": False,
            })
    elif device == "mps":
        info.update({
            "vendor": "Apple (MPS)",
            "name": "Apple Silicon GPU",
            "is_high_end": False,
        })

    return info


def is_gpu(device: str) -> bool:
    """Return True for GPU-backed devices (cuda, xpu, mps)."""
    return device in ("cuda", "xpu", "mps")


def supports_streams(device: str) -> bool:
    """Return True only for NVIDIA CUDA (not AMD ROCm)."""
    return device == "cuda" and not (
        hasattr(torch.version, "hip") and torch.version.hip
    )
