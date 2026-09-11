"""Device management utilities for PyTorch and PennyLane."""

import os
import torch
import pennylane as qml


def get_torch_device(preferred_device: str = None) -> torch.device:
    """Returns torch device (CUDA if available, MPS if on Apple Silicon, fallback to CPU)."""
    if preferred_device:
        return torch.device(preferred_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_device_info():
    """Prints diagnostic information about the active compute device."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        total_mem = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
        free_mem = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        print(f"[Device Info] Using CUDA Device {current_device}/{device_count}: {device_name}")
        print(f"[Device Info] VRAM: {free_mem:.2f} GB free / {total_mem:.2f} GB total")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device Info] Using Apple Silicon MPS device.")
    else:
        print("[Device Info] CUDA/MPS not available. Running on CPU.")


def get_quantum_device(wires: int = 6, backend: str = "default.qubit"):
    """Initializes and returns a PennyLane quantum simulation device."""
    return qml.device(backend, wires=wires)


if __name__ == "__main__":
    log_device_info()
    dev = get_quantum_device(6)
    print(f"[Quantum Device] Initialized {dev.name} with {len(dev.wires)} wires.")
