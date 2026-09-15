"""Focused numerical-kernel benchmarks, separate from training workloads."""

import time

import torch

from AADL.anderson_acceleration import anderson_qr_factorization


def benchmark_anderson_kernel(
    parameter_count=100_000, history_depth=10, repeats=10,
    sketch_fraction=1.0, dtype=torch.float32, device="cpu", seed=0,
):
    """Time coefficient estimation while retaining full extrapolation work."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    history = torch.randn(parameter_count, history_depth, generator=generator, dtype=dtype)
    history = history.to(device)
    row_indices = None
    if sketch_fraction < 1.0:
        rows = max(history_depth - 2, int(parameter_count * sketch_fraction))
        row_indices = torch.linspace(0, parameter_count - 1, rows, device=device).long()
    # Warm up lazy libraries and accelerator kernels.
    anderson_qr_factorization(
        history, 1.0, 1e-8, dtype, row_indices=row_indices,
    )
    if history.device.type == "cuda":
        torch.cuda.synchronize(history.device)
    started = time.perf_counter()
    for _ in range(repeats):
        anderson_qr_factorization(
            history, 1.0, 1e-8, dtype, row_indices=row_indices,
        )
    if history.device.type == "cuda":
        torch.cuda.synchronize(history.device)
    elapsed = time.perf_counter() - started
    return {
        "parameter_count": parameter_count,
        "history_depth": history_depth,
        "repeats": repeats,
        "sketch_fraction": sketch_fraction,
        "seconds": elapsed,
        "seconds_per_call": elapsed / repeats,
    }
