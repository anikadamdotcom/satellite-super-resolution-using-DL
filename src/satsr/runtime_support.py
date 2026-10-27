"""Keep CPU inference responsive on ordinary laptops."""
import os
import torch


def configure_cpu_threads():
    # Small overlapping tiles run poorly with a very large OpenMP thread pool.
    # Keep a caller's smaller limit (including test settings) and avoid changing
    # inter-op threads, which PyTorch only allows before parallel work begins.
    limit = min(4, os.cpu_count() or 1, torch.get_num_threads())
    if torch.get_num_threads() != limit:
        torch.set_num_threads(limit)
    return limit
