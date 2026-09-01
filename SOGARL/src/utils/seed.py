"""
seed.py

Reproducibility utilities for SOGARL.

Handles:
    - Python random seed
    - NumPy random seed
    - PyTorch CPU seed
    - PyTorch CUDA seed
    - optional deterministic behavior

This module does NOT:
    - sample scenarios
    - select files
    - build prompts
    - generate responses
    - calculate rewards
    - perform GRPO
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(
    seed: int = 42,
    deterministic: bool = False,
) -> None:
    """
    Set the global random seed for the SOGARL experiment.

    Args:
        seed:
            Experiment seed.

        deterministic:
            If True, request deterministic PyTorch behavior
            where supported.

    Note:
        Deterministic mode can reduce performance and some
        CUDA operations may still have implementation-specific
        behavior.
    """

    if not isinstance(seed, int):
        raise TypeError(
            "seed must be an integer."
        )

    if seed < 0:
        raise ValueError(
            "seed must be non-negative."
        )

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:

        configure_deterministic_mode()


def configure_deterministic_mode() -> None:
    """
    Request deterministic PyTorch behavior.

    This is mainly useful for debugging and controlled
    experiments rather than maximum training performance.
    """

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(
            True
        )
    except RuntimeError:
        pass


def configure_performance_mode() -> None:
    """
    Configure CUDA/cuDNN for normal training performance.

    This is the preferred mode for the main Kaggle training run.
    """

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    try:
        torch.use_deterministic_algorithms(
            False
        )
    except RuntimeError:
        pass


def get_rng_state() -> dict:
    """
    Return the current global RNG states.

    Useful when additional manual experiment-state handling
    is required.

    CheckpointManager already performs this automatically
    when saving checkpoints.
    """

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():

        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state: dict,
) -> None:
    """
    Restore previously captured RNG states.
    """

    if not isinstance(state, dict):
        raise TypeError(
            "state must be a dictionary."
        )

    if "python" in state:

        random.setstate(
            state["python"]
        )

    if "numpy" in state:

        np.random.set_state(
            state["numpy"]
        )

    if "torch" in state:

        torch.set_rng_state(
            state["torch"]
        )

    if (
        "cuda" in state
        and torch.cuda.is_available()
    ):

        torch.cuda.set_rng_state_all(
            state["cuda"]
        )


def get_seed_info(
    seed: int,
) -> dict:
    """
    Return basic seed information for experiment logging.
    """

    return {
        "seed": seed,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": (
            torch.cuda.device_count()
            if torch.cuda.is_available()
            else 0
        ),
    }