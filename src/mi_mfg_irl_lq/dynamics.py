"""Accepted Euler-Maruyama state update."""
import numpy as np
from .config import ModelConfig as ExperimentConfig

def euler_maruyama_update(
    states: np.ndarray,
    actions: np.ndarray,
    standard_normals: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """Apply one project-source Euler--Maruyama state update."""

    states = np.asarray(states, dtype=float)
    actions = np.asarray(actions, dtype=float)
    standard_normals = np.asarray(standard_normals, dtype=float)
    if states.shape != actions.shape or states.shape != standard_normals.shape:
        raise ValueError("States, actions, and normal increments must share one shape.")
    return (
        states
        + (
            config.b_alpha * actions
            + config.b_x * states
        )
        * config.delta_t
        + config.sigma * np.sqrt(config.delta_t) * standard_normals
    )

