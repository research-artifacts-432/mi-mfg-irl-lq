"""Core data structures and shared scalar ODE utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .config import ModelConfig as ExperimentConfig


@dataclass(frozen=True)
class AffinePolicy:
    """Time-grid representation of alpha(t, x) = slope(t) * x + intercept(t)."""

    time: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray


@dataclass(frozen=True)
class MeanTrajectory:
    """Time-grid representation of a population mean trajectory."""

    time: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ASolution:
    """Cached Riccati coefficient and its numerical diagnostics."""

    time: np.ndarray
    values: np.ndarray
    diagnostics: dict[str, float | int | bool | str]


def linear_value(time: np.ndarray, values: np.ndarray, query: float | np.ndarray) -> np.ndarray:
    """Evaluate a time-grid array using the approved linear interpolation."""

    return np.interp(query, time, values)


def a_dpp_rhs(a: float | np.ndarray, config: ExperimentConfig) -> float | np.ndarray:
    """Right-hand side of -dot(A) in the DPP equation."""

    return (
        -2.0 * config.b_alpha**2 / config.c_alpha * a**2
        + 2.0 * config.b_x * a
        + config.c_x / 2.0
    )


def b_dpp_rhs(
    a: float | np.ndarray,
    b: float | np.ndarray,
    candidate_lambda: float,
    config: ExperimentConfig,
) -> float | np.ndarray:
    """Right-hand side of -dot(B), including the mandatory A B cross term."""

    return (
        2.0 * config.b_x * b
        - 4.0 * config.b_alpha**2 / config.c_alpha * a * b
        - 2.0 * config.b_alpha**2 / config.c_alpha * b**2
        + config.c_mu / 2.0 * (1.0 - candidate_lambda) ** 2
    )


def max_ivp_simpson_defect(
    solution: object,
    rhs: Callable[[float, np.ndarray], np.ndarray],
    time_grid: np.ndarray,
) -> float:
    """Measure the maximum per-unit-time Simpson defect on the reporting grid."""

    dense = getattr(solution, "sol")
    left = time_grid[:-1]
    right = time_grid[1:]
    middle = 0.5 * (left + right)
    y_left = np.asarray(dense(left))
    y_middle = np.asarray(dense(middle))
    y_right = np.asarray(dense(right))
    f_left = np.column_stack([rhs(float(t), y_left[:, i]) for i, t in enumerate(left)])
    f_middle = np.column_stack(
        [rhs(float(t), y_middle[:, i]) for i, t in enumerate(middle)]
    )
    f_right = np.column_stack([rhs(float(t), y_right[:, i]) for i, t in enumerate(right)])
    widths = right - left
    defect = y_right - y_left - widths * (f_left + 4.0 * f_middle + f_right) / 6.0
    return float(np.max(np.abs(defect / widths)))


def solve_a(config: ExperimentConfig) -> ASolution:
    """Solve the lambda-independent Riccati equation backward once."""

    time = config.time_grid

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray([-a_dpp_rhs(y[0], config)])

    solution = solve_ivp(
        rhs,
        (config.T, 0.0),
        np.asarray([0.0]),
        method=config.ivp_method,
        rtol=config.ivp_rtol,
        atol=config.ivp_atol,
        dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(f"A solve failed: {solution.message}")
    values = np.asarray(solution.sol(time)[0])
    diagnostics: dict[str, float | int | bool | str] = {
        "success": bool(solution.success),
        "message": str(solution.message),
        "nfev": int(solution.nfev),
        "terminal_residual": float(abs(values[-1])),
        "max_ode_simpson_defect": max_ivp_simpson_defect(solution, rhs, time),
    }
    return ASolution(time=time, values=values, diagnostics=diagnostics)
