"""Exact expert generation by reduced and independent full DPP systems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp

from .config import ModelConfig as ExperimentConfig
from .model import (
    ASolution,
    AffinePolicy,
    MeanTrajectory,
    linear_value,
    solve_a,
)


@dataclass(frozen=True)
class ExpertEquilibrium:
    """Exact expert and the independent full-system cross-check coefficients."""

    policy: AffinePolicy
    mean: MeanTrajectory
    a: np.ndarray
    q: np.ndarray
    full_a: np.ndarray
    full_b: np.ndarray
    full_c: np.ndarray
    full_d: np.ndarray
    full_mean: np.ndarray
    q_from_full: np.ndarray
    diagnostics: dict[str, object]


def _solve_reduced_system(config: ExperimentConfig, a_solution: ASolution) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Solve the reduced two-point boundary-value system directly in (q, mean)."""

    time = config.time_grid

    def equations(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = linear_value(time, a_solution.values, t)
        q = y[0]
        mean = y[1]
        q_dot = (
            2.0 * config.b_alpha**2 / config.c_alpha * a * q
            - config.b_x * q
            - config.c_mu * (1.0 - config.lambda_star) * mean
        )
        mean_dot = (
            -2.0 * config.b_alpha**2 / config.c_alpha * a * mean
            - config.b_alpha**2 / config.c_alpha * q
            + config.b_x * mean
        )
        return np.vstack((q_dot, mean_dot))

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray([right[0], left[1] - config.initial_mean])

    initial_guess = np.vstack(
        (
            config.c_mu
            * (1.0 - config.lambda_star)
            * config.initial_mean
            * (config.T - time),
            config.initial_mean * np.exp(config.b_x * time),
        )
    )
    solution = solve_bvp(
        equations,
        boundary,
        time,
        initial_guess,
        tol=config.bvp_tol,
        max_nodes=config.bvp_max_nodes,
        verbose=0,
    )
    if not solution.success:
        raise RuntimeError(f"Reduced expert BVP failed: {solution.message}")
    q, mean = np.asarray(solution.sol(time))
    boundary_values = boundary(solution.sol(0.0), solution.sol(config.T))
    diagnostics: dict[str, object] = {
        "success": bool(solution.success),
        "message": str(solution.message),
        "status": int(solution.status),
        "nodes": int(solution.x.size),
        "max_rms_ode_residual": float(np.max(solution.rms_residuals)),
        "max_boundary_residual": float(np.max(np.abs(boundary_values))),
        "q_terminal_residual": float(abs(q[-1])),
        "mean_initial_residual": float(abs(mean[0] - config.initial_mean)),
    }
    return q, mean, diagnostics


def _solve_full_consistency_system(
    config: ExperimentConfig, a_solution: ASolution
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Independently solve the full consistency DPP equations for B, C, D, and mean."""

    time = config.time_grid

    # This literal full-system B equation intentionally does not call the oracle B RHS.
    def b_rhs(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = linear_value(time, a_solution.values, t)
        b = y[0]
        minus_b_dot = (
            2.0 * config.b_x * b
            - 4.0 * config.b_alpha**2 / config.c_alpha * a * b
            - 2.0 * config.b_alpha**2 / config.c_alpha * b**2
            + config.c_mu / 2.0 * (1.0 - config.lambda_star) ** 2
        )
        return np.vstack((-minus_b_dot,))

    def b_boundary(_left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray([right[0]])

    b_guess = np.asarray(
        [
            config.c_mu
            / 2.0
            * (1.0 - config.lambda_star) ** 2
            * (config.T - time)
        ]
    )
    b_solution = solve_bvp(
        b_rhs,
        b_boundary,
        time,
        b_guess,
        tol=config.bvp_tol,
        max_nodes=config.bvp_max_nodes,
        verbose=0,
    )
    if not b_solution.success:
        raise RuntimeError(f"Full expert B solve failed: {b_solution.message}")
    b = np.asarray(b_solution.sol(time)[0])

    def coupled_equations(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = linear_value(time, a_solution.values, t)
        b_at_t = linear_value(time, b, t)
        c = y[0]
        mean = y[1]
        c_dot = (
            -config.b_x * c
            + 2.0 * config.b_alpha**2 / config.c_alpha * (a + b_at_t) * c
            - config.c_mu * config.lambda_star * (1.0 - config.lambda_star) * mean
        )
        mean_dot = (
            config.b_x * mean
            - config.b_alpha**2
            / config.c_alpha
            * (2.0 * (a + b_at_t) * mean + c)
        )
        return np.vstack((c_dot, mean_dot))

    def coupled_boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray([right[0], left[1] - config.initial_mean])

    coupled_guess = np.vstack(
        (
            config.c_mu
            * config.lambda_star
            * (1.0 - config.lambda_star)
            * config.initial_mean
            * (config.T - time),
            config.initial_mean * np.exp(config.b_x * time),
        )
    )
    coupled_solution = solve_bvp(
        coupled_equations,
        coupled_boundary,
        time,
        coupled_guess,
        tol=config.bvp_tol,
        max_nodes=config.bvp_max_nodes,
        verbose=0,
    )
    if not coupled_solution.success:
        raise RuntimeError(f"Full expert C/mean BVP failed: {coupled_solution.message}")
    c, mean = np.asarray(coupled_solution.sol(time))

    def d_rhs(t: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = linear_value(time, a_solution.values, t)
        c_at_t = linear_value(time, c, t)
        mean_at_t = linear_value(time, mean, t)
        minus_d_dot = (
            config.sigma**2 * a
            - config.b_alpha**2 / (2.0 * config.c_alpha) * c_at_t**2
            + config.c_mu / 2.0 * config.lambda_star**2 * mean_at_t**2
        )
        return np.vstack((-minus_d_dot,))

    def d_boundary(_left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.asarray([right[0]])

    d_guess = np.asarray(
        [
            config.c_mu
            / 2.0
            * config.lambda_star**2
            * mean**2
            * (config.T - time)
        ]
    )
    d_solution = solve_bvp(
        d_rhs,
        d_boundary,
        time,
        d_guess,
        tol=config.bvp_tol,
        max_nodes=config.bvp_max_nodes,
        verbose=0,
    )
    if not d_solution.success:
        raise RuntimeError(f"Full expert D solve failed: {d_solution.message}")
    d = np.asarray(d_solution.sol(time)[0])
    boundary_values = coupled_boundary(coupled_solution.sol(0.0), coupled_solution.sol(config.T))
    diagnostics: dict[str, object] = {
        "success": True,
        "b_message": str(b_solution.message),
        "b_nodes": int(b_solution.x.size),
        "b_terminal_residual": float(abs(b[-1])),
        "b_max_rms_ode_residual": float(np.max(b_solution.rms_residuals)),
        "coupled_message": str(coupled_solution.message),
        "coupled_nodes": int(coupled_solution.x.size),
        "coupled_max_rms_ode_residual": float(np.max(coupled_solution.rms_residuals)),
        "coupled_max_boundary_residual": float(np.max(np.abs(boundary_values))),
        "c_terminal_residual": float(abs(c[-1])),
        "mean_initial_residual": float(abs(mean[0] - config.initial_mean)),
        "d_message": str(d_solution.message),
        "d_nodes": int(d_solution.x.size),
        "d_terminal_residual": float(abs(d[-1])),
        "d_max_rms_ode_residual": float(np.max(d_solution.rms_residuals)),
    }
    return b, c, d, mean, diagnostics


def generate_exact_expert(model: ExperimentConfig, lambda_star: float = 0.9) -> ExpertEquilibrium:
    """Generate the exact expert and perform an independent full-DPP cross-check."""

    from types import SimpleNamespace
    from dataclasses import asdict
    config = SimpleNamespace(**asdict(model), lambda_star=lambda_star, time_grid=model.time_grid)
    a_solution = solve_a(config)
    q, reduced_mean, reduced_diagnostics = _solve_reduced_system(config, a_solution)
    full_b, full_c, full_d, full_mean, full_diagnostics = _solve_full_consistency_system(
        config, a_solution
    )
    q_from_full = 2.0 * full_b * full_mean + full_c
    policy = AffinePolicy(
        time=config.time_grid,
        slope=-2.0 * config.b_alpha / config.c_alpha * a_solution.values,
        intercept=-config.b_alpha / config.c_alpha * q,
    )
    diagnostics: dict[str, object] = {
        "a": a_solution.diagnostics,
        "reduced_system": reduced_diagnostics,
        "full_system": full_diagnostics,
        "max_q_identity_error": float(np.max(np.abs(q - q_from_full))),
        "max_reduced_full_mean_error": float(np.max(np.abs(reduced_mean - full_mean))),
    }
    return ExpertEquilibrium(
        policy=policy,
        mean=MeanTrajectory(time=config.time_grid, values=reduced_mean),
        a=a_solution.values,
        q=q,
        full_a=a_solution.values.copy(),
        full_b=full_b,
        full_c=full_c,
        full_d=full_d,
        full_mean=full_mean,
        q_from_full=q_from_full,
        diagnostics=diagnostics,
    )
