"""Algorithm 2': pre-consistency model-based LQ best-response oracle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_bvp

from .config import ModelConfig as ExperimentConfig
from .model import (
    ASolution,
    AffinePolicy,
    MeanTrajectory,
    b_dpp_rhs,
    linear_value,
    solve_a,
)


@dataclass(frozen=True)
class BestResponse:
    """The intentionally narrow output of Algorithm 2'."""

    policy: AffinePolicy
    mean: MeanTrajectory


@dataclass(frozen=True)
class OracleValidationRecord:
    """Internal DPP coefficients retained solely for validation and diagnostics."""

    candidate_lambda: float
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    diagnostics: dict[str, object]


class BestResponseOracle:
    """Exact fixed-expert-flow best-response oracle with a cached A solution."""

    def __init__(self, config: ExperimentConfig, cached_a: ASolution | None = None):
        self.config = config
        if cached_a is not None and not np.array_equal(cached_a.time, config.time_grid):
            raise ValueError("A cached solution must use the configured time grid.")
        self._a_solution = cached_a or solve_a(config)
        self._last_validation: OracleValidationRecord | None = None
        self.solve_calls = 0

    @property
    def last_validation(self) -> OracleValidationRecord:
        """Return the most recent internal validation record without changing the oracle output."""

        if self._last_validation is None:
            raise RuntimeError("The oracle has not been solved yet.")
        return self._last_validation

    @property
    def cached_a(self) -> ASolution:
        return self._a_solution

    def solve(self, candidate_lambda: float, frozen_expert_mean: MeanTrajectory) -> BestResponse:
        """Return only the affine policy and its induced mean trajectory."""

        if not 0.0 <= candidate_lambda <= 1.0:
            raise ValueError("candidate_lambda must lie in [0, 1].")
        time = self.config.time_grid
        if not np.array_equal(frozen_expert_mean.time, time):
            raise ValueError("The frozen expert mean must use the configured time grid.")
        if frozen_expert_mean.values.shape != time.shape or not np.isfinite(frozen_expert_mean.values).all():
            raise ValueError("Malformed frozen expert mean.")
        self.solve_calls += 1

        b, b_solution, b_rhs = self._solve_b(candidate_lambda)
        c, d, cd_solution, cd_rhs = self._solve_c_and_d(
            candidate_lambda, b, frozen_expert_mean.values
        )
        mean, mean_solution, mean_rhs = self._solve_mean(b, c)

        policy = AffinePolicy(
            time=time,
            slope=-2.0
            * self.config.b_alpha
            / self.config.c_alpha
            * self._a_solution.values,
            intercept=-self.config.b_alpha
            / self.config.c_alpha
            * (2.0 * b * mean + c),
        )
        diagnostics: dict[str, object] = {
            "success": True,
            "a": self._a_solution.diagnostics,
            "b": {
                "success": bool(b_solution.success),
                "message": str(b_solution.message),
                "nodes": int(b_solution.x.size),
                "terminal_residual": float(abs(b[-1])),
                "max_rms_ode_residual": float(np.max(b_solution.rms_residuals)),
            },
            "c_d": {
                "success": bool(cd_solution.success),
                "message": str(cd_solution.message),
                "nodes": int(cd_solution.x.size),
                "c_terminal_residual": float(abs(c[-1])),
                "d_terminal_residual": float(abs(d[-1])),
                "max_rms_ode_residual": float(np.max(cd_solution.rms_residuals)),
            },
            "mean": {
                "success": bool(mean_solution.success),
                "message": str(mean_solution.message),
                "nodes": int(mean_solution.x.size),
                "initial_residual": float(abs(mean[0] - self.config.initial_mean)),
                "max_rms_ode_residual": float(np.max(mean_solution.rms_residuals)),
            },
        }
        self._last_validation = OracleValidationRecord(
            candidate_lambda=candidate_lambda,
            a=self._a_solution.values.copy(),
            b=b,
            c=c,
            d=d,
            diagnostics=diagnostics,
        )
        return BestResponse(
            policy=policy,
            mean=MeanTrajectory(time=time, values=mean),
        )

    def _solve_b(self, candidate_lambda: float) -> tuple[np.ndarray, object, object]:
        time = self.config.time_grid

        def rhs(t: np.ndarray, y: np.ndarray) -> np.ndarray:
            a = linear_value(time, self._a_solution.values, t)
            return np.vstack((-b_dpp_rhs(a, y[0], candidate_lambda, self.config),))

        def boundary(_left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.asarray([right[0]])

        guess = np.asarray(
            [
                self.config.c_mu
                / 2.0
                * (1.0 - candidate_lambda) ** 2
                * (self.config.T - time)
            ]
        )
        solution = solve_bvp(
            rhs,
            boundary,
            time,
            guess,
            tol=self.config.bvp_tol,
            max_nodes=self.config.bvp_max_nodes,
            verbose=0,
        )
        if not solution.success:
            raise RuntimeError(f"B solve failed at lambda={candidate_lambda}: {solution.message}")
        return np.asarray(solution.sol(time)[0]), solution, rhs

    def _solve_c_and_d(
        self, candidate_lambda: float, b: np.ndarray, frozen_mean: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, object, object]:
        time = self.config.time_grid

        def rhs(t: np.ndarray, y: np.ndarray) -> np.ndarray:
            a = linear_value(time, self._a_solution.values, t)
            b_at_t = linear_value(time, b, t)
            mean_at_t = linear_value(time, frozen_mean, t)
            c = y[0]
            minus_c_dot = (
                self.config.b_x * c
                - 2.0
                * self.config.b_alpha**2
                / self.config.c_alpha
                * (a + b_at_t)
                * c
                + self.config.c_mu
                * candidate_lambda
                * (1.0 - candidate_lambda)
                * mean_at_t
            )
            minus_d_dot = (
                self.config.sigma**2 * a
                - self.config.b_alpha**2 / (2.0 * self.config.c_alpha) * c**2
                + self.config.c_mu / 2.0 * candidate_lambda**2 * mean_at_t**2
            )
            return np.vstack((-minus_c_dot, -minus_d_dot))

        def boundary(_left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.asarray([right[0], right[1]])

        guess = np.vstack(
            (
                self.config.c_mu
                * candidate_lambda
                * (1.0 - candidate_lambda)
                * frozen_mean
                * (self.config.T - time),
                self.config.c_mu
                / 2.0
                * candidate_lambda**2
                * frozen_mean**2
                * (self.config.T - time),
            )
        )
        solution = solve_bvp(
            rhs,
            boundary,
            time,
            guess,
            tol=self.config.bvp_tol,
            max_nodes=self.config.bvp_max_nodes,
            verbose=0,
        )
        if not solution.success:
            raise RuntimeError(
                f"C/D solve failed at lambda={candidate_lambda}: {solution.message}"
            )
        c, d = np.asarray(solution.sol(time))
        return c, d, solution, rhs

    def _solve_mean(self, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, object, object]:
        time = self.config.time_grid

        def rhs(t: np.ndarray, y: np.ndarray) -> np.ndarray:
            a = linear_value(time, self._a_solution.values, t)
            b_at_t = linear_value(time, b, t)
            c_at_t = linear_value(time, c, t)
            mean = y[0]
            mean_dot = (
                self.config.b_x * mean
                - self.config.b_alpha**2
                / self.config.c_alpha
                * (2.0 * (a + b_at_t) * mean + c_at_t)
            )
            return np.vstack((mean_dot,))

        def boundary(left: np.ndarray, _right: np.ndarray) -> np.ndarray:
            return np.asarray([left[0] - self.config.initial_mean])

        guess = np.asarray([self.config.initial_mean * np.exp(self.config.b_x * time)])
        solution = solve_bvp(
            rhs,
            boundary,
            time,
            guess,
            tol=self.config.bvp_tol,
            max_nodes=self.config.bvp_max_nodes,
            verbose=0,
        )
        if not solution.success:
            raise RuntimeError(f"Mean solve failed: {solution.message}")
        return np.asarray(solution.sol(time)[0]), solution, rhs
