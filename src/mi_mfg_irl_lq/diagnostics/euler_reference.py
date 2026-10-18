"""Discrete affine Euler moments; continuous solver flow is only law information."""
from dataclasses import dataclass

import numpy as np

from ..best_response import BestResponse
from ..config import ModelConfig as ExperimentConfig
from ..model import MeanTrajectory


@dataclass(frozen=True)
class EulerMomentEvaluation:
    local_cost: float
    fixed_law_cost: float
    self_law_cost: float
    fixed_objective: float
    self_objective: float
    process_mean: np.ndarray
    second_moment: np.ndarray
    expected_control_square: np.ndarray


from ..costs import validate_policy_inputs, law_costs, left_integral


def evaluate_euler_moments(config: ExperimentConfig, candidate_lambda: float,
                           best_response: BestResponse, expert_mean: MeanTrajectory
                           ) -> EulerMomentEvaluation:
    validate_policy_inputs(config, candidate_lambda, best_response, expert_mean)
    policy = best_response.policy
    mean = np.empty(config.num_time_steps+1)
    second = np.empty_like(mean)
    mean[0] = config.initial_mean
    second[0] = config.initial_variance + config.initial_mean**2
    dt = config.delta_t
    for i in range(config.num_time_steps):
        f = 1 + (config.b_x + config.b_alpha*policy.slope[i])*dt
        g = config.b_alpha*policy.intercept[i]*dt
        mean[i+1] = f*mean[i] + g
        second[i+1] = f**2*second[i] + 2*f*g*mean[i] + g**2 + config.sigma**2*dt
    control2 = policy.slope**2*second + 2*policy.slope*policy.intercept*mean + policy.intercept**2
    local = left_integral(.5*(config.c_alpha*control2+config.c_x*second), config.time_grid)
    fixed, own = law_costs(config, candidate_lambda, best_response, expert_mean)
    if not np.isfinite([local, fixed, own]).all():
        raise FloatingPointError('Nonfinite Euler reference.')
    return EulerMomentEvaluation(local, fixed, own, local+fixed, local+own, mean, second, control2)
