"""Shared validation and solver-flow law costs."""
import numpy as np
from .config import ModelConfig as ExperimentConfig
from .best_response import BestResponse
from .model import MeanTrajectory

def left_integral(values, time):
    time=np.asarray(time,dtype=float); values=np.asarray(values,dtype=float)
    if time.ndim!=1 or time.size<2 or values.shape!=time.shape:
        raise ValueError('Left cost needs matching grid arrays.')
    dt=(time[-1]-time[0])/(time.size-1)
    if dt<=0 or not np.allclose(np.diff(time),dt,rtol=0,atol=1e-14):
        raise ValueError('Left cost requires the uniform model grid.')
    if not np.isfinite(values).all(): raise FloatingPointError('Nonfinite running costs.')
    return float(dt*np.sum(values[:-1]))

def validate_policy_inputs(config, candidate_lambda, best_response, expert_mean):
    if not 0 <= candidate_lambda <= 1:
        raise ValueError('candidate_lambda must lie in [0, 1].')
    for a in (best_response.policy.time, best_response.mean.time, expert_mean.time):
        if not np.array_equal(a, config.time_grid):
            raise ValueError('All inputs must use the configured grid.')
    for a in (best_response.policy.slope, best_response.policy.intercept,
              best_response.mean.values, expert_mean.values):
        if np.asarray(a).shape != config.time_grid.shape or not np.isfinite(a).all():
            raise ValueError('Nonfinite or malformed policy/flow.')
    if config.initial_variance < 0 or not np.isfinite([
            config.initial_mean, config.initial_variance, config.sigma]).all():
        raise ValueError('Invalid Gaussian initial law or diffusion.')


def law_costs(config: ExperimentConfig, candidate_lambda: float,
              best_response: BestResponse, expert_mean: MeanTrajectory):
    """No sampled-process input is accepted here; solver mean is mandatory."""
    mixed = candidate_lambda * expert_mean.values + (1-candidate_lambda) * best_response.mean.values
    return (left_integral(.5*config.c_mu*mixed**2, config.time_grid),
            left_integral(.5*config.c_mu*best_response.mean.values**2, config.time_grid))

