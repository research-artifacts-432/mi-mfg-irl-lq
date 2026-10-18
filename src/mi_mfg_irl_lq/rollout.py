"""Stream sampled affine policies with left running cost and solver law means."""
from dataclasses import dataclass

import numpy as np

from .best_response import BestResponse, BestResponseOracle
from .config import ModelConfig as ExperimentConfig
from .costs import law_costs, validate_policy_inputs
from .model import MeanTrajectory
from .dynamics import euler_maruyama_update


@dataclass(frozen=True)
class RandomBank:
    initial_standard_normals: np.ndarray
    brownian_standard_normals: np.ndarray
    seed_words: tuple[int, ...]


def make_random_bank(num_time_steps: int, paths: int, seed_words: tuple[int, ...]) -> RandomBank:
    if num_time_steps <= 0 or paths < 2:
        raise ValueError('Positive steps and at least two paths are required.')
    generator = np.random.default_rng(np.random.SeedSequence(seed_words))
    initial = generator.standard_normal(paths)
    brownian = generator.standard_normal((num_time_steps, paths))
    initial.flags.writeable = brownian.flags.writeable = False
    return RandomBank(initial, brownian, tuple(seed_words))


@dataclass(frozen=True)
class SampledPolicyEvaluation:
    paths: int
    local_cost: float
    local_cost_standard_error: float
    fixed_law_cost: float
    self_law_cost: float
    fixed_objective: float
    self_objective: float
    sampled_mean: np.ndarray
    sampled_state_second_moment: np.ndarray
    sampled_action_second_moment: np.ndarray


@dataclass(frozen=True)
class LocalPolicyEvaluation:
    paths: int
    local_cost: float
    local_cost_standard_error: float
    sampled_mean: np.ndarray
    sampled_state_second_moment: np.ndarray
    sampled_action_second_moment: np.ndarray


class SampledPolicyEvaluator:
    """Stream one process per candidate; all nested budgets/fixed/self share it."""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.evaluation_calls = 0
        self.rollout_calls = 0
        self.rollout_paths = 0
        self.simulated_time_steps = 0

    def evaluate_budgets(self, candidate_lambda: float, best_response: BestResponse,
                         expert_mean: MeanTrajectory, bank: RandomBank,
                         budgets: tuple[int, ...]) -> dict[int, SampledPolicyEvaluation]:
        config = self.config
        validate_policy_inputs(config, candidate_lambda, best_response, expert_mean)
        fixed, own = law_costs(config, candidate_lambda, best_response, expert_mean)
        locals = self.evaluate_local_policy_budgets(best_response.policy, bank, budgets)
        return {n: SampledPolicyEvaluation(n, value.local_cost,
            value.local_cost_standard_error, fixed, own, value.local_cost+fixed,
            value.local_cost+own, value.sampled_mean, value.sampled_state_second_moment,
            value.sampled_action_second_moment) for n, value in locals.items()}

    def evaluate_local_policy_budgets(self, policy, bank: RandomBank, budgets):
        """Shared fixed-policy EM engine; no candidate or law-cost argument.

        The pre-existing BR engine's arithmetic, counters and path prefixes are
        preserved. Expert-policy adapters supply their own explicit plug-in law cost.
        """
        config = self.config
        for array in (policy.time,):
            if not np.array_equal(array, config.time_grid):
                raise ValueError('Policy grid mismatch.')
        for array in (policy.slope, policy.intercept):
            if np.asarray(array).shape != config.time_grid.shape or not np.isfinite(array).all():
                raise ValueError('Nonfinite or malformed policy.')
        budgets = tuple(sorted(set(budgets)))
        if not budgets or budgets[0] < 2 or any(int(n) != n for n in budgets):
            raise ValueError('Budgets must be integer path counts >= 2.')
        largest = max(budgets)
        if (bank.initial_standard_normals.ndim != 1
            or bank.initial_standard_normals.size < largest
            or bank.brownian_standard_normals.shape != (config.num_time_steps, bank.initial_standard_normals.size)):
            raise ValueError('Random bank shape/budget does not match configuration.')
        if not np.isfinite(bank.initial_standard_normals[:largest]).all() or not np.isfinite(bank.brownian_standard_normals[:, :largest]).all():
            raise ValueError('Random bank must be finite.')
        # Copies isolate this candidate's state process from the shared CRN bank.
        states = config.initial_mean + np.sqrt(config.initial_variance)*bank.initial_standard_normals[:largest]
        costs = np.zeros(largest)
        diagnostics = {n: np.empty((3, config.num_time_steps+1)) for n in budgets}
        self.rollout_calls += 1
        self.rollout_paths += largest
        self.simulated_time_steps += largest*config.num_time_steps
        for i in range(config.num_time_steps+1):
            actions = policy.slope[i]*states + policy.intercept[i]
            state2, action2 = states**2, actions**2
            weight = config.delta_t if i < config.num_time_steps else 0.
            costs += weight*.5*(config.c_alpha*action2 + config.c_x*state2)
            for n in budgets:
                diagnostics[n][:, i] = [np.mean(states[:n]), np.mean(state2[:n]), np.mean(action2[:n])]
            if i < config.num_time_steps:
                states = euler_maruyama_update(states, actions,
                            bank.brownian_standard_normals[i, :largest], config)
        if not np.isfinite(costs).all():
            raise FloatingPointError('Nonfinite sampled local cost.')
        self.evaluation_calls += len(budgets)
        results = {}
        for n in budgets:
            local = float(np.mean(costs[:n]))
            se = float(np.std(costs[:n], ddof=1)/np.sqrt(n))
            arrays = diagnostics[n]
            results[n] = LocalPolicyEvaluation(n, local, se, arrays[0], arrays[1], arrays[2])
        return results
