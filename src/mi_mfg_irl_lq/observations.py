"""Observed sufficient statistics and deterministic-round-seed expert sampling."""
from dataclasses import dataclass
import numpy as np
from .dynamics import euler_maruyama_update
from .costs import left_integral
from .expert_data import ExpertData

@dataclass(frozen=True)
class Observations:
    time: np.ndarray
    mean: np.ndarray
    state_second: np.ndarray
    action_second: np.ndarray
    count: int


def preprocess(model, observations):
    o=observations
    if not np.array_equal(o.time,model.time_grid) or o.count<=0: raise ValueError('Invalid observed grid/count.')
    for a in (o.mean,o.state_second,o.action_second):
        if a.shape!=o.time.shape or not np.isfinite(a).all(): raise ValueError('Malformed observed statistics.')
    if np.any(o.state_second<0) or np.any(o.action_second<0): raise ValueError('Negative observed squares.')
    running=.5*(model.c_x*o.state_second+model.c_alpha*o.action_second+model.c_mu*o.mean**2)
    return ExpertData(o.time,o.mean,left_integral(running,o.time))


def simulate_observations(model, policy, *, dataset_seed, random_seed_base, setting_index, num_agents, num_rounds):
    """Preserve original RNG draw order: X0, then one normal vector per EM step."""
    if not np.array_equal(policy.time,model.time_grid): raise ValueError('Policy grid mismatch.')
    if min(num_agents,num_rounds)<=0: raise ValueError('Positive observation counts required.')
    totals=np.zeros((3,model.num_time_steps+1))
    for r in range(num_rounds):
        rng=np.random.default_rng(np.random.SeedSequence([random_seed_base,setting_index,dataset_seed,r]))
        x=rng.normal(loc=model.initial_mean,scale=np.sqrt(model.initial_variance),size=num_agents)
        for n in range(model.num_time_steps+1):
            a=policy.slope[n]*x+policy.intercept[n]
            totals[:,n]+=[float(np.sum(x)),float(np.dot(x,x)),float(np.dot(a,a))]
            if n<model.num_time_steps:
                x=euler_maruyama_update(x,a,rng.standard_normal(num_agents),model)
    totals/=num_agents*num_rounds
    return Observations(model.time_grid,totals[0],totals[1],totals[2],num_agents*num_rounds)
