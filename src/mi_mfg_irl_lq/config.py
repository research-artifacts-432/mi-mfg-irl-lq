"""Known model information, independent of synthetic generation truth."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class ModelConfig:
    T: float = 1.
    b_alpha: float = 1.
    b_x: float = .5
    c_alpha: float = 1.
    c_x: float = 1.
    c_mu: float = 1.
    sigma: float = .1
    initial_mean: float = .5
    initial_variance: float = .01
    initial_second_moment: float = .26
    num_time_steps: int = 1000
    ivp_method: str = 'DOP853'
    ivp_rtol: float = 1e-10
    ivp_atol: float = 1e-12
    bvp_tol: float = 1e-9
    bvp_max_nodes: int = 50000

    def __post_init__(self):
        numbers = [v for v in self.__dict__.values() if isinstance(v, (int, float))]
        if not np.isfinite(numbers).all(): raise ValueError('Model parameters must be finite.')
        if (self.T <= 0 or self.num_time_steps < 2 or int(self.num_time_steps) != self.num_time_steps
            or min(self.c_alpha,self.c_x,self.c_mu,self.ivp_rtol,self.ivp_atol,self.bvp_tol) <= 0
            or min(self.sigma,self.initial_variance) < 0): raise ValueError('Invalid model parameters.')
        if not np.isclose(self.initial_second_moment,self.initial_mean**2+self.initial_variance,rtol=0,atol=1e-14):
            raise ValueError('Inconsistent known initial moments.')

    @property
    def time_grid(self): return np.linspace(0.,self.T,self.num_time_steps+1)
    @property
    def delta_t(self): return self.T/self.num_time_steps

    @classmethod
    def load(cls, path): return cls(**json.loads(Path(path).read_text()))
