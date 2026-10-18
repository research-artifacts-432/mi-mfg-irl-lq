"""Narrow immutable inverse inputs: grid, empirical mean and observed J only."""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from .model import MeanTrajectory

@dataclass(frozen=True, slots=True)
class ExpertData:
    time: np.ndarray
    mean: np.ndarray
    J_E: float
    sJ: float = field(init=False)
    Q: float = field(init=False)

    def __post_init__(self):
        time=np.array(self.time,dtype=float,copy=True); mean=np.array(self.mean,dtype=float,copy=True)
        if (time.ndim!=1 or time.size<3 or mean.shape!=time.shape
            or not np.isfinite(time).all() or not np.isfinite(mean).all()
            or np.any(np.diff(time)<=0) or not np.isfinite(self.J_E)):
            raise ValueError('Malformed expert mean/J inputs.')
        sj=float(self.J_E**2); q=float(np.trapezoid(mean**2,time))
        if sj<=1e-14 or q<=1e-14: raise ValueError('Degenerate normalization; no flooring.')
        time.flags.writeable=mean.flags.writeable=False
        object.__setattr__(self,'time',time);object.__setattr__(self,'mean',mean)
        object.__setattr__(self,'sJ',sj);object.__setattr__(self,'Q',q)

    @property
    def mean_flow(self): return MeanTrajectory(self.time,self.mean)


def load_expert(path: str | Path, model) -> ExpertData:
    with np.load(path,allow_pickle=False) as z:
        expert=ExpertData(z['time'],z['empirical_mean'],float(z['J_left']))
        cached_sj=float(z['sJ']); cached_q=float(z['Q'])
        # Cached scales verify the inputs; ExpertData supplies the optimizer scales.
        if (not np.isfinite([cached_sj,cached_q]).all()
            or not np.isclose(cached_sj,expert.sJ,rtol=0,atol=1e-15)
            or not np.isclose(cached_q,expert.Q,rtol=0,atol=1e-15)):
            raise ValueError('Stale cached dataset scales.')
    if not np.array_equal(expert.time,model.time_grid): raise ValueError('Dataset/model grid mismatch.')
    return expert
