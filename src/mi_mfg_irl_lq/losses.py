"""Final relative V1 and unnormalized V2 objectives."""
import numpy as np

def components(expert, evaluation, br_mean):
    if not np.array_equal(br_mean.time,expert.time): raise ValueError('Mean grid mismatch.')
    dc=float(np.trapezoid((br_mean.values-expert.mean)**2,expert.time))
    result=dict(J_fixed=float(evaluation.fixed_objective),J_self=float(evaluation.self_objective),
        D_C=dc,A_rel=(expert.J_E-evaluation.fixed_objective)**2/expert.sJ,
        C_rel=dc/expert.Q,V2=(expert.J_E-evaluation.self_objective)**2)
    if not np.isfinite(list(result.values())).all(): raise FloatingPointError('Nonfinite objective.')
    return result

def loss(values,version,beta=None):
    if version=='V1' and beta is not None and np.isfinite(beta) and beta>=0:
        return float(values['A_rel']+beta*values['C_rel'])
    if version=='V2' and beta is None: return float(values['V2'])
    raise ValueError('V1 needs nonnegative beta; V2 has no beta.')
