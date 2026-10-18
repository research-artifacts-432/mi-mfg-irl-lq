"""Fixed-rate projected sampled finite differences with fresh paired random banks."""
from time import perf_counter
import numpy as np
from .best_response import BestResponseOracle
from .rollout import SampledPolicyEvaluator, make_random_bank
from .losses import components,loss
from .diagnostics.euler_reference import evaluate_euler_moments

H=1e-4

def finite_difference_stencil(candidate_lambda,step=H):
    if not np.isfinite([candidate_lambda,step]).all() or step<=0 or not 0<=candidate_lambda<=1:
        raise ValueError('Invalid finite-difference inputs.')
    if candidate_lambda-step<0:
        low=candidate_lambda;high=min(1.,candidate_lambda+step);scheme='forward'
    elif candidate_lambda+step>1:
        low=max(0.,candidate_lambda-step);high=candidate_lambda;scheme='backward'
    else: return candidate_lambda-step,candidate_lambda+step,2*step,'central'
    return low,high,high-low,scheme

def projected_update(current,eta,gradient):
    if not np.isfinite([current,eta,gradient]).all() or not 0<=current<=1 or eta<=0:
        raise ValueError('Invalid update inputs.')
    raw=current-eta*gradient
    if not np.isfinite(raw): raise FloatingPointError('Nonfinite update.')
    return raw,float(np.clip(raw,0,1))

def sampled_gradient(model,expert,oracle,evaluator,anchor,bank,paths,version,beta=None):
    low,high,denom,scheme=finite_difference_stencil(anchor)
    points=[]
    for candidate in (low,high):
        br=oracle.solve(candidate,expert.mean_flow)
        ev=evaluator.evaluate_budgets(candidate,br,expert.mean_flow,bank,(paths,))[paths]
        point=components(expert,ev,br.mean)
        point['loss']=loss(point,version,beta);point['local_se']=ev.local_cost_standard_error
        points.append(point)
    return (points[1]['loss']-points[0]['loss'])/denom,low,high,denom,scheme,points

def train(model,expert,*,dataset_id,run_id,version,eta,beta=None,updates=10,paths=4096,
          initial_lambda=.5,optimizer_seed=0,checkpoint_updates=None,progress_callback=None,start_update=0):
    loss(dict(A_rel=0.,C_rel=0.,V2=0.),version,beta)
    if updates<=0 or int(updates)!=updates: raise ValueError('Positive integer update count required.')
    if int(start_update)!=start_update or start_update<0: raise ValueError("Nonnegative integer start update required.")
    projected_update(initial_lambda,eta,0.)
    if not np.array_equal(expert.time,model.time_grid): raise ValueError('Expert/model grid mismatch.')
    selected=(start_update,start_update+updates) if checkpoint_updates is None else tuple(checkpoint_updates)
    if (len(set(selected))!=len(selected) or any(int(k)!=k or k<start_update or k>start_update+updates for k in selected)):
        raise ValueError('Checkpoint updates must be distinct integers within the run.')
    oracle=BestResponseOracle(model);evaluator=SampledPolicyEvaluator(model)
    history=[];checkpoints=[];current=initial_lambda;start=perf_counter()
    def checkpoint(k):
        br=oracle.solve(current,expert.mean_flow)
        ev=evaluate_euler_moments(model,current,br,expert.mean_flow)
        values=components(expert,ev,br.mean)
        checkpoints.append(dict(update=k,lambda_value=current,reference_loss=loss(values,version,beta),**values))
    def snapshot(completed=False,failure=None):
        return dict(run_id=run_id,version=version,beta=beta,eta=eta,updates=len(history),paths=paths,
        final_lambda=current,wall_seconds=perf_counter()-start,br_calls=oracle.solve_calls,
        rollout_calls=evaluator.rollout_calls,rollout_paths=evaluator.rollout_paths,
        simulated_steps=evaluator.simulated_time_steps,random_banks=len(history),
        checkpoints=checkpoints,history=history,completed=completed,failure=failure)
    try:
        if start_update in selected: checkpoint(start_update)
        if progress_callback: progress_callback(snapshot())
        for iteration in range(start_update,start_update+updates):
            tick=perf_counter()
            words=(20261016,int(dataset_id),int(optimizer_seed),iteration)
            bank=make_random_bank(model.num_time_steps,paths,words)
            gradient,low,high,denom,scheme,points=sampled_gradient(model,expert,oracle,evaluator,current,bank,paths,version,beta)
            raw,nxt=projected_update(current,eta,gradient)
            row=dict(run_id=run_id,version=version,beta=beta,eta=eta,update=iteration+1,
                lambda_before=current,raw_next_lambda=raw,lambda_after=nxt,gradient=gradient,
                lower_lambda=low,upper_lambda=high,denominator=denom,scheme=scheme,
                projected=raw!=nxt,J_E=expert.J_E,sJ=expert.sJ,Q=expert.Q,seed_words=str(words),
                wall_seconds=perf_counter()-tick)
            for prefix,point in zip(('lower','upper'),points):
                row.update({prefix+'_'+k:float(v) for k,v in point.items()})
            history.append(row);current=nxt
            if iteration+1 in selected:
                checkpoint(iteration+1)
                if progress_callback: progress_callback(snapshot())
    except BaseException as error:
        if progress_callback:
            progress_callback(snapshot(failure=dict(type=type(error).__name__,message=str(error))))
        raise
    result=snapshot(completed=True)
    if progress_callback: progress_callback(result)
    return result
