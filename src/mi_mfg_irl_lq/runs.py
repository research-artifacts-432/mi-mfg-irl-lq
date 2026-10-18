"""Common experiment configuration, sampled run execution and accounting."""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
from .config import ModelConfig
from .expert_data import load_expert
from .optimization import train,H
from .io import write_json

def load_experiment_config(path):
    spec=json.loads(Path(path).read_text())
    def integer(value, minimum=0):
        return type(value) is int and value>=minimum
    def number(value, minimum=0, positive=False):
        return (type(value) in (int,float) and np.isfinite(value)
                and (value>minimum if positive else value>=minimum))
    def unique_integers(values):
        return (isinstance(values,list) and bool(values)
                and all(integer(v) for v in values) and len(set(values))==len(values))
    if not unique_integers(spec['dataset_ids']): raise ValueError('Invalid dataset IDs.')
    if 'optimizer_seeds' in spec and 'optimizer_seed' in spec:
        raise ValueError('Use optimizer_seeds or optimizer_seed, not both.')
    if not unique_integers(optimizer_seeds(spec)): raise ValueError('Invalid optimizer seed IDs.')
    if not integer(spec['updates'],1) or not integer(spec['paths'],2):
        raise ValueError('Updates and paths must be positive integer budgets (paths >= 2).')
    if not number(spec['initial_lambda']) or spec['initial_lambda']>1 or spec['h']!=H:
        raise ValueError('Initial lambda must be bounded and h must remain canonical.')
    cp=spec['checkpoints']
    if (not unique_integers(cp) or cp!=sorted(cp) or cp[0]!=0 or cp[-1]!=spec['updates']):
        raise ValueError('Checkpoints must be ordered, unique and include initial/final updates.')
    if not isinstance(spec['families'],list) or not spec['families']: raise ValueError('No loss families.')
    seen=set()
    for family in spec['families']:
        version,beta=family['version'],family['beta']
        if not ((version=='V1' and number(beta)) or (version=='V2' and beta is None)):
            raise ValueError('Invalid V1 beta or V2 family.')
        key=(version,beta)
        if key in seen: raise ValueError('Duplicate loss family.')
        seen.add(key)
        etas=family['etas']
        if not isinstance(etas,list) or not etas or not all(number(e,positive=True) for e in etas):
            raise ValueError('Eta candidates must be finite positive numbers.')
    ids=[run_identifier(d,**g,seed=seed) for d in spec['dataset_ids']
         for seed in optimizer_seeds(spec) for g in configurations(spec)]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate run IDs or eta candidates.')
    return spec

def optimizer_seeds(spec):
    return spec['optimizer_seeds'] if 'optimizer_seeds' in spec else [spec['optimizer_seed']]

def configurations(spec):
    return [dict(version=f['version'],beta=f['beta'],eta=eta)
            for f in spec['families'] for eta in f['etas']]

def family_name(version,beta):
    return 'V2' if version=='V2' else f'V1_beta{beta:g}'

def run_identifier(dataset,version,beta,eta,seed):
    return f'd{dataset}_{family_name(version,beta)}_eta{eta:g}_s{seed}'

def _execute_run(root,output,spec,dataset,group,optimizer_seed=None):
    root=Path(root);output=Path(output)
    seed=optimizer_seeds(spec)[0] if optimizer_seed is None else optimizer_seed
    identifier=run_identifier(dataset,**group,seed=seed)
    path=output/'runs'/f'{identifier}.json'
    if path.exists():
        previous=json.loads(path.read_text())
        # Simple rerun protection: reuse completed runs; never silently resume a partial run.
        if previous['completed']: return identifier
        raise RuntimeError(f'Partial run retained at {path}; inspect before explicitly restarting it.')
    started=perf_counter()
    def save(run):
        write_json(path,dict(run,dataset_id=dataset,optimizer_seed=seed,
            requested_updates=spec['updates'],elapsed_seconds=perf_counter()-started))
    try:
        model=ModelConfig.load(root/'configs/model.json')
        expert=load_expert(root/spec['data_directory']/f'dataset_{dataset}.npz',model)
        train(model,expert,dataset_id=dataset,run_id=identifier,**group,updates=spec['updates'],
            paths=spec['paths'],initial_lambda=spec['initial_lambda'],optimizer_seed=seed,
            checkpoint_updates=spec['checkpoints'],progress_callback=save)
    except Exception as error:
        if path.exists():
            run=json.loads(path.read_text())
            run.update(completed=False,failure=dict(type=type(error).__name__,message=str(error)))
        else:
            run=dict(run_id=identifier,**group,updates=0,paths=spec['paths'],final_lambda=spec['initial_lambda'],
                wall_seconds=perf_counter()-started,br_calls=0,rollout_calls=0,rollout_paths=0,
                simulated_steps=0,random_banks=0,history=[],checkpoints=[],completed=False,
                failure=dict(type=type(error).__name__,message=str(error)))
        save(run)
    return identifier

def validate_run(run,spec,steps):
    """Ordinary checks for complete accounting and checkpoint/last-iterate reconstruction."""
    history=run['history'];cp=run['checkpoints']
    if run['updates']!=len(history): raise ValueError('Update count mismatch.')
    if [r['update'] for r in history]!=list(range(1,len(history)+1)):
        raise ValueError('Nonsequential history.')
    previous=spec['initial_lambda']
    for row in history:
        if row['lambda_before']!=previous: raise ValueError('Discontinuous parameter history.')
        if row['lambda_after']!=float(np.clip(previous-run['eta']*row['gradient'],0,1)):
            raise ValueError('Projected update mismatch.')
        previous=row['lambda_after']
    if previous!=run['final_lambda']: raise ValueError('Last iterate mismatch.')
    for row in cp:
        k=row['update']
        expected=spec['initial_lambda'] if k==0 else history[k-1]['lambda_after']
        if row['lambda_value']!=expected: raise ValueError('Checkpoint reconstruction mismatch.')
    if run['completed']:
        if run['failure'] or len(history)!=spec['updates'] or [r['update'] for r in cp]!=spec['checkpoints']:
            raise ValueError('Incomplete run marked completed.')
        if (run['rollout_calls']!=2*spec['updates']
            or run['rollout_paths']!=2*spec['updates']*spec['paths']
            or run['simulated_steps']!=2*spec['updates']*spec['paths']*steps
            or run['br_calls']!=2*spec['updates']+len(spec['checkpoints'])):
            raise ValueError('Complete run work accounting mismatch.')
    return True

def summarize_run(run,checkpoint_updates=None):
    cp={r['update']:r for r in run['checkpoints']};h=run['history']
    initial=cp.get(0,{}).get('reference_loss');final=cp.get(run['requested_updates'],{}).get('reference_loss')
    absolute=None if initial is None or final is None else initial-final
    relative=None if absolute is None or initial<=1e-14 else absolute/initial
    tail=np.array([r['lambda_after'] for r in h[-100:]],dtype=float)
    sd=float(np.std(tail,ddof=1)) if len(tail)>1 else None
    drift=float(tail[-1]-tail[0]) if len(tail)>1 else None
    projections=sum(r['projected'] for r in h)
    classification='inconclusive';reason='Incomplete or insufficient evidence of progress/stability.'
    if run['completed']:
        if final>initial+max(1e-10,.1*initial):
            classification='clearly inadequate';reason='Final objective regressed by more than 10% of initial objective.'
        elif sd>.15 or projections>.25*len(h):
            classification='clearly inadequate';reason='Large tail variation or projection on over 25% of updates.'
        elif relative is not None and relative>=.1 and abs(drift)<=max(.02,2*sd):
            classification='provisionally usable';reason='At least 10% objective progress with limited endpoint tail drift.'
    row=dict(run_id=run['run_id'],dataset_id=run['dataset_id'],optimizer_seed=run.get('optimizer_seed',0),family=family_name(run['version'],run['beta']),
        version=run['version'],beta=run['beta'],eta=run['eta'],completed=run['completed'],updates=run['updates'],
        initial_lambda=cp.get(0,{}).get('lambda_value'),final_lambda=run['final_lambda'],
        initial_objective=initial,final_objective=final,absolute_progress=absolute,relative_progress=relative,
        final_sampled_gradient=h[-1]['gradient'] if h else None,
        tail_count=len(tail),tail_mean=float(np.mean(tail)) if len(tail) else None,
        tail_sample_sd=sd,tail_drift=drift,
        tail_min=float(np.min(tail)) if len(tail) else None,tail_max=float(np.max(tail)) if len(tail) else None,
        tail_range=float(np.ptp(tail)) if len(tail) else None,
        tail_mean_abs_update=float(np.mean([abs(r['lambda_after']-r['lambda_before']) for r in h[-100:]])) if h else None,
        tail_total_variation=float(np.sum([abs(r['lambda_after']-r['lambda_before']) for r in h[-100:]])) if h else None,
        tail_projection_count=sum(r['projected'] for r in h[-100:]),projection_count=projections,
        failure_type=run['failure']['type'] if run['failure'] else None,
        failure_message=run['failure']['message'] if run['failure'] else None,
        wall_seconds=run['elapsed_seconds'],simulated_steps=run['simulated_steps'],
        classification=classification,classification_reason=reason)
    for k in (cp if checkpoint_updates is None else checkpoint_updates):
        for field in ('lambda_value','reference_loss','A_rel','C_rel','D_C','J_fixed','J_self','V2'):
            row[f'k{k}_{field}']=cp.get(k,{}).get(field)
    selected=list(cp) if checkpoint_updates is None else checkpoint_updates
    for lower,upper in zip(selected[:-1],selected[1:]):
        row[f'objective_progress_{lower}_{upper}']=(cp[lower]['reference_loss']-cp[upper]['reference_loss']) if lower in cp and upper in cp else None
    return row
