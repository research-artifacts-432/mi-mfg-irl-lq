"""Predeclared recovery selection within each family; never used by training."""
import json
import numpy as np


def summarize_candidate(records,expected_dataset_ids,target,success_threshold):
    """Each expert dataset contributes one last-iterate result."""
    valid=[r for r in records if r['valid_completed'] and r['lambda_hat'] is not None
           and np.isfinite([r['lambda_hat'],r['tail_sample_sd'],r['terminal_objective']]).all()
           and 0<=r['lambda_hat']<=1]
    complete=(len(records)==len(expected_dataset_ids) and
              {r['dataset_id'] for r in records}==set(expected_dataset_ids) and len(valid)==len(records))
    errors=np.array([abs(r['lambda_hat']-target) for r in valid])
    parameters=np.array([r['lambda_hat'] for r in valid])
    # Partial descriptive metrics retain the denominator; admissibility still requires every dataset.
    return dict(admissible=complete,valid_completed_runs=len(valid),expected_runs=len(expected_dataset_ids),
        success_count=int(np.sum(errors<=success_threshold)),
        mean_lambda=float(np.mean(parameters)) if len(valid) else None,
        sample_sd_lambda=float(np.std(parameters,ddof=1)) if len(valid)>1 else None,
        MAE=float(np.mean(errors)) if len(valid) else None,RMSE=float(np.sqrt(np.mean(errors**2))) if len(valid) else None,
        MaxAE=float(np.max(errors)) if len(valid) else None,median_absolute_error=float(np.median(errors)) if len(valid) else None,
        total_projection_count=sum(r['projection_count'] for r in records),
        median_tail_SD=float(np.median([r['tail_sample_sd'] for r in valid])) if valid else None,
        median_terminal_objective=float(np.median([r['terminal_objective'] for r in valid])) if valid else None)


def select_within_families(summaries,rule):
    """Sequential filtering; objectives are deliberately absent from tie-breaking."""
    selected=[]
    for family in dict.fromkeys(r['family'] for r in summaries):
        group=[r for r in summaries if r['family']==family]
        remaining=[r for r in group if r['admissible']]
        trace=[dict(step='admissibility',etas=[r['eta'] for r in remaining])]
        deciding='admissibility' if len(remaining)==1 else None
        if not remaining:
            selected.append(dict(family=family,status='unresolved: no admissible candidate',winner_eta=None,
                deciding_step=None,success_count=None,MAE=None,RMSE=None,MaxAE=None,total_projection_count=None,
                median_tail_SD=None,selection_trace=json.dumps(trace)));continue
        steps=[('maximize_success_count','success_count','max',0),
               ('MAE_practical_equivalence','MAE','min',rule['MAE_tolerance']),
               ('RMSE_practical_equivalence','RMSE','min',rule['RMSE_tolerance']),
               ('MaxAE_practical_equivalence','MaxAE','min',rule['MaxAE_tolerance']),
               ('minimize_projection_count','total_projection_count','min',0),
               ('minimize_median_tail_SD','median_tail_SD','min',0),
               ('smaller_eta','eta','min',0)]
        for name,key,direction,tolerance in steps:
            before=len(remaining)
            optimum=(max if direction=='max' else min)(r[key] for r in remaining)
            remaining=[r for r in remaining if (r[key]==optimum if direction=='max' else r[key]<=optimum+tolerance)]
            trace.append(dict(step=name,optimum=optimum,tolerance=tolerance,etas=[r['eta'] for r in remaining]))
            if before>1 and len(remaining)==1 and deciding is None:deciding=name
        if len(remaining)!=1:raise ValueError('Candidate eta IDs must be unique within a family.')
        winner=remaining[0]
        selected.append(dict(family=family,status='proposed development winner; independent review pending',
            winner_eta=winner['eta'],deciding_step=deciding,
            **{k:winner[k] for k in ['success_count','MAE','RMSE','MaxAE','total_projection_count','median_tail_SD']},
            selection_trace=json.dumps(trace)))
    return selected


def evaluate_candidates(records,spec):
    from .runs import configurations,family_name
    summaries=[]
    for group in configurations(spec):
        family=family_name(group['version'],group['beta'])
        subset=[r for r in records if r['family']==family and r['eta']==group['eta']]
        summaries.append(dict(family=family,**group,**summarize_candidate(subset,spec['dataset_ids'],
            spec['selection']['target_lambda'],spec['selection']['success_threshold'])))
    return summaries,select_within_families(summaries,spec['selection'])


def reselect_from_csv(output):
    """Reproduce selection from packet CSVs without optimizer runs or snapshots."""
    import csv
    from pathlib import Path
    from .io import write_csv
    output=Path(output);spec=json.loads((output/'config.json').read_text());records=[]
    with (output/'estimates.csv').open() as f:
        for serialized in csv.DictReader(f):
            r=dict(serialized)
            for key in ['dataset_id','optimizer_seed','updates','projection_count']:r[key]=int(r[key])
            for key in ['eta','lambda_hat','tail_sample_sd','terminal_objective']:
                r[key]=float(r[key]) if r[key] else None
            r['valid_completed']=(r['valid_completed']=='True' and r['completed']=='True' and r['updates']==spec['updates'])
            records.append(r)
    summaries,selection=evaluate_candidates(records,spec)
    write_csv(output/'candidates.csv',summaries);write_csv(output/'selection.csv',selection)
    return selection
