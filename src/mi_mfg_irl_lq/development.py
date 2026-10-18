"""Formal development orchestration using the accepted generator and trainer."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from dataclasses import asdict
from time import perf_counter
import csv,gzip,json
import numpy as np
from .config import ModelConfig
from .generation import generate_exact_expert
from .observations import simulate_observations,preprocess,Observations
from .expert_data import load_expert
from .runs import (load_experiment_config,configurations,family_name,run_identifier,
                          _execute_run,validate_run,summarize_run)
from .selection import evaluate_candidates
from .io import write_csv,write_json,write_npz,atomic_write


def load_development_config(path):
    spec=load_experiment_config(path)
    ModelConfig(**spec['model'])
    if spec.get('optimizer_seed')!=0 or 'optimizer_seeds' in spec:raise ValueError('Development uses optimizer seed 0.')
    if Path(spec['data_directory']).is_absolute() or '..' in Path(spec['data_directory']).parts:
        raise ValueError('Development data must use a project-relative directory.')
    gen=spec['data_generation'];rule=spec['selection']
    if (not 0<=gen['lambda_star']<=1 or any(type(gen[k]) is not int or gen[k]<=0 for k in ['num_agents','num_rounds'])
        or any(type(gen[k]) is not int or gen[k]<0 for k in ['random_seed_base','setting_index'])):
        raise ValueError('Invalid observation settings.')
    if not (0<=rule['target_lambda']<=1 and rule['success_threshold']>=0 and
            all(np.isfinite(rule[k]) and rule[k]>=0 for k in ['success_threshold','MAE_tolerance','RMSE_tolerance','MaxAE_tolerance'])):
        raise ValueError('Invalid selection constants.')
    expected=['maximize_success_count','MAE_practical_equivalence','RMSE_practical_equivalence',
              'MaxAE_practical_equivalence','minimize_projection_count','minimize_median_tail_SD','smaller_eta']
    if rule['ordered_steps']!=expected:raise ValueError('Selection order must remain predeclared.')
    return spec


def training_specification(spec):
    """Worker inputs contain no data-generation truth or recovery selection rule."""
    return {key:spec[key] for key in ['dataset_ids','optimizer_seed','updates','paths','initial_lambda','h',
                                      'checkpoints','families','data_directory']}


def prepare_development_data(root,output,spec,crosscheck_tolerance=1e-7):
    root=Path(root);output=Path(output);directory=root/spec['data_directory']
    directory.mkdir(parents=True,exist_ok=True);output.mkdir(parents=True,exist_ok=True)
    model=ModelConfig.load(root/'configs/model.json')
    if asdict(model)!=spec['model']:raise ValueError('Recorded and runtime model settings disagree.')
    settings={k:v for k,v in spec['data_generation'].items() if k!='lambda_star'}
    started=perf_counter();exact=None;rows=[];new_ids=[];dataset_seconds={}
    for dataset in spec['dataset_ids']:
        path=directory/f'dataset_{dataset}.npz';tick=perf_counter()
        if path.exists():
            # Ordinary restart: validate the retained observed statistics; never regenerate them.
            with np.load(path,allow_pickle=False) as z:
                if int(z['dataset_id'])!=dataset or int(z['dataset_seed'])!=dataset or int(z['observation_count'])!=settings['num_agents']*settings['num_rounds']:
                    raise ValueError('Retained dataset identity/count mismatch.')
                for key,value in settings.items():
                    if int(z[key])!=value:raise ValueError('Retained observation settings mismatch.')
                observed=Observations(z['time'],z['empirical_mean'],z['observed_mean_X_squared'],z['observed_mean_action_squared'],int(z['observation_count']))
                reconstructed=preprocess(model,observed)
                loaded=load_expert(path,model)
                if reconstructed.J_E!=loaded.J_E:raise ValueError('Observed left-J mismatch.')
        else:
            if exact is None:
                exact=generate_exact_expert(model,spec['data_generation']['lambda_star'])
                if max(exact.diagnostics['max_q_identity_error'],exact.diagnostics['max_reduced_full_mean_error'])>crosscheck_tolerance:
                    raise RuntimeError('Independent expert DPP systems disagree.')
            observed=simulate_observations(model,exact.policy,dataset_seed=dataset,**settings)
            expert=preprocess(model,observed)
            write_npz(path,time=observed.time,empirical_mean=observed.mean,
                observed_mean_X_squared=observed.state_second,observed_mean_action_squared=observed.action_second,
                J_left=expert.J_E,sJ=expert.sJ,Q=expert.Q,observation_count=observed.count,
                dataset_id=dataset,dataset_seed=dataset,**settings)
            new_ids.append(dataset)
        dataset_seconds[str(dataset)]=perf_counter()-tick
        row=dict(dataset_id=dataset,dataset_seed=dataset,split='development' if dataset>=1000 else 'smoke_fixture',
            path=path.relative_to(root).as_posix(),**settings,lambda_star=spec['data_generation']['lambda_star'],
            observation_count_per_time=settings['num_agents']*settings['num_rounds'],time_intervals=model.num_time_steps,
            provenance='new_independent_synthetic_observations' if dataset>=1000 else 'bounded_smoke_fixture',
            cost_quadrature='left_riemann',consistency_quadrature='trapezoidal')
        rows.append(row);write_csv(directory/'registry.csv',rows)
        print(f'Expert dataset {dataset} ready ({row["observation_count_per_time"]} observations per node).',flush=True)
    previous_path=output/'generation_accounting.json'
    if previous_path.exists() and not new_ids:return json.loads(previous_path.read_text())
    accounting=dict(new_dataset_ids=new_ids,dataset_ids=spec['dataset_ids'],new_dataset_count=len(new_ids),
        generation_elapsed_seconds=perf_counter()-started,dataset_seconds=dataset_seconds,
        observation_count_per_dataset_per_time=settings['num_agents']*settings['num_rounds'],
        new_expert_trajectory_steps=len(new_ids)*settings['num_agents']*settings['num_rounds']*model.num_time_steps,
        expert_diagnostics=exact.diagnostics if exact is not None else None)
    write_json(previous_path,accounting);return accounting


def collect_development(root,output,spec,elapsed,workers):
    """Stream complete raw histories to gzip; keep only small summaries in memory."""
    root=Path(root);output=Path(output);model=ModelConfig.load(root/'configs/model.json')
    summaries=[];datasets=[];checkpoints=[];metadata=[];failures=[]
    def stream_history(binary):
        import io
        with gzip.GzipFile(fileobj=binary,mode='wb') as zipped:
            text=io.TextIOWrapper(zipped,encoding='utf-8',newline='');writer=None
            for dataset in spec['dataset_ids']:
                for group in configurations(spec):
                    identifier=run_identifier(dataset,**group,seed=0)
                    path=output/'runs'/f'{identifier}.json'
                    if not path.exists():
                        failures.append(dict(run_id=identifier,type='MissingRun',message='No retained run output.'));continue
                    run=json.loads(path.read_text());validation_error=None
                    try:validate_run(run,spec,model.num_time_steps)
                    except ValueError as error:validation_error=str(error)
                    meta={k:v for k,v in run.items() if k not in ('history','checkpoints')};metadata.append(meta)
                    if run['failure']:failures.append(dict(run_id=identifier,**run['failure']))
                    if validation_error:failures.append(dict(run_id=identifier,type='InvalidRun',message=validation_error))
                    summary=summarize_run(run,spec['checkpoints']);summaries.append(summary)
                    for row in run['history']:
                        values=dict(row,dataset_id=dataset,optimizer_seed=0)
                        if writer is None:writer=csv.DictWriter(text,fieldnames=list(values));writer.writeheader()
                        writer.writerow(values)
                    family=family_name(group['version'],group['beta'])
                    common=dict(run_id=identifier,dataset_id=dataset,optimizer_seed=0,family=family,**group)
                    checkpoints.extend(dict(common,**point) for point in run['checkpoints'])
                    numeric_valid=all(np.isfinite(v) for point in run['checkpoints'] for v in point.values())
                    valid=(run['completed'] and run['failure'] is None and validation_error is None and numeric_valid
                           and np.isfinite(run['final_lambda']) and 0<=run['final_lambda']<=1)
                    target=spec['selection']['target_lambda']
                    error=abs(run['final_lambda']-target) if valid else None
                    datasets.append(dict(common,valid_completed=valid,completed=run['completed'],updates=run['updates'],
                        lambda_hat=run['final_lambda'] if valid else None,absolute_error=error,
                        success=(error<=spec['selection']['success_threshold']) if valid else None,
                        projection_count=summary['projection_count'],tail_sample_sd=summary['tail_sample_sd'],
                        tail_mean=summary['tail_mean'],tail_drift=summary['tail_drift'],tail_range=summary['tail_range'],
                        terminal_objective=summary['final_objective'],
                        failure_type=run['failure']['type'] if run['failure'] else ('InvalidRun' if not valid else None),
                        failure_message=run['failure']['message'] if run['failure'] else validation_error,
                        wall_seconds=run['elapsed_seconds'],simulated_steps=run['simulated_steps']))
            text.flush();text.detach()
    atomic_write(output/'training_history.csv.gz',stream_history,True)
    candidate_rows,selection=evaluate_candidates(datasets,spec)
    for name,rows in [('checkpoints',checkpoints),('candidates',candidate_rows),('estimates',datasets),('selection',selection),('run_summary',summaries)]:
        if rows:write_csv(output/f'{name}.csv',rows)
    runs=len(spec['dataset_ids'])*len(configurations(spec))
    accounting=dict(planned_configurations=len(configurations(spec)),planned_runs=runs,planned_updates=runs*spec['updates'],
        available_runs=len(metadata),completed_runs=sum(r['completed'] for r in metadata),valid_completed_runs=sum(r['valid_completed'] for r in datasets),
        completed_updates=sum(r['updates'] for r in metadata),failures=failures,elapsed_seconds=elapsed,workers=workers,
        sum_run_seconds=sum(r['elapsed_seconds'] for r in metadata),rollout_calls=sum(r['rollout_calls'] for r in metadata),
        rollout_paths=sum(r['rollout_paths'] for r in metadata),simulated_steps=sum(r['simulated_steps'] for r in metadata),
        br_calls=sum(r['br_calls'] for r in metadata),random_banks=sum(r['random_banks'] for r in metadata),
        deterministic_checkpoints=len(checkpoints),expert_dataset_units=len(spec['dataset_ids']),runs=metadata)
    write_json(output/'run_accounting.json',accounting)
    return accounting


def run_development(root,output,config_path,workers=3,crosscheck_tolerance=1e-7):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    spec=load_development_config(config_path)
    if ModelConfig.load(root/'configs/model.json')!=ModelConfig(**spec['model']):raise ValueError('Recorded model differs from runtime configuration.')
    saved=output/'config.json'
    if saved.exists() and json.loads(saved.read_text())!=spec:raise ValueError('Output belongs to a different development configuration.')
    write_json(saved,spec)
    prepare_development_data(root,output,spec,crosscheck_tolerance)
    prior=output/'run_accounting.json'
    if prior.exists():
        previous=json.loads(prior.read_text())
        expected=len(spec['dataset_ids'])*len(configurations(spec))
        if previous['completed_runs']==expected and previous['completed_updates']==expected*spec['updates']:return previous
    started=perf_counter();futures=[];training=training_specification(spec)
    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for dataset in spec['dataset_ids']:
                for group in configurations(spec):
                    futures.append(executor.submit(_execute_run,str(root),str(output),training,dataset,group,0))
            for index,future in enumerate(as_completed(futures),1):
                print(f'{index}/{len(futures)} saved: {future.result()}',flush=True)
    finally:result=collect_development(root,output,spec,perf_counter()-started,workers)
    return result
