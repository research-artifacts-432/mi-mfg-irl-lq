"""Frozen unseen evaluation and cumulative dispersion stopping; no model selection."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import csv,gzip,io,json
import numpy as np
from .config import ModelConfig
from .generation import generate_exact_expert
from .observations import simulate_observations,preprocess,Observations
from .expert_data import load_expert
from .runs import load_experiment_config,configurations,family_name,run_identifier,_execute_run,validate_run,summarize_run
from .io import write_csv,write_json,write_npz,atomic_write

FROZEN=[('V1',b,e) for b,e in zip([0,.001,.01,.03,.1,.3,1,10,100],[20,20,5,2,1,.5,.2,.02,.002])]+[('V2',None,20)]
TRAIN_KEYS=['dataset_ids','optimizer_seed','updates','paths','initial_lambda','h','checkpoints','families','data_directory']


def load_unseen_config(path):
    spec=load_experiment_config(path)
    expected=[dict(version=v,beta=b,eta=e) for v,b,e in FROZEN]
    if configurations(spec)!=expected:raise ValueError('Unseen configurations must match the ten approved winners.')
    if spec['dataset_ids']!=list(range(2000,2200)):raise ValueError('Unseen sequence must be 2000–2199.')
    if (spec['updates'],spec['paths'],spec['initial_lambda'],spec['optimizer_seed'],spec['checkpoints'])!=(1000,4096,.5,0,[0,250,500,750,1000]):raise ValueError('Frozen training settings differ.')
    if spec['stopping']!=dict(M_fixed=40,M_min=40,batch_size=5,M_max=200,consecutive_confirmations=2,relative_threshold=.05,width_multiplier=2.58,width_floor=1e-12):raise ValueError('Approved stopping settings differ.')
    if spec['data_generation']!=dict(lambda_star=.9,random_seed_base=20260818,setting_index=1,num_agents=200,num_rounds=100):raise ValueError('Approved observations differ.')
    if spec['reporting']['target_lambda']!=.9 or spec['reporting']['recovery_threshold']!=.01 or spec['reporting']['sd_ddof']!=1:raise ValueError('Recovery reporting differs.')
    if spec['model']!=asdict(ModelConfig()):raise ValueError('Accepted scientific model differs.')
    if spec['data_directory']!='data/unseen':raise ValueError('Use the standalone unseen data directory.')
    return spec


def width_statistics(estimates):
    """No truth or recovery metrics enter dispersion or stopping calculations."""
    values=np.asarray(estimates,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all():raise ValueError('At least two finite estimates required.')
    sd=float(np.std(values,ddof=1))
    return dict(mean_lambda=float(values.mean()),sample_sd=sd,se=sd/np.sqrt(len(values)),half_width=2.58*sd)


def stopping_trace(estimates,settings):
    lengths={len(v) for v in estimates.values()}
    if len(lengths)!=1 or not estimates:raise ValueError('Complete paired prefixes required.')
    count=next(iter(lengths));rows=[];previous={}
    for m in range(settings['batch_size'],count+1,settings['batch_size']):
        current=[]
        for family,values in estimates.items():
            stats=width_statistics(values[:m]);old=previous.get(family)
            delta=None if old is None else abs(stats['half_width']-old)/max(old,settings['width_floor'])
            current.append(dict(M=m,family=family,**stats,previous_half_width=old,relative_width_change=delta))
            previous[family]=stats['half_width']
        worst=max(r['relative_width_change'] for r in current) if current[0]['relative_width_change'] is not None else None
        rows.extend(dict(r,Delta=worst,passing=worst is not None and worst<=settings['relative_threshold']) for r in current)
    return rows


def stopping_decision(trace,m,settings):
    delta={r['M']:r['Delta'] for r in trace}
    pair=[delta.get(m-settings['batch_size']),delta.get(m)]
    stable=(m>=settings['M_min'] and all(x is not None and x<=settings['relative_threshold'] for x in pair))
    if stable:return dict(stop=True,reason='two_consecutive_stability_checks',previous_Delta=pair[0],current_Delta=pair[1])
    if m>=settings['M_max']:return dict(stop=True,reason='hard_cap',previous_Delta=pair[0],current_Delta=pair[1])
    return dict(stop=False,reason=None,previous_Delta=pair[0],current_Delta=pair[1])


def preserve_csv(path,rows):
    """Fixed M40 tables are written once; later batches must reproduce them exactly."""
    text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    expected=text.getvalue();path=Path(path)
    if path.exists():
        with path.open(newline='') as f:
            if f.read()!=expected:raise ValueError('Retained fixed-40 table differs.')
    else:atomic_write(path,lambda f:f.write(expected))


def prepare_unseen_data(root,output,spec,ids,exact=None):
    root=Path(root);output=Path(output);directory=root/spec['data_directory'];directory.mkdir(parents=True,exist_ok=True)
    registry=directory/'registry.csv'
    rows=list(csv.DictReader(registry.open())) if registry.exists() else []
    indexed={int(r['dataset_id']):r for r in rows};model=ModelConfig.load(root/'configs/model.json')
    settings={k:v for k,v in spec['data_generation'].items() if k!='lambda_star'}
    for dataset in ids:
        tick=perf_counter();path=directory/f'dataset_{dataset}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if int(z['dataset_id'])!=dataset or int(z['dataset_seed'])!=dataset:raise ValueError('Retained unseen identity differs.')
                for key,value in settings.items():
                    if int(z[key])!=value:raise ValueError('Retained observation settings differ.')
                observed=Observations(z['time'],z['empirical_mean'],z['observed_mean_X_squared'],z['observed_mean_action_squared'],int(z['observation_count']))
                if observed.count!=settings['num_agents']*settings['num_rounds']:raise ValueError('Wrong observed count.')
                if preprocess(model,observed).J_E!=load_expert(path,model).J_E:raise ValueError('Observed left-J differs.')
        else:
            if exact is None:
                exact=generate_exact_expert(model,spec['data_generation']['lambda_star'])
                if max(exact.diagnostics['max_q_identity_error'],exact.diagnostics['max_reduced_full_mean_error'])>1e-7:raise RuntimeError('Expert DPP systems disagree.')
            observed=simulate_observations(model,exact.policy,dataset_seed=dataset,**settings)
            expert=preprocess(model,observed)
            write_npz(path,time=observed.time,empirical_mean=observed.mean,observed_mean_X_squared=observed.state_second,
                observed_mean_action_squared=observed.action_second,J_left=expert.J_E,sJ=expert.sJ,Q=expert.Q,
                observation_count=observed.count,dataset_id=dataset,dataset_seed=dataset,**settings)
        if dataset not in indexed:
            indexed[dataset]=dict(dataset_id=dataset,dataset_seed=dataset,split='artificial_fixture' if spec.get('artificial_fixture') else 'unseen',**settings,
                lambda_star=spec['data_generation']['lambda_star'],observation_count_per_time=observed.count,
                time_intervals=model.num_time_steps,path=path.relative_to(root).as_posix(),generation_status='complete',
                provenance='artificial_sufficient_statistics_fixture' if spec.get('artificial_fixture') else 'new_independent_synthetic_observations',cost_quadrature='left_riemann',consistency_quadrature='trapezoidal',
                generation_seconds=perf_counter()-tick,expert_trajectory_steps=observed.count*model.num_time_steps)
        ordered=[indexed[k] for k in sorted(indexed)];write_csv(registry,ordered);write_csv(output/'dataset_registry.csv',ordered)
        print(f'Unseen expert dataset {dataset} ready.',flush=True)
    if exact is not None:write_json(output/'expert_generation_diagnostics.json',exact.diagnostics)
    return exact


def verify_run(root,output,spec,dataset,group):
    identifier=run_identifier(dataset,**group,seed=spec['optimizer_seed']);path=Path(output)/'runs'/f'{identifier}.json'
    run=json.loads(path.read_text());validate_run(run,spec,spec['model']['num_time_steps'])
    if not run['completed'] or run['failure']:raise RuntimeError(f'Failed or partial scientific run: {identifier}: {run["failure"]}')
    for key,value in dict(run_id=identifier,dataset_id=dataset,optimizer_seed=spec['optimizer_seed'],paths=spec['paths'],requested_updates=spec['updates'],**group).items():
        if run.get(key)!=value:raise ValueError(f'Run identity/configuration mismatch: {key}')
    for row in run['history']:
        if row['seed_words']!=str((20261016,dataset,spec['optimizer_seed'],row['update']-1)):raise ValueError('Wrong bank index.')
        if row['gradient']!=(row['upper_loss']-row['lower_loss'])/row['denominator']:raise ValueError('FD reconstruction differs.')
        if not np.isfinite([v for v in row.values() if isinstance(v,(float,int))]).all():raise ValueError('Nonfinite history.')
    if not all(np.isfinite(v) for point in run['checkpoints'] for v in point.values()):raise ValueError('Nonfinite checkpoint.')
    return run


def execute_frozen(root,output,spec,dataset,group):
    path=Path(output)/'runs'/f'{run_identifier(dataset,**group,seed=spec["optimizer_seed"])}.json'
    reused=path.exists()
    if reused:verify_run(root,output,spec,dataset,group)
    else:_execute_run(root,output,{k:spec[k] for k in TRAIN_KEYS},dataset,group,spec['optimizer_seed'])
    run=verify_run(root,output,spec,dataset,group)
    return dict(run_id=run['run_id'],reused=reused)


def execute_batch(root,output,spec,ids,workers):
    """At most three in-flight jobs; a genuine failure stops further submission."""
    jobs=iter((d,g) for d in ids for g in configurations(spec));pool=ProcessPoolExecutor(max_workers=workers);pending={};statuses=[]
    try:
        for _ in range(workers):
            item=next(jobs,None)
            if item is not None:pending[pool.submit(execute_frozen,str(root),str(output),spec,*item)]=item
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                dataset,group=pending.pop(future);status=future.result();statuses.append(status)
                print(f'Batch {ids[0]}–{ids[-1]}: {len(statuses)}/{10*len(ids)} complete: {status}',flush=True)
            for _ in done:
                item=next(jobs,None)
                if item is not None:pending[pool.submit(execute_frozen,str(root),str(output),spec,*item)]=item
    finally:
        for future in pending:future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
    return statuses


def dataset_records(root,output,spec,ids):
    rows=[];metadata=[]
    for dataset in ids:
        for group in configurations(spec):
            run=verify_run(root,output,spec,dataset,group);summary=summarize_run(run,spec['checkpoints'])
            error=abs(run['final_lambda']-spec['reporting']['target_lambda'])
            rows.append(dict(run_id=run['run_id'],dataset_id=dataset,family=family_name(group['version'],group['beta']),**group,
                optimizer_seed=spec['optimizer_seed'],completed=True,updates=run['updates'],lambda_hat=run['final_lambda'],
                absolute_error=error,success=error<=spec['reporting']['recovery_threshold'],projection_count=summary['projection_count'],
                tail_mean=summary['tail_mean'],tail_sample_sd=summary['tail_sample_sd'],tail_drift=summary['tail_drift'],tail_range=summary['tail_range'],
                terminal_objective=summary['final_objective'],wall_seconds=run['elapsed_seconds'],simulated_steps=run['simulated_steps'],failure=None))
            metadata.append({k:v for k,v in run.items() if k not in ['history','checkpoints']})
    return rows,metadata


def summarize_prefix(rows,spec,m):
    prefix=set(spec['dataset_ids'][:m]);result=[]
    for group in configurations(spec):
        family=family_name(group['version'],group['beta']);rr=[r for r in rows if int(r['dataset_id']) in prefix and r['family']==family]
        if len(rr)!=m or len({int(r['dataset_id']) for r in rr})!=m:raise ValueError('Incomplete/duplicate dataset-family prefix.')
        values=np.array([float(r['lambda_hat']) for r in rr]);stats=width_statistics(values)
        errors=np.abs(values-spec['reporting']['target_lambda']);altruism=1-stats['mean_lambda']
        result.append(dict(M=m,family=family,**group,**stats,mean_altruism=altruism,
            dispersion_lower=altruism-stats['half_width'],dispersion_upper=altruism+stats['half_width'],
            MAE=float(errors.mean()),RMSE=float(np.sqrt(np.mean(errors**2))),MaxAE=float(errors.max()),median_absolute_error=float(np.median(errors)),
            success_count=int(np.sum(errors<=spec['reporting']['recovery_threshold'])),success_rate=float(np.mean(errors<=spec['reporting']['recovery_threshold'])),
            min_lambda=float(values.min()),max_lambda=float(values.max()),total_projections=sum(int(r['projection_count']) for r in rr),
            median_tail_sd=float(np.median([float(r['tail_sample_sd']) for r in rr])),
            median_abs_tail_drift=float(np.median([abs(float(r['tail_drift'])) for r in rr])),
            median_tail_range=float(np.median([float(r['tail_range']) for r in rr])),
            median_terminal_objective=float(np.median([float(r['terminal_objective']) for r in rr]))))
    return result


def aggregate(rows,spec):
    count=len({int(r['dataset_id']) for r in rows});families=[family_name(g['version'],g['beta']) for g in configurations(spec)]
    if len(rows)!=len(families)*count:raise ValueError('Incomplete paired data.')
    estimates={f:[float(r['lambda_hat']) for r in sorted(rows,key=lambda r:int(r['dataset_id'])) if r['family']==f] for f in families}
    trace=stopping_trace(estimates,spec['stopping']);cumulative=[]
    for m in range(spec['stopping']['batch_size'],count+1,spec['stopping']['batch_size']):cumulative.extend(summarize_prefix(rows,spec,m))
    decision=stopping_decision(trace,count,spec['stopping'])
    return cumulative,trace,decision


def consolidate_history(root,output,spec,ids):
    checkpoints=[]
    def writer(binary):
        with gzip.GzipFile(fileobj=binary,mode='wb',mtime=0) as zipped:
            text=io.TextIOWrapper(zipped,encoding='utf-8',newline='');csv_writer=None
            for dataset in ids:
                for group in configurations(spec):
                    run=verify_run(root,output,spec,dataset,group);family=family_name(group['version'],group['beta'])
                    for row in run['history']:
                        value=dict(dataset_id=dataset,family=family,**row)
                        if csv_writer is None:csv_writer=csv.DictWriter(text,fieldnames=list(value));csv_writer.writeheader()
                        csv_writer.writerow(value)
                    checkpoints.extend(dict(run_id=run['run_id'],dataset_id=dataset,family=family,**group,**point) for point in run['checkpoints'])
            text.flush();text.detach()
    atomic_write(Path(output)/'training_history.csv.gz',writer,True)
    write_csv(Path(output)/'checkpoints.csv',checkpoints)
    return len(checkpoints)


def run_unseen(root,output,config,workers=3,spec_override=None):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    spec=load_unseen_config(config) if spec_override is None else spec_override
    if ModelConfig.load(root/'configs/model.json')!=ModelConfig(**spec['model']):raise ValueError('Runtime model differs.')
    path=output/'config.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Official output belongs to another configuration.')
    write_json(path,spec);started=perf_counter();batch_path=output/'batch_accounting.json'
    batches=json.loads(batch_path.read_text()) if batch_path.exists() else [];exact=None
    sizes=[spec['stopping']['M_fixed']]+list(range(spec['stopping']['M_fixed']+spec['stopping']['batch_size'],spec['stopping']['M_max']+1,spec['stopping']['batch_size']))
    for m in sizes:
        ids=spec['dataset_ids'][:m];new_ids=ids[:spec['stopping']['M_fixed']] if m==spec['stopping']['M_fixed'] else ids[-spec['stopping']['batch_size']:]
        tick=perf_counter()
        try:
            exact=prepare_unseen_data(root,output,spec,new_ids,exact)
            statuses=execute_batch(root,output,spec,new_ids,workers)
            rows,meta=dataset_records(root,output,spec,ids)
            cumulative,trace,decision=aggregate(rows,spec)
            write_csv(output/'estimates.csv',rows);write_csv(output/'cumulative.csv',cumulative);write_csv(output/'stopping.csv',trace)
            fixed=[r for r in cumulative if r['M']==spec['stopping']['M_fixed']]
            preserve_csv(output/'summary.csv',fixed)
            if not any(b['M']==m for b in batches):
                batches.append(dict(M=m,dataset_ids=new_ids,completed_dataset_prefix=ids,completed_runs=len(rows),
                    accumulated_updates=sum(r['updates'] for r in meta),simulated_steps=sum(r['simulated_steps'] for r in meta),
                    rollout_calls=sum(r['rollout_calls'] for r in meta),wall_seconds=perf_counter()-tick,
                    current_Delta=decision['current_Delta'],previous_Delta=decision['previous_Delta'],stop=decision['stop'],reason=decision['reason'],
                    newly_executed_runs=sum(not s['reused'] for s in statuses)))
                write_json(batch_path,batches)
            print(f'Completed prefix M={m}; stopping decision {decision}',flush=True)
            if decision['stop']:
                cps=consolidate_history(root,output,spec,ids)
                registry=list(csv.DictReader((root/spec['data_directory']/'registry.csv').open()))
                accounting=dict(completed=True,M_stop=m,last_dataset_id=ids[-1],stopping_reason=decision['reason'],
                    previous_Delta=decision['previous_Delta'],current_Delta=decision['current_Delta'],unused_reserved_datasets=len(spec['dataset_ids'])-m,
                    completed_runs=len(rows),completed_updates=sum(r['updates'] for r in meta),fd_endpoint_evaluations=sum(r['rollout_calls'] for r in meta),
                    sampled_BR_paths=sum(r['rollout_paths'] for r in meta),primary_BR_Euler_steps=sum(r['simulated_steps'] for r in meta),
                    deterministic_checkpoints=cps,br_calls=sum(r['br_calls'] for r in meta),projections=sum(r['projection_count'] for r in rows),
                    failures=[],workers=workers,batches=batches,elapsed_this_invocation_seconds=perf_counter()-started,
                    batch_execution_seconds=sum(b['wall_seconds'] for b in batches),sum_run_seconds=sum(r['elapsed_seconds'] for r in meta),
                    generation_seconds=sum(float(r['generation_seconds']) for r in registry),expert_trajectory_steps=sum(int(r['expert_trajectory_steps']) for r in registry),
                    dataset_ids=ids)
                write_json(output/'run_accounting.json',accounting);return accounting
            if m>=40:
                raise RuntimeError('Frozen M40 stopping condition failed. IDs 2040–2199 are not authorized; retained all completed evidence.')
        except BaseException as error:
            write_json(output/'run_accounting.json',dict(completed=False,failed_prefix_M=m,dataset_ids=ids,
                failure=dict(type=type(error).__name__,message=str(error)),batches=batches,elapsed_this_invocation_seconds=perf_counter()-started))
            raise
