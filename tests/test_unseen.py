from pathlib import Path
import copy,csv,json
from dataclasses import asdict
import numpy as np
import pytest
from mi_mfg_irl_lq.unseen import (FROZEN,load_unseen_config,width_statistics,stopping_trace,stopping_decision,preserve_csv,
                                aggregate,execute_frozen,verify_run,prepare_unseen_data)
from mi_mfg_irl_lq.runs import configurations,family_name
import os
ROOT=Path(__file__).resolve().parents[1]
HISTORY=Path(os.environ.get('LQ_UNSEEN_HISTORY',str(ROOT/'results/unseen/training_history.csv.gz'))).resolve()
SPEC=load_unseen_config(ROOT/'configs/unseen.json')


def test_frozen_configuration_and_truth_boundary(tmp_path):
    assert [(g['version'],g['beta'],g['eta']) for g in configurations(SPEC)]==FROZEN
    assert SPEC['dataset_ids']==list(range(2000,2200))
    from mi_mfg_irl_lq.unseen import TRAIN_KEYS
    assert 'data_generation' not in TRAIN_KEYS and 'reporting' not in TRAIN_KEYS
    bad=copy.deepcopy(SPEC);bad['families'][0]['etas']=[10]
    path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):load_unseen_config(path)
    assert SPEC['data_generation']['random_seed_base']==20260818
    assert SPEC['data_generation']['setting_index']==1


def test_sample_sd_width_and_worst_family():
    values=np.arange(1.,11.);stats=width_statistics(values)
    assert stats['sample_sd']==np.std(values,ddof=1)
    assert stats['se']==stats['sample_sd']/np.sqrt(10)
    assert stats['half_width']==2.58*stats['sample_sd']
    estimates={'a':values,'b':np.r_[np.ones(5),np.arange(5.)]}
    trace=stopping_trace(estimates,SPEC['stopping'])
    last=[r for r in trace if r['M']==10]
    assert all(r['Delta']==max(x['relative_width_change'] for x in last) for r in last)
    assert last[1]['relative_width_change']>1e10 # Denominator floor at zero preceding width.
    with pytest.raises(ValueError):stopping_trace({'a':[1]*10,'b':[1]*9},SPEC['stopping'])


@pytest.mark.parametrize('m,a,b,stop,reason',[
    (35,.01,.01,False,None),(40,.05,.05,True,'two_consecutive_stability_checks'),
    (40,.051,.01,False,None),(40,.01,.051,False,None),
    (45,.01,.02,True,'two_consecutive_stability_checks'),
    (200,.1,.1,True,'hard_cap'),(200,.01,.01,True,'two_consecutive_stability_checks')])
def test_exact_stopping_boundary(m,a,b,stop,reason):
    trace=[dict(M=m-5,Delta=a),dict(M=m,Delta=b)]
    result=stopping_decision(trace,m,SPEC['stopping'])
    assert (result['stop'],result['reason'])==(stop,reason)


def fake_rows(count=45):
    rows=[]
    for i,d in enumerate(SPEC['dataset_ids'][:count]):
        for g in configurations(SPEC):
            rows.append(dict(dataset_id=d,family=family_name(g['version'],g['beta']),lambda_hat=.9+(-1)**i*.002,
                projection_count=0,tail_sample_sd=.001,tail_drift=.001,tail_range=.003,terminal_objective=1e-8))
    return rows


def test_aggregate_pairing_fixed40_and_reconstruction(tmp_path):
    first=fake_rows(40);more=fake_rows(45)
    a,trace,decision=aggregate(first,SPEC);b,_,_=aggregate(more,SPEC)
    assert len(a)==80 and len(b)==90
    fixed_a=[r for r in a if r['M']==40];fixed_b=[r for r in b if r['M']==40]
    assert fixed_a==fixed_b and decision['stop']
    path=tmp_path/'fixed40.csv';preserve_csv(path,fixed_a);before=path.read_bytes();preserve_csv(path,fixed_b)
    assert path.read_bytes()==before
    changed=copy.deepcopy(fixed_b);changed[0]['MAE']=99
    with pytest.raises(ValueError):preserve_csv(path,changed)
    with pytest.raises(ValueError):aggregate(first[:-1],SPEC)
    duplicate=first+[first[0]]
    with pytest.raises(ValueError):aggregate(duplicate,SPEC)
    # Exactly reconstruct the report inputs from an ordinary CSV roundtrip.
    from mi_mfg_irl_lq.io import write_csv
    write_csv(tmp_path/'rows.csv',first)
    with (tmp_path/'rows.csv').open() as f:loaded=list(csv.DictReader(f))
    assert aggregate(loaded,SPEC)==(a,trace,decision)


def test_runner_uses_accepted_trainer_and_validated_resume(tmp_path,monkeypatch):
    from mi_mfg_irl_lq import unseen
    from mi_mfg_irl_lq.config import ModelConfig
    from mi_mfg_irl_lq.expert_data import ExpertData
    from mi_mfg_irl_lq.io import write_npz,write_json
    from mi_mfg_irl_lq.optimization import train
    model=ModelConfig(num_time_steps=20);(tmp_path/'configs').mkdir();write_json(tmp_path/'configs/model.json',asdict(model))
    data=tmp_path/'data/unseen/dataset_42.npz';expert=ExpertData(model.time_grid,np.full(21,.5),.3)
    write_npz(data,time=expert.time,empirical_mean=expert.mean,J_left=expert.J_E,sJ=expert.sJ,Q=expert.Q)
    spec=copy.deepcopy(SPEC);spec.update(dataset_ids=[42],updates=3,paths=32,checkpoints=[0,3],model=asdict(model))
    output=tmp_path/'output';g=dict(version='V1',beta=.1,eta=1)
    result=execute_frozen(tmp_path,output,spec,42,g)
    run=verify_run(tmp_path,output,spec,42,g)
    assert not result['reused'] and run['history'][0]['seed_words']==str((20261016,42,0,0))
    direct=train(model,expert,dataset_id=42,run_id=run['run_id'],**g,updates=3,paths=32,checkpoint_updates=[0,3])
    assert direct['final_lambda']==run['final_lambda']
    monkeypatch.setattr(unseen,'_execute_run',lambda *args:pytest.fail('Completed run must not execute twice.'))
    assert execute_frozen(tmp_path,output,spec,42,g)['reused']
    path=next((output/'runs').glob('*.json'));run['completed']=False;write_json(path,run)
    with pytest.raises(RuntimeError):execute_frozen(tmp_path,output,spec,42,g)


def test_generation_wrapper_uses_round_settings_and_observed_left_j(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from mi_mfg_irl_lq import unseen
    from mi_mfg_irl_lq.config import ModelConfig
    from mi_mfg_irl_lq.observations import Observations
    from mi_mfg_irl_lq.io import write_json
    model=ModelConfig(num_time_steps=20);(tmp_path/'configs').mkdir();write_json(tmp_path/'configs/model.json',asdict(model))
    output=tmp_path/'output';output.mkdir();spec=copy.deepcopy(SPEC);spec.update(model=asdict(model),dataset_ids=[42],artificial_fixture=True)
    spec['data_generation'].update(num_agents=4,num_rounds=2);calls=[]
    def simulate(model,policy,**settings):
        calls.append(settings)
        return Observations(model.time_grid,np.full(21,.6),np.r_[np.full(20,.5),999.],np.full(21,.2),8)
    monkeypatch.setattr(unseen,'simulate_observations',simulate)
    exact=SimpleNamespace(policy=None,diagnostics={})
    prepare_unseen_data(tmp_path,output,spec,[42],exact)
    prepare_unseen_data(tmp_path,output,spec,[42],exact)
    assert calls==[dict(dataset_seed=42,random_seed_base=20260818,setting_index=1,num_agents=4,num_rounds=2)]
    from mi_mfg_irl_lq.expert_data import load_expert
    expert=load_expert(tmp_path/'data/unseen/dataset_42.npz',model)
    assert expert.J_E==pytest.approx(.5*(.5+.2+.36))
    assert expert.mean[0]==.6 and not hasattr(expert,'state_second')


def test_reporting_tables_reconstruct_all_figure_inputs(tmp_path):
    from mi_mfg_irl_lq.unseen_reporting import reconstruct_tables,BAND_CAPTION
    from mi_mfg_irl_lq.io import write_json,write_csv
    write_json(tmp_path/'config.json',SPEC)
    write_json(tmp_path/'run_accounting.json',dict(completed=True,M_stop=40,stopping_reason='two_consecutive_stability_checks'))
    rows=fake_rows(40);write_csv(tmp_path/'estimates.csv',rows)
    result=reconstruct_tables(tmp_path)
    assert len(result[3])==10 and len(result[4])==80 and len(result[5])==80 and len(result[6])==10
    assert 'not confidence intervals for the mean' in BAND_CAPTION
    for r in result[4]:
        assert r['mean_altruism']==1-r['mean_lambda']
        assert r['half_width']==2.58*r['sample_sd']
        assert r['dispersion_lower']==r['mean_altruism']-r['half_width']
        assert r['dispersion_upper']==r['mean_altruism']+r['half_width']
    before=(tmp_path/'summary.csv').read_bytes()
    reconstruct_tables(tmp_path)
    assert (tmp_path/'summary.csv').read_bytes()==before
    write_json(tmp_path/'run_accounting.json',dict(completed=True,M_stop=45,stopping_reason='two_consecutive_stability_checks'))
    write_csv(tmp_path/'estimates.csv',fake_rows(45))
    with pytest.raises(ValueError,match='earliest'):reconstruct_tables(tmp_path)


def test_beta_recovery_uses_only_canonical_fixed40_csvs(tmp_path):
    from mi_mfg_irl_lq.unseen_reporting import v1_beta_recovery_data,V1_BETAS
    from mi_mfg_irl_lq.io import write_csv
    rows=fake_rows(40)
    cumulative,_,_=aggregate(rows,SPEC)
    fixed=[r for r in cumulative if r['M']==40]
    write_csv(tmp_path/'estimates.csv',rows)
    write_csv(tmp_path/'summary.csv',fixed)
    groups=v1_beta_recovery_data(tmp_path)
    assert [g['beta'] for g in groups]==V1_BETAS==[0,.001,.01,.03,.1,.3,1,10,100]
    assert [g['eta'] for g in groups]==[20,20,5,2,1,.5,.2,.02,.002]
    for group,summary in zip(groups,fixed[:9]):
        assert len(group['values'])==40
        assert group['mean']==summary['mean_lambda']==np.mean(group['values'])
        assert group['sd']==summary['sample_sd']==np.std(group['values'],ddof=1)


@pytest.mark.skipif(not HISTORY.exists(),reason='Complete 400,000-update history is deliberately external to the compact repository; set LQ_UNSEEN_HISTORY to test it.')
def test_completed_unseen_serialized_evidence():
    """Reconstruct actual updates, last iterates, summaries and earliest stopping from CSV only."""
    import gzip,ast,io
    from collections import defaultdict,deque
    from mi_mfg_irl_lq.unseen_reporting import read_csv
    out=ROOT/'results/unseen';a=json.loads((out/'run_accounting.json').read_text())
    assert a['completed'];m=a['M_stop'];ids=SPEC['dataset_ids'][:m]
    assert a['completed_runs']==10*m and a['completed_updates']==10000*m
    assert a['fd_endpoint_evaluations']==20000*m
    assert a['primary_BR_Euler_steps']==1000*4096*a['fd_endpoint_evaluations']
    assert a['deterministic_checkpoints']==50*m
    previous={};counts=defaultdict(int);tails=defaultdict(lambda:deque(maxlen=100));endpoints={};scales={};projections=defaultdict(int)
    with gzip.open(HISTORY,'rt') as f:
        total=0
        for r in csv.DictReader(f):
            total+=1;d=int(r['dataset_id']);key=(d,r['family']);k=int(r['update']);counts[key]+=1
            assert d in ids and k==counts[key]
            assert ast.literal_eval(r['seed_words'])==(20261016,d,0,k-1)
            before=float(r['lambda_before']);after=float(r['lambda_after']);gradient=float(r['gradient']);eta=float(r['eta'])
            assert before==previous.get(key,.5)
            assert gradient==(float(r['upper_loss'])-float(r['lower_loss']))/float(r['denominator'])
            raw=before-eta*gradient
            assert raw==float(r['raw_next_lambda']) and after==float(np.clip(raw,0,1))
            projections[key]+=raw!=after
            for prefix in ['lower_','upper_']:
                own=(float(r[prefix+'V2']) if r['version']=='V2' else float(r[prefix+'A_rel'])+float(r['beta'])*float(r[prefix+'C_rel']))
                assert own==float(r[prefix+'loss'])
            scale=(float(r['J_E']),float(r['sJ']),float(r['Q']));assert scales.setdefault(d,scale)==scale
            previous[key]=after;tails[key].append(after)
            if k in [250,500,750,1000]:endpoints[(key,k)]=after
    assert total==a['completed_updates'] and len(counts)==10*m and set(counts.values())=={1000}
    rows=read_csv(out/'estimates.csv');assert len(rows)==10*m
    for r in rows:
        key=(int(r['dataset_id']),r['family']);value=previous[key];tail=np.array(tails[key])
        assert value==float(r['lambda_hat'])
        assert abs(value-.9)==float(r['absolute_error']) and r['success']==str(abs(value-.9)<=.01)
        assert int(r['projection_count'])==projections[key]
        assert float(r['tail_sample_sd'])==np.std(tail,ddof=1)
    cps=read_csv(out/'checkpoints.csv');assert len(cps)==50*m
    for r in cps:
        key=(int(r['dataset_id']),r['family']);k=int(r['update'])
        assert float(r['lambda_value'])==(.5 if k==0 else endpoints[(key,k)])
        expected=float(r['V2']) if r['version']=='V2' else float(r['A_rel'])+float(r['beta'])*float(r['C_rel'])
        assert expected==float(r['reference_loss'])
    cumulative,trace,decision=aggregate(rows,SPEC)
    eligible=range(40,m+1,5)
    assert next(n for n in eligible if stopping_decision(trace,n,SPEC['stopping'])['stop'])==m
    assert decision['reason']==a['stopping_reason']
    def canonical(rr):
        text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(rr[0]));writer.writeheader();writer.writerows(rr);text.seek(0)
        return list(csv.DictReader(text))
    assert canonical(cumulative)==read_csv(out/'cumulative.csv')
    assert canonical(trace)==read_csv(out/'stopping.csv')
    assert canonical([r for r in cumulative if r['M']==40])==read_csv(out/'summary.csv')
    assert m==40 and canonical([r for r in cumulative if r['M']==m])==read_csv(out/'summary.csv')
    assert {int(r['dataset_id']) for r in read_csv(ROOT/'data/unseen/registry.csv')}==set(ids)
    assert {p.name for p in (ROOT/'data/unseen').glob('dataset_*.npz')}=={f'dataset_{d}.npz' for d in ids}
    for d in ids:
        with np.load(ROOT/f'data/unseen/dataset_{d}.npz',allow_pickle=False) as z:
            assert int(z['observation_count'])==20000 and int(z['dataset_seed'])==d
            running=.5*(z['observed_mean_X_squared']+z['observed_mean_action_squared']+z['empirical_mean']**2)
            assert float(z['J_left'])==.001*float(np.sum(running[:-1]))
            assert float(z['sJ'])==pytest.approx(float(z['J_left'])**2,rel=0,abs=1e-15)
            assert scales[d]==(float(z['J_left']),float(z['sJ']),float(z['Q']))
