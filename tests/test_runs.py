from pathlib import Path
import copy,json
import numpy as np
import pytest
from mi_mfg_irl_lq.config import ModelConfig
from mi_mfg_irl_lq.expert_data import ExpertData
from mi_mfg_irl_lq.optimization import train
from mi_mfg_irl_lq.runs import validate_run,summarize_run,load_experiment_config,configurations,run_identifier

def fixture_run(**extra):
    model=ModelConfig(num_time_steps=20)
    expert=ExpertData(model.time_grid,np.full(21,.5),.4)
    run=train(model,expert,dataset_id=100,run_id='test',version='V1',beta=.1,eta=2,
        updates=3,paths=32,**extra)
    return dict(run,dataset_id=100,optimizer_seed=0,requested_updates=3,elapsed_seconds=run['wall_seconds'])

def test_selected_checkpoints_preserve_training():
    default=fixture_run();selected=fixture_run(checkpoint_updates=[0,1,3])
    for a,b in zip(default['history'],selected['history']):
        for key in a:
            if key!='wall_seconds': assert a[key]==b[key]
    assert [r['update'] for r in default['checkpoints']]==[0,3]
    assert [r['update'] for r in selected['checkpoints']]==[0,1,3]
    for a,b in zip(default['checkpoints'],[selected['checkpoints'][0],selected['checkpoints'][-1]]):
        assert a==b
    spec=dict(initial_lambda=.5,updates=3,paths=32,checkpoints=[0,1,3])
    assert validate_run(selected,spec,20)
    broken=copy.deepcopy(selected);broken['checkpoints'][1]['lambda_value']+=.01
    with pytest.raises(ValueError): validate_run(broken,spec,20)
    broken=copy.deepcopy(selected);broken['simulated_steps']-=1
    with pytest.raises(ValueError): validate_run(broken,spec,20)
    broken=copy.deepcopy(selected);broken['completed']=True;broken['history'].pop();broken['updates']=2
    with pytest.raises(ValueError): validate_run(broken,spec,20)

def test_partial_snapshot_on_numerical_failure(monkeypatch):
    from mi_mfg_irl_lq import optimization
    original=optimization.sampled_gradient;calls=0;snapshots=[]
    def fail_second(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2: raise FloatingPointError('test numerical failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(optimization,'sampled_gradient',fail_second)
    def save(r): snapshots.append(copy.deepcopy(r))
    with pytest.raises(FloatingPointError): fixture_run(progress_callback=save)
    final=snapshots[-1]
    assert not final['completed'] and final['updates']==1
    assert final['failure']['type']=='FloatingPointError'
    assert final['rollout_calls']==2 and final['simulated_steps']==1280
    assert final['history'][0]['update']==1

def test_partial_summary_keeps_configured_checkpoint_columns():
    run=fixture_run(checkpoint_updates=[0,1,3])
    partial=copy.deepcopy(run)
    partial.update(completed=False,updates=2,history=run['history'][:2],
        checkpoints=run['checkpoints'][:2],final_lambda=run['history'][1]['lambda_after'],
        failure=dict(type='FloatingPointError',message='fixture failure'))
    complete=summarize_run(run,[0,1,3]);incomplete=summarize_run(partial,[0,1,3])
    assert complete.keys()==incomplete.keys()
    assert incomplete['k3_reference_loss'] is None
    assert not incomplete['completed'] and incomplete['updates']==2

def test_tail_range_and_checkpoint_progress():
    run=fixture_run(checkpoint_updates=[0,1,3]);summary=summarize_run(run,[0,1,3])
    tail=[r['lambda_after'] for r in run['history']]
    assert summary['tail_range']==max(tail)-min(tail)
    assert summary['tail_min']==min(tail) and summary['tail_max']==max(tail)
    assert summary['objective_progress_1_3']==run['checkpoints'][1]['reference_loss']-run['checkpoints'][2]['reference_loss']

ROOT=Path(__file__).resolve().parents[1]

def test_experiment_configuration_validation(tmp_path):
    spec=load_experiment_config(ROOT/'configs/development.json')
    assert len(configurations(spec))==30
    ids=[run_identifier(d,**g,seed=0) for d in spec['dataset_ids'] for g in configurations(spec)]
    assert len(ids)==len(set(ids))==600
    assert len(ids)*spec['updates']==600000
    # Configuration validation also supports bounded artificial fixtures.
    changed=copy.deepcopy(spec)
    changed.update(dataset_ids=[100,102],updates=30,paths=32,checkpoints=[0,10,30],optimizer_seeds=[11,12])
    changed.pop('optimizer_seed')
    changed['families'][0]['etas']=[.5,1,2,5,10,20]
    p=tmp_path/'new.json';p.write_text(json.dumps(changed))
    assert load_experiment_config(p)==changed
    for field,value in [('paths',1),('updates',True),('dataset_ids',[100,100]),
                        ('optimizer_seeds',[-1]),('checkpoints',[0,31]),('initial_lambda',2),('h',.01)]:
        bad=copy.deepcopy(changed);bad[field]=value;p.write_text(json.dumps(bad))
        with pytest.raises(ValueError): load_experiment_config(p)
    for etas in ([1,1],[0,1],[1,float('inf')],[1,1.00000001]):
        bad=copy.deepcopy(changed);bad['families'][0]['etas']=etas;p.write_text(json.dumps(bad))
        with pytest.raises(ValueError): load_experiment_config(p)


@pytest.mark.parametrize('version,beta',[('V1',0),('V1',.001),('V2',None)])
def test_exact_split_training_and_absolute_banks(version,beta):
    model=ModelConfig(num_time_steps=20)
    expert=ExpertData(model.time_grid,np.full(21,.5),.3)
    args=dict(dataset_id=42,run_id='split',version=version,beta=beta,eta=20,paths=32,optimizer_seed=0)
    full=train(model,expert,**args,updates=6,checkpoint_updates=[3,6])
    first=train(model,expert,**args,updates=3,checkpoint_updates=[3])
    second=train(model,expert,**args,updates=3,start_update=3,initial_lambda=first['final_lambda'],checkpoint_updates=[6])
    for actual,expected in zip(first['history']+second['history'],full['history']):
        assert {k:v for k,v in actual.items() if k!='wall_seconds'}=={k:v for k,v in expected.items() if k!='wall_seconds'}
    assert second['history'][0]['update']==4
    assert second['history'][0]['seed_words']==str((20261016,42,0,3))
    assert second['checkpoints'][-1]==full['checkpoints'][-1]
    with pytest.raises(ValueError):train(model,expert,**args,updates=3,start_update=3,checkpoint_updates=[0])
    with pytest.raises(ValueError):train(model,expert,**args,start_update=-1)
