"""Exact development matrix, strict ordered selection and bounded orchestration."""
from pathlib import Path
import copy,json
import numpy as np
import pytest
from mi_mfg_irl_lq.development import load_development_config,training_specification
from mi_mfg_irl_lq.runs import configurations,run_identifier
from mi_mfg_irl_lq.selection import summarize_candidate,select_within_families
ROOT=Path(__file__).resolve().parents[1]
SPEC=load_development_config(ROOT/'configs/development.json')
RULE=SPEC['selection']


def test_exact_development_matrix_and_budgets():
    s=SPEC
    assert s['dataset_ids']==list(range(1000,1020)) and s['optimizer_seed']==0
    assert (s['updates'],s['paths'],s['h'],s['initial_lambda'])==(1000,4096,1e-4,.5)
    assert s['checkpoints']==[0,250,500,750,1000]
    assert [f['etas'] for f in s['families']]==[[5,10,20],[5,10,20],[5,10,20],[2,5,10],[1,2,5],
        [.5,1,2],[.2,.5,1],[.02,.05,.1],[.002,.005,.01],[5,10,20]]
    assert [(f['version'],f['beta']) for f in s['families']]==[('V1',b) for b in [0,.001,.01,.03,.1,.3,1,10,100]]+[('V2',None)]
    ids=[run_identifier(d,**g,seed=0) for d in s['dataset_ids'] for g in configurations(s)]
    assert len(ids)==len(set(ids))==600
    assert len(ids)*s['updates']*2*s['paths']*1000==4915200000000
    assert s['data_generation']==dict(lambda_star=.9,num_agents=200,num_rounds=100,random_seed_base=20260818,setting_index=1)
    assert RULE['target_lambda']==.9 and RULE['success_threshold']==.01
    assert all(RULE[k]==1e-4 for k in ['MAE_tolerance','RMSE_tolerance','MaxAE_tolerance'])
    scientific=json.loads((ROOT/'configs/scientific.json').read_text())
    assert scientific['common_final_K']==1000 and list(scientific['eta_grids'].values())==[f['etas'] for f in s['families']]


def test_worker_inputs_exclude_generation_and_selection_truth():
    training=training_specification(SPEC)
    assert 'data_generation' not in training and 'selection' not in training and 'model' not in training
    assert 'lambda_star' not in str(training) and 'target_lambda' not in str(training)
    assert training['data_directory']=='data/development'


def candidate(eta=1,**kwargs):
    row=dict(family='V1_beta0.1',eta=eta,admissible=True,success_count=20,MAE=.002,RMSE=.003,MaxAE=.005,
             total_projection_count=0,median_tail_SD=.001,median_terminal_objective=999.)
    row.update(kwargs);return row


def test_success_count_strictly_precedes_all_error_metrics():
    a=candidate(1,success_count=20,MAE=.009,RMSE=.009,MaxAE=.01)
    b=candidate(2,success_count=19,MAE=.001,RMSE=.001,MaxAE=.011)
    selected=select_within_families([a,b],RULE)[0]
    assert selected['winner_eta']==1 and selected['deciding_step']=='maximize_success_count'


@pytest.mark.parametrize('change,step',[
    (dict(MAE=.0022),'MAE_practical_equivalence'),
    (dict(MAE=.00205,RMSE=.0032),'RMSE_practical_equivalence'),
    (dict(MAE=.00205,RMSE=.00305,MaxAE=.0052),'MaxAE_practical_equivalence'),
    (dict(total_projection_count=1),'minimize_projection_count'),
    (dict(median_tail_SD=.002),'minimize_median_tail_SD'),
    ({},'smaller_eta')])
def test_ordered_practical_equivalence_and_deciding_steps(change,step):
    a=candidate(1);b=candidate(2,**change)
    result=select_within_families([b,a],RULE)[0]
    assert result['winner_eta']==1 and result['deciding_step']==step


def test_tolerances_filter_sequentially_and_objective_never_breaks_ties():
    a=candidate(1,MAE=.002,RMSE=.0032,median_terminal_objective=0.)
    b=candidate(2,MAE=.00205,RMSE=.003,median_terminal_objective=100.)
    c=candidate(3,MAE=.0022,RMSE=.0001,median_terminal_objective=-1.)
    result=select_within_families([a,b,c],RULE)[0]
    assert result['winner_eta']==2 and result['deciding_step']=='RMSE_practical_equivalence'
    trace=json.loads(result['selection_trace'])
    assert trace[2]['etas']==[1,2] and trace[3]['etas']==[2]


def test_admissibility_and_family_separation():
    results=select_within_families([candidate(1,admissible=False),candidate(2),
        candidate(1,family='V2',success_count=0),candidate(2,family='V2',admissible=False)],RULE)
    assert [(r['family'],r['winner_eta']) for r in results]==[('V1_beta0.1',2),('V2',1)]
    unresolved=select_within_families([candidate(1,admissible=False)],RULE)[0]
    assert unresolved['winner_eta'] is None and unresolved['status'].startswith('unresolved')


def test_candidate_recovery_metrics_use_twenty_dataset_units():
    records=[dict(dataset_id=i,valid_completed=True,lambda_hat=x,projection_count=1,
        tail_sample_sd=.001,terminal_objective=1.) for i,x in enumerate([.89,.91,.9])]
    r=summarize_candidate(records,[0,1,2],.9,.01)
    errors=np.abs(np.array([.89,.91,.9])-.9)
    assert r['admissible'] and r['valid_completed_runs']==3 and r['total_projection_count']==3
    assert r['success_count']==int(np.sum(errors<=.01))
    assert r['sample_sd_lambda']==np.std([.89,.91,.9],ddof=1)
    assert r['MAE']==np.mean(errors) and r['RMSE']==np.sqrt(np.mean(errors**2))
    assert not summarize_candidate(records,[0,1,2,3],.9,.01)['admissible']
    records[0]['valid_completed']=False
    assert not summarize_candidate(records,[0,1,2],.9,.01)['admissible']


def test_practical_equivalence_boundary_and_projection_priority():
    a=candidate(1,MAE=.002,RMSE=.003,total_projection_count=3,median_tail_SD=.01)
    b=candidate(2,MAE=.002+1e-4,RMSE=.003+1e-4,total_projection_count=0,median_tail_SD=.02)
    r=select_within_families([a,b],RULE)[0]
    assert r['winner_eta']==2 and r['deciding_step']=='minimize_projection_count'
    a.update(success_count=20);b.update(success_count=19,MAE=0.,RMSE=0.,MaxAE=0.)
    assert select_within_families([a,b],RULE)[0]['winner_eta']==1


def test_nonfinite_or_out_of_bounds_parameters_are_not_admissible():
    for value in [float('nan'),float('inf'),-1.,2.]:
        r=dict(dataset_id=42,valid_completed=True,lambda_hat=value,tail_sample_sd=.001,terminal_objective=1.,projection_count=0)
        assert not summarize_candidate([r],[42],.9,.01)['admissible']


def test_smoke_packet_selection_reproduces_without_snapshots(tmp_path):
    import csv
    from mi_mfg_irl_lq.io import write_csv
    from mi_mfg_irl_lq.selection import reselect_from_csv,evaluate_candidates
    spec=copy.deepcopy(SPEC);spec.update(dataset_ids=[42,43],families=[dict(version='V1',beta=.1,etas=[1,2])])
    (tmp_path/'config.json').write_text(json.dumps(spec))
    rows=[]
    for d in [42,43]:
        for eta in [1,2]:
            rows.append(dict(family='V1_beta0.1',dataset_id=d,optimizer_seed=0,eta=eta,completed=True,
                valid_completed=True,updates=1000,lambda_hat=.9+eta*.001,projection_count=0,
                tail_sample_sd=.001,terminal_objective=1.))
    write_csv(tmp_path/'estimates.csv',rows)
    expected_summary,expected_selection=evaluate_candidates(rows,spec)
    actual=reselect_from_csv(tmp_path)
    assert actual==expected_selection
    with (tmp_path/'candidates.csv').open() as f:saved=list(csv.DictReader(f))
    assert [float(r['MAE']) for r in saved]==[r['MAE'] for r in expected_summary]
    assert actual[0]['deciding_step']=='MAE_practical_equivalence'
