"""Official compact evidence remains testable without private training histories."""
from pathlib import Path
import csv,io,json,shutil
import numpy as np
import pytest
from mi_mfg_irl_lq.config import ModelConfig
from mi_mfg_irl_lq.expert_data import load_expert
from mi_mfg_irl_lq.selection import reselect_from_csv
from mi_mfg_irl_lq.unseen_reporting import reconstruct_tables,read_csv
ROOT=Path(__file__).resolve().parents[1]

def serialized(rows):
    stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);stream.seek(0)
    return list(csv.DictReader(stream))

def forbid_training(monkeypatch):
    from mi_mfg_irl_lq import optimization,runs
    def forbidden(*a,**k):pytest.fail('Report reconstruction must not invoke optimization.')
    monkeypatch.setattr(optimization,'train',forbidden);monkeypatch.setattr(runs,'train',forbidden)

def test_all_compact_observations_and_information_boundary():
    model=ModelConfig.load(ROOT/'configs/model.json')
    for split,ids in [('development',range(1000,1020)),('unseen',range(2000,2040))]:
        registry=read_csv(ROOT/'data'/split/'registry.csv')
        assert {int(r['dataset_id']) for r in registry}==set(ids)
        assert {p.name for p in (ROOT/'data'/split).glob('*.npz')}=={f'dataset_{d}.npz' for d in ids}
        for d in ids:
            path=ROOT/'data'/split/f'dataset_{d}.npz';expert=load_expert(path,model)
            with np.load(path,allow_pickle=False) as z:
                assert len(z['time'])==1001 and int(z['dataset_seed'])==d
                assert int(z['observation_count'])==20000 and int(z['num_rounds'])==100 and int(z['num_agents'])==200
                assert int(z['random_seed_base'])==20260818 and int(z['setting_index'])==1
                assert expert.J_E==.001*float(np.sum((.5*(z['observed_mean_X_squared']+z['observed_mean_action_squared']+z['empirical_mean']**2))[:-1]))
                assert expert.Q==np.trapezoid(z['empirical_mean']**2,z['time'])
                np.testing.assert_array_equal(expert.mean,z['empirical_mean'])
                assert not hasattr(expert,'state_second') and not hasattr(expert,'lambda_star')

def test_compact_development_exact_selection(tmp_path,monkeypatch):
    forbid_training(monkeypatch)
    source=ROOT/'results/development'
    for name in ['config.json','estimates.csv']:shutil.copy2(source/name,tmp_path/name)
    decisions=reselect_from_csv(tmp_path)
    assert len(read_csv(source/'estimates.csv'))==600 and len(read_csv(tmp_path/'candidates.csv'))==30
    assert read_csv(tmp_path/'candidates.csv')==read_csv(source/'candidates.csv')
    assert serialized(decisions)==read_csv(source/'selection.csv')
    assert [r['winner_eta'] for r in decisions]==[20,20,5,2,1,.5,.2,.02,.002,20]
    assert [r['deciding_step'] for r in decisions]==['maximize_success_count']*2+['minimize_median_tail_SD']*7+['maximize_success_count']

def test_compact_unseen_summaries_and_stopping(tmp_path,monkeypatch):
    forbid_training(monkeypatch)
    source=ROOT/'results/unseen'
    for name in ['config.json','run_accounting.json','estimates.csv','summary.csv']:shutil.copy2(source/name,tmp_path/name)
    spec,account,rows,fixed,cumulative,trace,stopped=reconstruct_tables(tmp_path)
    assert len(rows)==400 and len(fixed)==10 and len(cumulative)==80 and len(trace)==80
    assert serialized(fixed)==read_csv(source/'summary.csv')==serialized(stopped)
    assert serialized(cumulative)==read_csv(source/'cumulative.csv')
    assert serialized(trace)==read_csv(source/'stopping.csv')
    assert account['M_stop']==40
    assert trace[-1]['Delta']==account['current_Delta']==.029786041500306042
    assert next(r['Delta'] for r in trace if r['M']==35)==account['previous_Delta']==.04505515704763263
    assert [r['success_count'] for r in fixed]==[16,19,40,40,39,39,39,39,39,15]

def test_existing_corrupt_history_is_not_skipped(tmp_path,monkeypatch):
    import gzip,test_unseen
    bad=tmp_path/'history.csv.gz';bad.write_bytes(b'not a gzip history')
    monkeypatch.setattr(test_unseen,'HISTORY',bad)
    with pytest.raises(gzip.BadGzipFile):test_unseen.test_completed_unseen_serialized_evidence()
