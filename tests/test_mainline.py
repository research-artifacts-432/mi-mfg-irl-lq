from dataclasses import replace,fields
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from mi_mfg_irl_lq.config import ModelConfig
from mi_mfg_irl_lq.model import AffinePolicy,MeanTrajectory,solve_a
from mi_mfg_irl_lq.best_response import BestResponse,BestResponseOracle
from mi_mfg_irl_lq.generation import generate_exact_expert
from mi_mfg_irl_lq.costs import left_integral,law_costs
from mi_mfg_irl_lq.observations import Observations,preprocess,simulate_observations
from mi_mfg_irl_lq.expert_data import ExpertData,load_expert
from mi_mfg_irl_lq.rollout import make_random_bank,SampledPolicyEvaluator,RandomBank
from mi_mfg_irl_lq.optimization import finite_difference_stencil,projected_update,sampled_gradient,train
from mi_mfg_irl_lq.losses import components,loss
from mi_mfg_irl_lq.diagnostics.euler_reference import evaluate_euler_moments

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def model(): return ModelConfig(num_time_steps=20)


def zero_br(m,mean=2.):
    t=m.time_grid;z=np.zeros_like(t)
    return BestResponse(AffinePolicy(t,z,z),MeanTrajectory(t,np.full_like(t,mean)))


def test_mixed_quadrature_and_fixed_scales(model):
    t=model.time_grid;f=t**2
    assert left_integral(f,t)==pytest.approx(model.delta_t*np.sum(f[:-1]))
    assert left_integral(np.r_[np.ones(20),1e12],t)==pytest.approx(1.)
    e=ExpertData(t,t+1,2.)
    assert e.Q==pytest.approx(np.trapezoid((t+1)**2,t))
    assert e.Q!=pytest.approx(left_integral((t+1)**2,t))
    ev=evaluate_euler_moments(model,.4,zero_br(model),e.mean_flow)
    c=components(e,ev,zero_br(model).mean)
    assert c['D_C']==pytest.approx(np.trapezoid((2-e.mean)**2,t))
    for beta in (0,.1,100): loss(c,'V1',beta)
    assert (e.J_E,e.sJ,e.Q)==(2.,4.,float(np.trapezoid((t+1)**2,t)))
    with pytest.raises(ValueError): loss(c,'V2',0)
    with pytest.raises(ValueError): e.mean[0]=0


def test_observed_preprocessing(model):
    t=model.time_grid;mean=np.full_like(t,.6);x2=np.full_like(t,.5);a2=np.full_like(t,.2)
    x2[-1]=100.
    o=Observations(t,mean,x2,a2,16);e=preprocess(model,o)
    assert e.J_E==pytest.approx(.5*(.5+.2+.36))
    assert e.mean[0]==.6
    assert not hasattr(e,'state_second') and not hasattr(e,'policy')
    assert {f.name for f in fields(e)}=={'time','mean','J_E','sJ','Q'}
    with pytest.raises(ValueError): preprocess(model,replace(o,action_second=-a2))


def test_known_law_br_and_kernel_residuals(model):
    flow=MeanTrajectory(model.time_grid,np.full(21,.73))
    oracle=BestResponseOracle(model)
    for candidate in (0.,.5,1.):
        br=oracle.solve(candidate,flow);record=oracle.last_validation
        assert br.mean.values[0]==pytest.approx(.5,abs=1e-12)
        assert np.isfinite(br.policy.intercept).all()
        assert br.policy.slope[-1]==pytest.approx(0.,abs=1e-12)
        assert max(record.diagnostics[k]['max_rms_ode_residual'] for k in ('b','c_d','mean'))<2e-9
        assert abs(record.b[-1])+abs(record.c[-1])+abs(record.d[-1])<1e-12
    assert oracle.cached_a.diagnostics['max_ode_simpson_defect']<1e-6


def test_exact_expert_independent_dpp(model):
    exact=generate_exact_expert(ModelConfig(),.9)
    assert exact.diagnostics['max_q_identity_error']<1e-7
    assert exact.diagnostics['max_reduced_full_mean_error']<1e-7
    assert not hasattr(model,'lambda_star')


def test_sampled_cost_manual_em_and_solver_law(model):
    br=zero_br(model,2.);flow=MeanTrajectory(model.time_grid,np.full(21,3.))
    bank=make_random_bank(20,8,(10,))
    ev=SampledPolicyEvaluator(model).evaluate_budgets(.4,br,flow,bank,(8,))[8]
    x=.5+.1*bank.initial_standard_normals;cost=np.zeros(8)
    for n in range(20):
        cost+=.5*x*x*model.delta_t
        x=x+.5*x*model.delta_t+.1*np.sqrt(model.delta_t)*bank.brownian_standard_normals[n]
    assert ev.local_cost==pytest.approx(np.mean(cost),abs=1e-14)
    assert ev.fixed_law_cost==pytest.approx(.5*(.4*3+.6*2)**2)
    assert ev.self_law_cost==pytest.approx(2.)
    assert abs(ev.sampled_mean[0]-2)>1


def test_euler_reference_matches_sampled_expectation(model):
    br=BestResponseOracle(model).solve(.5,MeanTrajectory(model.time_grid,np.full(21,.5)))
    flow=MeanTrajectory(model.time_grid,np.full(21,.5))
    ref=evaluate_euler_moments(model,.5,br,flow)
    bank=make_random_bank(20,50000,(17,))
    ev=SampledPolicyEvaluator(model).evaluate_budgets(.5,br,flow,bank,(50000,))[50000]
    assert abs(ref.local_cost-ev.local_cost)<5*ev.local_cost_standard_error
    assert ref.local_cost==pytest.approx(left_integral(.5*(ref.second_moment+ref.expected_control_square),model.time_grid))
    assert ref.fixed_law_cost==ev.fixed_law_cost
    assert ref.self_law_cost==ev.self_law_cost
    assert ref.second_moment[0]==.26

@pytest.mark.parametrize('anchor,expected',[(0.,(0.,.0001,'forward')),(.00005,(.00005,.00015,'forward')),(.5,(.4999,.5001,'central')),(.99995,(.99985,.99995,'backward')),(1.,(.9999,1.,'backward'))])
def test_endpoint_convention(anchor,expected):
    low,high,d,scheme=finite_difference_stencil(anchor)
    assert (low,high)==pytest.approx(expected[:2]);assert scheme==expected[2]
    assert d==pytest.approx(high-low)
    assert projected_update(.5,10,1)==(-9.5,0.)
    assert projected_update(.5,10,-1)==(10.5,1.)


def test_crn_pairing_and_fresh_banks(model):
    e=ExpertData(model.time_grid,np.full(21,.5),.4)
    oracle=BestResponseOracle(model);engine=SampledPolicyEvaluator(model)
    class RecordingEvaluator:
        def __init__(self): self.banks=[]
        def evaluate_budgets(self,c,br,flow,bank,budgets):
            self.banks.append(bank)
            return engine.evaluate_budgets(c,br,flow,bank,budgets)
    rec=RecordingEvaluator();bank=make_random_bank(20,32,(20261016,100,0,0))
    a=sampled_gradient(model,e,oracle,rec,.5,bank,32,'V1',.1)
    assert len(rec.banks)==2 and rec.banks[0] is rec.banks[1]
    same=make_random_bank(20,32,(20261016,100,0,0));fresh=make_random_bank(20,32,(20261016,100,0,1))
    assert np.array_equal(bank.brownian_standard_normals,same.brownian_standard_normals)
    assert not np.array_equal(bank.initial_standard_normals,fresh.initial_standard_normals)
    r=train(model,e,dataset_id=100,run_id='test',version='V2',eta=10,updates=2,paths=32)
    assert r['rollout_calls']==4 and r['simulated_steps']==2560
    assert r['history'][0]['seed_words']!=r['history'][1]['seed_words']
    assert r['history'][1]['lambda_before']==r['history'][0]['lambda_after']
    for row in r['history']:
        assert row['lambda_after']==np.clip(row['lambda_before']-10*row['gradient'],0,1)


def test_observation_seed_semantics(model):
    p=zero_br(model).policy
    settings=dict(dataset_seed=42,random_seed_base=20260818,setting_index=1,num_agents=4,num_rounds=2)
    obs=simulate_observations(model,p,**settings)
    initial=[]
    for r in range(2):
        rng=np.random.default_rng(np.random.SeedSequence([20260818,1,42,r]))
        initial.extend(rng.normal(.5,.1,4))
    assert obs.mean[0]==pytest.approx(np.mean(initial))
    assert obs.state_second[0]==pytest.approx(np.mean(np.square(initial)))
    assert obs.count==8


def test_portable_observed_data_loading(tmp_path):
    from mi_mfg_irl_lq.io import write_npz
    m=ModelConfig()
    observed=Observations(m.time_grid,np.full(1001,.501),np.full(1001,.27),np.full(1001,.1),20000)
    expert=preprocess(m,observed)
    path=tmp_path/'observed_fixture.npz'
    write_npz(path,time=observed.time,empirical_mean=observed.mean,
        observed_mean_X_squared=observed.state_second,observed_mean_action_squared=observed.action_second,
        observation_count=observed.count,J_left=expert.J_E,sJ=expert.sJ,Q=expert.Q)
    loaded=load_expert(path,m)
    with np.load(path) as z:
        roundtrip=Observations(z['time'],z['empirical_mean'],z['observed_mean_X_squared'],z['observed_mean_action_squared'],int(z['observation_count']))
        assert roundtrip.count==20000
        np.testing.assert_array_equal(roundtrip.state_second,observed.state_second)
        np.testing.assert_array_equal(roundtrip.action_second,observed.action_second)
    assert preprocess(m,roundtrip).J_E==loaded.J_E==expert.J_E
    assert loaded.sJ==expert.sJ and loaded.Q==expert.Q
    assert loaded.mean[0]!=m.initial_mean


def test_invalid_inputs(model):
    with pytest.raises(ValueError): ModelConfig(initial_variance=.1)
    with pytest.raises(ValueError): ExpertData(model.time_grid,np.zeros(21),1.)
    with pytest.raises(ValueError): projected_update(.5,1,np.nan)
    with pytest.raises(ValueError): make_random_bank(20,1,(1,))
    bad=zero_br(model)
    bad.policy.slope[2]=np.nan
    with pytest.raises(ValueError): evaluate_euler_moments(model,.5,bad,MeanTrajectory(model.time_grid,np.ones(21)))


def test_standalone_import_from_outside_workspace():
    import json
    command="""import importlib, importlib.metadata, json, pathlib, pkgutil
import mi_mfg_irl_lq as package
names=[package.__name__]+[m.name for m in pkgutil.walk_packages(package.__path__,package.__name__+'.')]
print(json.dumps({'base':str(importlib.metadata.distribution('mi-mfg-irl-lq').locate_file('mi_mfg_irl_lq').resolve()),
 'files':{name:str(pathlib.Path(importlib.import_module(name).__file__).resolve()) for name in names}}))"""
    out=subprocess.check_output([sys.executable,'-I','-c',command],cwd='/tmp',text=True)
    evidence=json.loads(out);base=Path(evidence['base'])
    assert base.is_relative_to(Path(sys.prefix).resolve())
    assert not base.is_relative_to(ROOT/'src')  # Require the actual non-editable installation.
    expected={p.relative_to(ROOT/'src/mi_mfg_irl_lq').as_posix():p for p in (ROOT/'src/mi_mfg_irl_lq').rglob('*.py')}
    imported={Path(name).relative_to(base).as_posix():Path(name) for name in evidence['files'].values()}
    assert imported.keys()==expected.keys()
    for relative,installed in imported.items():
        assert installed.is_relative_to(base)
        assert installed.read_bytes()==expected[relative].read_bytes()
    assert 'evaluator-dev' not in out
