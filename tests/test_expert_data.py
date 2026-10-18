"""Tight cached-scale verification tolerates rounding, never replaces inverse scales."""
import numpy as np
import pytest
from mi_mfg_irl_lq.config import ModelConfig
from mi_mfg_irl_lq.expert_data import ExpertData,load_expert

@pytest.fixture
def dataset(tmp_path):
    model=ModelConfig(num_time_steps=20)
    expert=ExpertData(model.time_grid,.5+.01*model.time_grid,.24)
    path=tmp_path/'expert.npz'
    arrays=dict(time=expert.time,empirical_mean=expert.mean,J_left=expert.J_E,sJ=expert.sJ,Q=expert.Q)
    return model,expert,path,arrays

def save_and_load(dataset):
    model,expert,path,arrays=dataset
    np.savez(path,**arrays)
    return load_expert(path,model)

def test_correct_cached_scales_load(dataset):
    loaded=save_and_load(dataset);expert=dataset[1]
    assert loaded.J_E==expert.J_E and loaded.sJ==expert.sJ and loaded.Q==expert.Q
    np.testing.assert_array_equal(loaded.mean,expert.mean)

@pytest.mark.parametrize('key',['sJ','Q'])
@pytest.mark.parametrize('direction',[0.,np.inf])
def test_one_ulp_cache_rounding_is_verification_only(dataset,key,direction):
    expert=dataset[1];arrays=dataset[3];original=arrays[key]
    arrays[key]=np.nextafter(original,direction)
    assert arrays[key]!=original and 0<abs(arrays[key]-original)<1e-15
    loaded=save_and_load(dataset)
    assert loaded.sJ==expert.sJ and loaded.Q==expert.Q
    assert getattr(loaded,key)==original and getattr(loaded,key)!=arrays[key]

@pytest.mark.parametrize('key',['sJ','Q'])
def test_materially_stale_cached_scale_is_rejected(dataset,key):
    dataset[3][key]+=1e-6
    with pytest.raises(ValueError,match='Stale cached dataset scales'):save_and_load(dataset)

@pytest.mark.parametrize('key',['sJ','Q'])
@pytest.mark.parametrize('invalid',[np.nan,np.inf,-np.inf,0.,-1.])
def test_invalid_cached_scale_is_rejected(dataset,key,invalid):
    dataset[3][key]=invalid
    with pytest.raises(ValueError,match='Stale cached dataset scales'):save_and_load(dataset)
