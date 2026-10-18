from pathlib import Path
import json
import pytest
from mi_mfg_irl_lq.workspace import prepare_workspace
from mi_mfg_irl_lq.unseen import load_unseen_config,run_unseen,aggregate
from test_unseen import fake_rows
ROOT=Path(__file__).resolve().parents[1]


def test_workspace_model_and_no_silent_overwrite(tmp_path):
    prepare_workspace(tmp_path,ROOT)
    p=tmp_path/'configs/model.json';before=p.read_bytes()
    prepare_workspace(tmp_path,ROOT);assert p.read_bytes()==before
    data=json.loads(p.read_text());data['sigma']=.2;p.write_text(json.dumps(data))
    with pytest.raises(ValueError):prepare_workspace(tmp_path,ROOT)


def test_frozen_reproduction_never_extends_to_unused_ids(tmp_path,monkeypatch):
    from mi_mfg_irl_lq import unseen
    spec=load_unseen_config(ROOT/'configs/unseen.json')
    prepare_workspace(tmp_path,ROOT);prepared=[]
    monkeypatch.setattr(unseen,'prepare_unseen_data',lambda root,out,spec,ids,exact:prepared.extend(ids))
    monkeypatch.setattr(unseen,'execute_batch',lambda *args:[])
    rows=fake_rows(40)
    monkeypatch.setattr(unseen,'dataset_records',lambda *args:(rows,[]))
    def require_extension(rows,spec):
        cumulative,trace,decision=aggregate(rows,spec)
        decision.update(stop=False,reason=None,previous_Delta=.1,current_Delta=.1)
        return cumulative,trace,decision
    monkeypatch.setattr(unseen,'aggregate',require_extension)
    with pytest.raises(RuntimeError,match='IDs 2040'):
        run_unseen(tmp_path,tmp_path/'output',ROOT/'configs/unseen.json',workers=1)
    assert prepared==list(range(2000,2040))
