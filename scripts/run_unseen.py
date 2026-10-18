#!/usr/bin/env python3
"""Execute the frozen unseen protocol or a separate artificial implementation smoke."""
import argparse,json
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
from mi_mfg_irl_lq.config import ModelConfig
from mi_mfg_irl_lq.observations import Observations,preprocess
from mi_mfg_irl_lq.io import write_npz,write_json
from mi_mfg_irl_lq.unseen import run_unseen,load_unseen_config
from mi_mfg_irl_lq.workspace import prepare_workspace
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/unseen.json'))
    p.add_argument('--workspace',type=Path,default=ROOT);p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,choices=[1,2,3],default=3)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    config=args.config if args.config.is_absolute() else ROOT/args.config
    root=args.workspace.resolve();prepare_workspace(root,ROOT)
    output=(args.output if args.output.is_absolute() else root/args.output) if args.output else root/'results/unseen'
    override=None
    if args.smoke:
        output=output/'smoke';root=output/'fixture_project';(root/'configs').mkdir(parents=True,exist_ok=True)
        model=replace(ModelConfig.load(ROOT/'configs/model.json'),num_time_steps=20)
        spec=load_unseen_config(config);spec.update(model=asdict(model),dataset_ids=[42,43],updates=2,paths=32,checkpoints=[0,2],artificial_fixture=True)
        spec['data_generation'].update(num_agents=4,num_rounds=2)
        spec['stopping'].update(M_fixed=2,M_min=2,batch_size=2,M_max=2)
        write_json(root/'configs/model.json',asdict(model));settings={k:v for k,v in spec['data_generation'].items() if k!='lambda_star'}
        for i,dataset in enumerate(spec['dataset_ids']):
            path=root/spec['data_directory']/f'dataset_{dataset}.npz'
            if not path.exists():
                observed=Observations(model.time_grid,np.full(21,.5+.001*i),np.full(21,.26),np.full(21,.01),8)
                expert=preprocess(model,observed)
                write_npz(path,time=observed.time,empirical_mean=observed.mean,observed_mean_X_squared=observed.state_second,
                    observed_mean_action_squared=observed.action_second,J_left=expert.J_E,sJ=expert.sJ,Q=expert.Q,observation_count=8,
                    dataset_id=dataset,dataset_seed=dataset,**settings)
        override=spec
    result=run_unseen(root,output,config,args.workers,override)
    print(f"Completed M={result['M_stop']}, runs={result['completed_runs']}, reason={result['stopping_reason']}")

if __name__=='__main__':main()
