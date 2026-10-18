#!/usr/bin/env python3
"""Run the prescribed formal development selection or a bounded separate smoke."""
import argparse,json
from pathlib import Path
from dataclasses import asdict,replace
from mi_mfg_irl_lq.config import ModelConfig
from mi_mfg_irl_lq.development import run_development
from mi_mfg_irl_lq.workspace import prepare_workspace
ROOT=Path(__file__).resolve().parents[1]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/development.json')
    p.add_argument('--workspace',type=Path,default=ROOT)
    p.add_argument('--output',type=Path)
    p.add_argument('--workers',type=int,choices=[1,2,3],default=3)
    p.add_argument('--select-only',action='store_true',help='Reproduce the ordered selection from consolidated dataset results; no simulation.')
    p.add_argument('--smoke',action='store_true',help='Separate seed-42 fixture: 20 steps, 8 agents, 2 rounds, 3 updates, 32 paths.')
    args=p.parse_args(argv);config=args.config if args.config.is_absolute() else ROOT/args.config
    root=args.workspace.resolve();prepare_workspace(root,ROOT)
    output=(args.output if args.output.is_absolute() else root/args.output) if args.output else root/'results/development'
    tolerance=1e-7
    if args.select_only:
        if args.smoke:p.error('Use --select-only with an existing output directory, not --smoke.')
        from mi_mfg_irl_lq.selection import reselect_from_csv
        selected=reselect_from_csv(output)
        print(f'Reconstructed {len(selected)} within-family decisions from CSV.');return
    if args.smoke:
        output=output/'smoke';root=output/'fixture_project';(root/'configs').mkdir(parents=True,exist_ok=True)
        model=replace(ModelConfig.load(ROOT/'configs/model.json'),num_time_steps=20)
        spec=json.loads(config.read_text());spec.update(model=asdict(model),dataset_ids=[42],updates=3,paths=32,checkpoints=[0,3],families=[dict(version='V1',beta=.1,etas=[1,2]),dict(version='V2',beta=None,etas=[5])])
        spec['data_generation'].update(num_agents=8,num_rounds=2)
        (root/'configs/model.json').write_text(json.dumps(asdict(model)))
        config=root/'configs/smoke.json';config.write_text(json.dumps(spec));tolerance=2e-6
    result=run_development(root,output,config,args.workers,tolerance)
    print(f"Completed {result['completed_runs']}/{result['planned_runs']} runs; primary steps={result['simulated_steps']}")

if __name__=='__main__':main()
