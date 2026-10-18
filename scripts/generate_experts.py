#!/usr/bin/env python3
"""Generate exactly the accepted development or frozen unseen observations."""
from pathlib import Path
import argparse
from mi_mfg_irl_lq.workspace import prepare_workspace
from mi_mfg_irl_lq.development import load_development_config,prepare_development_data
from mi_mfg_irl_lq.unseen import load_unseen_config,prepare_unseen_data
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['development','unseen'],required=True)
    p.add_argument('--workspace',type=Path,required=True);args=p.parse_args()
    root=prepare_workspace(args.workspace.resolve(),ROOT)
    output=root/'results'/args.split;output.mkdir(parents=True,exist_ok=True)
    if args.split=='development':
        spec=load_development_config(ROOT/'configs/development.json')
        prepare_development_data(root,output,spec)
    else:
        spec=load_unseen_config(ROOT/'configs/unseen.json')
        prepare_unseen_data(root,output,spec,list(range(2000,2040)))

if __name__=='__main__':main()
