#!/usr/bin/env python3
"""Reconstruct frozen unseen tables, figures and report without training."""
import argparse
from pathlib import Path
from mi_mfg_irl_lq.unseen_reporting import reconstruct_tables,scientific_figures
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('results/unseen'));args=p.parse_args()
    output=args.output if args.output.is_absolute() else ROOT/args.output
    inputs=reconstruct_tables(output);figures=scientific_figures(output,inputs)
    print(f'Reconstructed tables and {len(figures)} figure files from retained estimates; no scientific training.')

if __name__=='__main__':main()
