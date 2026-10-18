#!/usr/bin/env python3
"""Recompute within-family development selection from compact estimates; no training."""
import argparse
from pathlib import Path
from mi_mfg_irl_lq.selection import reselect_from_csv
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/development')
    args=parser.parse_args();selected=reselect_from_csv(args.output)
    print(f'Reconstructed {len(selected)} within-family selections; no optimization.')
if __name__=='__main__':main()
