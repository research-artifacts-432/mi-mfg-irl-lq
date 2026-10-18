"""Small atomic file writers for data and run results."""
from pathlib import Path
import csv
import json
import os
import tempfile
import numpy as np

def atomic_write(path, writer, binary=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb' if binary else 'w') as f: writer(f)
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)

def write_json(path,value): atomic_write(path,lambda f:json.dump(value,f,indent=2,allow_nan=False))
def write_npz(path,**arrays): atomic_write(path,lambda f:np.savez_compressed(f,**arrays),True)
def write_csv(path,rows):
    def writer(f):
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_write(path,writer)
