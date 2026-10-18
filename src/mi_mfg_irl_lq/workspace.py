"""Keep generated data/results separate from the installed scientific source."""
from pathlib import Path
import json
from .io import write_json


def prepare_workspace(workspace,source):
    workspace=Path(workspace);source=Path(source)
    model=json.loads((source/'configs/model.json').read_text())
    path=workspace/'configs/model.json'
    if path.exists():
        if json.loads(path.read_text())!=model:raise ValueError('Workspace model configuration differs.')
    else:write_json(path,model)
    return workspace
