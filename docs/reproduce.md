# Reproduction

Run commands from the repository root with the non-editable Python environment described in README. All paths below are relative; the environment and experimental workspace are separate from the repository.

## Quick reconstruction: no training

```sh
../lq-env/bin/python -m pip check
../lq-env/bin/python -m pytest -q -rs
../lq-env/bin/python scripts/report_development.py
../lq-env/bin/python scripts/report_unseen.py
```

Development reporting reads estimates.csv/config.json and recomputes candidates.csv/selection.csv. Unseen reporting reads estimates.csv/config.json/run_accounting.json, verifies the earliest stopping decision and reconstructs summary.csv,cumulative.csv,stopping.csv and five PNG/PDF figure pairs. It does not invoke generation, BR solves or optimization. Fixed40 and stopped40 coincide, so there is one summary and one estimate table. Full-precision CSVs, not rounded README values, are authoritative.

The optional full-history test runs when results/unseen/training_history.csv.gz exists or LQ_UNSEEN_HISTORY names an external gzip CSV. If absent, only that test skips explicitly. An existing corrupt/inconsistent history fails. All compact-data, numerical, selection and stopping tests run without histories. Full stochastic histories are deliberately excluded from the repository; any full experimental rerun writes its own.

## Optional expensive formal reproduction

Use an empty `../lq-work` workspace for fresh observations and histories. The commands use the included accepted generator directly:

```sh
../lq-env/bin/python scripts/generate_experts.py --split development --workspace ../lq-work
../lq-env/bin/python scripts/run_development.py --workspace ../lq-work --workers 3
../lq-env/bin/python scripts/report_development.py --output ../lq-work/results/development
# After verifying development selection, run only the frozen40-dataset unseen experiment.
../lq-env/bin/python scripts/generate_experts.py --split unseen --workspace ../lq-work
../lq-env/bin/python scripts/run_unseen.py --workspace ../lq-work --workers 3
../lq-env/bin/python scripts/report_unseen.py --output ../lq-work/results/unseen
```

To train on retained compact observations rather than regenerate them, first create ../lq-work/data and copy data/development and data/unseen there; the same commands validate and reuse the observations. Missing observations are generated once. Workspace model configuration is copied/validated automatically. Existing genuine completed runs in that workspace may be reused on restart; partial/failed runs remain explicit. Do not point expensive training into the supplied official results directories.

The fixed matrix is600 development plus400 unseen trajectories, each K1000/4096 paths per endpoint/h1e-4/seed0/initial.5. Expected work:1,000,000 updates,2,000,000 FD endpoints,8,192,000,000 sampled BR paths and8,192,000,000,000 primary BR Euler trajectory-steps; additionally60×20,000 expert paths×1000 steps and5000 diagnostic checkpoints. Expect hours with up to three workers, not a quick test. Full-history and per-run restart outputs stay in ../lq-work; no old histories are optimizer inputs.

A bounded fixture is available without formal dataset exposure:

```sh
../lq-env/bin/python scripts/run_development.py --workspace ../lq-fixture --smoke --workers 1
```

This uses only seed42,20 steps,8 agents×2 rounds,3 configurations×3 updates and32 paths per endpoint. It is a software/scientific implementation check, not a shortened formal experiment or calibration choice. No calibration search, eta refinement, K2000 continuation or additional unseen IDs are authorized by these reproduction commands.
