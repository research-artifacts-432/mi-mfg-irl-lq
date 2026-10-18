# mi-mfg-irl-lq

Version **1.0.0**: **LQ inverse learning with closed-form BR oracle**. The BR policy comes from the accepted closed-form oracle, BR costs use sampled Monte Carlo trajectories, and inverse optimization uses sampled finite-difference gradients. The planned release tag is `v1.0.0`; no release has been published.

Scalar-LQ mean-field-game inverse learning with a closed-form LQ best-response (BR) solver and **sampled Euler–Maruyama BR cost evaluation**. The analytic BR is obtained through the retained numerical ODE/boundary-value solver; training costs are sampled, not moment substituted. This version includes expert generation, formal development selection, frozen unseen evaluation, compact official evidence and figure reconstruction.

## Method

T=1, b_alpha=1, b_x=0.5, c_alpha=c_x=c_mu=1, sigma=0.1. Dynamics are dX=(b_alpha*alpha+b_x*X)dt+sigma*dW. The known Gaussian initial law has mean0.5, variance0.01 (standard deviation0.1) and second moment0.26. The grid has1000 intervals/1001 nodes, dt=.001.

Inverse expert inputs are only time, empirical mean flow and observed empirical J, together with known model information. Empirical means are not recentered. Expert J is computed directly from observed state/action second moments; these preprocessing arrays and generating truth are not optimizer inputs. Candidate state/action costs are sampled under the known law. Law-dependent costs use the solver-provided BR mean, never the BR sample mean. Fixed-flow cost uses lambda*m_E+(1-lambda)*m_BR; self-flow cost uses m_BR, with the same BR policy.

Expert/candidate J use exactly1000 **left-Riemann** running-cost terms, excluding t=T; no terminal cost. Consistency and Q use **trapezoidal** integration:

```text
sJ = J_E**2
A_rel = (J_E - J_BR_fixed)**2 / sJ
C_rel = trapezoid((m_BR - m_E)**2, time) / trapezoid(m_E**2, time)
V1 = A_rel + beta*C_rel
V2 = (J_E - J_BR_self)**2
```

V2 is unnormalized and beta-free. Fixed-eta projected optimization uses paired-CRN bounded finite differences h=1e-4. Formal fits use initial lambda=.5, seed0, **K1000**, **4096 paths per endpoint** and the last iterate. No scheduling, averaging or early stopping is used. Euler moment checkpoints are diagnostics only.

## Official configurations and results

| Family | Development eta candidates | Frozen eta | Development successes/20 | Unseen successes/40 |
|---|---|---:|---:|---:|
| V1 beta=0 | 5,10,20 | 20 | 9 | 16 |
| V1 beta=0.001 | 5,10,20 | 20 | 13 | 19 |
| V1 beta=0.01 | 5,10,20 | 5 | 20 | 40 |
| V1 beta=0.03 | 2,5,10 | 2 | 20 | 40 |
| V1 beta=0.1 | 1,2,5 | 1 | 20 | 39 |
| V1 beta=0.3 | .5,1,2 | .5 | 20 | 39 |
| V1 beta=1 | .2,.5,1 | .2 | 20 | 39 |
| V1 beta=10 | .02,.05,.1 | .02 | 20 | 39 |
| V1 beta=100 | .002,.005,.01 | .002 | 20 | 39 |
| V2 | 5,10,20 | 20 | 13 | 15 |

Success means abs(lambda_hat-.9)<=.01. Full-precision estimates/errors,30 candidate summaries and10 decisions are in `results/development/{estimates,candidates,selection}.csv`. Unseen400 estimates, fixed-M40 summaries, cumulative statistics and all-family stopping traces are in `results/unseen/{estimates,summary,cumulative,stopping}.csv`. Five PNG/PDF pairs are in `results/unseen/figures/`. Weak recovery remains visible; no overall winner across families is selected.

Development uses20 independent expert datasets1000–1019; unseen uses40 datasets2000–2039. Each has100 rounds ×200 agents ×1000 steps. Compact observations and registries are supplied under `data/`; no path tensors or full training histories are included. The same generator can recreate them from seeds in a separate workspace. Each dataset is reused across configurations;600 development fits are not600 independent expert datasets.

Within each family, admissibility requires all20 valid K1000 fits. Selection sequentially maximizes success count, retains MAE within1e-4 of its surviving minimum, then RMSE within1e-4, then MaxAE within1e-4, minimizes projections, minimizes median final100-update sample SD, then chooses smaller eta. Objective loss is not a selection tie-break. See [configuration](docs/configuration.md).

Unseen widths are2.58×sample SD (ddof=1), not confidence intervals for the mean. Every five datasets, relative width changes are computed for all ten families; Delta is their maximum. Earliest stop40 requires both Delta(35) and Delta(40)<=.05. The observed values are .04505515704763263 and .029786041500306042, yielding **M_stop=40**. Fixed40 and stopped40 summaries coincide. Stopping does not use recovery truth, and width stability does not establish recovery accuracy. The executable frozen workflow is limited to IDs2000–2039.

Independent calibration screening and confirmation established the eta candidate grids and common K before formal development selection. The reduced project intentionally excludes historical parameter-search runners/results; **complete calibration-sweep reproduction is outside this repository’s reproduction scope**.

## Install and inspect

Use Python3.12 (validated3.12.2); dependencies pin NumPy2.5.4, SciPy1.18.1, pytest9.1.1 and Matplotlib3.10.8. A non-editable environment may live outside this repository:

```sh
python3.12 -m venv ../lq-env
../lq-env/bin/python -m pip install '.[dev,plots]'
../lq-env/bin/python -m pytest -q -rs
../lq-env/bin/python scripts/report_development.py
../lq-env/bin/python scripts/report_unseen.py
```

These reporting commands recreate tables/figures from the supplied compact estimates and never train. Only the optional full-history integration test skips when the deliberately external400,000-update history is absent. [Reproduction guide](docs/reproduce.md) covers fresh generation, retained-data training and expensive full reproduction:600+400 trajectories,1,000,000 updates and approximately8.192 trillion primary BR Euler trajectory-steps.
