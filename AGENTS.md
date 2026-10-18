# Scientific maintenance

Preserve the accepted model, known initial law, inverse information boundary, solver tolerances, sampled rollout and fixed projected optimizer. J uses exactly1000 left-Riemann terms; consistency/Q use trapezoidal quadrature. Generation truth never enters inverse training. Do not tune on development or unseen results or expose unused unseen IDs without explicit approval.

Keep official compact data/results in the repository and expensive training outputs in a separate workspace. Use ordinary tests, finite-value/solver checks, types and atomic writes. No checksum manifests or layered audit frameworks. Diagnose scientific changes before any commit or publication. Every task return must include concrete paths and actual validation results.
