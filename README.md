# Reduced JAX PF-CP Core for Review-Stage Transparency

This directory contains a reduced, documentation-oriented subset of a JAX-based phase-field/crystal-plasticity implementation prepared for review-stage transparency.

The purpose of this package is to expose a small core of the numerical structure behind the single-crystal PF-CP model without releasing the full research workflow during peer review.

## What is included

- `reduced_pfcp_core.py`
  - rotation utilities
  - material and thermodynamic parameter helpers
  - smooth interpolation / barrier functions
  - isotropic and anisotropic Laplacian operators
  - a small softmax helper used in variant-competition logic
- `requirements.txt`

## What is intentionally not included

- full production solver
- full job-control and monitoring workflow
- data-processing scripts used for all figures
- remote submission helpers
- full polycrystal workflow
- complete benchmark inputs and output datasets

## Review-stage scope

This release is intentionally incomplete. It is meant to document a limited core of the model structure rather than reproduce the entire computational campaign. The complete codebase will be curated and released after manuscript acceptance.

## Suggested GitHub upload steps

If you want to publish this directory as a standalone repository after logging in on this machine:

```bash
cd /Users/fudaixin/SynologyDrive/Program/Zircaloy/Simulation/KKShydride/jax/github_partial_release
git init
git add .
git commit -m "Add reduced JAX PF-CP core for review-stage release"
gh auth login
gh repo create <repo-name> --public --source=. --remote=origin --push
```

If you prefer to keep the repository anonymous during review, create a neutral repository name and avoid profile information that directly identifies the authors.
