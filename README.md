# Reduced JAX PF-CP Core

This repository provides a reduced, documentation-oriented implementation of
the JAX phase-field/crystal-plasticity (PF-CP) model used in the accepted
manuscript:

> *Plasticity-Enabled Growth and Variant Competition of Delta Hydrides in
> Single-Crystal Alpha-Zr*, Journal of Nuclear Materials,
> manuscript reference JNUMA-D-26-00757 (production reference NUMA 156967).

## Scope

`reduced_pfcp_core.py` contains representative numerical components of the
model:

- crystal-orientation and tensor-rotation utilities;
- the 12-system alpha-Zr slip geometry used in the model;
- resolved-shear-stress, phase-weighted CRSS, and rate-dependent slip helpers;
- material and KKS-type thermodynamic parameter helpers;
- phase-field interpolation functions and finite-difference operators; and
- the softmax operation used in the variant-competition implementation.

This compact release is intended to document the principal implementation
choices. It is not the complete production solver and does not include cluster
job control, figure-processing workflows, or the full simulation datasets.
Consequently, it is not a stand-alone reproduction package for every figure in
the article.

## Installation

Python 3.10 or later is recommended. For a CPU installation:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

JAX accelerator builds are platform-specific; consult the JAX installation
documentation when GPU support is required.

## Verification

Run the included lightweight checks after installing the dependencies:

```bash
python smoke_test.py
```

The script checks the orientation matrix, alpha-Zr Schmid tensors,
phase-weighted slip resistance, phase-field operators, and variant softmax.

## Repository contents

- `reduced_pfcp_core.py`: reduced model utilities
- `smoke_test.py`: lightweight numerical checks
- `requirements.txt`: Python dependencies
