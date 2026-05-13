"""Reduced JAX PF-CP core extracted for review-stage code sharing.

This file exposes a small subset of the local solver utilities:
- orientation / tensor rotation helpers
- material and thermodynamic parameter helpers
- smooth phase-field interpolation functions
- finite-difference Laplacian operators
- a simple softmax utility used in variant competition

It is intentionally not a full runnable solver.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
VOIGT_MAP = {
    (0, 0): 0,
    (1, 1): 1,
    (2, 2): 2,
    (1, 2): 3,
    (2, 1): 3,
    (0, 2): 4,
    (2, 0): 4,
    (0, 1): 5,
    (1, 0): 5,
}


def get_rotation_matrix(euler_deg):
    p1, P, p2 = [np.deg2rad(x) for x in euler_deg]
    c1, s1 = np.cos(p1), np.sin(p1)
    c, s = np.cos(P), np.sin(P)
    c2, s2 = np.cos(p2), np.sin(p2)
    return np.array(
        [
            [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
            [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
            [s1 * s, -c1 * s, c],
        ]
    )


def rotation_about_z(angle_deg):
    th = np.deg2rad(angle_deg)
    ct, st = np.cos(th), np.sin(th)
    return np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])


def strain_tensor_to_voigt(eps):
    return np.array(
        [
            eps[0, 0],
            eps[1, 1],
            eps[2, 2],
            2.0 * eps[1, 2],
            2.0 * eps[0, 2],
            2.0 * eps[0, 1],
        ]
    )


def rotate_stiffness_matrix(C_voigt, R):
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = C_voigt[VOIGT_MAP[(i, j)], VOIGT_MAP[(k, l)]]
    C_rot = np.einsum("im,jn,kp,lq,mnpq->ijkl", R, R, R, R, C)
    return np.array(
        [[C_rot[i, j, k, l] for (k, l) in VOIGT_PAIRS] for (i, j) in VOIGT_PAIRS]
    )


def get_material_constants():
    base_eps = np.diag([0.0477, 0.0477, 0.0745])
    base_eps[1, 2] = base_eps[2, 1] = 0.09715

    Cm = np.zeros((6, 6))
    Ch = np.zeros((6, 6))

    Cm[0, 0] = Cm[1, 1] = 155.4e9
    Cm[2, 2] = 172.5e9
    Cm[0, 1] = Cm[1, 0] = 67.2e9
    Cm[0, 2] = Cm[2, 0] = Cm[1, 2] = Cm[2, 1] = 64.6e9
    Cm[3, 3] = Cm[4, 4] = 36.3e9
    Cm[5, 5] = 44.1e9

    Ch[0, 0] = Ch[1, 1] = 2.04e11
    Ch[2, 2] = 2.18e11
    Ch[0, 1] = Ch[1, 0] = 1.0e11
    Ch[0, 2] = Ch[2, 0] = Ch[1, 2] = Ch[2, 1] = 8.6e10
    Ch[3, 3] = Ch[4, 4] = 3.8e10
    Ch[5, 5] = 5.2e10
    Ch[0, 3] = Ch[3, 0] = -1.98e10
    Ch[1, 3] = Ch[3, 1] = 1.98e10
    Ch[4, 5] = Ch[5, 4] = -1.98e10

    lmda_local = np.diag([0.0329, 0.0329, 0.0542])
    return base_eps, Cm, Ch, lmda_local


def get_thermo_params(T_sim, scale, speed_up):
    V_m = 1.4e-5
    a1 = 206.6 * np.exp(4429.0 / T_sim)
    b1 = 40.63 * T_sim - 89180.0
    c1 = -55.56 * T_sim + 6484.0
    a2 = -771.3 * T_sim + 830500.0
    b2 = 1021.0 * T_sim - 1109000.0
    c2 = -361.2 * T_sim + 321600.0

    lam_a = scale * 2.0 * a1 / V_m
    lam_d = scale * 2.0 * a2 / V_m
    c_v_a = -b1 / (2.0 * a1)
    c_v_d = -b2 / (2.0 * a2)
    f_0_a = scale * (c1 - b1 * b1 / (4.0 * a1)) / V_m
    f_0_d = scale * (c2 - b2 * b2 / (4.0 * a2)) / V_m
    D_alpha = 1.0e-10 * speed_up
    M_mob = D_alpha / lam_a
    return lam_a, lam_d, c_v_a, c_v_d, f_0_a, f_0_d, M_mob


def h_func(eta):
    return eta**3 * (10.0 - 15.0 * eta + 6.0 * eta**2)


def dh_func(eta):
    return 30.0 * eta**2 * (1.0 - eta) ** 2


def dg_func(eta):
    return 2.0 * eta * (1.0 - eta) * (1.0 - 2.0 * eta)


def scalar_laplacian(f, dx):
    return (
        jnp.roll(f, 1, 0)
        + jnp.roll(f, -1, 0)
        + jnp.roll(f, 1, 1)
        + jnp.roll(f, -1, 1)
        + jnp.roll(f, 1, 2)
        + jnp.roll(f, -1, 2)
        - 6.0 * f
    ) / (dx * dx)


def anisotropic_laplacian(eta, kappa, dx):
    eta_xp = jnp.roll(eta, -1, axis=0)
    eta_xm = jnp.roll(eta, 1, axis=0)
    eta_yp = jnp.roll(eta, -1, axis=1)
    eta_ym = jnp.roll(eta, 1, axis=1)
    eta_zp = jnp.roll(eta, -1, axis=2)
    eta_zm = jnp.roll(eta, 1, axis=2)

    dxx = (eta_xp - 2.0 * eta + eta_xm) / (dx * dx)
    dyy = (eta_yp - 2.0 * eta + eta_ym) / (dx * dx)
    dzz = (eta_zp - 2.0 * eta + eta_zm) / (dx * dx)
    return kappa[0, 0] * dxx + kappa[1, 1] * dyy + kappa[2, 2] * dzz


def softmax_axis0(logits):
    shifted = logits - jnp.max(logits, axis=0, keepdims=True)
    exp_vals = jnp.exp(shifted)
    return exp_vals / (jnp.sum(exp_vals, axis=0, keepdims=True) + 1.0e-12)


__all__ = [
    "VOIGT_MAP",
    "VOIGT_PAIRS",
    "anisotropic_laplacian",
    "dg_func",
    "dh_func",
    "get_material_constants",
    "get_rotation_matrix",
    "get_thermo_params",
    "h_func",
    "rotate_stiffness_matrix",
    "rotation_about_z",
    "scalar_laplacian",
    "softmax_axis0",
    "strain_tensor_to_voigt",
]
