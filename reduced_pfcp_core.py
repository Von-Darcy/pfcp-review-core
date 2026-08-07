"""Reduced JAX PF-CP core accompanying the accepted manuscript.

This file exposes a small subset of the local solver utilities:
- orientation / tensor rotation helpers
- alpha-Zr slip-system and rate-dependent plasticity helpers
- material and thermodynamic parameter helpers
- smooth phase-field interpolation functions
- finite-difference Laplacian operators
- a simple softmax utility used in variant competition

It is intentionally not the complete production solver.
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


def get_rotated_schmid_tensors(rotation_matrix):
    """Return the three prism, three basal, and six pyramidal Schmid tensors."""
    sqrt3_2 = np.sqrt(3.0) / 2.0
    half = 0.5

    slip_directions = [
        np.array([1.0, 0.0, 0.0]),
        np.array([half, sqrt3_2, 0.0]),
        np.array([-half, sqrt3_2, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([half, sqrt3_2, 0.0]),
        np.array([-half, sqrt3_2, 0.0]),
    ]
    plane_normals = [
        np.array([0.0, 1.0, 0.0]),
        np.array([-sqrt3_2, half, 0.0]),
        np.array([sqrt3_2, half, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    sin_a, cos_a = 0.8746, 0.4848
    for theta in np.deg2rad([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]):
        plane_normals.append(
            np.array([sin_a * np.cos(theta), sin_a * np.sin(theta), cos_a])
        )
        slip_directions.append(
            np.array([cos_a * np.cos(theta), cos_a * np.sin(theta), -sin_a])
        )

    directions = (rotation_matrix @ np.vstack(slip_directions).T).T
    normals = (rotation_matrix @ np.vstack(plane_normals).T).T
    tensors = np.zeros((12, 6))
    for index, (direction, normal) in enumerate(zip(directions, normals)):
        symmetric_tensor = 0.5 * (
            np.outer(direction, normal) + np.outer(normal, direction)
        )
        tensors[index] = strain_tensor_to_voigt(symmetric_tensor)
    return tensors


def resolved_shear_stress(stress_voigt, schmid_tensors):
    """Resolve a Voigt stress field onto the supplied slip systems."""
    return jnp.einsum(
        "i...,ki->k...", jnp.asarray(stress_voigt), jnp.asarray(schmid_tensors)
    )


def phase_weighted_crss(
    hydride_fraction,
    crss_prism=100.0e6,
    crss_basal=150.0e6,
    crss_pyramidal=250.0e6,
    crss_hydride=400.0e6,
):
    """Interpolate the baseline slip resistance across the diffuse interface."""
    hydride_fraction = jnp.asarray(hydride_fraction)
    matrix_crss = jnp.asarray(
        [crss_prism] * 3 + [crss_basal] * 3 + [crss_pyramidal] * 6
    )
    reshape = (12,) + (1,) * hydride_fraction.ndim
    matrix_crss = matrix_crss.reshape(reshape)
    return (
        matrix_crss * (1.0 - hydride_fraction)[None, ...]
        + crss_hydride * hydride_fraction[None, ...]
    )


def power_law_slip_increment(
    resolved_shear,
    current_crss,
    dt,
    reference_slip_rate=1.0e5,
    rate_sensitivity=0.05,
    shear_modulus=36.3e9,
):
    """Compute the capped rate-dependent slip increment used by the model."""
    resolved_shear = jnp.asarray(resolved_shear)
    current_crss = jnp.asarray(current_crss)
    overstress = jnp.maximum(jnp.abs(resolved_shear) - current_crss, 0.0)
    maximum_increment = overstress / (shear_modulus + 1.0e-6)
    stress_ratio = jnp.clip(
        jnp.abs(resolved_shear) / (current_crss + 1.0e-6), 0.0, 2.0
    )
    slip_rate = (
        reference_slip_rate
        * jnp.power(stress_ratio, 1.0 / rate_sensitivity)
        * jnp.sign(resolved_shear)
    )
    return jnp.sign(resolved_shear) * jnp.minimum(
        jnp.abs(slip_rate * dt), maximum_increment
    )


def equivalent_strain_voigt(strain_voigt):
    """Return the von-Mises equivalent strain for engineering-shear Voigt data."""
    exx, eyy, ezz, eyz, exz, exy = strain_voigt
    mean = (exx + eyy + ezz) / 3.0
    deviatoric_square = (
        (exx - mean) ** 2
        + (eyy - mean) ** 2
        + (ezz - mean) ** 2
        + 0.5 * (eyz**2 + exz**2 + exy**2)
    )
    return jnp.sqrt(jnp.maximum((2.0 / 3.0) * deviatoric_square, 0.0))


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


def get_material_constants(shear_scale=1.0):
    base_eps = np.diag([0.0477, 0.0477, 0.0745])
    base_eps[1, 2] = base_eps[2, 1] = 0.09715 * shear_scale

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
    D_alpha = 2.06e-10 * speed_up
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
    dxy = (
        jnp.roll(eta, (-1, -1), (0, 1))
        + jnp.roll(eta, (1, 1), (0, 1))
        - jnp.roll(eta, (-1, 1), (0, 1))
        - jnp.roll(eta, (1, -1), (0, 1))
    ) / (4.0 * dx * dx)
    dxz = (
        jnp.roll(eta, (-1, -1), (0, 2))
        + jnp.roll(eta, (1, 1), (0, 2))
        - jnp.roll(eta, (-1, 1), (0, 2))
        - jnp.roll(eta, (1, -1), (0, 2))
    ) / (4.0 * dx * dx)
    dyz = (
        jnp.roll(eta, (-1, -1), (1, 2))
        + jnp.roll(eta, (1, 1), (1, 2))
        - jnp.roll(eta, (-1, 1), (1, 2))
        - jnp.roll(eta, (1, -1), (1, 2))
    ) / (4.0 * dx * dx)
    return (
        kappa[0, 0] * dxx
        + kappa[1, 1] * dyy
        + kappa[2, 2] * dzz
        + 2.0 * kappa[0, 1] * dxy
        + 2.0 * kappa[0, 2] * dxz
        + 2.0 * kappa[1, 2] * dyz
    )


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
    "equivalent_strain_voigt",
    "get_material_constants",
    "get_rotation_matrix",
    "get_rotated_schmid_tensors",
    "get_thermo_params",
    "h_func",
    "rotate_stiffness_matrix",
    "rotation_about_z",
    "scalar_laplacian",
    "softmax_axis0",
    "strain_tensor_to_voigt",
    "phase_weighted_crss",
    "power_law_slip_increment",
    "resolved_shear_stress",
]
