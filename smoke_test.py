"""Lightweight numerical checks for the reduced PF-CP utilities."""

import jax.numpy as jnp
import numpy as np

import reduced_pfcp_core as core


def main():
    rotation = core.get_rotation_matrix([15.0, 25.0, 35.0])
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1.0e-12)

    schmid = core.get_rotated_schmid_tensors(rotation)
    assert schmid.shape == (12, 6)
    resolved = core.resolved_shear_stress(jnp.ones(6), schmid)
    assert resolved.shape == (12,)

    hydride_fraction = jnp.asarray([0.0, 1.0])
    crss = core.phase_weighted_crss(hydride_fraction)
    assert crss.shape == (12, 2)
    np.testing.assert_allclose(np.asarray(crss[:3, 0]), 100.0e6)
    np.testing.assert_allclose(np.asarray(crss[:, 1]), 400.0e6)

    constant = jnp.ones((4, 4, 4))
    np.testing.assert_allclose(core.scalar_laplacian(constant, 1.0), 0.0)
    np.testing.assert_allclose(
        core.anisotropic_laplacian(constant, jnp.eye(3), 1.0), 0.0
    )

    probabilities = core.softmax_axis0(jnp.asarray([[1.0], [2.0], [3.0]]))
    np.testing.assert_allclose(np.asarray(probabilities.sum(axis=0)), 1.0)
    print("All reduced PF-CP checks passed.")


if __name__ == "__main__":
    main()

