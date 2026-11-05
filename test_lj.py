"""Tests for lj.py.

Tests of the TODO(you) functions fail with NotImplementedError until those
functions are written.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lj

P = lj.ARGON
SIGMA, EPS = P["sigma"], P["epsilon"]
R_MIN = 2 ** (1 / 6) * SIGMA
BOX = 20 * SIGMA  # large enough that a single pair never sees its image


def pair(r):
    """Two particles at x = 0 and x = r."""
    return jnp.array([0.0, r])


# --- potential ------------------------------------------------------------


def test_energy_zero_at_sigma():
    assert lj.pair_energy(SIGMA, P) == pytest.approx(0.0, abs=1e-12)


def test_energy_minimum_is_minus_epsilon():
    assert lj.pair_energy(R_MIN, P) == pytest.approx(-EPS, rel=1e-12)


def test_potential_energy_matches_pair_energy_for_two_particles():
    r = 1.3 * SIGMA
    assert lj.potential_energy(pair(r), BOX, P) == pytest.approx(lj.pair_energy(r, P), rel=1e-12)


def test_translation_invariance_under_box_shift():
    pos = jnp.array([0.0, 1.3, 2.9, 4.4]) * SIGMA
    box = 6 * SIGMA
    shifted = pos.at[1].add(box)  # one particle moved by a full box length
    assert float(lj.potential_energy(shifted, box, P)) == pytest.approx(
        float(lj.potential_energy(pos, box, P)), rel=1e-10
    )


# --- forces ---------------------------------------------------------------


def test_analytic_force_zero_at_minimum():
    assert lj.lj_force_analytic(R_MIN, P) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("r_over_sigma", [0.95, 1.0, 1.12, 1.5, 2.0, 3.0])
def test_autodiff_force_matches_analytic(r_over_sigma):
    r = r_over_sigma * SIGMA
    f = lj.forces(pair(r), BOX, P)
    # particle 1 sits at +r, so a repulsive pair force pushes it to +x
    assert f[1] == pytest.approx(float(lj.lj_force_analytic(r, P)), rel=1e-10)
    assert float(f[0]) == pytest.approx(float(-f[1]), rel=1e-10)


def test_total_force_vanishes():
    pos = jnp.array([0.1, 1.2, 2.9, 4.0, 5.3]) * SIGMA
    assert jnp.sum(lj.forces(pos, 7 * SIGMA, P)) == pytest.approx(0.0, abs=1e-8)


# --- dynamics -------------------------------------------------------------


def test_verlet_step_keeps_symmetric_pair_symmetric():
    state = lj.init_state(pair(1.3 * SIGMA), BOX, P)
    new = lj.velocity_verlet_step(state, BOX, 0.002, P)
    assert float(new.vel[0]) == pytest.approx(float(-new.vel[1]), rel=1e-10)


def test_run_conserves_energy_for_bound_pair():
    state = lj.init_state(pair(1.3 * SIGMA), BOX, P)  # E = U(1.3 sigma) < 0
    _, traj = lj.run(state, BOX, 0.002, 5000, P)
    drift = jnp.max(jnp.abs(traj.total - traj.total[0])) / abs(traj.total[0])
    assert drift < 1e-3
    assert traj.pos.shape == (5000, 2)


# --- observables (independent of the TODO functions) ----------------------


def test_g_of_r_is_one_for_uniform_random_positions():
    key = jax.random.PRNGKey(0)
    box = 30.0
    pos = jax.random.uniform(key, (4000, 20), minval=0.0, maxval=box)
    _, g = lj.g_of_r(pos, box, n_bins=15)
    np.testing.assert_allclose(np.asarray(g), 1.0, atol=0.05)


def test_init_lattice_is_reproducible_and_inside_jitter():
    key = jax.random.PRNGKey(1)
    a = lj.init_lattice(10, 20.0, 0.1, key)
    b = lj.init_lattice(10, 20.0, 0.1, key)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert jnp.max(jnp.abs(a - 2.0 * jnp.arange(10))) <= 0.1 * 2.0 + 1e-12
