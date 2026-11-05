"""Minimal 1D Lennard-Jones molecular dynamics in JAX (argon).

The system is N identical, uncharged particles on a periodic line of length
``box`` interacting through the 12-6 Lennard-Jones potential,

    U(r) = 4 * eps * [(sigma / r)**12 - (sigma / r)**6].

Dynamics are NVE (velocity Verlet). Forces come from automatic differentiation
of a single energy function, cross-checked against the analytic force.

Units
-----
The code works in an "Angstrom - Kelvin - atomic mass unit" system:

* length in Angstrom
* energy in Kelvin (i.e. energy / k_B, so k_B = 1 and T equals k_B * T)
* mass in u

Time is measured in the unit that follows from these three, with no
conversion provided. All physical constants are passed around in a plain
``params`` dict (see ``ARGON``) so that no function hard-codes sigma, epsilon
or the mass.

Functions marked ``TODO(you)`` are intentionally left unimplemented.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

# Float32 noise would dominate energy-drift measurements, so use float64.
jax.config.update("jax_enable_x64", True)

Array = jax.Array
Params = dict[str, float]

# --------------------------------------------------------------------------
# Parameters and units
# --------------------------------------------------------------------------

ARGON: Params = {
    "sigma": 3.405,  # Angstrom, LJ length scale
    "epsilon": 119.8,  # Kelvin (epsilon / k_B), LJ energy scale
    "mass": 39.948,  # u
}


# --------------------------------------------------------------------------
# State containers
# --------------------------------------------------------------------------


class State(NamedTuple):
    """Instantaneous state of the system.

    Attributes
    ----------
    pos : Array, shape (N,)
        Particle positions (not wrapped into the box).
    vel : Array, shape (N,)
        Particle velocities.
    force : Array, shape (N,)
        Forces at ``pos``. Carried along so that each Verlet step needs only
        one new force evaluation.
    """

    pos: Array
    vel: Array
    force: Array


class Trajectory(NamedTuple):
    """Time series recorded by :func:`run`, one entry per step.

    Attributes
    ----------
    pos : Array, shape (n_steps, N)
    kinetic, potential, total : Array, shape (n_steps,)
    """

    pos: Array
    kinetic: Array
    potential: Array
    total: Array


# --------------------------------------------------------------------------
# Core physics (TODO(you))
# --------------------------------------------------------------------------


def minimum_image(d: Array, box: float) -> Array:
    """Map displacements onto the nearest periodic image, in [-box/2, box/2]."""
    return d - box * jnp.round(d / box)


def pair_energy(r: Array, params: Params) -> Array:
    """Lennard-Jones energy of a pair at separation ``r``.

    Parameters
    ----------
    r : Array
        Pair separation(s), r > 0.
    params : dict
        Must contain ``sigma`` and ``epsilon``.

    Returns
    -------
    Array
        4 * eps * [(sigma/r)**12 - (sigma/r)**6], same shape as ``r``.
    """
    # TODO(you)
    raise NotImplementedError


def total_energy(pos: Array, box: float, params: Params) -> Array:
    """Total potential energy of all particle pairs (minimum image).

    Sum ``pair_energy`` over all unique pairs i < j, using
    :func:`minimum_image` for the separations.

    Hints: ``jnp.triu_indices(N, k=1)`` gives the pairs without the diagonal,
    so no r = 0 term (which would give NaN gradients) appears.

    Parameters
    ----------
    pos : Array, shape (N,)
    box : float
        Length of the periodic box.
    params : dict

    Returns
    -------
    Array
        Scalar potential energy.
    """
    # TODO(you)
    raise NotImplementedError


def forces(pos: Array, box: float, params: Params) -> Array:
    """Forces on all particles, F = -dU/dx, via automatic differentiation.

    Returns
    -------
    Array, shape (N,)
    """
    # TODO(you): one line with jax.grad
    raise NotImplementedError


def lj_force_analytic(r: Array, params: Params) -> Array:
    """Analytic LJ pair force along the separation, F(r) = -dU/dr.

    Positive values are repulsive. Used as a reference for :func:`forces`.

    Parameters
    ----------
    r : Array
        Pair separation(s), r > 0.
    params : dict

    Returns
    -------
    Array
        24 * eps / r * [2 (sigma/r)**12 - (sigma/r)**6].
    """
    # TODO(you)
    raise NotImplementedError


def velocity_verlet_step(state: State, box: float, dt: float, params: Params) -> State:
    """Advance the system by one velocity-Verlet step.

    1. v(t + dt/2) = v(t) + dt/2 * F(t) / m
    2. x(t + dt)   = x(t) + dt * v(t + dt/2)
    3. F(t + dt)   = forces(x(t + dt))
    4. v(t + dt)   = v(t + dt/2) + dt/2 * F(t + dt) / m

    Parameters
    ----------
    state : State
        Current positions, velocities and forces F(t).
    box : float
    dt : float
        Time step in the time unit of the unit system (see module docstring).
    params : dict

    Returns
    -------
    State
        State at t + dt, including the new forces.
    """
    # TODO(you)
    raise NotImplementedError


def run(
    state: State, box: float, dt: float, n_steps: int, params: Params
) -> tuple[State, Trajectory]:
    """Integrate the equations of motion for ``n_steps`` steps (NVE).

    Suggested steps:

    1. Define a scan body ``step(state, _)`` that calls
       :func:`velocity_verlet_step` and returns ``(new_state, record)``, with
       the record holding positions and the kinetic, potential and total
       energies (:func:`kinetic_energy`, :func:`total_energy`).
    2. Call ``jax.lax.scan(step, state, None, length=n_steps)``.
    3. Return the final state and a :class:`Trajectory`.
    4. Compile with ``jax.jit(run, static_argnames=("n_steps",))``.

    Positions stay unwrapped during the run; wrap them only for plotting or
    histograms (``jnp.mod(pos, box)``).

    Parameters
    ----------
    state : State
        Initial state, e.g. from :func:`init_state`.
    box : float
    dt : float
    n_steps : int
    params : dict

    Returns
    -------
    final_state : State
    trajectory : Trajectory
    """
    # TODO(you)
    raise NotImplementedError


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def kinetic_energy(vel: Array, params: Params) -> Array:
    """Total kinetic energy, 0.5 * m * sum(v**2)."""
    return 0.5 * params["mass"] * jnp.sum(vel**2)


def temperature(vel: Array, params: Params) -> Array:
    """Instantaneous temperature in Kelvin from the kinetic energy.

    Uses N - 1 degrees of freedom because the total momentum is conserved
    (centre of mass fixed). Energies are in Kelvin, so k_B = 1.
    """
    n = vel.shape[0]
    return 2.0 * kinetic_energy(vel, params) / (n - 1)


def init_lattice(n: int, box: float, jitter: float, key: Array) -> Array:
    """Equally spaced positions with a random displacement.

    A perfect lattice with zero velocities never moves, because all forces
    cancel by symmetry. The displacement stores potential energy that is
    converted into motion, which fixes the total energy of the run (the
    temperature is measured afterwards, not chosen).

    Parameters
    ----------
    n : int
        Number of particles.
    box : float
        Box length; the spacing is ``box / n``.
    jitter : float
        Maximum displacement as a fraction of the spacing. Must be below 0.5,
        otherwise neighbouring particles can coincide (r = 0, NaN forces).
    key : jax.Array
        PRNG key (``jax.random.PRNGKey(seed)``).

    Returns
    -------
    Array, shape (n,)
    """
    spacing = box / n
    base = spacing * jnp.arange(n)
    shift = jax.random.uniform(key, (n,), minval=-jitter, maxval=jitter)
    return base + spacing * shift


def init_state(pos: Array, box: float, params: Params) -> State:
    """Build a :class:`State` with zero velocities and the forces at ``pos``."""
    return State(pos=pos, vel=jnp.zeros_like(pos), force=forces(pos, box, params))


@jax.jit(static_argnames=("max_iter",))
def _minimize_loop(
    pos: Array, box: float, params: Params, max_iter: int, tol: float, alpha: float, cap: float
) -> tuple[Array, Array]:
    def cond(carry):
        _, i, fmax = carry
        return (i < max_iter) & (fmax > tol)

    def body(carry):
        x, i, _ = carry
        f = forces(x, box, params)
        dx = jnp.clip(alpha * f, -cap, cap)
        return x + dx, i + 1, jnp.max(jnp.abs(f))

    x, n_iter, _ = jax.lax.while_loop(cond, body, (pos, 0, jnp.inf))
    return x, n_iter


def minimize_energy(
    pos: Array,
    box: float,
    params: Params,
    tol: float = 1e-8,
    max_iter: int = 20_000,
    step: float = 0.005,
    max_disp: float = 0.1,
) -> tuple[Array, Array]:
    """Energy minimisation by steepest descent.

    Repeats ``x <- x + step * F(x)`` until the largest force component is
    below ``tol``. This is a first-order downhill walk with no momentum,
    unlike the dynamics of :func:`run`. It finds the nearest *local* minimum.

    Parameters
    ----------
    pos : Array, shape (N,)
    box : float
    params : dict
    tol : float
        Convergence threshold on max|F|, in units of epsilon / sigma.
    max_iter : int
    step : float
        Step size alpha, in units of sigma**2 / epsilon. Must stay below
        about 2 / (largest force constant); 0.005 is safe for chains.
    max_disp : float
        Cap on the displacement per step, in units of sigma. Protects against
        overshooting into the r**-12 wall.

    Returns
    -------
    pos : Array, shape (N,)
        Relaxed positions.
    n_iter : Array
        Number of iterations used.
    """
    sigma, eps = params["sigma"], params["epsilon"]
    return _minimize_loop(
        pos, box, params, max_iter, tol * eps / sigma, step * sigma**2 / eps, max_disp * sigma
    )


# --------------------------------------------------------------------------
# Observables and theory
# --------------------------------------------------------------------------


def g_of_r(
    pos_traj: Array, box: float, n_bins: int = 100, r_max: float | None = None
) -> tuple[Array, Array]:
    """Radial distribution function g(r) of a 1D periodic system.

    Normalised so that g -> 1 for an ideal gas. For N particles, an
    unordered pair separated by |d| in a shell of width dr is expected
    N (N - 1) dr / box times per frame in an ideal gas.

    Parameters
    ----------
    pos_traj : Array, shape (n_frames, N)
    box : float
    n_bins : int
    r_max : float, optional
        Upper limit, at most box / 2 (default).

    Returns
    -------
    r : Array, shape (n_bins,)
        Bin centres.
    g : Array, shape (n_bins,)
    """
    n_frames, n = pos_traj.shape
    r_max = box / 2 if r_max is None else r_max
    i, j = jnp.triu_indices(n, k=1)
    d = jnp.abs(minimum_image(pos_traj[:, i] - pos_traj[:, j], box))
    counts, edges = jnp.histogram(d.ravel(), bins=n_bins, range=(0.0, r_max))
    dr = edges[1] - edges[0]
    r = 0.5 * (edges[1:] + edges[:-1])
    g = counts * box / (n_frames * n * (n - 1) * dr)
    return r, g


def predicted_density(r: Array, energy: float, params: Params) -> Array:
    """Predicted time-density P(r) of a bound pair with relative energy ``energy``.

    For two particles with the centre of mass at rest, the relative motion
    has reduced mass mu = m / 2 and

        P(r) ~ 1 / |dr/dt| = 1 / sqrt(2 * (E - U(r)) / mu)

    between the turning points (zero outside). Normalised on the grid ``r``.
    The turning points are integrable singularities, so compare against a
    histogram with moderately coarse bins.

    Parameters
    ----------
    r : Array
        Fine, increasing grid of separations.
    energy : float
        Total energy of the pair (kinetic + potential).
    params : dict

    Returns
    -------
    Array
        Probability density with ``trapezoid(P, r) = 1``.
    """
    mu = params["mass"] / 2.0
    kin = energy - pair_energy(r, params)
    inside = kin > 0
    p = jnp.where(inside, 1.0 / jnp.sqrt(2.0 * jnp.where(inside, kin, 1.0) / mu), 0.0)
    return p / jnp.trapezoid(p, r)


def boltzmann_gr(r: Array, temp: float, params: Params) -> Array:
    """Low-density limit of the radial distribution function, exp(-U(r) / T).

    Energies are in Kelvin, so k_B = 1 and ``temp`` is in Kelvin.
    """
    return jnp.exp(-pair_energy(r, params) / temp)
