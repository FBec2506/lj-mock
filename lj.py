"""Minimal 1D Lennard-Jones molecular dynamics in JAX.

The system is N identical, uncharged particles on a periodic line of length
``box`` interacting through the 12-6 Lennard-Jones potential,

    U(r) = 4 * eps * [(sigma / r)**12 - (sigma / r)**6].

Dynamics are NVE (velocity Verlet). Forces come from automatic differentiation
of a single energy function, cross-checked against the analytic force.

Units
-----
The code works in atomic units:

* length in Angstrom
* energy in Kelvin (i.e. energy / k_B, so k_B = 1 and T equals k_B * T)
* mass in u
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

Array = jax.Array
AtomParams = dict[str, float]

# --------------------------------------------------------------------------
# Parameters and units
# --------------------------------------------------------------------------

ARGON: AtomParams = {
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
# Core physics
# --------------------------------------------------------------------------


def minimum_image(d: Array, box: float) -> Array:
    """Map displacements onto the nearest periodic image, in [-box/2, box/2]."""
    return d - box * jnp.round(d / box)


def pair_energy(r: Array, atom_params: AtomParams) -> Array:
    """Lennard-Jones energy of a pair at separation ``r``.

    Parameters
    ----------
    r : Array
        Pair separation(s), r > 0.
    atom_params : dict
        Must contain ``sigma`` and ``epsilon``.

    Returns
    -------
    Array
        4 * eps * [(sigma/r)**12 - (sigma/r)**6], same shape as ``r``.
    """
    ratio_sigma_r_6 = (atom_params["sigma"]/r)**6 # original implementation, saves less resources in jax than in naive impelmentations
    return (4 * atom_params["epsilon"] * (ratio_sigma_r_6**2 - ratio_sigma_r_6))


def potential_energy(pos: Array, box: float, atom_params: AtomParams) -> Array:
    """Total potential energy of all particle pairs (minimum image).
    Parameters
    ----------
    pos : Array, shape (N,)
    box : float
        Length of the periodic box.
    atom_params : dict

    Returns
    -------
    Array
        Scalar potential energy.
    """
    N = pos.shape[0] # extract number of particles
    i, j = jnp.triu_indices(N, k=1) # get all pairwise indices except for self interaction (i,i)
    d = minimum_image(pos[i] - pos[j], box) # calculate distances with PBC
    return jnp.sum(pair_energy(jnp.abs(d), atom_params)) # sum all pairwise energy contributions


def forces(pos: Array, box: float, atom_params: AtomParams) -> Array:
    """Forces on all particles, F = -dU/dx, via automatic differentiation.

    Returns
    -------
    Array, shape (N,)
    """
    return -1 * jax.grad(potential_energy, argnums=0)(pos, box, atom_params)


def lj_force_analytic(r: Array, atom_params: AtomParams) -> Array:
    """Analytic LJ pair force along the separation, F(r) = -dU/dr.

    Parameters
    ----------
    r : Array
        Pair separation(s), r > 0.
    atom_params : dict

    Returns
    -------
    Array
        24 * eps / r * [2 (sigma/r)**12 - (sigma/r)**6].
    """
    ratio_sigma_r_6 = (atom_params["sigma"]/r)**6 # original implementation, saves less resources in jax than in naive impelmentations
    return 24 * atom_params["epsilon"]/r * (2*ratio_sigma_r_6**2 - ratio_sigma_r_6)


def velocity_verlet_step(initial_state: State, box: float, dt: float, atom_params: AtomParams) -> State:
    """Advance the system by one velocity-Verlet step(half step formulation).

    1. v(t + dt/2) = v(t) + dt/2 * F(t) / m
    2. x(t + dt)   = x(t) + dt * v(t + dt/2)
    3. F(t + dt)   = forces(x(t + dt))
    4. v(t + dt)   = v(t + dt/2) + dt/2 * F(t + dt) / m

    Parameters
    ----------
    initial_state : State
        Current positions, velocities and forces F(t).
    box : float
    dt : float
        Time step in the time unit of the unit system (see module docstring).
    atom_params : dict

    Returns
    -------
    State
        State at t + dt, including the new forces.
    """
    vel_half = initial_state.vel + dt/2 * initial_state.force/atom_params["mass"]
    pos = initial_state.pos + dt * vel_half
    force = forces(pos=pos, box=box, atom_params=atom_params)
    vel_full = vel_half + dt/2 * force/atom_params["mass"]

    return State(pos=pos, vel=vel_full, force=force)


@jax.jit(static_argnames=("n_steps",)) # allows compilation if n_steps is constant
def run(
    initial_state: State, box: float, dt: float, n_steps: int, atom_params: AtomParams
) -> tuple[State, Trajectory]:
    """Integrate the equations of motion for ``n_steps`` steps (NVE) in a jit-way.

    Parameters
    ----------
    initial_state : State
    box : float
    dt : float
    n_steps : int
    atom_params : dict

    Returns
    -------
    final_state : State
    trajectory : Trajectory
    """

    # scan calls step(carry, x) with exactly two arguments; box, dt and
    # atom_params are taken from the enclosing scope.
    def step(state: State, _) -> tuple[State, tuple[Array, Array, Array]]:
        new_state = velocity_verlet_step(initial_state=state, box=box, dt=dt, atom_params=atom_params) # refer to values from run's signature

        kinetic_energy_val = kinetic_energy(vel=new_state.vel, atom_params=atom_params)
        potential_energy_val = potential_energy(pos=new_state.pos, box=box, atom_params=atom_params)

        # (new carry, record for this step)
        return new_state, (new_state.pos, kinetic_energy_val, potential_energy_val)

    # scan stacks the records along a new first axis: pos (n_steps, N), energies (n_steps,)
    carry_state, (pos_vals, kinetic_energy_vals, potential_energy_vals) = jax.lax.scan(
        step, initial_state, None, length=n_steps
    )

    return carry_state, Trajectory(
        pos=pos_vals,
        kinetic=kinetic_energy_vals,
        potential=potential_energy_vals,
        total=kinetic_energy_vals + potential_energy_vals,
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def kinetic_energy(vel: Array, atom_params: AtomParams) -> Array:
    """Total kinetic energy, 0.5 * m * sum(v**2)."""
    return 0.5 * atom_params["mass"] * jnp.sum(vel**2)


def temperature(vel: Array, atom_params: AtomParams) -> Array:
    """Instantaneous temperature in Kelvin from the kinetic energy.

    Uses N - 1 degrees of freedom because the total momentum is conserved
    (centre of mass fixed). Energies are in Kelvin, so k_B = 1.
    """
    n = vel.shape[0]
    return 2.0 * kinetic_energy(vel, atom_params) / (n - 1) # def of instantaneous T


def init_lattice(n: int, box: float, jitter: float, key: Array) -> Array:
    """Equally spaced positions with a random displacement.

    A perfect lattice with zero velocities never moves, because all forces
    cancel by symmetry. The displacement stores potential energy that is
    converted into motion, which fixes the total energy of the run (the
    temperature is measured afterwards, not chosen).

    Parameters
    ----------
    n : int
        Number of particles per unit box.
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
    base = jnp.linspace(start=0, stop=box, num=n, endpoint=False)
    shift = jax.random.uniform(key, (n,), minval=-jitter, maxval=jitter)
    return base + box / n * shift


def init_state(pos: Array, box: float, atom_params: AtomParams) -> State:
    """Build a :class:`State` with zero velocities and the forces at ``pos``."""
    return State(pos=pos, vel=jnp.zeros_like(pos), force=forces(pos, box, atom_params))


# --------------------------------------------------------------------------
# Observables
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
    counts, edges = jnp.histogram(d.ravel(), bins=n_bins, range=(0.0, r_max)) # flatten distance array and produce histogram
    dr = edges[1] - edges[0]
    r = 0.5 * (edges[1:] + edges[:-1]) # get bin centers
    g = counts * box / (n_frames * n * (n - 1) * dr)
    return r, g
