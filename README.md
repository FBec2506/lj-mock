# 1D Lennard-Jones molecular dynamics for argon (JAX)

This repository acts as a code example from me, Felix Beckmann. It is strongly simplified and should be regarded as a finger exercise.

A small, readable MD code: identical uncharged particles on a periodic line, interacting through the 12-6 Lennard-Jones potential with argon parameters, integrated with velocity Verlet in an NVE ensemble. Forces come from automatic differentiation of the energy function and are cross-checked against the analytic force. The demo notebook compares each result with a theoretical expectation (equilibrium distance, oscillation frequency, energy drift scaling, pair distribution).

## Files
| File | Content                                                             |
|---|---------------------------------------------------------------------|
| `lj.py` | energy, forces, integrator, `run`, observables |
| `test_lj.py` | tests (`pytest`)                                                    |
| `demo.ipynb` | Explanation and execution of the code                               |
| `PLAN.md` | design notes from the implementation process                        |

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab demo.ipynb
```

## Scope and simplifications
These simplifications were made to keep this mock example at a reasonable size
- 1D only
- single particle type
- no bonds or charges, so the force field is a single pair potential
- minimum-image periodic boundaries only (no cutoff, tail correction or sum over images)
- O(N^2) pair loop instead of neighbour list
- NVE only (no thermostat or barostat)
- zero-velocity start for interpretability
