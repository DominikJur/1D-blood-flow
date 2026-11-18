# Solver (scripts/solver.py + scripts/simulation_engine.py)

## Purpose
Simulate 1D blood flow in several arterial topologies (straight tube, Y-bifurcation,
split-merge, double-Y) with a finite-volume scheme, and export results for PINN training
and visualization.

## Model Summary
- Variables: cross-sectional area A, flow rate Q, pressure P.
- Tube law (nonlinear): P = beta * (sqrt(A) - sqrt(A0)).
- Wave speed: c = sqrt(beta / (2 * rho)) * A^(1/4).
- Riemann invariants: W1 = u + 4c, W2 = u - 4c, u = Q / A.

## Governing Equations (A, Q form)
- Continuity: dA/dt + dQ/dz = 0
- Momentum: dQ/dt + d(Q^2/A)/dz + A/rho * dP/dz = -KR * Q / A

## Numerical Scheme
- Spatial: MUSCL reconstruction with minmod limiter.
- Flux: Rusanov (Lax-Friedrichs).
- Time: explicit Adams-Bashforth 2nd order.
- CFL = 0.30 (scripts/topology_config.py). CFL = 0.45 is unstable with AB2 here:
  2*dx odd-even oscillations develop and A drops below zero; CFL <= 0.30 runs the full
  0.75 s cleanly.

## Boundary Conditions
- Inlet (parent, z=0): prescribed Q_in(t), solve for A using W2 extrapolation.
- Outlet (daughters, z=1): non-reflective boundary with reflection coefficient Rt.
- Junction: Newton-Raphson solve of 6 equations:
  - Mass conservation.
  - Total pressure continuity (parent vs daughters).
  - Riemann invariant matching (W1 for parent, W2 for daughters).

## Inputs and Outputs
- Inputs: per-topology vessel geometry, junctions, inflow and T_out; defined in
  scripts/topology_config.py (CFL lives there too).
- Outputs:
  - data/<topology>_blood_flow_data.csv with columns: t, vessel, z, A, Q
    (data/ is gitignored; regenerate with scripts/solver.py).
  - gifs/<topology>_blood_flow.gif 3D animation (requires PyVista, off-screen).

## Running
- `uv run python scripts/solver.py` runs all topologies and writes CSV + GIF per topology.
- If you change the tube law or parameters, regenerate data/*.csv before PINN training.

## Notes
- The solver uses a nonlinear tube law (P proportional to sqrt(A)). If you switch to another model,
  update pressure, wave speed, flux, and Riemann invariants consistently.
- PyVista is required for the GIF export in scripts/solver.py.
