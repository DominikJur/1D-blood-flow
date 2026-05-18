# Solver (solver.py)

## Purpose
Simulate 1D blood flow in a Y-shaped bifurcation using a finite-volume scheme and export
results for PINN training and visualization.

## Model Summary
- Variables: cross-sectional area A, flow rate Q, pressure P.
- Tube law (linear): P = beta * (A - A0).
- Wave speed: c = sqrt(beta * A / rho).
- Riemann invariants: W1 = u + 2c, W2 = u - 2c, u = Q / A.

## Governing Equations (A, Q form)
- Continuity: dA/dt + dQ/dz = 0
- Momentum: dQ/dt + d(Q^2/A)/dz + A/rho * dP/dz = -KR * Q / A

## Numerical Scheme
- Spatial: MUSCL reconstruction with minmod limiter.
- Flux: Rusanov (Lax-Friedrichs).
- Time: explicit Adams-Bashforth 2nd order.

## Boundary Conditions
- Inlet (parent, z=0): prescribed Q_in(t), solve for A using W2 extrapolation.
- Outlet (daughters, z=1): non-reflective boundary with reflection coefficient Rt.
- Junction: Newton-Raphson solve of 6 equations:
  - Mass conservation.
  - Total pressure continuity (parent vs daughters).
  - Riemann invariant matching (W1 for parent, W2 for daughters).

## Inputs and Outputs
- Inputs: vessel lengths, radii, c0, time horizon, CFL; defined in the file.
- Outputs:
  - blood_flow_data.csv with columns: t, vessel, z, A, Q.
  - blood_flow.gif visualization (requires PyVista).

## Running
- Run the script directly: it performs the simulation and writes outputs.
- If you change the tube law or parameters, regenerate blood_flow_data.csv before PINN training.

## Notes
- The solver uses the linear tube law from the paper. If you switch to another model,
  update pressure, wave speed, flux, and Riemann invariants consistently.
- PyVista is required for the GIF; you can comment out the visualization section
  if you only need CSV output.
