# Implementation Alignment Report

This report summarizes required changes to align the solver and PINN with:
- [research_papers/2508.18484v1.pdf](research_papers/2508.18484v1.pdf)
- [research_papers/1711.10561v1.pdf](research_papers/1711.10561v1.pdf)

It compares the current implementations in:
- [solver.py](solver.py)
- [train_pinn.py](train_pinn.py)

## Paper Summary (Paraphrased)

### 2508.18484 (1D blood flow review)
- Governing equations in $(A, Q)$ form: continuity and momentum with friction.
- A constitutive tube law closes the system; the review presents a linear $P= P_{ext} + \beta (A - A_0)$, but its later flux term and characteristic relations are consistent with the common nonlinear $P \propto \sqrt{A}$ form.
- Characteristic (Riemann) invariants: $W_1 = u + 4c$, $W_2 = u - 4c$ with $c$ defined from the constitutive law.
- Non-reflective outlets use extrapolated $W_1$ and a reflected or reference $W_2$ with a reflection coefficient.
- Bifurcations enforce mass conservation and total pressure continuity, plus characteristic matching for outgoing/incoming waves.
- Numerical scheme uses MUSCL with Rusanov flux and AB2 time integration (for the solver, not a PINN).

### 1711.10561 (PINN methodology)
- PINNs minimize a composite loss: data misfit + PDE residual(s) + boundary/initial constraints.
- Automatic differentiation is used for spatial/temporal derivatives.
- Any training schedule is acceptable as long as the final optimization minimizes the full physics-informed loss.

## Current Alignment: solver.py vs 2508.18484

### Matches
- Governing equations (continuity + momentum) and viscous term in conservative form.
- MUSCL reconstruction and Rusanov flux.
- Adams-Bashforth 2nd order time stepping.
- Characteristic invariants and use in inlet/outlet boundary conditions.
- Bifurcation junction system: mass + total pressure continuity + characteristic extrapolation.

### Potential Mismatch / Ambiguity
- The paper text presents a linear tube law, but the solver uses a $\sqrt{A}$ tube law. This affects pressure, wave speed, Riemann invariants, and the junction system.
- The flux term in the paper uses $A^{3/2}$, which is consistent with the $\sqrt{A}$ tube law. This suggests the review mixes two tube-law forms in different sections.

**Decision Required**: Choose one tube-law model and apply it consistently across solver and PINN.

## Current Alignment: train_pinn.py vs 1711.10561 and 2508.18484

### Matches
- Composite loss includes data, PDE residuals, and boundary losses.
- Uses automatic differentiation for time and space derivatives.
- Boundary conditions enforce inlet flow, non-reflective outlets (via $W_2$), and bifurcation constraints.
- Uses normalized $z$ with explicit $1/L$ scaling in the PDE residual, which is consistent if all terms are scaled the same way.

### Required Changes (for consistency with solver + paper)

1. **Make the PINN tube law match the solver**
   - If the solver keeps $P \propto \sqrt{A}$, then:
     - Use the same $\beta$ definition as the solver.
     - Use $\partial P/\partial z = (\beta/(2\sqrt{A})) A_z$.
   - If you change the solver to a linear tube law, update:
     - Pressure, wave speed, flux term, Riemann invariants, and junction conditions in the solver.
     - The PINN’s $\partial P/\partial z$ and any $c(A)$ definition.

2. **Ensure the PDE residual uses vessel-consistent parameters**
   - With one-hot vessel IDs, all physical parameters used in the PDE residual must be derived from the same one-hot assignment.

3. **Ensure training optimizes the full PINN loss**
   - If you use L-BFGS or other second-stage optimizers, they must minimize the combined loss (data + PDE + BC), not PDE-only.

4. **Align evaluation with the trained network**
   - If training uses a custom PyTorch loop, evaluation and prediction should use the same network state (not a stale DeepXDE model state).

## Normalization Decision (z vs z/L)

- Keeping $z/L$: stable input scaling across vessels, but requires consistent $1/L$ factors in every PDE term and any BC formula that uses spatial derivatives.
- Using physical $z$: matches the paper equations directly; but input scales vary by vessel length and may need other normalization.

## Recommended Path

- **Option A (solver stays as-is)**: Treat the $\sqrt{A}$ tube law as authoritative (it matches solver + flux term in the paper).
  - Update PINN equations and parameters to match solver.
- **Option B (strict paper linear tube law)**: Change solver and PINN to linear tube law with corresponding changes to flux and characteristics.

## Appendix: Files to Update (If You Choose Each Option)

### Option A (keep solver tube law)
- [train_pinn.py](train_pinn.py): align $\beta$, $P(A)$, $\partial P/\partial z$, and ensure full-loss optimization.

### Option B (switch to linear tube law)
- [solver.py](solver.py): pressure function, wave speed, flux term, Riemann invariants, from_riemann, outlet and junction relations.
- [train_pinn.py](train_pinn.py): PDE residual terms, $P(A)$, $c(A)$, and boundary conditions referencing $W_1/W_2$.
