"""
Core simulation engine for 1D blood flow with arbitrary topologies.
MUSCL + Adams-Bashforth 2nd order.
Based on: D. Kim, J. Tithof (2025) - sections 5, 6, 7
"""
import numpy as np
from topology_config import rho, KR, CFL, compute_vessel_params


class Vessel:
    def __init__(self, cfg):
        p = compute_vessel_params(cfg)
        self.L, self.r, self.N = p["L"], p["r"], p["N"]
        self.name = p["name"]
        self.A0, self.beta, self.dx = p["A0"], p["beta"], p["dx"]
        self.x = np.linspace(self.dx/2, self.L - self.dx/2, self.N)
        self.A = np.ones(self.N) * self.A0
        self.Q = np.zeros(self.N)
        self.A_hist, self.Q_hist = [], []
        self.Phi_prev = None


def pressure(A, A0, beta):
    return beta * (np.sqrt(A) - np.sqrt(A0))

def wave_speed(A, A0, beta):
    return np.sqrt(beta / (2.0 * rho)) * np.power(A, 0.25)

def total_pressure(A, Q, A0, beta):
    return pressure(A, A0, beta) + 0.5 * rho * (Q / A)**2

def flux_func(A, Q, A0, beta):
    return np.array([Q, Q**2 / A + beta / (3.0 * rho) * np.power(A, 1.5)])

def riemann_W1(A, Q, A0, beta):
    return Q / A + 4.0 * wave_speed(A, A0, beta)

def riemann_W2(A, Q, A0, beta):
    return Q / A - 4.0 * wave_speed(A, A0, beta)

def from_riemann(W1, W2, A0, beta):
    c = (W1 - W2) / 8.0
    A = np.power(2.0 * rho * c**2 / beta, 2.0)
    Q = A * (W1 + W2) / 2.0
    return A, Q


def muscl_reconstruction(vessel):
    A, Q, N, dx = vessel.A, vessel.Q, vessel.N, vessel.dx
    U = np.vstack((A, Q))
    U_pad = np.pad(U, ((0,0),(1,1)), mode='edge')
    dU_L = U_pad[:, 1:-1] - U_pad[:, 0:-2]
    dU_R = U_pad[:, 2:] - U_pad[:, 1:-1]
    DUi = np.zeros_like(U)
    mp = (dU_L > 0) & (dU_R > 0)
    mn = (dU_L < 0) & (dU_R < 0)
    DUi[mp] = np.minimum(dU_L[mp], dU_R[mp]) / dx
    DUi[mn] = np.maximum(dU_L[mn], dU_R[mn]) / dx
    UL = np.zeros((2, N+1)); UR = np.zeros((2, N+1))
    UL[:, 1:N] = U[:, :N-1] + 0.5*dx*DUi[:, :N-1]
    UR[:, 1:N] = U[:, 1:] - 0.5*dx*DUi[:, 1:]
    return UL, UR


def muscl_update(vessel, UL, UR):
    A0, beta, dx, N = vessel.A0, vessel.beta, vessel.dx, vessel.N
    FL = flux_func(UL[0], UL[1], A0, beta)
    FR = flux_func(UR[0], UR[1], A0, beta)
    cL = wave_speed(UL[0], A0, beta)
    cR = wave_speed(UR[0], A0, beta)
    S_max = np.maximum(np.abs(UL[1]/UL[0]) + cL, np.abs(UR[1]/UR[0]) + cR)
    F_interf = 0.5*(FL+FR) - 0.5*S_max*(UR-UL)
    S = np.zeros((2, N)); S[1,:] = -KR * vessel.Q / vessel.A
    return -(F_interf[:,1:] - F_interf[:,:-1]) / dx + S


def apply_inlet_BC(vessel, t, dt, UL, UR, inflow_func):
    A0, beta = vessel.A0, vessel.beta
    Q_in = inflow_func(t)
    W2_0 = riemann_W2(vessel.A[0], vessel.Q[0], A0, beta)
    W2_1 = riemann_W2(vessel.A[1], vessel.Q[1], A0, beta)
    u0 = vessel.Q[0] / vessel.A[0]
    c0 = wave_speed(vessel.A[0], A0, beta)
    W2_new = W2_0 + dt * (W2_1 - W2_0) / vessel.dx * (-(u0 - c0))
    A_g = max(vessel.A[0], A0)
    for _ in range(15):
        cA = wave_speed(A_g, A0, beta)
        F_val = Q_in / A_g - 4.0*cA - W2_new
        if abs(F_val) < 1e-14: break
        dF = -Q_in / A_g**2 - cA / A_g
        A_g = max(A_g - F_val/dF, A0*0.1)
    UL[:,0] = [A_g, Q_in]; UR[:,0] = [A_g, Q_in]
    return UL, UR


def apply_outlet_BC(vessel, dt, UL, UR, Rt=0.0):
    A0, beta, N = vessel.A0, vessel.beta, vessel.N
    W1_L = riemann_W1(vessel.A[-1], vessel.Q[-1], A0, beta)
    W1_Lm1 = riemann_W1(vessel.A[-2], vessel.Q[-2], A0, beta)
    uL = vessel.Q[-1] / vessel.A[-1]
    cL = wave_speed(vessel.A[-1], A0, beta)
    W1_new = W1_L + dt * (W1_Lm1 - W1_L) / vessel.dx * (uL + cL)
    c0 = wave_speed(A0, A0, beta)
    W2_new = (-4.0*c0) - Rt * (W1_new - 4.0*c0)
    A_out, Q_out = from_riemann(W1_new, W2_new, A0, beta)
    UL[:,N] = [A_out, Q_out]; UR[:,N] = [A_out, Q_out]
    return UL, UR


def solve_bifurcation(vp, vd1, vd2, dt, tol=1e-8, max_iter=50):
    """Solve Y-junction: 1 parent -> 2 daughters (Section 7, Kim & Tithof)."""
    A0_p, b_p = vp.A0, vp.beta
    A0_d1, b_d1 = vd1.A0, vd1.beta
    A0_d2, b_d2 = vd2.A0, vd2.beta

    # Extrapolate W1 from parent (outgoing)
    W1p = riemann_W1(vp.A[-1], vp.Q[-1], A0_p, b_p)
    W1pm = riemann_W1(vp.A[-2], vp.Q[-2], A0_p, b_p)
    uL = vp.Q[-1]/vp.A[-1]; cL = wave_speed(vp.A[-1], A0_p, b_p)
    W1_ext = W1p + dt*(W1pm-W1p)/vp.dx*(uL+cL)

    # Extrapolate W2 from daughters (incoming)
    def ext_W2(v, A0, beta):
        W2_0 = riemann_W2(v.A[0], v.Q[0], A0, beta)
        W2_1 = riemann_W2(v.A[1], v.Q[1], A0, beta)
        u0 = v.Q[0]/v.A[0]; c0 = wave_speed(v.A[0], A0, beta)
        return W2_0 + dt*(W2_1-W2_0)/v.dx*(-(u0-c0))

    W2_d1 = ext_W2(vd1, A0_d1, b_d1)
    W2_d2 = ext_W2(vd2, A0_d2, b_d2)

    x = np.array([vp.A[-1], vp.Q[-1], vd1.A[0], vd1.Q[0], vd2.A[0], vd2.Q[0]])
    c0_ref = wave_speed(A0_p, A0_p, b_p)
    scale = np.array([1/A0_p, 1/(rho*c0_ref**2), 1/(rho*c0_ref**2),
                      1/c0_ref, 1/c0_ref, 1/c0_ref])

    for _ in range(max_iter):
        Ap,Qp,Ad1,Qd1,Ad2,Qd2 = x
        cp = wave_speed(Ap, A0_p, b_p)
        cd1 = wave_speed(Ad1, A0_d1, b_d1)
        cd2 = wave_speed(Ad2, A0_d2, b_d2)
        Pp = total_pressure(Ap, Qp, A0_p, b_p)
        Pd1 = total_pressure(Ad1, Qd1, A0_d1, b_d1)
        Pd2 = total_pressure(Ad2, Qd2, A0_d2, b_d2)

        F = np.array([Qp-Qd1-Qd2, Pp-Pd1, Pp-Pd2,
                      Qp/Ap+4*cp-W1_ext, Qd1/Ad1-4*cd1-W2_d1, Qd2/Ad2-4*cd2-W2_d2])
        if np.max(np.abs(F*scale)) < tol: break

        def dPt_dA(A,Q,A0,b): return b/(2*np.sqrt(A)) - rho*Q**2/A**3
        def dPt_dQ(A,Q,A0,b): return rho*Q/A**2

        J = np.zeros((6,6))
        J[0,1]=1; J[0,3]=-1; J[0,5]=-1
        J[1,0]=dPt_dA(Ap,Qp,A0_p,b_p); J[1,1]=dPt_dQ(Ap,Qp,A0_p,b_p)
        J[1,2]=-dPt_dA(Ad1,Qd1,A0_d1,b_d1); J[1,3]=-dPt_dQ(Ad1,Qd1,A0_d1,b_d1)
        J[2,0]=dPt_dA(Ap,Qp,A0_p,b_p); J[2,1]=dPt_dQ(Ap,Qp,A0_p,b_p)
        J[2,4]=-dPt_dA(Ad2,Qd2,A0_d2,b_d2); J[2,5]=-dPt_dQ(Ad2,Qd2,A0_d2,b_d2)
        J[3,0]=-Qp/Ap**2+cp/Ap; J[3,1]=1/Ap
        J[4,2]=-Qd1/Ad1**2-cd1/Ad1; J[4,3]=1/Ad1
        J[5,4]=-Qd2/Ad2**2-cd2/Ad2; J[5,5]=1/Ad2

        Js = J * scale[:,np.newaxis]
        try: dx_nr = np.linalg.solve(Js, -F*scale)
        except: dx_nr = np.linalg.lstsq(Js, -F*scale, rcond=None)[0]

        mr = max(1.0, max(abs(dx_nr[i])/(0.2*x[i]) for i in [0,2,4]))
        x += dx_nr / mr
        x[0]=max(x[0],A0_p*0.1); x[2]=max(x[2],A0_d1*0.1); x[4]=max(x[4],A0_d2*0.1)
    return x


def solve_merge(vp1, vp2, vd, dt, tol=1e-8, max_iter=50):
    """Solve anastomosis: 2 parents -> 1 daughter (Section 7, Kim & Tithof)."""
    A0_p1, b_p1 = vp1.A0, vp1.beta
    A0_p2, b_p2 = vp2.A0, vp2.beta
    A0_d, b_d = vd.A0, vd.beta

    # W1 from both parents (outgoing, at end)
    def ext_W1(v, A0, beta):
        W1_L = riemann_W1(v.A[-1], v.Q[-1], A0, beta)
        W1_Lm = riemann_W1(v.A[-2], v.Q[-2], A0, beta)
        uL = v.Q[-1]/v.A[-1]; cL = wave_speed(v.A[-1], A0, beta)
        return W1_L + dt*(W1_Lm-W1_L)/v.dx*(uL+cL)

    W1_p1 = ext_W1(vp1, A0_p1, b_p1)
    W1_p2 = ext_W1(vp2, A0_p2, b_p2)

    # W2 from daughter (incoming, at start)
    W2_0 = riemann_W2(vd.A[0], vd.Q[0], A0_d, b_d)
    W2_1 = riemann_W2(vd.A[1], vd.Q[1], A0_d, b_d)
    u0 = vd.Q[0]/vd.A[0]; c0 = wave_speed(vd.A[0], A0_d, b_d)
    W2_d = W2_0 + dt*(W2_1-W2_0)/vd.dx*(-(u0-c0))

    x = np.array([vp1.A[-1], vp1.Q[-1], vp2.A[-1], vp2.Q[-1], vd.A[0], vd.Q[0]])
    c0_ref = wave_speed(A0_d, A0_d, b_d)
    scale = np.array([1/A0_p1, 1/(rho*c0_ref**2), 1/(rho*c0_ref**2),
                      1/c0_ref, 1/c0_ref, 1/c0_ref])

    for _ in range(max_iter):
        Ap1,Qp1,Ap2,Qp2,Ad,Qd = x
        cp1 = wave_speed(Ap1, A0_p1, b_p1)
        cp2 = wave_speed(Ap2, A0_p2, b_p2)
        cd = wave_speed(Ad, A0_d, b_d)
        Pp1 = total_pressure(Ap1, Qp1, A0_p1, b_p1)
        Pp2 = total_pressure(Ap2, Qp2, A0_p2, b_p2)
        Pd = total_pressure(Ad, Qd, A0_d, b_d)

        F = np.array([Qp1+Qp2-Qd, Pp1-Pd, Pp2-Pd,
                      Qp1/Ap1+4*cp1-W1_p1, Qp2/Ap2+4*cp2-W1_p2, Qd/Ad-4*cd-W2_d])
        if np.max(np.abs(F*scale)) < tol: break

        def dPt_dA(A,Q,A0,b): return b/(2*np.sqrt(A)) - rho*Q**2/A**3
        def dPt_dQ(A,Q,A0,b): return rho*Q/A**2

        J = np.zeros((6,6))
        J[0,1]=1; J[0,3]=1; J[0,5]=-1
        J[1,0]=dPt_dA(Ap1,Qp1,A0_p1,b_p1); J[1,1]=dPt_dQ(Ap1,Qp1,A0_p1,b_p1)
        J[1,4]=-dPt_dA(Ad,Qd,A0_d,b_d); J[1,5]=-dPt_dQ(Ad,Qd,A0_d,b_d)
        J[2,2]=dPt_dA(Ap2,Qp2,A0_p2,b_p2); J[2,3]=dPt_dQ(Ap2,Qp2,A0_p2,b_p2)
        J[2,4]=-dPt_dA(Ad,Qd,A0_d,b_d); J[2,5]=-dPt_dQ(Ad,Qd,A0_d,b_d)
        J[3,0]=-Qp1/Ap1**2+cp1/Ap1; J[3,1]=1/Ap1
        J[4,2]=-Qp2/Ap2**2+cp2/Ap2; J[4,3]=1/Ap2
        J[5,4]=-Qd/Ad**2-cd/Ad; J[5,5]=1/Ad

        Js = J * scale[:,np.newaxis]
        try: dx_nr = np.linalg.solve(Js, -F*scale)
        except: dx_nr = np.linalg.lstsq(Js, -F*scale, rcond=None)[0]

        mr = max(1.0, max(abs(dx_nr[i])/(0.2*x[i]) for i in [0,2,4]))
        x += dx_nr / mr
        x[0]=max(x[0],A0_p1*0.1); x[2]=max(x[2],A0_p2*0.1); x[4]=max(x[4],A0_d*0.1)
    return x


def solve_conjunction(vp, vd, dt, tol=1e-8, max_iter=50):
    """Solve single conjunction (turn): 1 parent -> 1 daughter (Section 7, Kim & Tithof).
    4 unknowns: (A_p, Q_p, A_d, Q_d)
    Equations:
      1. Mass conservation: Q_p - Q_d = 0
      2. Total pressure continuity: P_tot_p = P_tot_d
      3. Riemann W1 from parent (outgoing)
      4. Riemann W2 from daughter (incoming)
    """
    A0_p, b_p = vp.A0, vp.beta
    A0_d, b_d = vd.A0, vd.beta

    # Extrapolate W1 from parent (outgoing, at end)
    W1_L = riemann_W1(vp.A[-1], vp.Q[-1], A0_p, b_p)
    W1_Lm = riemann_W1(vp.A[-2], vp.Q[-2], A0_p, b_p)
    uL = vp.Q[-1]/vp.A[-1]; cL = wave_speed(vp.A[-1], A0_p, b_p)
    W1_ext = W1_L + dt*(W1_Lm-W1_L)/vp.dx*(uL+cL)

    # Extrapolate W2 from daughter (incoming, at start)
    W2_0 = riemann_W2(vd.A[0], vd.Q[0], A0_d, b_d)
    W2_1 = riemann_W2(vd.A[1], vd.Q[1], A0_d, b_d)
    u0 = vd.Q[0]/vd.A[0]; c0 = wave_speed(vd.A[0], A0_d, b_d)
    W2_ext = W2_0 + dt*(W2_1-W2_0)/vd.dx*(-(u0-c0))

    x = np.array([vp.A[-1], vp.Q[-1], vd.A[0], vd.Q[0]])
    c0_ref = wave_speed(A0_p, A0_p, b_p)
    scale = np.array([1/A0_p, 1/(rho*c0_ref**2), 1/c0_ref, 1/c0_ref])

    for _ in range(max_iter):
        Ap, Qp, Ad, Qd = x
        cp = wave_speed(Ap, A0_p, b_p)
        cd = wave_speed(Ad, A0_d, b_d)
        Pp = total_pressure(Ap, Qp, A0_p, b_p)
        Pd = total_pressure(Ad, Qd, A0_d, b_d)

        F = np.array([Qp - Qd, Pp - Pd,
                      Qp/Ap + 4*cp - W1_ext, Qd/Ad - 4*cd - W2_ext])
        if np.max(np.abs(F*scale)) < tol: break

        def dPt_dA(A,Q,A0,b): return b/(2*np.sqrt(A)) - rho*Q**2/A**3
        def dPt_dQ(A,Q,A0,b): return rho*Q/A**2

        J = np.zeros((4,4))
        J[0,1] = 1; J[0,3] = -1
        J[1,0] = dPt_dA(Ap,Qp,A0_p,b_p); J[1,1] = dPt_dQ(Ap,Qp,A0_p,b_p)
        J[1,2] = -dPt_dA(Ad,Qd,A0_d,b_d); J[1,3] = -dPt_dQ(Ad,Qd,A0_d,b_d)
        J[2,0] = -Qp/Ap**2 + cp/Ap; J[2,1] = 1/Ap
        J[3,2] = -Qd/Ad**2 - cd/Ad; J[3,3] = 1/Ad

        Js = J * scale[:,np.newaxis]
        try: dx_nr = np.linalg.solve(Js, -F*scale)
        except: dx_nr = np.linalg.lstsq(Js, -F*scale, rcond=None)[0]

        mr = max(1.0, max(abs(dx_nr[i])/(0.2*x[i]) for i in [0,2]))
        x += dx_nr / mr
        x[0] = max(x[0], A0_p*0.1); x[2] = max(x[2], A0_d*0.1)
    return x


def run_simulation(topo_name, topo_cfg):
    """Run full MUSCL simulation for a topology. Returns (vessels, dt, Nt)."""
    print(f"\n{'='*60}")
    print(f"Running simulation: {topo_name}")
    print(f"{'='*60}")

    vessels = [Vessel(vc) for vc in topo_cfg["vessels"]]
    T_out = topo_cfg["T_out"]
    inflow = topo_cfg["inflow_func"]
    inlet_idx = topo_cfg["inlet_vessel_idx"]
    outlet_idxs = topo_cfg["outlet_vessel_indices"]
    junctions = topo_cfg["junctions"]

    for v in vessels:
        print(f"  {v.name}: L={v.L:.2f}m, r={v.r*1000:.1f}mm, N={v.N}")

    dx_min = min(v.dx for v in vessels)
    c0 = topo_cfg["vessels"][0]["c0"]
    dt = CFL * dx_min / c0
    Nt = int(np.ceil(T_out / dt))
    dt = T_out / Nt
    print(f"  dt={dt:.6f}s, Nt={Nt}, T_out={T_out:.3f}s\n")

    for n in range(Nt):
        t = n * dt
        for v in vessels:
            v.A_hist.append(v.A.copy())
            v.Q_hist.append(v.Q.copy())

        # MUSCL reconstruction for all vessels
        ULs, URs = {}, {}
        for i, v in enumerate(vessels):
            ULs[i], URs[i] = muscl_reconstruction(v)

        # Inlet BC
        ULs[inlet_idx], URs[inlet_idx] = apply_inlet_BC(
            vessels[inlet_idx], t, dt, ULs[inlet_idx], URs[inlet_idx], inflow)

        # Outlet BCs
        for oi in outlet_idxs:
            ULs[oi], URs[oi] = apply_outlet_BC(vessels[oi], dt, ULs[oi], URs[oi])

        # Junction coupling
        for junc in junctions:
            if junc["type"] == "bifurcation":
                pi = junc["parent_idx"]
                di1, di2 = junc["daughter_indices"]
                res = solve_bifurcation(vessels[pi], vessels[di1], vessels[di2], dt)
                ULs[pi][:, vessels[pi].N] = [res[0], res[1]]
                URs[pi][:, vessels[pi].N] = [res[0], res[1]]
                ULs[di1][:, 0] = [res[2], res[3]]
                URs[di1][:, 0] = [res[2], res[3]]
                ULs[di2][:, 0] = [res[4], res[5]]
                URs[di2][:, 0] = [res[4], res[5]]
            elif junc["type"] == "merge":
                pi1, pi2 = junc["parent_indices"]
                di = junc["daughter_idx"]
                res = solve_merge(vessels[pi1], vessels[pi2], vessels[di], dt)
                ULs[pi1][:, vessels[pi1].N] = [res[0], res[1]]
                URs[pi1][:, vessels[pi1].N] = [res[0], res[1]]
                ULs[pi2][:, vessels[pi2].N] = [res[2], res[3]]
                URs[pi2][:, vessels[pi2].N] = [res[2], res[3]]
                ULs[di][:, 0] = [res[4], res[5]]
                URs[di][:, 0] = [res[4], res[5]]
            elif junc["type"] == "conjunction":
                pi = junc["parent_idx"]
                di = junc["daughter_idx"]
                res = solve_conjunction(vessels[pi], vessels[di], dt)
                ULs[pi][:, vessels[pi].N] = [res[0], res[1]]
                URs[pi][:, vessels[pi].N] = [res[0], res[1]]
                ULs[di][:, 0] = [res[2], res[3]]
                URs[di][:, 0] = [res[2], res[3]]

        # Compute spatial operators
        Phis = {}
        for i, v in enumerate(vessels):
            Phis[i] = muscl_update(v, ULs[i], URs[i])

        # Adams-Bashforth time integration
        for i, v in enumerate(vessels):
            if v.Phi_prev is None:
                v.A += dt * Phis[i][0,:]
                v.Q += dt * Phis[i][1,:]
            else:
                v.A += dt * (1.5*Phis[i][0,:] - 0.5*v.Phi_prev[0,:])
                v.Q += dt * (1.5*Phis[i][1,:] - 0.5*v.Phi_prev[1,:])
            v.Phi_prev = Phis[i].copy()

        if (n+1) % max(1, Nt//10) == 0:
            print(f"  step {n+1:6d}/{Nt} (t = {t+dt:.5f}s)")

    print("Simulation complete.")
    return vessels, dt, Nt
