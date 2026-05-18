"""
Solver 1D przepływu krwi z bifurkacją (Y-shaped bifurcation)
Metoda: Objętości skończone (MUSCL + Adams-Bashforth 2. rzędu)
Koncepcja złącza: Rozwiązywanie układu 6 równań nieliniowych (Newton-Raphson)
Wizualizacja 3D: PyVista
na podstawie: D. Kim, J. Tithof. "One-Dimensional Modeling of Blood Flow:
A Comprehensive Yet Concise Review" (2025) - sekcje 5, 6, 7
"""

import numpy as np
import pyvista as pv

# ============================================================
# STAŁE FIZYCZNE
# ============================================================
rho = 1060.0       # Gęstość krwi [kg/m^3]
mu = 4.0e-3        # Lepkość krwi [Pa·s]
KR = 8.0 * np.pi * mu / rho  # Współczynnik tarcia (Poiseuille, gamma=2)
CFL = 0.45
T_out = 0.75       # Czas symulacji [s]

# ============================================================
# NACZYNIA
# ============================================================
class Vessel:
    """Pojedynczy segment naczynia krwionośnego."""
    def __init__(self, L, r, c0, N=50, name=""):
        self.L = L
        self.r = r
        self.A0 = np.pi * r**2
        self.N = N
        self.name = name
        self.dx = L / N
        self.x = np.linspace(self.dx/2, L - self.dx/2, N)
        # beta zgodnie z publikacją (liniowa relacja P = beta*(A - A0))
        # c0^2 = beta * A0 / rho
        self.beta = rho * c0**2 / self.A0
        self.A = np.ones(N) * self.A0
        self.Q = np.zeros(N)
        self.A_hist = []
        self.Q_hist = []
        self.Phi_prev = None  # dla Adams-Bashforth


# ============================================================
# FUNKCJE POMOCNICZE (zgodne z publikacją)
# ============================================================
def pressure(A, A0, beta):
    # P = Pext + beta * (A - A0), Pext=0
    return beta * (A - A0)

def wave_speed(A, A0, beta):
    # c = sqrt(beta * A / rho)
    return np.sqrt(beta * A / rho)

def total_pressure(A, Q, A0, beta):
    return pressure(A, A0, beta) + 0.5 * rho * (Q / A)**2

def flux_func(A, Q, A0, beta):
    F1 = Q
    # F2 = Q^2/A + beta/(2*rho) * A^2
    F2 = Q**2 / A + beta / (2.0 * rho) * A**2
    return np.array([F1, F2])

def riemann_W1(A, Q, A0, beta):
    c = wave_speed(A, A0, beta)
    return Q / A + 2.0 * c

def riemann_W2(A, Q, A0, beta):
    c = wave_speed(A, A0, beta)
    return Q / A - 2.0 * c

def from_riemann(W1, W2, A0, beta):
    # c = (W1 - W2)/4
    # A = rho * c^2 / beta
    c = (W1 - W2) / 4.0
    A = rho * c**2 / beta
    u = (W1 + W2) / 2.0
    Q = A * u
    return A, Q


# ============================================================
# MUSCL RECONSTRUCTION (bez half-step, zgodnie z sekcją 5.1)
# ============================================================
def muscl_reconstruction(vessel):
    A, Q = vessel.A, vessel.Q
    N = vessel.N
    dx = vessel.dx
    U = np.vstack((A, Q))
    # Padding edge dla minmod
    U_pad = np.pad(U, ((0, 0), (1, 1)), mode='edge')
    dU_L = U_pad[:, 1:-1] - U_pad[:, 0:-2]   # U_i - U_{i-1}
    dU_R = U_pad[:, 2:]   - U_pad[:, 1:-1]   # U_{i+1} - U_i

    # Minmod slope limiter [str. 15]
    DUi = np.zeros_like(U)
    mask_pos = (dU_L > 0) & (dU_R > 0)
    mask_neg = (dU_L < 0) & (dU_R < 0)
    DUi[mask_pos] = np.minimum(dU_L[mask_pos], dU_R[mask_pos]) / dx
    DUi[mask_neg] = np.maximum(dU_L[mask_neg], dU_R[mask_neg]) / dx

    UL = np.zeros((2, N + 1))
    UR = np.zeros((2, N + 1))
    # Wewnętrzne interfejsy: 1 .. N-1
    UL[:, 1:N] = U[:, :N-1] + 0.5 * dx * DUi[:, :N-1]
    UR[:, 1:N] = U[:, 1:]   - 0.5 * dx * DUi[:, 1:]
    # Interfejsy 0 (inlet) i N (outlet) ustawiane przez BC
    return UL, UR


# ============================================================
# OBLICZENIE OPERATORA PRZESTRZENNEGO PHI + człon źródłowy
# ============================================================
def muscl_update(vessel, UL, UR):
    A0, beta, dx, N = vessel.A0, vessel.beta, vessel.dx, vessel.N
    AL, QL = UL
    AR, QR = UR

    # Rusanov (Lax-Friedrichs) flux na wszystkich interfejsach 0..N
    FL = flux_func(AL, QL, A0, beta)
    FR = flux_func(AR, QR, A0, beta)

    uL = QL / AL
    cL = wave_speed(AL, A0, beta)
    uR = QR / AR
    cR = wave_speed(AR, A0, beta)
    S_max = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)

    F_interf = 0.5 * (FL + FR) - 0.5 * S_max * (UR - UL)

    # Człon źródłowy (tarcie lepkie) [str. 14, 16]
    S = np.zeros((2, N))
    S[1, :] = -KR * vessel.Q / vessel.A

    # Operator przestrzenny Phi
    Phi = -(F_interf[:, 1:] - F_interf[:, :-1]) / dx + S
    return Phi


# ============================================================
# WARUNKI BRZEGOWE
# ============================================================
def apply_inlet_BC(vessel, t, dt, UL, UR):
    A0, beta, N = vessel.A0, vessel.beta, vessel.N
    T_pulse = 0.02
    if t <= T_pulse:
        Q_in = 5e-5 * np.sin(np.pi * t / T_pulse)
    elif 0.1 <= t <= 0.1 + T_pulse:
        Q_in = 5e-5 * np.sin(np.pi * (t - 0.1) / T_pulse)
    else:
        Q_in = 0.0

    A0_cell = vessel.A[0]
    A1_cell = vessel.A[1]
    Q0_cell = vessel.Q[0]
    Q1_cell = vessel.Q[1]

    W2_0 = riemann_W2(A0_cell, Q0_cell, A0, beta)
    W2_dz = riemann_W2(A1_cell, Q1_cell, A0, beta)
    u0 = Q0_cell / A0_cell
    c0 = wave_speed(A0_cell, A0, beta)
    # Ekstrapolacja W2 (charakterystyka wychodząca, lambda2 < 0) [str. 19]
    W2_new = W2_0 + dt * (W2_dz - W2_0) / vessel.dx * (-(u0 - c0))

    # Rozwiązanie nieliniowe: Q_in/A - 2*c(A) - W2_new = 0
    A_guess = max(A0_cell, A0)
    for _ in range(15):
        cA = wave_speed(A_guess, A0, beta)
        F_val = Q_in / A_guess - 2.0 * cA - W2_new
        if abs(F_val) < 1e-14:
            break
        dF = -Q_in / A_guess**2 - cA / A_guess
        A_guess = A_guess - F_val / dF
        A_guess = max(A_guess, A0 * 0.1)

    UL[:, 0] = [A_guess, Q_in]
    UR[:, 0] = [A_guess, Q_in]
    return UL, UR


def apply_outlet_BC(vessel, dt, UL, UR, Rt=0.0):
    A0, beta, N = vessel.A0, vessel.beta, vessel.N
    A_last = vessel.A[-1]
    A_prev = vessel.A[-2]
    Q_last = vessel.Q[-1]
    Q_prev = vessel.Q[-2]

    W1_L = riemann_W1(A_last, Q_last, A0, beta)
    W1_Lm1 = riemann_W1(A_prev, Q_prev, A0, beta)
    uL = Q_last / A_last
    cL = wave_speed(A_last, A0, beta)

    # POPRAWKA ZNAKU: ekstrapolacja W1 wzdłuż charakterystyki lambda1 > 0 [str. 21]
    W1_new = W1_L + dt * (W1_Lm1 - W1_L) / vessel.dx * (uL + cL)

    u0 = 0.0
    c0 = wave_speed(A0, A0, beta)
    W1_0 = u0 + 2.0 * c0
    W2_0 = u0 - 2.0 * c0
    W2_new = W2_0 - Rt * (W1_new - W1_0)

    A_out, Q_out = from_riemann(W1_new, W2_new, A0, beta)
    UL[:, N] = [A_out, Q_out]
    UR[:, N] = [A_out, Q_out]
    return UL, UR


# ============================================================
# SOLVER BIFURKACJI (NEWTON-RAPHSON) - sekcja 7
# ============================================================
def solve_bifurcation(vessel_p, vessel_d1, vessel_d2, dt, tol=1e-12, max_iter=30):
    A0_p, beta_p = vessel_p.A0, vessel_p.beta
    A0_d1, beta_d1 = vessel_d1.A0, vessel_d1.beta
    A0_d2, beta_d2 = vessel_d2.A0, vessel_d2.beta

    # Ekstrapolacja W1 parent (outgoing)
    A_L, A_Lm1 = vessel_p.A[-1], vessel_p.A[-2]
    Q_L, Q_Lm1 = vessel_p.Q[-1], vessel_p.Q[-2]
    W1_p = riemann_W1(A_L, Q_L, A0_p, beta_p)
    W1_p_m1 = riemann_W1(A_Lm1, Q_Lm1, A0_p, beta_p)
    u_L = Q_L / A_L
    c_L = wave_speed(A_L, A0_p, beta_p)
    W1_p_ext = W1_p + dt * (W1_p - W1_p_m1) / vessel_p.dx * (u_L + c_L)

    # Ekstrapolacja W2 daughter 1 (incoming)
    A_0, A_1 = vessel_d1.A[0], vessel_d1.A[1]
    Q_0, Q_1 = vessel_d1.Q[0], vessel_d1.Q[1]
    W2_d1 = riemann_W2(A_0, Q_0, A0_d1, beta_d1)
    W2_d1_p1 = riemann_W2(A_1, Q_1, A0_d1, beta_d1)
    u_0 = Q_0 / A_0
    c_0 = wave_speed(A_0, A0_d1, beta_d1)
    W2_d1_ext = W2_d1 + dt * (W2_d1_p1 - W2_d1) / vessel_d1.dx * (-(u_0 - c_0))

    # Ekstrapolacja W2 daughter 2 (incoming)
    A_0, A_1 = vessel_d2.A[0], vessel_d2.A[1]
    Q_0, Q_1 = vessel_d2.Q[0], vessel_d2.Q[1]
    W2_d2 = riemann_W2(A_0, Q_0, A0_d2, beta_d2)
    W2_d2_p1 = riemann_W2(A_1, Q_1, A0_d2, beta_d2)
    u_0 = Q_0 / A_0
    c_0 = wave_speed(A_0, A0_d2, beta_d2)
    W2_d2_ext = W2_d2 + dt * (W2_d2_p1 - W2_d2) / vessel_d2.dx * (-(u_0 - c_0))

    x = np.array([vessel_p.A[-1], vessel_p.Q[-1],
                  vessel_d1.A[0], vessel_d1.Q[0],
                  vessel_d2.A[0], vessel_d2.Q[0]])

    for it in range(max_iter):
        A_p, Q_p, A_d1, Q_d1, A_d2, Q_d2 = x
        u_p = Q_p / A_p
        u_d1 = Q_d1 / A_d1
        u_d2 = Q_d2 / A_d2
        c_p = wave_speed(A_p, A0_p, beta_p)
        c_d1 = wave_speed(A_d1, A0_d1, beta_d1)
        c_d2 = wave_speed(A_d2, A0_d2, beta_d2)

        P_tot_p = total_pressure(A_p, Q_p, A0_p, beta_p)
        P_tot_d1 = total_pressure(A_d1, Q_d1, A0_d1, beta_d1)
        P_tot_d2 = total_pressure(A_d2, Q_d2, A0_d2, beta_d2)

        F = np.array([
            Q_p - Q_d1 - Q_d2,
            P_tot_p - P_tot_d1,
            P_tot_p - P_tot_d2,
            u_p + 2.0 * c_p - W1_p_ext,
            u_d1 - 2.0 * c_d1 - W2_d1_ext,
            u_d2 - 2.0 * c_d2 - W2_d2_ext,
        ])
        if np.max(np.abs(F)) < tol:
            break

        J = np.zeros((6, 6))
        J[0, 1] = 1.0
        J[0, 3] = -1.0
        J[0, 5] = -1.0

        # Pochodne ciśnienia dla liniowej relacji P = beta*(A - A0)
        def dPtot_dA(A_, Q_, A0_, beta_):
            return beta_ - rho * Q_**2 / A_**3

        def dPtot_dQ(A_, Q_, A0_, beta_):
            return rho * Q_ / A_**2

        J[1, 0] = dPtot_dA(A_p, Q_p, A0_p, beta_p)
        J[1, 1] = dPtot_dQ(A_p, Q_p, A0_p, beta_p)
        J[1, 2] = -dPtot_dA(A_d1, Q_d1, A0_d1, beta_d1)
        J[1, 3] = -dPtot_dQ(A_d1, Q_d1, A0_d1, beta_d1)

        J[2, 0] = dPtot_dA(A_p, Q_p, A0_p, beta_p)
        J[2, 1] = dPtot_dQ(A_p, Q_p, A0_p, beta_p)
        J[2, 4] = -dPtot_dA(A_d2, Q_d2, A0_d2, beta_d2)
        J[2, 5] = -dPtot_dQ(A_d2, Q_d2, A0_d2, beta_d2)

        J[3, 0] = -Q_p / A_p**2 + c_p / A_p
        J[3, 1] = 1.0 / A_p

        J[4, 2] = -Q_d1 / A_d1**2 - c_d1 / A_d1
        J[4, 3] = 1.0 / A_d1

        J[5, 4] = -Q_d2 / A_d2**2 - c_d2 / A_d2
        J[5, 5] = 1.0 / A_d2

        try:
            dx_nr = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            dx_nr = np.linalg.lstsq(J, -F, rcond=None)[0]
        x = x + dx_nr
        x[0] = max(x[0], A0_p * 0.1)
        x[2] = max(x[2], A0_d1 * 0.1)
        x[4] = max(x[4], A0_d2 * 0.1)

    return x


# ============================================================
# DEFINICJA NACZYŃ (Y-shaped bifurcation)
# ============================================================
L_p, r_p, c0, N_p = 0.10, 0.005, 5.0, 60
vessel_p = Vessel(L_p, r_p, c0, N=N_p, name="Parent (Aorta)")
L_d1, r_d1, N_d1 = 0.08, 0.0035, 48
vessel_d1 = Vessel(L_d1, r_d1, c0, N=N_d1, name="Daughter 1")
L_d2, r_d2, N_d2 = 0.09, 0.0040, 54
vessel_d2 = Vessel(L_d2, r_d2, c0, N=N_d2, name="Daughter 2")

dt_min = CFL * min(vessel_p.dx, vessel_d1.dx, vessel_d2.dx) / c0
Nt = int(np.ceil(T_out / dt_min))
dt = T_out / Nt

print(f"Bifurkacja Y: 1 parent -> 2 daughters")
print(f"  Parent:     L={L_p:.2f}m, r={r_p*1000:.1f}mm, N={N_p}")
print(f"  Daughter 1: L={L_d1:.2f}m, r={r_d1*1000:.1f}mm, N={N_d1}")
print(f"  Daughter 2: L={L_d2:.2f}m, r={r_d2*1000:.1f}mm, N={N_d2}")
print(f"  dt={dt:.6f}s, Nt={Nt}, T_out={T_out:.3f}s")
print()


# ============================================================
# PĘTLA GŁÓWNA SYMULACJI (Adams-Bashforth 2. rzędu)
# ============================================================
print("Symulacja...")
for n in range(Nt):
    t = n * dt
    # Zapis historii (stan przed update)
    vessel_p.A_hist.append(vessel_p.A.copy())
    vessel_p.Q_hist.append(vessel_p.Q.copy())
    vessel_d1.A_hist.append(vessel_d1.A.copy())
    vessel_d1.Q_hist.append(vessel_d1.Q.copy())
    vessel_d2.A_hist.append(vessel_d2.A.copy())
    vessel_d2.Q_hist.append(vessel_d2.Q.copy())

    # Rekonstrukcja MUSCL
    UL_p, UR_p = muscl_reconstruction(vessel_p)
    UL_d1, UR_d1 = muscl_reconstruction(vessel_d1)
    UL_d2, UR_d2 = muscl_reconstruction(vessel_d2)

    # Warunki brzegowe
    UL_p, UR_p = apply_inlet_BC(vessel_p, t, dt, UL_p, UR_p)
    UL_d1, UR_d1 = apply_outlet_BC(vessel_d1, dt, UL_d1, UR_d1, Rt=0.0)
    UL_d2, UR_d2 = apply_outlet_BC(vessel_d2, dt, UL_d2, UR_d2, Rt=0.0)

    # Bifurkacja: rozwiązanie złącza i nadpisanie interfejsów
    result = solve_bifurcation(vessel_p, vessel_d1, vessel_d2, dt)
    UL_p[:, vessel_p.N] = [result[0], result[1]]
    UR_p[:, vessel_p.N] = [result[0], result[1]]
    UL_d1[:, 0] = [result[2], result[3]]
    UR_d1[:, 0] = [result[2], result[3]]
    UL_d2[:, 0] = [result[4], result[5]]
    UR_d2[:, 0] = [result[4], result[5]]

    # Obliczenie operatorów przestrzennych Phi
    Phi_p = muscl_update(vessel_p, UL_p, UR_p)
    Phi_d1 = muscl_update(vessel_d1, UL_d1, UR_d1)
    Phi_d2 = muscl_update(vessel_d2, UL_d2, UR_d2)

    # Adams-Bashforth 2. rzędu [str. 16]
    if vessel_p.Phi_prev is None:
        # Pierwszy krok: Euler do przodu
        vessel_p.A += dt * Phi_p[0, :]
        vessel_p.Q += dt * Phi_p[1, :]
        vessel_d1.A += dt * Phi_d1[0, :]
        vessel_d1.Q += dt * Phi_d1[1, :]
        vessel_d2.A += dt * Phi_d2[0, :]
        vessel_d2.Q += dt * Phi_d2[1, :]
    else:
        vessel_p.A += dt * (1.5 * Phi_p[0, :] - 0.5 * vessel_p.Phi_prev[0, :])
        vessel_p.Q += dt * (1.5 * Phi_p[1, :] - 0.5 * vessel_p.Phi_prev[1, :])
        vessel_d1.A += dt * (1.5 * Phi_d1[0, :] - 0.5 * vessel_d1.Phi_prev[0, :])
        vessel_d1.Q += dt * (1.5 * Phi_d1[1, :] - 0.5 * vessel_d1.Phi_prev[1, :])
        vessel_d2.A += dt * (1.5 * Phi_d2[0, :] - 0.5 * vessel_d2.Phi_prev[0, :])
        vessel_d2.Q += dt * (1.5 * Phi_d2[1, :] - 0.5 * vessel_d2.Phi_prev[1, :])

    # Zapamiętanie Phi dla następnego kroku
    vessel_p.Phi_prev = Phi_p.copy()
    vessel_d1.Phi_prev = Phi_d1.copy()
    vessel_d2.Phi_prev = Phi_d2.copy()

    if (n + 1) % max(1, Nt // 10) == 0:
        print(f"  krok {n+1:6d}/{Nt} (t = {t+dt:.5f}s)")

print("Symulacja zakończona.")
print()


# ============================================================
# EKSPORT DANYCH DO CSV
# ============================================================
import csv

csv_filename = "blood_flow_data.csv"
time_vals_full = np.linspace(0, T_out, Nt)

with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["t", "vessel", "z", "A", "Q"])

    for n in range(Nt):
        t = time_vals_full[n]
        for i in range(N_p):
            writer.writerow([f"{t:.8e}", "parent", f"{vessel_p.x[i]:.8e}",
                             f"{vessel_p.A_hist[n][i]:.8e}", f"{vessel_p.Q_hist[n][i]:.8e}"])
        for i in range(N_d1):
            writer.writerow([f"{t:.8e}", "daughter1", f"{vessel_d1.x[i]:.8e}",
                             f"{vessel_d1.A_hist[n][i]:.8e}", f"{vessel_d1.Q_hist[n][i]:.8e}"])
        for i in range(N_d2):
            writer.writerow([f"{t:.8e}", "daughter2", f"{vessel_d2.x[i]:.8e}",
                             f"{vessel_d2.A_hist[n][i]:.8e}", f"{vessel_d2.Q_hist[n][i]:.8e}"])

n_rows = Nt * (N_p + N_d1 + N_d2)
print(f"Zapisano {csv_filename} ({n_rows} wierszy)")
print()


# ============================================================
# WIZUALIZACJA 3D - PyVista
# ============================================================
deformation_scale = 3.0
bif_angle = np.deg2rad(30)

# Geometria 3D
x_p = vessel_p.x
y_p = np.zeros_like(x_p)
z_p = np.zeros_like(x_p)

s_d1 = np.linspace(0, L_d1, N_d1)
x_d1 = L_p + s_d1 * np.cos(bif_angle)
y_d1 = s_d1 * np.sin(bif_angle)
z_d1 = np.zeros_like(s_d1)

s_d2 = np.linspace(0, L_d2, N_d2)
x_d2 = L_p + s_d2 * np.cos(bif_angle)
y_d2 = np.zeros_like(s_d2)
z_d2 = -s_d2 * np.sin(bif_angle)

N_theta = 20
theta_vals = np.linspace(0, 2 * np.pi, N_theta, endpoint=True)

R0_p = np.sqrt(vessel_p.A0 / np.pi)
R0_d1 = np.sqrt(vessel_d1.A0 / np.pi)
R0_d2 = np.sqrt(vessel_d2.A0 / np.pi)

Qmax_global = 0
for Q_hist in [vessel_p.Q_hist, vessel_d1.Q_hist, vessel_d2.Q_hist]:
    for q in Q_hist:
        Qmax_global = max(Qmax_global, np.max(np.abs(q)))
Qmax_global = max(Qmax_global, 1e-10)
Qmax = Qmax_global * 1e6
print(f"Globalne max Q: {Qmax:.2f} ml/s")


def build_vessel_grid(x_c, y_c, z_c, R_vals):
    N_z = len(x_c)
    dx = np.gradient(x_c)
    dy = np.gradient(y_c)
    dz = np.gradient(z_c)
    pts = np.zeros((N_theta, N_z, 3))

    for i in range(N_z):
        t = np.array([dx[i], dy[i], dz[i]])
        tnorm = np.linalg.norm(t)
        if tnorm < 1e-15:
            t = np.array([1.0, 0.0, 0.0])
        else:
            t = t / tnorm
        if abs(t[0]) < 0.9:
            u = np.cross(t, np.array([1.0, 0.0, 0.0]))
        else:
            u = np.cross(t, np.array([0.0, 1.0, 0.0]))
        unorm = np.linalg.norm(u)
        if unorm > 1e-15:
            u = u / unorm
        else:
            u = np.array([0.0, 1.0, 0.0])
        v = np.cross(t, u)
        center = np.array([x_c[i], y_c[i], z_c[i]])
        R = R_vals[i]
        for j in range(N_theta):
            theta = theta_vals[j]
            pt = center + R * (np.cos(theta) * u + np.sin(theta) * v)
            pts[j, i] = pt

    grid = pv.StructuredGrid(pts[:, :, 0], pts[:, :, 1], pts[:, :, 2])
    return grid


# Pre-komputacja klatek
step = max(1, Nt // 200)
frame_indices = list(range(0, Nt, step))
time_vals = np.linspace(0, T_out, Nt)

all_grids_p = []
all_grids_d1 = []
all_grids_d2 = []

for idx in frame_indices:
    A_p_f = np.array(vessel_p.A_hist[idx])
    Q_p_f = np.array(vessel_p.Q_hist[idx])
    A_d1_f = np.array(vessel_d1.A_hist[idx])
    Q_d1_f = np.array(vessel_d1.Q_hist[idx])
    A_d2_f = np.array(vessel_d2.A_hist[idx])
    Q_d2_f = np.array(vessel_d2.Q_hist[idx])

    R_p_f = R0_p + (np.sqrt(A_p_f / np.pi) - R0_p) * deformation_scale
    R_d1_f = R0_d1 + (np.sqrt(A_d1_f / np.pi) - R0_d1) * deformation_scale
    R_d2_f = R0_d2 + (np.sqrt(A_d2_f / np.pi) - R0_d2) * deformation_scale

    g_p = build_vessel_grid(x_p, y_p, z_p, R_p_f)
    g_d1 = build_vessel_grid(x_d1, y_d1, z_d1, R_d1_f)
    g_d2 = build_vessel_grid(x_d2, y_d2, z_d2, R_d2_f)

    Qp_flat = np.tile(Q_p_f * 1e6, (N_theta, 1)).ravel(order='F')
    Qd1_flat = np.tile(Q_d1_f * 1e6, (N_theta, 1)).ravel(order='F')
    Qd2_flat = np.tile(Q_d2_f * 1e6, (N_theta, 1)).ravel(order='F')

    g_p.point_data['Q'] = Qp_flat
    g_d1.point_data['Q'] = Qd1_flat
    g_d2.point_data['Q'] = Qd2_flat

    all_grids_p.append(g_p)
    all_grids_d1.append(g_d1)
    all_grids_d2.append(g_d2)

# Eksport GIF
print("Eksportowanie animacji do GIF...")
pv.set_plot_theme('dark')
p = pv.Plotter(window_size=[1200, 800], off_screen=True)

bifurcation_point = (L_p, 0.0, 0.0)
p.camera_position = [(0.36, -0.24, 0.15),
                     bifurcation_point,
                     (0.0, 0.0, 1.0)]
p.enable_anti_aliasing()
p.add_axes()

p.open_gif("blood_flow.gif", fps=25)

n_cycles = 1
for cycle in range(n_cycles):
    for idx, frame_idx in enumerate(frame_indices):
        p.clear()
        p.add_mesh(all_grids_p[idx], scalars='Q', clim=[0, Qmax],
                   cmap='plasma', show_edges=False, lighting=True,
                   smooth_shading=True)
        p.add_mesh(all_grids_d1[idx], scalars='Q', clim=[0, Qmax],
                   cmap='plasma', show_edges=False, lighting=True,
                   smooth_shading=True)
        p.add_mesh(all_grids_d2[idx], scalars='Q', clim=[0, Qmax],
                   cmap='plasma', show_edges=False, lighting=True,
                   smooth_shading=True)

        t_val = time_vals[frame_idx]
        p.add_text(f"Czas: {t_val:.3f} s",
                   position='upper_left', font_size=14, color='white')
        p.write_frame()

    print(f"  Cykl {cycle+1}/{n_cycles} ({len(frame_indices)} klatek)")

p.close()
print(f"Zapisano blood_flow.gif ({len(frame_indices) * n_cycles} klatek, {n_cycles} cykli)")

print(f"\nPodsumowanie:")
print(f"  Średni przepływ na wylocie parent: {np.mean(np.array(vessel_p.Q_hist)[:, -1])*1e6:.3f} ml/s")
print(f"  Średni przepływ na wlocie d1:      {np.mean(np.array(vessel_d1.Q_hist)[:, 0])*1e6:.3f} ml/s")
print(f"  Średni przepływ na wlocie d2:      {np.mean(np.array(vessel_d2.Q_hist)[:, 0])*1e6:.3f} ml/s")