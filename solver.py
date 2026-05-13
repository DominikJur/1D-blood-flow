"""
Pojedynczy segment naczynia - solver 1D przepływu krwi
Metoda: Objętości skończonych (MUSCL-Hancock)
"""

import numpy as np
import plotly.graph_objects as go


L = 0.1          # Długość naczynia [m] (10 cm)
rho = 1060.0     # Gęstość krwi [kg/m^3]
A0 = np.pi * (0.01)**2 / 4.0  # Spoczynkowe pole przekroju [m^2] (promień 5 mm)
c0 = 5.0         # Nominalna prędkość fali [m/s]
beta = (4.0 / 3.0) * np.sqrt(np.pi) * rho * c0**2 * A0  # Współczynnik sztywności naczynia

N = 100    
dx = L / N 
CFL = 0.8       
T_out = 0.25    
dt = CFL * dx / c0  
Nt = int(np.ceil(T_out / dt))
dt = T_out / Nt  

x = np.linspace(dx/2, L - dx/2, N)
A_hist = np.zeros((Nt, N))
Q_hist = np.zeros((Nt, N))

A = np.ones(N) * A0
Q = np.zeros(N)

def pressure(A):
    """Tube law - ciśnienie jako funkcja pola przekroju."""
    return beta * (np.sqrt(A) - np.sqrt(A0)) / A0

def wave_speed(A):
    """Lokalna prędkość fali propagacyjnej c(A)."""
    return np.sqrt(beta / (2 * rho * A0)) * np.sqrt(np.sqrt(A))

def flux(A, Q):
    """
    Strumień dla równań 1D zachowawczych: U = [A, Q]^T
    F = [Q, Q^2/A + (beta/(3*rho*A0)) * A^{3/2}]^T
    """
    F1 = Q
    F2 = Q**2 / A + beta / (3.0 * rho * A0) * A**(1.5)
    return np.array([F1, F2])

def minmod(r):
    """Limiter MinMod dla nachylenia."""
    return np.maximum(0, np.minimum(1, r))

# =============================================================================
# SCHEMAT NUMERYCZNY (MUSCL-Hancock)
# =============================================================================
def minmod_slope(U_left, U_center, U_right):
    """Obliczanie ograniczonego nachylenia za pomocą limitera minmod."""
    dU_L = U_center - U_left
    dU_R = U_right - U_center
    
    r = np.zeros_like(dU_L)
    mask = dU_R != 0
    r[mask] = dU_L[mask] / dU_R[mask]
    
    phi = minmod(r)
    slope = phi * dU_R
    return slope

def rusanov_flux(UL, UR):
    """Solwer Riemanna: Strumień Rusanova (Local Lax-Friedrichs)."""
    AL, QL = UL
    AR, QR = UR
    
    FL = flux(AL, QL)
    FR = flux(AR, QR)
    
    uL = QL / AL
    cL = wave_speed(AL)
    
    uR = QR / AR
    cR = wave_speed(AR)
    
    S_max = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)
    
    F = 0.5 * (FL + FR) - 0.5 * S_max * (UR - UL)
    return F

# =============================================================================
# PĘTLA GŁÓWNA SYMULACJI
# =============================================================================
print("Rozpoczynam symulację przepływu...")

for n in range(Nt):
    # --- Zapis historii ---
    A_hist[n, :] = A.copy()
    Q_hist[n, :] = Q.copy()
    
    U = np.vstack((A, Q))  # Kształt: (2, N)
    
    # --- 1. REKONSTRUKCJA I KROK PREDYKCJI (Przestrzeń) - WEKTORYZACJA ---
    U_L_interf = np.zeros((2, N+1))
    U_R_interf = np.zeros((2, N+1))
    
    U_pad = np.pad(U, ((0, 0), (1, 1)), mode='edge') 
    
    dU_L = U_pad[:, 1:-1] - U_pad[:, 0:-2]
    dU_R = U_pad[:, 2:]   - U_pad[:, 1:-1]
    
    r = np.zeros_like(dU_L)
    mask = dU_R != 0
    r[mask] = dU_L[mask] / dU_R[mask]
    
    phi = np.maximum(0, np.minimum(1, r)) # Limitmin mod
    slope = phi * dU_R
    
    UL_i = U - 0.5 * slope
    UR_i = U + 0.5 * slope
    
    F_UL_i = np.array([UL_i[1], UL_i[1]**2 / UL_i[0] + beta / (3.0 * rho * A0) * UL_i[0]**(1.5)])
    F_UR_i = np.array([UR_i[1], UR_i[1]**2 / UR_i[0] + beta / (3.0 * rho * A0) * UR_i[0]**(1.5)])
    
    U_half_i = U - 0.5 * (dt / dx) * (F_UR_i - F_UL_i)
    
    U_L_interf[:, 1:] = U_half_i + 0.5 * slope
    U_R_interf[:, :-1]   = U_half_i - 0.5 * slope
    
    # WARUNKI BRZEGOWE (Riemann Invariants)
    t = n * dt
    T_pulse = 0.02
    
    # 2 impulsy (jeden na początku, drugi w t = 0.1)
    if t <= T_pulse:
        Q_in = 5e-5 * np.sin(np.pi * t / T_pulse)
    elif 0.1 <= t <= 0.1 + T_pulse:
        Q_in = 5e-5 * np.sin(np.pi * (t - 0.1) / T_pulse)
    else:
        Q_in = 0.0
    
    A_interior_L = U[0, 0]
    u_interior_L = U[1, 0] / A_interior_L
    c_interior_L = wave_speed(A_interior_L)
    W2_L = u_interior_L - 4 * c_interior_L
    
    A_in = A_interior_L
    for _ in range(5):
        c_in = wave_speed(A_in)
        u_in = Q_in / A_in
        res = u_in - 4 * c_in - W2_L
        d_u = - Q_in / (A_in**2)
        # Poprawiona pochodna dc/dA (brakowało 1/4 i minusa nie powinno być)
        d_c = 0.25 * np.sqrt(beta / (2 * rho * A0)) * A_in**(-0.75)
        deriv = d_u - 4 * d_c
        A_in = A_in - res / deriv
    
    U_R_interf[:, 0] = [A_in, Q_in]
    U_L_interf[:, 0] = [A_in, Q_in]

    U_L_interf[:, N] = U_R_interf[:, N-1]
    U_R_interf[:, N] = U_R_interf[:, N-1]
    
    # --- 2. OBLICZANIE STRUMIENI i KROK KOREKCJI (WEKTORYZACJA) ---
    AL, QL = U_L_interf
    AR, QR = U_R_interf
    
    FL = np.array([QL, QL**2 / AL + beta / (3.0 * rho * A0) * AL**(1.5)])
    FR = np.array([QR, QR**2 / AR + beta / (3.0 * rho * A0) * AR**(1.5)])
    
    uL = QL / AL
    cL = np.sqrt(beta / (2 * rho * A0)) * AL**(0.25)
    
    uR = QR / AR
    cR = np.sqrt(beta / (2 * rho * A0)) * AR**(0.25)
    
    S_max = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)
    
    F_interf = 0.5 * (FL + FR) - 0.5 * S_max * (U_R_interf - U_L_interf)
    
    U -= (dt / dx) * (F_interf[:, 1:] - F_interf[:, :-1])
        
    A, Q = U[0, :], U[1, :]

print("Symulacja zakończona.")

# =============================================================================
# WIZUALIZACJA 3D (Animowana siatka naczynia krwionośnego / Vein Mesh)
# =============================================================================
time_vals = np.linspace(0, T_out, Nt)

# Zmniejszamy gęstość klatek do animacji, żeby wykres działał płynnie
step = max(1, Nt // 75)  # ok. 50 klatek w sumie dla całego przebiegu
frames = []

R0 = np.sqrt(A0 / np.pi)
deformation_scale = 1.0 # Wyolbrzymienie pulsacji (by falka pola przekroju była zauważalna dla oka)
theta = np.linspace(0, 2 * np.pi, 10) # pełny obwód naczynia (góra i dół)

for i in range(0, Nt, step):
    A_f = A_hist[i, :]
    Q_f = Q_hist[i, :]
    
    R = np.sqrt(A_f / np.pi)
    R_scaled = R0 + (R - R0) * deformation_scale
    
    theta_grid, z_grid = np.meshgrid(theta, x)
    R_grid = np.outer(R_scaled, np.ones(len(theta)))
    Q_grid = np.outer(Q_f, np.ones(len(theta)))
    
    # Generowanie rury - leżącej wzdłuż osi x (Z w Plotly)
    X = z_grid
    Y = R_grid * np.cos(theta_grid)
    Z = R_grid * np.sin(theta_grid)
    
    frames.append(go.Frame(
        data=[go.Surface(
            x=X, y=Y, z=Z, 
            surfacecolor=Q_grid * 1e6, 
            colorscale='Plasma', 
            cmin=0, 
            cmax=np.max(Q_hist)*1e6
        )],
        name=str(i)
    ))

# Parametry startowe dla pierwszej klatki
A_ini = A_hist[0, :]
R_ini = R0 + (np.sqrt(A_ini / np.pi) - R0) * deformation_scale
theta_grid, z_grid = np.meshgrid(theta, x)
R_grid_ini = np.outer(R_ini, np.ones(len(theta)))
X_ini = z_grid
Y_ini = R_grid_ini * np.cos(theta_grid)
Z_ini = R_grid_ini * np.sin(theta_grid)
Q_grid_ini = np.outer(Q_hist[0, :], np.ones(len(theta)))

fig = go.Figure(
    data=[go.Surface(
        x=X_ini, y=Y_ini, z=Z_ini,
        surfacecolor=Q_grid_ini * 1e6,
        colorscale='Plasma',
        cmin=0,
        cmax=np.max(Q_hist)*1e6,
        colorbar=dict(title='Przepływ Q [ml/s]')
    )],
    layout=go.Layout(
        title='3D Siatka naczynia krwionośnego (wyolbrzymiony puls tętna)',
        scene=dict(
            xaxis_title='Z (długość naczynia) [m]',
            yaxis_title='X [m]',
            zaxis_title='Y [m]',
            aspectratio=dict(x=3, y=1, z=1), # Wydłużamy rurę żeby nie była spłaszczona
            yaxis=dict(range=[-R0*3, R0*3]), # Stałe ograniczenia na osie przekroju, żeby rura nie skakała
            zaxis=dict(range=[-R0*3, R0*3])
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Odtwórz", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
                dict(label="Przerwij", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
            ]
        )],
        sliders=[dict(
            steps=[dict(method='animate', args=[[str(k)], dict(mode='immediate', frame=dict(duration=0, redraw=True))], label=f'{time_vals[k]:.3f}s') for k in range(0, Nt, step)]
        )]
    ),
    frames=frames
)

fig.show()
