"""
Shared topology configurations for 1D blood flow simulation.
Defines 4 vascular topologies with vessel geometry, junctions,
inflow functions, and 3D layout for visualization.

Used by both solver.py and train_pinn.py.
"""

import numpy as np


# ============================================================
# PHYSICAL CONSTANTS (shared between solver and PINN)
# ============================================================
rho = 1060.0        # Blood density [kg/m^3]
mu = 4.0e-3         # Blood viscosity [Pa·s]
KR = 8.0 * np.pi * mu / rho  # Friction coefficient (Poiseuille, gamma=2)
CFL = 0.30          # CFL number for Adams-Bashforth stability


# ============================================================
# INFLOW FUNCTIONS
# ============================================================
def inflow_double_sine(t):
    """Double sine pulse — two pulses at t=0 and t=0.1."""
    T_pulse = 0.02
    Q_peak = 5e-5
    if t <= T_pulse:
        return Q_peak * np.sin(np.pi * t / T_pulse)
    elif 0.1 <= t <= 0.1 + T_pulse:
        return Q_peak * np.sin(np.pi * (t - 0.1) / T_pulse)
    else:
        return 0.0


def inflow_single_sharp(t):
    """Single sharp sine pulse."""
    T_pulse = 0.02
    Q_peak = 5e-5
    if t <= T_pulse:
        return Q_peak * np.sin(np.pi * t / T_pulse)
    else:
        return 0.0


def inflow_gaussian(t):
    """Gaussian pulse centered at t=0.02 with width 0.008."""
    Q_peak = 5e-5
    t_center = 0.02
    sigma = 0.008
    return Q_peak * np.exp(-((t - t_center) / sigma) ** 2)


def inflow_multi_harmonic(t):
    """Multi-harmonic pulse: sum of two harmonics."""
    T_pulse = 0.04
    Q_peak = 3e-5
    if t <= T_pulse:
        return Q_peak * (np.sin(2 * np.pi * t / T_pulse)
                         + 0.5 * np.sin(4 * np.pi * t / T_pulse))
    else:
        return 0.0


# ============================================================
# 3D LAYOUT FUNCTIONS
# Each returns dict: vessel_idx -> (x_coords, y_coords, z_coords)
# ============================================================
def layout_straight_tube(vessels):
    """Single straight tube along x-axis."""
    v = vessels[0]
    x = np.linspace(v["dx"] / 2, v["L"] - v["dx"] / 2, v["N"])
    y = np.zeros_like(x)
    z = np.zeros_like(x)
    return {0: (x, y, z)}


def layout_y_bifurcation(vessels):
    """Y-bifurcation: parent along x, daughters branch at ±30 deg."""
    bif_angle = np.deg2rad(30)
    vp = vessels[0]
    vd1 = vessels[1]
    vd2 = vessels[2]

    # Parent along x-axis
    x_p = np.linspace(vp["dx"] / 2, vp["L"] - vp["dx"] / 2, vp["N"])
    y_p = np.zeros_like(x_p)
    z_p = np.zeros_like(x_p)

    # Daughter 1: branches up (y direction)
    s_d1 = np.linspace(0, vd1["L"], vd1["N"])
    x_d1 = vp["L"] + s_d1 * np.cos(bif_angle)
    y_d1 = s_d1 * np.sin(bif_angle)
    z_d1 = np.zeros_like(s_d1)

    # Daughter 2: branches down (z direction)
    s_d2 = np.linspace(0, vd2["L"], vd2["N"])
    x_d2 = vp["L"] + s_d2 * np.cos(bif_angle)
    y_d2 = np.zeros_like(s_d2)
    z_d2 = -s_d2 * np.sin(bif_angle)

    return {0: (x_p, y_p, z_p),
            1: (x_d1, y_d1, z_d1),
            2: (x_d2, y_d2, z_d2)}


def layout_split_merge(vessels):
    """Split-merge: parent -> bifurcation -> 2 outward branches -> turns ->
    2 inward branches -> merge -> merge vessel.
    Daughters arc outward then turn back via conjunction junctions.
    """
    vp = vessels[0]      # parent
    vb1_out = vessels[1]  # branch 1 outward
    vb2_out = vessels[2]  # branch 2 outward
    vb1_in = vessels[3]   # branch 1 inward (after turn)
    vb2_in = vessels[4]   # branch 2 inward (after turn)
    vm = vessels[5]       # merge vessel

    bif_angle = np.deg2rad(30)
    arc_half_span = 0.035  # half the horizontal span per branch segment

    # Parent along x-axis
    x_p = np.linspace(vp["dx"]/2, vp["L"] - vp["dx"]/2, vp["N"])
    y_p = np.zeros_like(x_p)
    z_p = np.zeros_like(x_p)

    # Branch 1 outward: goes up-right at bif_angle
    s1 = np.linspace(0, vb1_out["L"], vb1_out["N"])
    x_b1o = vp["L"] + s1 * np.cos(bif_angle)
    y_b1o = s1 * np.sin(bif_angle)
    z_b1o = np.zeros_like(s1)

    # Branch 2 outward: goes down-right at -bif_angle (in z)
    s2 = np.linspace(0, vb2_out["L"], vb2_out["N"])
    x_b2o = vp["L"] + s2 * np.cos(bif_angle)
    y_b2o = np.zeros_like(s2)
    z_b2o = -s2 * np.sin(bif_angle)

    # Turn points (end of outward branches)
    turn1 = np.array([x_b1o[-1], y_b1o[-1], z_b1o[-1]])
    turn2 = np.array([x_b2o[-1], y_b2o[-1], z_b2o[-1]])

    # Merge point: same x as turns, y=0, z=0
    merge_x = turn1[0] + vb1_in["L"] * np.cos(bif_angle)

    # Branch 1 inward: turns back from turn1 toward merge point
    s1i = np.linspace(0, vb1_in["L"], vb1_in["N"])
    x_b1i = turn1[0] + s1i * np.cos(bif_angle)
    y_b1i = turn1[1] - s1i * np.sin(bif_angle)
    z_b1i = np.zeros_like(s1i)

    # Branch 2 inward: turns back from turn2 toward merge point
    s2i = np.linspace(0, vb2_in["L"], vb2_in["N"])
    x_b2i = turn2[0] + s2i * np.cos(bif_angle)
    y_b2i = np.zeros_like(s2i)
    z_b2i = turn2[2] + s2i * np.sin(bif_angle)

    # Merge vessel continues straight from convergence point
    x_m = np.linspace(merge_x, merge_x + vm["L"], vm["N"])
    y_m = np.zeros_like(x_m)
    z_m = np.zeros_like(x_m)

    return {0: (x_p, y_p, z_p),
            1: (x_b1o, y_b1o, z_b1o),
            2: (x_b2o, y_b2o, z_b2o),
            3: (x_b1i, y_b1i, z_b1i),
            4: (x_b2i, y_b2i, z_b2i),
            5: (x_m, y_m, z_m)}


def layout_double_y(vessels):
    """Double-Y: parent -> 2 daughters -> each daughter splits into 2 granddaughters.
    Vessels: 0=parent, 1=d1, 2=d2, 3=gd1a, 4=gd1b, 5=gd2a, 6=gd2b
    """
    bif_angle1 = np.deg2rad(30)
    bif_angle2 = np.deg2rad(20)
    vp = vessels[0]
    vd1 = vessels[1]
    vd2 = vessels[2]
    vgd1a = vessels[3]
    vgd1b = vessels[4]
    vgd2a = vessels[5]
    vgd2b = vessels[6]

    # Parent along x-axis
    x_p = np.linspace(vp["dx"] / 2, vp["L"] - vp["dx"] / 2, vp["N"])
    y_p = np.zeros_like(x_p)
    z_p = np.zeros_like(x_p)

    # Daughter 1: branches up
    s_d1 = np.linspace(0, vd1["L"], vd1["N"])
    dir_d1 = np.array([np.cos(bif_angle1), np.sin(bif_angle1), 0.0])
    x_d1 = vp["L"] + s_d1 * dir_d1[0]
    y_d1 = s_d1 * dir_d1[1]
    z_d1 = np.zeros_like(s_d1)

    # Daughter 2: branches down (in z)
    s_d2 = np.linspace(0, vd2["L"], vd2["N"])
    dir_d2 = np.array([np.cos(bif_angle1), 0.0, -np.sin(bif_angle1)])
    x_d2 = vp["L"] + s_d2 * dir_d2[0]
    y_d2 = np.zeros_like(s_d2)
    z_d2 = s_d2 * dir_d2[2]

    # Granddaughters of D1 — branch in y-z plane from end of D1
    d1_end = np.array([x_d1[-1], y_d1[-1], z_d1[-1]])

    s_gd1a = np.linspace(0, vgd1a["L"], vgd1a["N"])
    dir_gd1a = np.array([np.cos(bif_angle1 + bif_angle2),
                          np.sin(bif_angle1 + bif_angle2), 0.0])
    x_gd1a = d1_end[0] + s_gd1a * dir_gd1a[0]
    y_gd1a = d1_end[1] + s_gd1a * dir_gd1a[1]
    z_gd1a = np.zeros_like(s_gd1a)

    s_gd1b = np.linspace(0, vgd1b["L"], vgd1b["N"])
    dir_gd1b = np.array([np.cos(bif_angle1 - bif_angle2),
                          np.sin(bif_angle1 - bif_angle2), 0.0])
    x_gd1b = d1_end[0] + s_gd1b * dir_gd1b[0]
    y_gd1b = d1_end[1] + s_gd1b * dir_gd1b[1]
    z_gd1b = np.zeros_like(s_gd1b)

    # Granddaughters of D2 — branch in x-z plane from end of D2
    d2_end = np.array([x_d2[-1], y_d2[-1], z_d2[-1]])

    s_gd2a = np.linspace(0, vgd2a["L"], vgd2a["N"])
    dir_gd2a = np.array([np.cos(bif_angle1), 0.0,
                          -np.sin(bif_angle1 + bif_angle2)])
    x_gd2a = d2_end[0] + s_gd2a * dir_gd2a[0]
    y_gd2a = np.zeros_like(s_gd2a)
    z_gd2a = d2_end[2] + s_gd2a * dir_gd2a[2]

    s_gd2b = np.linspace(0, vgd2b["L"], vgd2b["N"])
    dir_gd2b = np.array([np.cos(bif_angle1), 0.0,
                          -np.sin(bif_angle1 - bif_angle2)])
    x_gd2b = d2_end[0] + s_gd2b * dir_gd2b[0]
    y_gd2b = np.zeros_like(s_gd2b)
    z_gd2b = d2_end[2] + s_gd2b * dir_gd2b[2]

    return {
        0: (x_p, y_p, z_p),
        1: (x_d1, y_d1, z_d1),
        2: (x_d2, y_d2, z_d2),
        3: (x_gd1a, y_gd1a, z_gd1a),
        4: (x_gd1b, y_gd1b, z_gd1b),
        5: (x_gd2a, y_gd2a, z_gd2a),
        6: (x_gd2b, y_gd2b, z_gd2b),
    }


# ============================================================
# TOPOLOGY DEFINITIONS
# ============================================================
c0_default = 5.0  # Reference wave speed [m/s]

TOPOLOGIES = {
    # ----------------------------------------------------------
    # 1. Straight tube — single vessel, no junctions
    # ----------------------------------------------------------
    "straight_tube": {
        "vessels": [
            {"name": "Tube", "L": 0.20, "r": 0.005, "c0": c0_default, "N": 100},
        ],
        "junctions": [],
        "inlet_vessel_idx": 0,
        "outlet_vessel_indices": [0],
        "inflow_func": inflow_double_sine,
        "T_out": 0.75,
        "layout_func": layout_straight_tube,
    },

    # ----------------------------------------------------------
    # 2. Y-bifurcation — parent + 2 daughters (existing topology)
    # ----------------------------------------------------------
    "y_bifurcation": {
        "vessels": [
            {"name": "Parent (Aorta)", "L": 0.10, "r": 0.005, "c0": c0_default, "N": 60},
            {"name": "Daughter 1", "L": 0.08, "r": 0.0035, "c0": c0_default, "N": 48},
            {"name": "Daughter 2", "L": 0.09, "r": 0.0040, "c0": c0_default, "N": 54},
        ],
        "junctions": [
            {"type": "bifurcation", "parent_idx": 0, "daughter_indices": [1, 2]},
        ],
        "inlet_vessel_idx": 0,
        "outlet_vessel_indices": [1, 2],
        "inflow_func": inflow_single_sharp,
        "T_out": 0.75,
        "layout_func": layout_y_bifurcation,
    },

    # ----------------------------------------------------------
    # 3. Split-merge — parent -> 2 daughters -> merge vessel
    # ----------------------------------------------------------
    "split_merge": {
        "vessels": [
            {"name": "Parent", "L": 0.08, "r": 0.005, "c0": c0_default, "N": 48},
            {"name": "Branch 1 Out", "L": 0.04, "r": 0.0035, "c0": c0_default, "N": 24},
            {"name": "Branch 2 Out", "L": 0.04, "r": 0.0040, "c0": c0_default, "N": 24},
            {"name": "Branch 1 In", "L": 0.04, "r": 0.0035, "c0": c0_default, "N": 24},
            {"name": "Branch 2 In", "L": 0.04, "r": 0.0040, "c0": c0_default, "N": 24},
            {"name": "Merge", "L": 0.08, "r": 0.0048, "c0": c0_default, "N": 48},
        ],
        "junctions": [
            {"type": "bifurcation", "parent_idx": 0, "daughter_indices": [1, 2]},
            {"type": "conjunction", "parent_idx": 1, "daughter_idx": 3},
            {"type": "conjunction", "parent_idx": 2, "daughter_idx": 4},
            {"type": "merge", "parent_indices": [3, 4], "daughter_idx": 5},
        ],
        "inlet_vessel_idx": 0,
        "outlet_vessel_indices": [5],
        "inflow_func": inflow_gaussian,
        "T_out": 0.75,
        "layout_func": layout_split_merge,
    },

    # ----------------------------------------------------------
    # 4. Double-Y — parent -> 2 daughters -> 4 granddaughters
    # ----------------------------------------------------------
    "double_y": {
        "vessels": [
            {"name": "Parent (Aorta)", "L": 0.10, "r": 0.005, "c0": c0_default, "N": 60},
            {"name": "Daughter 1", "L": 0.06, "r": 0.0038, "c0": c0_default, "N": 36},
            {"name": "Daughter 2", "L": 0.06, "r": 0.0040, "c0": c0_default, "N": 36},
            {"name": "Granddaughter 1a", "L": 0.05, "r": 0.0025, "c0": c0_default, "N": 30},
            {"name": "Granddaughter 1b", "L": 0.05, "r": 0.0028, "c0": c0_default, "N": 30},
            {"name": "Granddaughter 2a", "L": 0.05, "r": 0.0027, "c0": c0_default, "N": 30},
            {"name": "Granddaughter 2b", "L": 0.05, "r": 0.0030, "c0": c0_default, "N": 30},
        ],
        "junctions": [
            {"type": "bifurcation", "parent_idx": 0, "daughter_indices": [1, 2]},
            {"type": "bifurcation", "parent_idx": 1, "daughter_indices": [3, 4]},
            {"type": "bifurcation", "parent_idx": 2, "daughter_indices": [5, 6]},
        ],
        "inlet_vessel_idx": 0,
        "outlet_vessel_indices": [3, 4, 5, 6],
        "inflow_func": inflow_multi_harmonic,
        "T_out": 0.75,
        "layout_func": layout_double_y,
    },
}


def get_topology(name):
    """Get a topology config by name."""
    if name not in TOPOLOGIES:
        raise ValueError(f"Unknown topology: {name}. "
                         f"Available: {list(TOPOLOGIES.keys())}")
    return TOPOLOGIES[name]


def get_all_topology_names():
    """Get list of all topology names."""
    return list(TOPOLOGIES.keys())


def compute_vessel_params(vessel_cfg):
    """Compute derived vessel parameters from config dict.
    Returns dict with A0, beta, dx added.
    """
    r = vessel_cfg["r"]
    L = vessel_cfg["L"]
    N = vessel_cfg["N"]
    c0 = vessel_cfg["c0"]

    A0 = np.pi * r ** 2
    beta = 2.0 * rho * c0 ** 2 / np.sqrt(A0)
    dx = L / N

    return {
        **vessel_cfg,
        "A0": A0,
        "beta": beta,
        "dx": dx,
    }
