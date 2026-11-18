"""
Main solver script — runs MUSCL simulation for all topologies,
exports CSV data and 3D PyVista GIF for each one.
"""
import numpy as np
import pyvista as pv
import csv, os

from topology_config import get_all_topology_names, get_topology, rho
from simulation_engine import run_simulation

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
GIFS_DIR = os.path.join(BASE_DIR, "gifs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(GIFS_DIR, exist_ok=True)

N_theta = 20
theta_vals = np.linspace(0, 2*np.pi, N_theta, endpoint=True)
deformation_scale = 15.0


def build_vessel_grid(x_c, y_c, z_c, R_vals):
    N_z = len(x_c)
    dx = np.gradient(x_c); dy = np.gradient(y_c); dz = np.gradient(z_c)
    pts = np.zeros((N_theta, N_z, 3))
    for i in range(N_z):
        t = np.array([dx[i], dy[i], dz[i]])
        tn = np.linalg.norm(t)
        t = t/tn if tn > 1e-15 else np.array([1,0,0])
        u = np.cross(t, [1,0,0]) if abs(t[0]) < 0.9 else np.cross(t, [0,1,0])
        un = np.linalg.norm(u)
        u = u/un if un > 1e-15 else np.array([0,1,0])
        v = np.cross(t, u)
        center = np.array([x_c[i], y_c[i], z_c[i]])
        for j in range(N_theta):
            pts[j, i] = center + R_vals[i]*(np.cos(theta_vals[j])*u + np.sin(theta_vals[j])*v)
    return pv.StructuredGrid(pts[:,:,0], pts[:,:,1], pts[:,:,2])


def export_csv(topo_name, vessels, Nt, T_out):
    csv_path = os.path.join(DATA_DIR, f"{topo_name}_blood_flow_data.csv")
    time_vals = np.arange(Nt) * (T_out / Nt)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["t", "vessel", "z", "A", "Q"])
        for n in range(Nt):
            t = time_vals[n]
            for vi, v in enumerate(vessels):
                for i in range(v.N):
                    w.writerow([f"{t:.8e}", v.name, f"{v.x[i]:.8e}",
                                f"{v.A_hist[n][i]:.8e}", f"{v.Q_hist[n][i]:.8e}"])
    n_rows = sum(Nt * v.N for v in vessels)
    print(f"Saved {csv_path} ({n_rows} rows)")


def export_gif(topo_name, topo_cfg, vessels, Nt, T_out):
    layout = topo_cfg["layout_func"]
    vessel_cfgs = topo_cfg["vessels"]
    # Build vessel configs with dx for layout
    for i, vc in enumerate(vessel_cfgs):
        vc["dx"] = vessels[i].dx
    coords = layout(vessel_cfgs)

    step = max(1, Nt // 200)
    frame_indices = list(range(0, Nt, step))
    time_vals = np.arange(Nt) * (T_out / Nt)

    # Compute global Q max
    Qmax = 1e-10
    for v in vessels:
        for q in v.Q_hist:
            Qmax = max(Qmax, np.max(np.abs(q)))
    Qmax_ml = Qmax * 1e6

    # Pre-compute grids
    all_grids = {vi: [] for vi in range(len(vessels))}
    for idx in frame_indices:
        for vi, v in enumerate(vessels):
            A_f = np.array(v.A_hist[idx])
            Q_f = np.array(v.Q_hist[idx])
            R0 = np.sqrt(v.A0 / np.pi)
            R_f = R0 + (np.sqrt(A_f/np.pi) - R0) * deformation_scale
            xc, yc, zc = coords[vi]
            g = build_vessel_grid(xc, yc, zc, R_f)
            g.point_data['Q'] = np.tile(Q_f*1e6, (N_theta,1)).ravel(order='F')
            all_grids[vi].append(g)

    # Render GIF
    gif_path = os.path.join(GIFS_DIR, f"{topo_name}_blood_flow.gif")
    pv.set_plot_theme('dark')
    p = pv.Plotter(window_size=[1200, 800], off_screen=True)

    # Camera: center on midpoint of all vessel coords
    all_x = np.concatenate([coords[vi][0] for vi in range(len(vessels))])
    all_y = np.concatenate([coords[vi][1] for vi in range(len(vessels))])
    all_z = np.concatenate([coords[vi][2] for vi in range(len(vessels))])
    cx, cy, cz = np.mean(all_x), np.mean(all_y), np.mean(all_z)
    extent = max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z), 0.01)
    p.camera_position = [(cx + extent*1.5, cy - extent*1.2, cz + extent*0.8),
                         (cx, cy, cz), (0, 0, 1)]
    p.enable_anti_aliasing()
    p.add_axes()
    p.open_gif(gif_path, fps=25)

    for fi, frame_idx in enumerate(frame_indices):
        p.clear()
        for vi in range(len(vessels)):
            p.add_mesh(all_grids[vi][fi], scalars='Q', clim=[0, Qmax_ml],
                       cmap='plasma', show_edges=False, lighting=True, smooth_shading=True)
        p.add_text(f"Time: {time_vals[frame_idx]:.3f} s  [{topo_name}]",
                   position='upper_left', font_size=14, color='white')
        p.write_frame()

    p.close()
    print(f"Saved {gif_path} ({len(frame_indices)} frames)")


if __name__ == "__main__":
    for topo_name in get_all_topology_names():
        topo_cfg = get_topology(topo_name)
        vessels, dt, Nt = run_simulation(topo_name, topo_cfg)
        export_csv(topo_name, vessels, Nt, topo_cfg["T_out"])
        export_gif(topo_name, topo_cfg, vessels, Nt, topo_cfg["T_out"])
    print("\nAll simulations complete!")