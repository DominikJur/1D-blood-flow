"""
PINN training for 1D blood flow — supports all topologies.
Trains a separate PINN per topology using DeepXDE + PyTorch.

Usage:
  uv run train_pinn.py --train                  # train all topologies
  uv run train_pinn.py --train --topo y_bifurcation  # train one
  uv run train_pinn.py --topo straight_tube      # eval only
"""
import numpy as np
import pandas as pd
import deepxde as dde
import torch
import os, argparse
import pyvista as pv

from topology_config import (get_all_topology_names, get_topology,
                              compute_vessel_params, rho, KR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
GIFS_DIR = os.path.join(BASE_DIR, "gifs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
for d in [MODELS_DIR, DATA_DIR, GIFS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

dde.config.set_default_float("float64")
torch.set_default_dtype(torch.float64)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--topo", default=None, help="Topology name (default: all)")
    return p.parse_args()

args = parse_args()

# ============================================================
# TORCH HELPERS
# ============================================================
def safe_sqrt(x, eps=1e-12):
    return torch.sqrt(torch.clamp(x, min=eps))

def safe_pow_025(x, eps=1e-12):
    return torch.pow(torch.clamp(x, min=eps), 0.25)

def t_pressure(A, A0, beta):
    bt = torch.as_tensor(beta, dtype=A.dtype, device=A.device)
    return bt * (safe_sqrt(A) - np.sqrt(A0))

def t_wave_speed(A, A0, beta):
    bt = torch.as_tensor(beta, dtype=A.dtype, device=A.device)
    return torch.sqrt(bt / (2.0 * rho)) * safe_pow_025(A)

def t_total_pressure(A, Q, A0, beta):
    Ac = torch.clamp(A, min=1e-10)
    return t_pressure(A, A0, beta) + 0.5 * rho * (Q / Ac)**2

def t_riemann_W2(A, Q, A0, beta):
    Ac = torch.clamp(A, min=1e-10)
    return Q/Ac - 4.0 * t_wave_speed(A, A0, beta)


# ============================================================
# PER-TOPOLOGY PINN
# ============================================================
def run_pinn_for_topology(topo_name, train=False):
    print(f"\n{'='*60}")
    print(f"PINN: {topo_name} ({'TRAIN' if train else 'EVAL'})")
    print(f"{'='*60}")

    cfg = get_topology(topo_name)
    vessel_cfgs = cfg["vessels"]
    n_vessels = len(vessel_cfgs)
    junctions = cfg["junctions"]
    inlet_idx = cfg["inlet_vessel_idx"]
    outlet_idxs = cfg["outlet_vessel_indices"]
    inflow_func = cfg["inflow_func"]
    T_out = cfg["T_out"]

    # Compute vessel params
    vparams = [compute_vessel_params(vc) for vc in vessel_cfgs]
    Q_peak = 5e-5
    T_scale = T_out
    A_scale = max(vp["A0"] for vp in vparams)
    Q_scale = Q_peak
    P_scale = rho * vparams[0]["c0"]**2
    c0_val = vparams[0]["c0"]
    input_dim = 2 + n_vessels  # t, z, one-hot

    # Load data
    data_path = os.path.join(DATA_DIR, f"{topo_name}_blood_flow_data.csv")
    print(f"Loading {data_path}...")
    df = pd.read_csv(data_path)

    # Map vessel names to indices
    vname_to_idx = {vparams[i]["name"]: i for i in range(n_vessels)}
    df["vessel_id"] = df["vessel"].map(vname_to_idx)
    df["z_norm"] = 0.0
    for i, vp in enumerate(vparams):
        mask = df["vessel_id"] == i
        df.loc[mask, "z_norm"] = df.loc[mask, "z"].values / vp["L"]

    # Subsample
    subsample = 50
    df_train = df.iloc[::subsample].reset_index(drop=True)
    print(f"  {len(df)} total -> {len(df_train)} train points")

    t_obs = df_train["t"].values / T_scale
    z_obs = df_train["z_norm"].values
    vid = df_train["vessel_id"].values.astype(int)
    A_obs = df_train["A"].values
    Q_obs = df_train["Q"].values

    onehot = np.zeros((len(vid), n_vessels))
    onehot[np.arange(len(vid)), vid] = 1.0
    X_obs = np.column_stack([t_obs, z_obs, onehot]).astype(np.float64)
    y_obs = np.column_stack([A_obs/A_scale, Q_obs/Q_scale]).astype(np.float64)

    # BC time points
    N_bc = 200
    t_bc = np.linspace(0, T_out, N_bc).astype(np.float64)
    t_bc_n = t_bc / T_scale

    def make_onehot_col(vi):
        oh = np.zeros((N_bc, n_vessels))
        oh[:, vi] = 1.0
        return oh

    def make_X(z_val, vi):
        return np.column_stack([t_bc_n, np.full(N_bc, z_val), make_onehot_col(vi)]).astype(np.float64)

    # Inlet BC
    X_inlet = make_X(0.0, inlet_idx)
    Q_inlet_target = np.array([[inflow_func(t)] for t in t_bc], dtype=np.float64)

    # Outlet BCs
    X_outlets = {oi: make_X(1.0, oi) for oi in outlet_idxs}

    # Junction BCs
    junc_data = []
    for junc in junctions:
        if junc["type"] == "bifurcation":
            pi = junc["parent_idx"]
            di1, di2 = junc["daughter_indices"]
            junc_data.append(("bif", make_X(1.0, pi), make_X(0.0, di1), make_X(0.0, di2),
                              pi, di1, di2))
        elif junc["type"] == "merge":
            pi1, pi2 = junc["parent_indices"]
            di = junc["daughter_idx"]
            junc_data.append(("merge", make_X(1.0, pi1), make_X(1.0, pi2), make_X(0.0, di),
                              pi1, pi2, di))
        elif junc["type"] == "conjunction":
            pi = junc["parent_idx"]
            di = junc["daughter_idx"]
            junc_data.append(("conj", make_X(1.0, pi), make_X(0.0, di), None,
                              pi, di, None))

    # Build network
    net = dde.nn.FNN([input_dim, 128, 128, 128, 128, 2], "tanh", "Glorot normal")

    def output_transform(x, y):
        return torch.cat([torch.nn.functional.softplus(y[:,0:1]) + 1e-4, y[:,1:2]], dim=1)
    net.apply_output_transform(output_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(MODELS_DIR, f"{topo_name}_pinn.pth")

    def denorm(y_hat):
        return y_hat[:,0:1]*A_scale, y_hat[:,1:2]*Q_scale

    if train:
        # Supervised pre-training
        data_sup = dde.data.DataSet(X_train=X_obs, y_train=y_obs,
                                     X_test=X_obs[:1000], y_test=y_obs[:1000], standardize=False)
        model_sup = dde.Model(data_sup, net)
        print("\nStep 1: Supervised pre-training")
        model_sup.compile("adam", lr=1e-3, loss="MSE")
        model_sup.train(iterations=15000, display_every=3000)
        model_sup.compile("L-BFGS", loss="MSE")
        model_sup.train(iterations=15000, display_every=3000)

        # Fine-tuning with physics + BC
        print(f"\nStep 2: Fine-tuning with PDE + BC (device: {device})")
        N_colloc = 20000
        n_each = N_colloc // n_vessels

        X_colloc_parts = []
        for vi in range(n_vessels):
            n_s = n_each if vi < n_vessels-1 else N_colloc - n_each*(n_vessels-1)
            t_c = np.random.uniform(0, 1, (n_s, 1))
            z_c = np.random.uniform(0, 1, (n_s, 1))
            oh = np.zeros((n_s, n_vessels)); oh[:, vi] = 1.0
            X_colloc_parts.append(np.hstack([t_c, z_c, oh]))
        X_colloc = torch.tensor(np.vstack(X_colloc_parts), dtype=torch.float64,
                                device=device, requires_grad=True)

        X_obs_t = torch.tensor(X_obs, dtype=torch.float64, device=device)
        y_obs_t = torch.tensor(y_obs, dtype=torch.float64, device=device)
        X_inlet_t = torch.tensor(X_inlet, dtype=torch.float64, device=device)
        Q_inlet_t = torch.tensor(Q_inlet_target, dtype=torch.float64, device=device)

        X_outlets_t = {k: torch.tensor(v, dtype=torch.float64, device=device) for k,v in X_outlets.items()}
        junc_t = []
        for jd in junc_data:
            junc_t.append((jd[0],
                           torch.tensor(jd[1], dtype=torch.float64, device=device),
                           torch.tensor(jd[2], dtype=torch.float64, device=device),
                           torch.tensor(jd[3], dtype=torch.float64, device=device) if jd[3] is not None else None,
                           *jd[4:]))

        pinn_net = net.to(device)
        pinn_net.train()

        def pde_loss(x_c, net_m):
            if not x_c.requires_grad: x_c.requires_grad_(True)
            y = net_m(x_c)
            Ah, Qh = y[:,0:1], y[:,1:2]
            A = Ah * A_scale; Q = Qh * Q_scale
            Ah_t = torch.autograd.grad(Ah.sum(), x_c, create_graph=True)[0][:,0:1]
            Qh_t = torch.autograd.grad(Qh.sum(), x_c, create_graph=True)[0][:,0:1]
            Ah_z = torch.autograd.grad(Ah.sum(), x_c, create_graph=True)[0][:,1:2]
            Qh_z = torch.autograd.grad(Qh.sum(), x_c, create_graph=True)[0][:,1:2]

            oh = x_c[:, 2:2+n_vessels]
            beta_v = sum(vparams[i]["beta"] * oh[:,i:i+1] for i in range(n_vessels)).detach()
            L_v = sum(vparams[i]["L"] * oh[:,i:i+1] for i in range(n_vessels)).detach()

            A_t = (A_scale/T_scale)*Ah_t; Q_t = (Q_scale/T_scale)*Qh_t
            A_z = (A_scale/L_v)*Ah_z; Q_z = (Q_scale/L_v)*Qh_z

            F_A = A_t + Q_z
            Ac = torch.clamp(A, min=1e-8)
            u = Q/Ac
            dF2 = 2*u*Q_z - u**2*A_z
            dPdz = beta_v/(2*safe_sqrt(Ac))*A_z
            F_Q = Q_t + dF2 + Ac/rho*dPdz + KR*u

            sA = Q_scale / vparams[0]["L"]
            sQ = Q_scale / T_scale
            return torch.mean((F_A/sA)**2) + torch.mean((F_Q/sQ)**2)

        def bc_loss(net_m):
            loss = torch.tensor(0.0, device=device, dtype=torch.float64)
            # Inlet
            yi = net_m(X_inlet_t); _, Qi = denorm(yi)
            loss = loss + torch.mean(((Qi - Q_inlet_t)/Q_scale)**2)
            # Outlets (non-reflective)
            for oi in outlet_idxs:
                yo = net_m(X_outlets_t[oi]); Ao, Qo = denorm(yo)
                W2 = t_riemann_W2(Ao, Qo, vparams[oi]["A0"], vparams[oi]["beta"])
                W2_0 = -4.0 * np.sqrt(vparams[oi]["beta"]/(2*rho)) * vparams[oi]["A0"]**0.25
                loss = loss + torch.mean(((W2 - W2_0)/c0_val)**2)
            # Junctions
            for jd in junc_t:
                if jd[0] == "bif":
                    y1 = net_m(jd[1]); y2 = net_m(jd[2]); y3 = net_m(jd[3])
                    A1,Q1 = denorm(y1); A2,Q2 = denorm(y2); A3,Q3 = denorm(y3)
                    loss = loss + torch.mean(((Q1-Q2-Q3)/Q_scale)**2)
                    Pt1 = t_total_pressure(A1,Q1,vparams[jd[4]]["A0"],vparams[jd[4]]["beta"])
                    Pt2 = t_total_pressure(A2,Q2,vparams[jd[5]]["A0"],vparams[jd[5]]["beta"])
                    Pt3 = t_total_pressure(A3,Q3,vparams[jd[6]]["A0"],vparams[jd[6]]["beta"])
                    loss = loss + torch.mean(((Pt1-Pt2)/P_scale)**2)
                    loss = loss + torch.mean(((Pt1-Pt3)/P_scale)**2)
                elif jd[0] == "merge":
                    y1 = net_m(jd[1]); y2 = net_m(jd[2]); y3 = net_m(jd[3])
                    A1,Q1 = denorm(y1); A2,Q2 = denorm(y2); A3,Q3 = denorm(y3)
                    loss = loss + torch.mean(((Q1+Q2-Q3)/Q_scale)**2)
                    Pt1 = t_total_pressure(A1,Q1,vparams[jd[4]]["A0"],vparams[jd[4]]["beta"])
                    Pt2 = t_total_pressure(A2,Q2,vparams[jd[5]]["A0"],vparams[jd[5]]["beta"])
                    Pt3 = t_total_pressure(A3,Q3,vparams[jd[6]]["A0"],vparams[jd[6]]["beta"])
                    loss = loss + torch.mean(((Pt1-Pt3)/P_scale)**2)
                    loss = loss + torch.mean(((Pt2-Pt3)/P_scale)**2)
                elif jd[0] == "conj":
                    y1 = net_m(jd[1]); y2 = net_m(jd[2])
                    A1,Q1 = denorm(y1); A2,Q2 = denorm(y2)
                    loss = loss + torch.mean(((Q1-Q2)/Q_scale)**2)
                    Pt1 = t_total_pressure(A1,Q1,vparams[jd[4]]["A0"],vparams[jd[4]]["beta"])
                    Pt2 = t_total_pressure(A2,Q2,vparams[jd[5]]["A0"],vparams[jd[5]]["beta"])
                    loss = loss + torch.mean(((Pt1-Pt2)/P_scale)**2)
            return loss

        optimizer = torch.optim.Adam(pinn_net.parameters(), lr=1e-4)
        N_iter = 20000
        for it in range(N_iter):
            optimizer.zero_grad()
            ld = torch.mean((pinn_net(X_obs_t) - y_obs_t)**2)
            lp = pde_loss(X_colloc, pinn_net)
            lb = bc_loss(pinn_net)
            lt = ld + lp + lb
            if torch.isnan(lt):
                print(f"NaN at iter {it+1}, stopping Adam."); break
            lt.backward()
            torch.nn.utils.clip_grad_norm_(pinn_net.parameters(), 1.0)
            optimizer.step()
            if (it+1) % 2000 == 0:
                print(f"  iter {it+1}/{N_iter}: total={lt.item():.4e} data={ld.item():.4e} "
                      f"pde={lp.item():.4e} bc={lb.item():.4e}")

        print("  L-BFGS fine-tuning...")
        lbfgs = torch.optim.LBFGS(pinn_net.parameters(), max_iter=5000, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            ld = torch.mean((pinn_net(X_obs_t)-y_obs_t)**2)
            lp = pde_loss(X_colloc, pinn_net)
            lb = bc_loss(pinn_net)
            lt = ld + lp + lb
            if not torch.isnan(lt): lt.backward()
            return lt
        try: lbfgs.step(closure)
        except Exception as e: print(f"  L-BFGS error: {e}")

        sd = {k: v.cpu() for k, v in pinn_net.state_dict().items()}
        torch.save(sd, model_path)
        print(f"  Model saved: {model_path}")
    else:
        pinn_net = net.to(device)
        pinn_net.load_state_dict(torch.load(model_path, map_location="cpu"))
        pinn_net.eval()
        print(f"  Loaded: {model_path}")

    # ============================================================
    # PREDICTION & GIF
    # ============================================================
    print("Generating predictions...")
    pinn_net.eval()
    pred_dev = next(pinn_net.parameters()).device

    c0 = vparams[0]["c0"]
    dx_min = min(vp["L"]/vp["N"] for vp in vparams)
    dt_sim = 0.3 * dx_min / c0
    Nt_sim = int(np.ceil(T_out / dt_sim))
    step_sim = max(1, Nt_sim // 200)
    frame_indices = list(range(0, Nt_sim, step_sim))
    t_grid = np.linspace(0, T_out, Nt_sim)[frame_indices]
    Nt_g = len(t_grid)
    # Predict for each vessel
    A_pred = {vi: np.zeros((Nt_g, vparams[vi]["N"])) for vi in range(n_vessels)}
    Q_pred = {vi: np.zeros((Nt_g, vparams[vi]["N"])) for vi in range(n_vessels)}
    rows = []

    for i, tv in enumerate(t_grid):
        for vi in range(n_vessels):
            z_nv = np.linspace(0, 1, vparams[vi]["N"])
            for j, zv in enumerate(z_nv):
                oh = [0]*n_vessels; oh[vi] = 1
                x_in = np.array([[tv/T_scale, zv] + oh], dtype=np.float64)
                with torch.no_grad():
                    y = pinn_net(torch.tensor(x_in, dtype=torch.float64, device=pred_dev)).cpu().numpy()
                Av = float(y[0,0]*A_scale); Qv = float(y[0,1]*Q_scale)
                A_pred[vi][i,j] = Av; Q_pred[vi][i,j] = Qv
                rows.append([tv, zv*vparams[vi]["L"], vparams[vi]["name"], Av, Qv])
        if (i+1) % 25 == 0: print(f"  Pred: {i+1}/{Nt_g}")

    df_pred = pd.DataFrame(rows, columns=["t","z","vessel","A","Q"])
    df_pred.to_csv(os.path.join(DATA_DIR, f"{topo_name}_pinn_prediction.csv"), index=False)

    # GIF
    print("Generating PINN GIF...")
    N_theta = 20
    theta_vals = np.linspace(0, 2*np.pi, N_theta, endpoint=True)
    deformation_scale = 15.0

    for vc in vessel_cfgs:
        vc["dx"] = vc["L"] / vc["N"]
    coords = cfg["layout_func"](vessel_cfgs)

    Qmax = max(np.max(np.abs(Q_pred[vi])) for vi in range(n_vessels))
    Qmax = max(Qmax, 1e-10) * 1e6

    def build_grid(xc, yc, zc, Rv):
        Nz = len(xc)
        dx = np.gradient(xc); dy = np.gradient(yc); dz = np.gradient(zc)
        pts = np.zeros((N_theta, Nz, 3))
        for ii in range(Nz):
            t = np.array([dx[ii], dy[ii], dz[ii]])
            tn = np.linalg.norm(t)
            t = t/tn if tn > 1e-15 else np.array([1,0,0])
            u = np.cross(t, [1,0,0]) if abs(t[0])<0.9 else np.cross(t, [0,1,0])
            un = np.linalg.norm(u); u = u/un if un>1e-15 else np.array([0,1,0])
            v = np.cross(t, u)
            c = np.array([xc[ii], yc[ii], zc[ii]])
            for jj in range(N_theta):
                pts[jj,ii] = c + Rv[ii]*(np.cos(theta_vals[jj])*u + np.sin(theta_vals[jj])*v)
        return pv.StructuredGrid(pts[:,:,0], pts[:,:,1], pts[:,:,2])

    all_grids = {vi: [] for vi in range(n_vessels)}
    for idx in range(Nt_g):
        for vi in range(n_vessels):
            R0 = np.sqrt(vparams[vi]["A0"]/np.pi)
            Af = np.maximum(A_pred[vi][idx], 1e-12)
            Rf = R0 + (np.sqrt(Af/np.pi) - R0) * deformation_scale
            xc, yc, zc = coords[vi]
            g = build_grid(xc, yc, zc, Rf)
            g.point_data['Q'] = np.tile(Q_pred[vi][idx]*1e6, (N_theta,1)).ravel(order='F')
            all_grids[vi].append(g)

    gif_path = os.path.join(GIFS_DIR, f"{topo_name}_pinn_prediction.gif")
    pv.set_plot_theme('dark')
    p = pv.Plotter(window_size=[1200,800], off_screen=True)
    all_x = np.concatenate([coords[vi][0] for vi in range(n_vessels)])
    all_y = np.concatenate([coords[vi][1] for vi in range(n_vessels)])
    all_z = np.concatenate([coords[vi][2] for vi in range(n_vessels)])
    cx,cy,cz = np.mean(all_x), np.mean(all_y), np.mean(all_z)
    ext = max(np.ptp(all_x), np.ptp(all_y), np.ptp(all_z), 0.01)
    p.camera_position = [(cx+ext*1.5, cy-ext*1.2, cz+ext*0.8), (cx,cy,cz), (0,0,1)]
    p.enable_anti_aliasing(); p.add_axes()
    p.open_gif(gif_path, fps=25)
    for fi in range(Nt_g):
        p.clear()
        for vi in range(n_vessels):
            p.add_mesh(all_grids[vi][fi], scalars='Q', clim=[0, Qmax],
                       cmap='plasma', show_edges=False, lighting=True, smooth_shading=True)
        p.add_text(f"Time: {t_grid[fi]:.3f} s  [{topo_name} PINN]",
                   position='upper_left', font_size=14, color='white')
        p.write_frame()
    p.close()
    print(f"Saved {gif_path}")


if __name__ == "__main__":
    topos = [args.topo] if args.topo else get_all_topology_names()
    for tn in topos:
        run_pinn_for_topology(tn, train=args.train)
    print("\nAll done!")