
"""
Synthetic Lens Surface Deformation Simulation
---------------------------------------------
A compact toolkit to model a base lens surface and add deformations/noise,
then visualize as a 3D surface and a contour map, and export data/figures.

Usage (as a module):
    from lens_deformation_sim import simulate_and_plot, default_config
    simulate_and_plot(default_config())

Or run this file directly (it will save sample outputs to ./outputs):
    python lens_deformation_sim.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import os
import json

# ------------------------
# Config
# ------------------------
def default_config():
    return {
        "grid": {
            "side_mm": 50.0,       # physical side length (mm) for square aperture
            "points": 201          # resolution along one axis (odd recommended so center exists)
        },
        "base_surface": {
            # z = a*(x^2 + y^2)
            "a": 1.0e-3            # curvature coefficient (mm^-1); adjust to match desired sag
        },
        "deformations": [
            # Gaussian center "pit" or "bump"
            {"type": "gaussian", "amplitude": -0.2, "sigma_mm": 8.0, "x0_mm": 0.0, "y0_mm": 0.0},
            # Ring-like edge bulge (gaussian ring)
            {"type": "gaussian_ring", "amplitude": 0.15, "r0_mm": 18.0, "sigma_mm": 3.0},
            # Simple astigmatism term k*(x^2 - y^2)
            {"type": "astigmatism", "k": 5e-5},
            # Trefoil-like term k*r^3*cos(3θ) (very small by default)
            {"type": "trefoil", "k": 5e-7}
        ],
        "noise": {
            "measurement_rms": 0.01,   # RMS of additive white noise (mm)
            "random_seed": 42
        },
        "output": {
            "dir": "./outputs",
            "basename": "lens_sim"
        }
    }

# ------------------------
# Helpers
# ------------------------
def make_grid(side_mm: float, points: int):
    half = side_mm / 2.0
    x = np.linspace(-half, half, points)
    y = np.linspace(-half, half, points)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    return xx, yy, rr, theta

def base_paraboloid(xx, yy, a: float):
    return a * (xx**2 + yy**2)

def deform_gaussian(xx, yy, amp, sigma_mm, x0=0.0, y0=0.0):
    return amp * np.exp(-(((xx - x0)**2 + (yy - y0)**2) / (2.0 * sigma_mm**2)))

def deform_gaussian_ring(rr, amp, r0_mm, sigma_mm):
    # radial gaussian centered at r0 (ring-shaped when seen on a 2D surface)
    return amp * np.exp(-((rr - r0_mm)**2) / (2.0 * sigma_mm**2))

def deform_astigmatism(xx, yy, k):
    return k * (xx**2 - yy**2)

def deform_trefoil(rr, theta, k):
    # simple trefoil-like polynomial in polar coordinates: k * r^3 * cos(3θ)
    return k * (rr**3) * np.cos(3.0 * theta)

def add_deformations(xx, yy, rr, theta, cfg):
    dz_total = np.zeros_like(xx)
    for d in cfg.get("deformations", []):
        t = d.get("type", "").lower()
        if t == "gaussian":
            dz = deform_gaussian(xx, yy, d["amplitude"], d["sigma_mm"], d.get("x0_mm", 0.0), d.get("y0_mm", 0.0))
        elif t == "gaussian_ring":
            dz = deform_gaussian_ring(rr, d["amplitude"], d["r0_mm"], d["sigma_mm"])
        elif t == "astigmatism":
            dz = deform_astigmatism(xx, yy, d["k"])
        elif t == "trefoil":
            dz = deform_trefoil(rr, theta, d["k"])
        else:
            raise ValueError(f"Unknown deformation type: {t}")
        dz_total += dz
    return dz_total

def add_noise(z, rms, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, rms, size=z.shape)
    return z + noise

def metrics(z):
    pv = float(np.max(z) - np.min(z))  # Peak-to-Valley
    rms = float(np.sqrt(np.mean((z - np.mean(z))**2)))
    return {"PV": pv, "RMS": rms, "Mean": float(np.mean(z))}

# ------------------------
# Core simulation
# ------------------------
def simulate(cfg=None):
    if cfg is None:
        cfg = default_config()

    side = float(cfg["grid"]["side_mm"])
    n = int(cfg["grid"]["points"])
    xx, yy, rr, theta = make_grid(side, n)

    base = base_paraboloid(xx, yy, float(cfg["base_surface"]["a"]))
    dz = add_deformations(xx, yy, rr, theta, cfg)
    z = base + dz

    noise_cfg = cfg.get("noise", {})
    z_noisy = add_noise(z, float(noise_cfg.get("measurement_rms", 0.0)), noise_cfg.get("random_seed", None))

    return {"x": xx, "y": yy, "z_clean": z, "z_noisy": z_noisy, "base": base, "dz": dz, "cfg": cfg}

# ------------------------
# Visualization (each chart on its own figure, no custom colors)
# ------------------------
def plot_surface(x, y, z, title="Surface (3D)"):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, linewidth=0, antialiased=True)
    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("sag (mm)")
    plt.tight_layout()
    return fig

def plot_contour(x, y, z, title="Surface (Contour)"):
    fig = plt.figure(figsize=(6, 5))
    cs = plt.contourf(x, y, z)
    plt.colorbar(cs, shrink=0.85, pad=0.05)
    plt.title(title)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.tight_layout()
    return fig

# ------------------------
# Save utilities
# ------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def export_all(sim, out_dir, basename):
    ensure_dir(out_dir)

    # Save numeric arrays
    np.save(os.path.join(out_dir, f"{basename}_x.npy"), sim["x"])
    np.save(os.path.join(out_dir, f"{basename}_y.npy"), sim["y"])
    np.save(os.path.join(out_dir, f"{basename}_z_clean.npy"), sim["z_clean"])
    np.save(os.path.join(out_dir, f"{basename}_z_noisy.npy"), sim["z_noisy"])
    np.save(os.path.join(out_dir, f"{basename}_dz.npy"), sim["dz"])

    # Save metrics
    m_clean = metrics(sim["z_clean"])
    m_noisy = metrics(sim["z_noisy"])
    with open(os.path.join(out_dir, f"{basename}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"clean": m_clean, "noisy": m_noisy, "config": sim["cfg"]}, f, indent=2, ensure_ascii=False)

def simulate_and_plot(cfg=None, show=True, save_dir=None, basename="lens_sim"):
    sim = simulate(cfg)
    # Metrics
    m_clean = metrics(sim["z_clean"])
    m_noisy = metrics(sim["z_noisy"])

    # Plots
    fig1 = plot_surface(sim["x"], sim["y"], sim["z_clean"], "Clean Surface (3D)")
    fig2 = plot_contour(sim["x"], sim["y"], sim["z_clean"], "Clean Surface (Contour)")
    fig3 = plot_surface(sim["x"], sim["y"], sim["z_noisy"], "Measured Surface (3D)")
    fig4 = plot_contour(sim["x"], sim["y"], sim["z_noisy"], "Measured Surface (Contour)")

    if save_dir is not None:
        ensure_dir(save_dir)
        fig1.savefig(os.path.join(save_dir, f"{basename}_clean_3d.png"), dpi=200, bbox_inches="tight")
        fig2.savefig(os.path.join(save_dir, f"{basename}_clean_contour.png"), dpi=200, bbox_inches="tight")
        fig3.savefig(os.path.join(save_dir, f"{basename}_noisy_3d.png"), dpi=200, bbox_inches="tight")
        fig4.savefig(os.path.join(save_dir, f"{basename}_noisy_contour.png"), dpi=200, bbox_inches="tight")
        export_all(sim, save_dir, basename)

    if show:
        plt.show()
    else:
        plt.close(fig1); plt.close(fig2); plt.close(fig3); plt.close(fig4)

    return {"clean": m_clean, "noisy": m_noisy, "config": sim["cfg"]}

# ------------------------
# CLI / Script entry
# ------------------------
if __name__ == "__main__":
    cfg = default_config()
    out = cfg["output"]
    out_dir = out.get("dir", "./outputs")
    basename = out.get("basename", "lens_sim")

    # Run and save
    result = simulate_and_plot(cfg, show=False, save_dir=out_dir, basename=basename)
    print(json.dumps(result, indent=2))
