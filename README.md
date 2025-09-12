
# Synthetic Lens Surface Deformation Simulation (Starter)

This is a minimal, ready-to-run project to simulate a lens surface, apply deformations (center pit, edge ring, astigmatism, trefoil), add measurement noise, and visualize 3D/contour maps. Outputs (figures + .npy arrays + metrics.json) are saved to `outputs/`.

## 1) Create & activate a virtual environment

### macOS / Linux (bash/zsh)
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 2) Install dependencies
```bash
pip install -r requirements.txt
```

## 3) Run the simulation
```bash
python src/lens_deformation_sim.py
```
Outputs will appear under `outputs/`:
- `*_clean_3d.png`, `*_clean_contour.png`
- `*_noisy_3d.png`, `*_noisy_contour.png`
- Arrays: `*_x.npy`, `*_y.npy`, `*_z_clean.npy`, `*_z_noisy.npy`, `*_dz.npy`
- Metrics: `*_metrics.json`

## 4) Customize parameters

Open `src/lens_deformation_sim.py` and edit `default_config()`:
- `grid.side_mm`, `grid.points`: aperture size & resolution
- `base_surface.a`: curvature coefficient (z = a*(x^2 + y^2))
- `deformations`: tweak types and strengths (gaussian, gaussian_ring, astigmatism, trefoil)
- `noise.measurement_rms`: measurement noise RMS (mm)

Then run again to regenerate outputs.

## 5) Optional: Use as a module in a notebook

```python
from src.lens_deformation_sim import simulate_and_plot, default_config
cfg = default_config()
cfg["base_surface"]["a"] = 8e-4
simulate_and_plot(cfg, show=True, save_dir="outputs", basename="try1")
```

---

### Quick tips
- Want a deeper center pit? Make the `gaussian.amplitude` more negative (e.g. `-0.3`).
- Want a sharper edge ring? Reduce `gaussian_ring.sigma_mm` or shift `r0_mm` closer to the edge.
- Increase `grid.points` (e.g. 401) for a finer mesh (slower but prettier).
