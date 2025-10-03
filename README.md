### 🔬 Lens Sim Starter — Synthetic Lens Surface Deformation

A compact toolkit to model a base **lens surface** and add **deformations/noise**,  
then visualize as 3D surface and contour maps. Exports figures and raw arrays for further analysis.

**Features**
- Base surface: paraboloid `z = a(x^2 + y^2)`
- Deformations: Gaussian pit/bump, Gaussian ring, Astigmatism, Trefoil
- Measurement noise (Gaussian RMS)
- Exports `.png` plots + `.npy` arrays + metrics `.json`

**Metrics Example (from ./outputs/lens_sim_metrics.json)**
- Clean: PV ≈ 1.4656 mm, RMS ≈ 0.2983 mm, Mean ≈ 0.4394 mm  
- Noisy: PV ≈ 1.4941 mm, RMS ≈ 0.2984 mm, Mean ≈ 0.4395 mm  

**Usage**
```python
from lens_deformation_sim import simulate_and_plot, default_config
cfg = default_config()
simulate_and_plot(cfg, show=False, save_dir="./outputs", basename="lens_sim")
