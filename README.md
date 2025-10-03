# 🔬 Lens Sim Starter  

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)  
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
[![Made with NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy)](https://numpy.org/)  
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8+-11557c?logo=plotly)](https://matplotlib.org/)  

Synthetic **lens surface deformation simulation**:  
simulate a paraboloid lens surface, add **custom deformations + noise**, and visualize in **3D + contour**.  
Exports plots and numerical data for further analysis.  

---

## 📂 Example Output  

### 🔹 Clean Lens Surface (No Deformation + Noise)  
<p align="center">
  <img src="outputs/lens_sim_clean_3d.png" alt="Lens Clean 3D" width="45%"/>
  <img src="outputs/lens_sim_clean_contour.png" alt="Lens Clean Contour" width="45%"/>
</p>  

---

### 🔹 Noisy Lens Surface (With Measurement Noise)  
<p align="center">
  <img src="outputs/lens_sim_noisy_3d.png" alt="Lens Noisy 3D" width="45%"/>
  <img src="outputs/lens_sim_noisy_contour.png" alt="Lens Noisy Contour" width="45%"/>
</p>  

---

## 📊 Metrics (from `lens_sim_metrics.json`)  
- **Clean Surface** → PV ≈ 1.4656 mm, RMS ≈ 0.2983 mm, Mean ≈ 0.4394 mm  
- **Noisy Surface** → PV ≈ 1.4941 mm, RMS ≈ 0.2984 mm, Mean ≈ 0.4395 mm  

---

## 🚀 Usage  

Run with default parameters:
```bash
python lens_deformation_sim.py
