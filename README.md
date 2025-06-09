
# SPSN-MVPS: Shifting Points, Silhouettes, and Neural Inverse for Solving Multiview Photometric Stereo

This repository contains the official code for the paper:

**SPSN-MVPS: Shifting Points, Silhouettes and Neural Inverse for Solving Multi-View Photometric Stereo**  
📄 Published in *Signal, Image and Video Processing*, 2025  
📌 [DOI: 10.1007/s11760-025-04304-3](https://doi.org/10.1007/s11760-025-04304-3)  
🔗 [Paper PDF (researchgate)](https://www.researchgate.net/publication/392440934_SPSN-MVPS_Shifting_points_silhouettes_and_neural_inverse_for_solving_multiview_photometric_stereo#fullTextFileContent)
🔗 [Paper PDF (view-only version - Springer)](https://rdcu.be/epFgW)



---

## 🔍 Overview

SPSN-MVPS is a novel method for 3D reconstruction from images using Multi-View Photometric Stereo (MVPS).  
It combines **Shape-from-Silhouette (SFS)** initialization and **normal information** from photometric stereo into a unified **implicit neural network** trained to predict surface occupancy.

Unlike NeRF-based models relying heavily on RGB values, this method only uses silhouettes and surface normals, ensuring faster and more robust convergence.

---

## 🧠 Core Features

- A single implicit MLP predicts object occupancy from 3D points.
- Uses **SFS** to initialize object shape.
- Incorporates **photometric normals** into the loss function.
- Incorporates a **Shifting Points** strategy used in our previous work.
- Compatible with **DiLiGenT-MV** dataset.

---

## 🧪 Requirements

```bash
Python 3.8+
PyTorch >= 1.10
numpy
scipy
imageio
matplotlib
```

---

## 📁 Dataset

Download the **DiLiGenT-MV dataset** from:  
🔗 https://sites.google.com/site/photometricstereodata/mv

Extract it under `dataset/`, so your structure looks like:

```
dataset/
└── cow/
    ├── img/
    ├── mask/
    ├── norm_mask/
    ├── normal_gt/
    ├── normal_ps_TIP19Li/
    └── params.json
```

---

## 🚀 Running the Code

Main training and reconstruction script:

```bash
python SPSN-MVPS.py
```

Parameters like object name, number of points, learning rate, etc., can be edited in the `GaS-MVPS.py` script.

---

## 📊 Visualization

- The visualization step uses the **UniSurf** (PS-NeRF) rendering method.
- Results are saved in the `out/` directory (e.g., `out/cow/test1/images/`).

> **Note:** This step is optional, as the 3D object can be generated directly from the grid.

---

## 📦 Output

The script will output:
- Reconstructed 3D surface predictions
- Intermediate checkpoint models
- Normal map visualizations (with PS and GT normals)
- MAE logs and metrics
- Use extract_mesh.py file to generate 3D object

---

## 📈 Performance

SPSN-MVPS achieves:
- Faster convergence (≈ 3 hours)
- Competitive or better MAE accuracy on multiple DiLiGenT-MV objects

Refer to the paper for detailed comparisons and analysis.

---

## 📜 Citation

If you use this code or paper, please cite:

```bibtex
@article{abada2025spsn,
  title={SPSN-MVPS: Shifting points, silhouettes and neural inverse for solving multiview photometric stereo},
  author={Abada, Lyes and Gacem, Tarek and Mezabiat, Aimen Said and Bourzam, Saadallah and Malki, Omar Chouaab and Mekkaoui, Mohamed},
  journal={Signal, Image and Video Processing},
  volume={19},
  number={694},
  year={2025},
  publisher={Springer}
}
```

You may also be interested in our related work:  
**[Enhancing PSNeRF with Shape-from-Silhouette for Efficient and Accurate 3D Reconstruction](https://link.springer.com/article/10.1007/s11042-024-20319-3)**  
(*Multimedia Tools and Applications*, 2024)

https://www.researchgate.net/publication/392440934_SPSN-MVPS_Shifting_points_silhouettes_and_neural_inverse_for_solving_multiview_photometric_stereo#fullTextFileContent

---

## 🙏 Acknowledgements

- Many parts of this code are inspired by **PS-NeRF** and **UniSurf**.

---

## 📧 Contact

For questions or suggestions, feel free to contact:

**Lyes Abada** — 📧 labada@usthb.dz  
Lab: Artificial Intelligence Laboratory (LRIA), USTHB, Algiers
