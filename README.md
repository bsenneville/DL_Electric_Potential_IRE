# *Rapid approximation of electric potential fields in Irreversible Electroporation (IRE) using deep learning and numerical refinement.*

---

## **📌 Overview**

This repository provides the **training implementation**, **synthetic data generation scripts**, and a **tutorial** for a **3D U-Net-based model** that rapidly approximates electric potential fields in **Irreversible Electroporation (IRE)** procedures. The framework combines:

- **Deep Learning (DL)**: A 3D CNN (U-Net) to predict basis-function fields from electrode configurations.
- **Numerical Refinement**: Lightweight iterative correction (Bi-CGSTAB solver) to enforce physical consistency.

---

## **📂 Repository Structure**

```bash
.
├── data_generation/                    # Scripts for synthetic data generation
│   ├── Bresenham3D.py                  # 3D line voxelization (Bresenham algorithm)
│   ├── Simulate_Basis_Fonctions.py     # Solves PDE for basis functions (Bi-CGSTAB)
│   ├── compute_gradient.py             # Computes the gradient of a 2D or 3D image using central differences for interior points and forward/backward differences for boundary points
│   ├── fill_matrix_sparsekronecker.py  # Constructs the 1D/2D/3D discrete Laplacian matrix with Neumann boundary conditions
│   └── simulation_data.py              # Main data generation pipeline
│
├── models/                    # Deep learning components
│   ├── UNet3D.py              # 3D U-Net architecture
│   ├── dataset.py             # Dataset for electrode configurations
│   ├── inference.py           # Model inference script
│   ├── loss.py                # Custom loss functions (e.g., MSE + physical constraints)
│   └── train.py               # Training script (supports small batch sizes)
│
├── LICENSE                    # License
└── README.md                  # This file
```

---

## **🔧 Installation**

### **Prerequisites**

- Python ≥ 3.9
- CUDA-compatible GPU (recommended for training)
- 16GB+ RAM (for high-resolution simulations)

### **Setup**

1. Clone the repository:
  ```bash
   git clone https://github.com/bsenneville/DL_Electric_Potential_IRE.git
   cd ire-hybrid-framework
  ```


---

## **🚀 Quick Start**

### **1. Generate Synthetic Training Data**

The dataset consists of **voxelized electrode configurations** and their corresponding **basis-function fields** (solutions to the elliptic PDE with $\sigma = 1$).

### **2. Train the 3D U-Net Model**

### **3. Inference: Predict Basis Functions**

**Run inference on a new electrode configuration**:

## **🔬 Methodology Details**

### **Model Input/Output**


| Component          | Description                                                                                  |
| ------------------ | -------------------------------------------------------------------------------------------- |
| **Input**          | 3D voxel grid: `-1` (inactive needle), `0` (background), `1` (active needle).                |
| **Output**         | Basis-function field $v_l$ (solution to $\nabla \cdot (\sigma \nabla v_l) = 0$).             |
| **PDE Solver**     | Bi-CGSTAB with **Dirichlet** (needle surfaces) and **Neumann** (domain boundary) conditions. |
| **Discretization** | 2nd-order finite differences.                                                                |


---

## **📜 Citation**

If you use this code or framework in your research, please cite our paper:

```bibtex
@article{desier2026hybrid,
  title={Hybrid Learning/Numerical Framework for Fast and Robust Electric Field Simulation in Irreversible Electroporation},
  author = {Kylian Desier and Olivier Sutter and Luc Lafitte and Laurent Facq and Olivier Seror and Clair Poignard and Baudouin {Denis de Senneville}},
  journal = {Computer Methods and Programs in Biomedicine},
  pages = {109408},
  year = {2026},
  issn = {0169-2607},
  doi = {https://doi.org/10.1016/j.cmpb.2026.109408},
  url = {https://www.sciencedirect.com/science/article/pii/S016926072600163X},
}

```

**Acknowledgments**:  
This work was supported by:

- Plan Cancer MECI (`PC_MECI_21CM119_00`)
- Institut National du Cancer (INCa, PLBIO `2023-156`)
- ANR projects: IMITATE, MIRE4VTACH, DEVIN
- GENCI/IDRIS (Grant `2023-AD010614857R1` on Jean Zay supercomputer).

