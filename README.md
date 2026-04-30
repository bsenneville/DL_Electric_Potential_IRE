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

