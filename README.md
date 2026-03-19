# NeuroVista: Early Alzheimer’s Prognosis using Deep Learning

---

## Overview

**NeuroVista** is a deep learning framework designed for the **early prognosis of Alzheimer’s Disease (AD)**.

By focusing on the **Mild Cognitive Impairment (MCI)** stage, NeuroVista identifies:
- **High-risk patients (pMCI)** → likely to progress to Alzheimer’s  
- **Stable patients (sMCI)** → unlikely to progress  

This prediction is made **years before irreversible brain damage occurs**, enabling early intervention.

---

## The Challenge

Traditional diagnostic methods struggle to:
- Distinguish between **Stable MCI (sMCI)** and **Progressive MCI (pMCI)**
- Capture subtle, early-stage brain changes  

NeuroVista addresses this by modeling the brain as a **dynamic functional network** using **Graph Neural Networks (GNNs)**.

---

## Key Innovations

### Multi-Topological Mapping
Captures brain connectivity from multiple perspectives:
- **Pearson Correlation (Spatial)** → Functional synchronization  
- **Spectral Coherence (Frequency)** → Rhythmic interactions (0.01–0.1 Hz)  
- **Granger Causality (Directional)** → Effective connectivity  

---

### GATv2 Architecture
- Uses **attention mechanisms**  
- Learns importance of different brain regions dynamically  

---

### Explainable AI (XAI)
- Identifies **top 10 brain regions** influencing predictions  
- Provides **clinical transparency**

---

## System Architecture

![Architecture](Images/archi-.png)

The NeuroVista pipeline transforms raw fMRI data into clinical insights through the following stages:

---

### Preprocessing & Denoising
Using **MATLAB, SPM12, and CONN Toolbox**:

- Convert **DICOM → NIfTI (BIDS standard)**  
- Structural processing:
  - Skull stripping  
  - Normalization to MNI space  
- Functional processing:
  - Slice-timing correction  
  - Motion realignment  
  - Spatial smoothing  

---

### ROI Extraction & Graph Construction

- Brain divided into **116 regions (AAL Atlas)**  
- Extract **mean BOLD time-series**  

Construct three connectivity matrices:
- **Pearson Correlation** → Static connectivity  
- **Spectral Coherence** → Frequency-based connectivity  
- **Granger Causality** → Directional connectivity  

---

### 3️Model Inference: NeuroVistaGNN

- Processes graphs using **GATv2 layers**  
- Uses **Multiplex Cross-Attention (MCA)** for fusion  
- Learns complex brain connectivity patterns  

---

## Performance Results

### Overall Metrics
- **Accuracy:** 80.91%  
- **ROC-AUC:** 0.9167  

---

### MCI Conversion Prediction (sMCI vs pMCI)

| Class | Precision | Recall | F1-Score |
|------|----------|--------|----------|
| Stable MCI (sMCI) | 0.79 | 0.84 | 0.81 |
| Progressive MCI (pMCI) | 0.83 | 0.78 | 0.80 |

Demonstrates strong ability to identify **high-risk patients early**

---

## Web Interface

The **NeuroVista Research Suite** allows clinicians to:
- Upload preprocessed fMRI scans  
- Receive **instant diagnostic predictions**  
- View **explainability insights**

![Dashboard](Images/web-res.png)

## Tech Stack

- **Deep Learning**: PyTorch, PyTorch Geometric (PyG)  
- **Neuroimaging**: Nilearn, SciPy, Statsmodels  
- **Backend**: Flask (Python)  
- **Frontend**: HTML5, Chart.js  
- **Preprocessing**: MATLAB, SPM12, CONN Toolbox  

---
## Project Team

**Members:**
- Darsana R  
- Diya Soyi  
- Helan Lophy  
- Aparna Sabu  

**Guide:** Prof. Shijin Knox G U  
**Institution:** Government Engineering College Palakkad  
