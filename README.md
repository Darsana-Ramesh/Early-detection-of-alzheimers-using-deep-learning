# Neurovista: A Deep Learning Framework for Early Detection of Alzheimer’s Disease using fMRI

Neurovista is a deep learning framework designed for the early prognosis of Alzheimer’s Disease (AD). Here, “early” refers to the pre-dementia phase—when patients are still in the Mild Cognitive Impairment (MCI) stage, often years before clinical Alzheimer’s is diagnosed and before significant, irreversible brain damage occurs.

At this stage, traditional diagnosis struggles because symptoms are subtle and progression is uncertain—not all MCI patients develop AD. NeuroVista addresses this challenge by focusing on disease progression, specifically distinguishing between stable MCI (sMCI) and progressive MCI (pMCI).

By leveraging Graph Neural Networks (GNNs) and resting-state fMRI (rs-fMRI) data, the system models the brain as a connectivity graph and captures subtle disruptions in functional network topology. These changes—often invisible in conventional analysis—serve as early indicators of neurodegeneration.

Through this approach, NeuroVista not only differentiates between cognitive stages (CN, MCI, AD) but also predicts which MCI patients are likely to convert to Alzheimer’s within a given time horizon, enabling earlier intervention, better patient monitoring, and more informed clinical decision-making.

## 👥 Project Team
* **Members**: Darsana R, Diya Soyi, Helan Lophy, Aparna Sabu
* **Guide**: Prof. Shijin Knox G U
* **Institution**: Government Engineering College Palakkad

---

## Research Goal & Objectives
**Goal:** To identify individuals at high risk of converting from Mild Cognitive Impairment (MCI) to Alzheimer's Disease using advanced neuroimaging-based computational techniques.

* **Innovate:** Develop a GNN-based technique to classify **progressive MCI (pMCI)** and **stable MCI (sMCI)**.
* **Analyze:** Enhance predictive performance via **multi-topological connectivity analysis** (Spatial, Spectral, and Directional).
* **Deploy:** Create a **user-friendly web platform** for clinicians to upload fMRI scans and obtain instant diagnostic insights.

---

## Key Features

* **Multi-Topological Mapping**: Goes beyond static connectivity by integrating three distinct brain network views:
    * **Pearson Correlation**: Captures static functional synchronization.
    * **Spectral Coherence**: Captures frequency-domain rhythmic interactions (0.01–0.1 Hz).
    * **Granger Causality**: Captures directional, effective connectivity via Vector Autoregressive (VAR) modeling.
* **Advanced GNN Architecture**: Uses **GATv2** layers to dynamically weigh the importance of neural connections and **Multiplex Cross-Attention (MCA)** to fuse information across modalities.
* **Explainable AI (XAI)**: It provides a real-time diagnostic classification and an "Explainability" panel that highlights the top 10 brain regions contributing to the result.
* **Clinical Relevance**: Specifically tuned to distinguish between **Stable MCI (sMCI)** and **Progressive MCI (pMCI)** for timely clinical intervention.

---

## System Architecture: End-to-End Pipeline

The framework operates through a standardized neuroimaging pipeline, moving from raw data to clinical insights.

### 1. Data Collection & Preparation
Data is sourced from the **ADNI (Alzheimer’s Disease Neuroimaging Initiative)** repository. 
* **Selection Criteria**: Subjects must have both T1-weighted structural MRI (for anatomical mapping) and rs-fMRI (for functional activity).
* **Format Conversion**: Raw DICOM (.dcm) files are converted to NIfTI (.nii) using **dcm2nii** and organized according to the **BIDS** (Brain Imaging Data Structure) standard.

### 2. Preprocessing Pipeline
Executed using **MATLAB**, **SPM12**, and the **CONN Toolbox**, the data undergoes a rigorous cleaning process:
* **Structural**: Skull stripping, segmentation, and normalization to MNI space.
* **Functional**: Slice-timing correction, motion realignment, coregistration with T1 images, and spatial smoothing.
* **Denoising**: Removal of physiological noise and motion artifacts to extract clean BOLD signals.

### 3. ROI Extraction & Graph Construction
The brain is parcellated into 116 regions using the **AAL Atlas**.
* **Signal Extraction**: Mean BOLD time-series are extracted for each region.
* **Graph Modeling**: These signals are transformed into three parallel adjacency matrices (Pearson, Granger, Wavelet), representing the brain as a sequence of weighted graphs.

### 4. Model Inference: NeuroVistaGNN
The core model processes these graphs through:

![Architecture](Images/archi-.png)

* **Spatial Feature Learning**: Independent GATv2 heads for each connectivity modality.
* **Feature Fusion**: The MCA module identifies consensus patterns across temporal, spectral, and directional data.
* **Early Detection**: Classification into CN, EMCI, LMCI, or AD, with a focus on predicting MCI-to-AD conversion.

---

## Visual Insights

### Interface & Analytics
![Research Suite](Images/web-res.png)

*Figure 1: **The NeuroVista Research Suite Dashboard.** This interface allows clinicians to upload preprocessed fMRI data. It provides a real-time diagnostic classification and an "Explainability" panel that highlights the top 10 brain regions contributing to the result.*

### Connectivity Mapping
![Connectivity Analysis](/conn.png)
*Figure 2: **Multi-topological Functional Connectivity.** This visualization compares Pearson correlation, frequency-based Wavelet coherence, and directional Granger causality to capture a holistic "fingerprint" of neural dysfunction.*

---

## Tech Stack

* **Backend**: Python (Flask)
* **Deep Learning**: PyTorch, PyTorch Geometric (PyG)
* **Neuroimaging**: Nilearn, Scipy (Signal processing)
* **Data Analysis**: Statsmodels (VAR models), Joblib
* **Frontend**: HTML5, Chart.js (Research Suite UI)
* **Preprocessing Tools**: MATLAB, SPM12, CONN Toolbox

---


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
