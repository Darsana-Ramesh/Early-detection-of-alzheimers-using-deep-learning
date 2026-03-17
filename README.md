# 🧠 NeuroVista: Multi-Modal GNN for Early Alzheimer’s Detection

**NeuroVista** is a cutting-edge deep learning framework designed for the early prognosis of Alzheimer’s Disease (AD). By leveraging **Graph Neural Networks (GNNs)** and **resting-state fMRI (rs-fMRI)** data, the system identifies subtle disruptions in brain network topology to differentiate between cognitive stages and predict the conversion from Mild Cognitive Impairment (MCI) to AD.

---

## 🚀 Key Features

* **Multi-Topological Mapping**: Goes beyond static connectivity by integrating three distinct brain network views:
    * **Pearson Correlation**: Captures static functional synchronization.
    * **Spectral Coherence**: Captures frequency-domain rhythmic interactions (0.01–0.1 Hz).
    * **Granger Causality**: Captures directional, effective connectivity via Vector Autoregressive (VAR) modeling.
* **Advanced GNN Architecture**: Uses **GATv2** layers to dynamically weigh the importance of neural connections and **Multiplex Cross-Attention (MCA)** to fuse information across modalities.
* **Explainable AI (XAI)**: Integrated attentional pooling and NIfTI-based heatmapping to pinpoint specific brain regions (AAL-116) driving the diagnosis.
* **Clinical Relevance**: Specifically tuned to distinguish between **Stable MCI (sMCI)** and **Progressive MCI (pMCI)** for timely clinical intervention.

---

## 🏗️ System Architecture

The framework operates through a standardized neuroimaging pipeline:

### 1. Data Processing
* **Standardization**: Raw DICOM data is converted to **NIfTI** format following the **BIDS** standard.
* **Preprocessing**: Executed via **CONN Toolbox** & **SPM12**, involving realignment, slice-timing correction, spatial normalization (MNI space), and Gaussian smoothing.
* **Parcellation**: The brain is divided into 116 anatomical regions using the **AAL Atlas**.

### 2. Model: NeuroVistaGNN
The core model processes three parallel graph streams:
* **Spatial Feature Learning**: Each graph (Pearson, Granger, Wavelet) is processed by independent GATv2 heads.
* **Feature Fusion**: The MCA module identifies consensus patterns across temporal, spectral, and directional data.
* **Aggregation**: Node-level representations are aggregated into a graph-level embedding for multi-class classification (CN, EMCI, LMCI, AD).

---
### 🖥️ Interface & Analytics
![Research Suite]()
*Figure 1: The NeuroVista Research Suite provides a user-friendly dashboard for fMRI analysis and explainable AI insights.*

### 🧠 Connectivity Mapping
![Connectivity Analysis]()
*Figure 2: Multi-topological mapping of brain functional connectivity, comparing Pearson, Granger, and Wavelet representations.*
## 💻 Tech Stack

* **Backend**: Python (Flask)
* **Deep Learning**: PyTorch, PyTorch Geometric (PyG)
* **Neuroimaging**: Nilearn, Scipy (Signal processing)
* **Data Analysis**: Statsmodels (VAR models), Joblib
* **Frontend**: HTML5, Chart.js (Research Suite UI)

---

## Project Members
- Darsana R  
- Diya Soyi  
- Helan Lophy  
- Aparna Sabu  

## Guide
- Prof. Shijin Knox G U  

---

## Research Goal
To identify individuals who are at risk of converting from Mild Cognitive Impairment (MCI) to Alzheimer's Disease using advanced neuroimaging-based computational techniques.

---

## Objectives
- Develop an innovative machine learning technique to classify **progressive MCI (pMCI)** and **stable MCI (sMCI)**.  
- Create a **user-friendly web-based platform** allowing clinicians and researchers to upload fMRI scans and obtain classification results.  
- Enhance predictive performance via **graph-based connectivity analysis** capturing inter-regional brain relationships.

---

## Dataset Source
- Dataset obtained from the ADNI data repository:  
  ➤ https://ida.loni.usc.edu/pages/access/search.jsp  
- Imaging modalities used:
  - **T1-weighted structural MRI**
  - **Resting-state fMRI (rs-fMRI)**  
- Includes diagnostic categories:
  - AD, CN, EMCI, LMCI, MCI
- Only subjects having both MRI and fMRI were retained.

---

## Dataset Preparation Workflow
1. Download subject-wise data from ADNI.
2. Extract folders and list unique subject IDs.
3. Identify and separate:
   - T1-weighted MRI folders
   - rs-fMRI folders
4. Convert DICOM (.dcm) images to NIfTI (.nii) format using **dcm2nii**.
5. Organize files according to subject identifiers and imaging types.

---

## Preprocessing Pipeline

### Tools & Frameworks
- `MATLAB`
- `SPM12`
- `CONN Toolbox`

### Steps Performed
#### Structural MRI (T1-weighted)
- Skull stripping  
- Segmentation  
- Spatial normalization to MNI space  

#### Resting-state fMRI
- Slice timing correction  
- Motion correction and realignment  
- Coregistration with T1 image  
- Normalization to MNI  
- Smoothing  
- Artifact/Noise removal  
- Confound estimation  

#### Final Outputs
- Preprocessed fMRI time series  
- Normalized structural images  
- Data ready for connectivity and machine learning analysis  

---

## Future Work
- Construction of functional connectivity networks  
- Graph-theoretical biomarker extraction  
- Machine learning classification of sMCI vs pMCI  
- Deployment of a web-based prediction platform  

---

## Repository Contents
- Scripts for data filtering and preprocessing  
- Utility codes for conversion and folder organization  
- Documentation for data handling pipeline  

---

## Contribution / Issues
Feel free to open an issue or pull request for improvements or discussions.

