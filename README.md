# Neurovista: Early Detection of Alzheimer's Disease using MRI and fMRI

This project focuses on early diagnosis and progression prediction of Alzheimer's Disease by analyzing neuroimaging data from ADNI. The work includes preprocessing, connectivity analysis, and machine learning-based classification of stable and progressive MCI.

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

