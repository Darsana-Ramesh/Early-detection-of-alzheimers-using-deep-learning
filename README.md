## 🚀 What is NeuroVista?

**NeuroVista** is a deep learning system that predicts **whether a person with Mild Cognitive Impairment (MCI) will develop Alzheimer’s Disease**.

Instead of just detecting Alzheimer’s after it happens, NeuroVista tries to answer a more important question:

> **“Who is likely to develop Alzheimer’s in the future?”**

---

## ❗ Why does this matter?

Alzheimer’s develops slowly over years.

- Many patients are first diagnosed with **MCI (Mild Cognitive Impairment)**
- But **not all MCI patients develop Alzheimer’s**
- Doctors currently **cannot reliably predict progression**

👉 This is where NeuroVista helps.

---

## 💡 What does NeuroVista do?

NeuroVista:
- Analyzes **brain activity (fMRI scans)**
- Studies how different brain regions **communicate with each other**
- Learns patterns that indicate **early disease progression**

### 🎯 It can:
- Classify patients as:
  - Normal (CN)
  - Stable MCI (sMCI)
  - Progressive MCI (pMCI)
  - Alzheimer’s (AD)
- Predict if an MCI patient will **convert to Alzheimer’s**

---

## 🧠 How does it work? (Simple View)

1. **Brain Scan Input (fMRI)**
   - Captures brain activity over time

2. **Build Brain Network**
   - Brain is treated like a **graph**
   - Regions = nodes  
   - Connections = edges  

3. **Deep Learning Model (GNN)**
   - Learns how brain connections change
   - Detects abnormal patterns

4. **Prediction**
   - Outputs diagnosis + future risk

---

## 🔬 What makes it different?

Instead of using just one type of brain connection, NeuroVista combines three:

- **Correlation** → Which regions activate together  
- **Frequency patterns** → How signals behave over time  
- **Causality** → Which region influences another  

👉 This gives a **more complete picture of brain function**

---

## 📊 Results

- **Accuracy:** 80.91%  
- **ROC-AUC:** 0.9167  

### sMCI vs pMCI Prediction

| Class | Precision | Recall | F1-Score |
|------|----------|--------|----------|
| sMCI | 0.79     | 0.84   | 0.81     |
| pMCI | 0.83     | 0.78   | 0.80     |

✅ Shows strong ability to identify **high-risk patients early**

---

## 🖥️ Demo

### 📊 Web Interface
![Dashboard](Images/web-res.png)

- Upload scan  
- Get prediction instantly  
- View important brain regions  

---

## 🏗️ Architecture (High-Level)

![Architecture](Images/archi-.png)

- Converts brain signals → graphs  
- Uses **Graph Neural Networks (GNNs)**  
- Combines multiple brain connectivity views  

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, PyTorch Geometric  
- **Backend**: Flask  
- **Frontend**: HTML, Chart.js  
- **Neuroimaging**: Nilearn, SPM, CONN  

---

## 👥 Team

- Darsana R  
- Diya Soyi  
- Helan Lophy  
- Aparna Sabu  

**Guide:** Prof. Shijin Knox G U  
**Institution:** Government Engineering College Palakkad  
