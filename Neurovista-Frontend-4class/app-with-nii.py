import os
import io
import base64
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib

# --- FIX 1: STOP MATPLOTLIB GUI ERRORS ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, render_template, request, jsonify
from nilearn.maskers import NiftiLabelsMasker
from statsmodels.tsa.api import VAR
from scipy import signal
from torch_geometric.nn import GATv2Conv, AttentionalAggregation, BatchNorm, GraphSizeNorm
from torch_geometric.data import Data

app = Flask(__name__)

# --- CONFIGURATION ---
NUM_NODES = 116
PCA_DIM = 32
CLASS_NAMES = ['AD', 'LMCI', 'EMCI', 'CN']
TR = 3.0  # fMRI Repetition Time

# --- 1. MODEL ARCHITECTURE ---
class NeuroVistaPrecisionNet(torch.nn.Module):
    def __init__(self, node_in, vector_in):
        super().__init__()
        self.gat_p = GATv2Conv(node_in, 64, heads=8, concat=True, dropout=0.1)
        self.gat_g = GATv2Conv(node_in, 64, heads=8, concat=True, dropout=0.1)
        self.gat_w = GATv2Conv(node_in, 64, heads=8, concat=True, dropout=0.1)
        self.gnorm = GraphSizeNorm()
        self.bn = BatchNorm(512)
        self.pool = AttentionalAggregation(nn.Sequential(nn.Linear(512, 1)))
        self.classifier = nn.Sequential(
            nn.Linear(512*3 + vector_in, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, data):
        p = F.elu(self.bn(self.gnorm(self.gat_p(data.x, data.edge_index), data.batch)))
        g = F.elu(self.bn(self.gnorm(self.gat_g(data.x, data.edge_granger), data.batch)))
        w = F.elu(self.bn(self.gnorm(self.gat_w(data.x, data.edge_wavelet), data.batch)))
        pooled = torch.cat([
            self.pool(p, data.batch),
            self.pool(g, data.batch),
            self.pool(w, data.batch),
            data.extra_feat
        ], dim=1)
        return self.classifier(pooled)

# --- 2. ASSET LOADING ---
device = torch.device("cpu")
model = NeuroVistaPrecisionNet(PCA_DIM, 1024)
model.load_state_dict(torch.load("neurovista_model.pth", map_location=device))
model.eval()

scaler = joblib.load("scaler.pkl")
masker = NiftiLabelsMasker(labels_img='ROI_MNI_V4.nii', standardize=True, detrend=True)

# --- 3. HELPER FUNCTIONS ---
def generate_heatmap_base64(matrix, title, cmap):
    plt.figure(figsize=(4, 4))
    sns.heatmap(matrix, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
    plt.title(title)
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close('all') # --- FIX: ENSURE PLOT IS CLOSED ---
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def extract_connectivity(nii_path):
    time_series = masker.fit_transform(nii_path)
    p_mat = np.corrcoef(time_series.T)
    np.nan_to_num(p_mat, copy=False)
    
    try:
        var_model = VAR(time_series)
        results = var_model.fit(1)
        g_mat = np.abs(results.coefs[0])
    except:
        g_mat = np.zeros((NUM_NODES, NUM_NODES))
        
    w_mat = np.zeros((NUM_NODES, NUM_NODES))
    fs = 1.0 / TR
    for i in range(NUM_NODES):
        for j in range(i+1, NUM_NODES):
            f, Cxy = signal.coherence(time_series[:, i], time_series[:, j], fs=fs, nperseg=64)
            mean_coh = np.mean(Cxy[(f >= 0.01) & (f <= 0.1)])
            w_mat[i, j] = w_mat[j, i] = mean_coh
        w_mat[i, i] = 1.0
        
    return p_mat, g_mat, w_mat

# --- 4. ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    # Use a fixed filename to avoid accumulating files
    file_path = os.path.join(os.getcwd(), "temp_subject.nii")
    file.save(file_path)

    try:
        # 1. Pipeline
        p_mat, g_mat, w_mat = extract_connectivity(file_path)
        
        # 2. Preprocessing
        combined_flat = np.hstack([p_mat.flatten(), g_mat.flatten(), w_mat.flatten()]).reshape(1, -1)
        extra_feat_scaled = scaler.transform(combined_flat)[:, :1024]
        
        # 3. Create Graph
        t = np.percentile(np.abs(p_mat), 85)
        graph_data = Data(
            x=torch.tensor(p_mat[:, :PCA_DIM], dtype=torch.float),
            edge_index=torch.tensor(np.argwhere(np.abs(p_mat) > t).T, dtype=torch.long),
            edge_granger=torch.tensor(np.argwhere(np.abs(g_mat) > t).T, dtype=torch.long),
            edge_wavelet=torch.tensor(np.argwhere(np.abs(w_mat) > t).T, dtype=torch.long),
            extra_feat=torch.tensor(extra_feat_scaled, dtype=torch.float),
            batch=torch.zeros(NUM_NODES, dtype=torch.long)
        )

        # 4. Final Prediction
        with torch.no_grad():
            output = model(graph_data)
            probs = torch.softmax(output, dim=1).numpy()[0]
            pred_idx = np.argmax(probs)

        # 5. Visualizations
        heatmaps = {
            'pearson': generate_heatmap_base64(p_mat, "Pearson", "RdBu_r"),
            'granger': generate_heatmap_base64(g_mat, "Granger", "magma"),
            'wavelet': generate_heatmap_base64(w_mat, "Coherence", "viridis")
        }

        # --- FIX 2: SAFER FILE CLEANUP FOR WINDOWS ---
        del p_mat, g_mat, w_mat # Clear from memory
        gc.collect() # Force garbage collection
        
        try:
            os.remove(file_path)
        except PermissionError:
            print("Note: temp_subject.nii is locked by Windows, will overwrite on next run.")

        return jsonify({
            'prediction': CLASS_NAMES[pred_idx],
            'confidence': f"{probs[pred_idx]*100:.2f}%",
            'all_probs': {CLASS_NAMES[i]: f"{probs[i]*100:.2f}%" for i in range(4)},
            'heatmaps': heatmaps
        })

    except Exception as e:
        print(f"Server Error: {str(e)}")
        # Cleanup even on failure
        try:
            gc.collect()
            if os.path.exists(file_path): os.remove(file_path)
        except: pass
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False) # Reloader can sometimes double-lock files