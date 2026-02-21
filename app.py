import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joblib
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from statsmodels.tsa.api import VAR
from scipy import signal
from torch_geometric.nn import GATv2Conv, AttentionalAggregation, BatchNorm, GraphSizeNorm
from torch_geometric.data import Data

# --- SYSTEM LOGGING ---
class ExecutionLogger:
    @staticmethod
    def log_info(message):
        print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} | {message}")

app = Flask(__name__)

# --- CONFIGURATION ---
NUM_NODES = 116
PCA_DIM = 32
CLASS_NAMES = ['AD', 'LMCI', 'EMCI', 'CN']
TR = 3.0  

# --- CORE ARCHITECTURE: MULTIPLEX CROSS-ATTENTION ---
class MultiplexCrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.scale = np.sqrt(in_channels)

    def forward(self, p, g, w):
        # Stacking topological modalities: [Nodes, 3, Channels]
        x = torch.stack([p, g, w], dim=1)
        q, k, v = self.query(x), self.key(x), self.value(x)
        
        # Identification of inter-modal consensus
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, v).mean(dim=1)

class NeuroVistaGNN(torch.nn.Module):
    def __init__(self, node_in, vector_in):
        super().__init__()
        self.gat_p = GATv2Conv(node_in, 64, heads=8, concat=True)
        self.gat_g = GATv2Conv(node_in, 64, heads=8, concat=True)
        self.gat_w = GATv2Conv(node_in, 64, heads=8, concat=True)
        
        # Variable name matches the saved state_dict "cross_modal_fusion"
        self.cross_modal_fusion = MultiplexCrossAttention(512)
        
        self.gnorm = GraphSizeNorm()
        self.bn = BatchNorm(512)
        self.pool = AttentionalAggregation(nn.Sequential(nn.Linear(512, 1)))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 + vector_in, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, data):
        p = F.elu(self.gat_p(data.x, data.edge_index))
        g = F.elu(self.gat_g(data.x, data.edge_granger))
        w = F.elu(self.gat_w(data.x, data.edge_wavelet))
        
        f = self.bn(self.gnorm(self.cross_modal_fusion(p, g, w), data.batch))
        graph_latent = self.pool(f, data.batch)
        combined = torch.cat([graph_latent, data.extra_feat], dim=1)
        return self.classifier(combined)

# --- GLOBAL ASSET INITIALIZATION ---
device = torch.device("cpu")
model = NeuroVistaGNN(PCA_DIM, 1024)
scaler = None

def load_assets():
    global scaler
    try:
        if os.path.exists("neurovista_model.pth"):
            model.load_state_dict(torch.load("neurovista_model.pth", map_location=device))
            model.eval()
            ExecutionLogger.log_info("Model weights loaded successfully.")
        
        if os.path.exists("scaler.pkl"):
            scaler = joblib.load("scaler.pkl")
            ExecutionLogger.log_info("StandardScaler loaded successfully.")
    except Exception as e:
        ExecutionLogger.log_info(f"Asset Load Error: {e}")

load_assets()

# --- TOPOLOGY CALCULATION ---
def get_topology(ts):
    """Generates Pearson, Granger, and Coherence matrices from fMRI timeseries."""
    if ts.shape[0] == NUM_NODES: ts = ts.T
    
    # Pearson Correlation
    p_mat = np.nan_to_num(np.corrcoef(ts.T))
    
    # Granger Causality (VAR-1 Approximation)
    g_mat = np.zeros((NUM_NODES, NUM_NODES))
    try:
        res = VAR(ts).fit(1)
        g_mat = np.abs(res.coefs[0])
    except: pass
    
    # Wavelet/Spectral Coherence
    w_mat = np.zeros((NUM_NODES, NUM_NODES))
    fs = 1.0 / TR
    for i in range(NUM_NODES):
        for j in range(i+1, NUM_NODES):
            f, Cxy = signal.coherence(ts[:, i], ts[:, j], fs=fs, nperseg=64)
            w_mat[i, j] = w_mat[j, i] = np.mean(Cxy[(f >= 0.01) & (f <= 0.1)])
        w_mat[i, i] = 1.0
        
    return p_mat, g_mat, w_mat

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global scaler
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    if scaler is None:
        return jsonify({'error': 'System assets (scaler) not initialized.'}), 500

    try:
        file = request.files['file']
        ExecutionLogger.log_info(f"Processing sequence: {file.filename}")
        ts = np.load(file)
        
        # 1. Feature Extraction (Topology Generation)
        p, g, w = get_topology(ts)
        
        # 2. Vectorization for Classifier Input
        flat = np.hstack([p.flatten(), g.flatten(), w.flatten()]).reshape(1, -1)
        scaled_vector = scaler.transform(flat)
        extra = torch.tensor(scaled_vector[:, :1024], dtype=torch.float)
        
        # 3. Graph Construction (Top 15% Sparsity)
        threshold = np.percentile(np.abs(p), 85)
        graph = Data(
            x=torch.tensor(p[:, :PCA_DIM], dtype=torch.float),
            edge_index=torch.tensor(np.argwhere(np.abs(p) > threshold).T, dtype=torch.long),
            edge_granger=torch.tensor(np.argwhere(np.abs(g) > threshold).T, dtype=torch.long),
            edge_wavelet=torch.tensor(np.argwhere(np.abs(w) > threshold).T, dtype=torch.long),
            extra_feat=extra,
            batch=torch.zeros(NUM_NODES, dtype=torch.long)
        )

        # 4. Consensus Inference
        with torch.no_grad():
            output = model(graph)
            probs = torch.softmax(output, dim=1).numpy()[0]
        
        # Cleanup
        gc.collect()
        
        return jsonify({
            'prediction': CLASS_NAMES[np.argmax(probs)],
            'confidence': f"{np.max(probs)*100:.1f}%",
            'all_probs': {CLASS_NAMES[i]: f"{probs[i]*100:.2f}%" for i in range(4)}
        })
        
    except Exception as e:
        ExecutionLogger.log_info(f"Prediction Failure: {str(e)}")
        return jsonify({'error': f"Internal Processing Error: {str(e)}"}), 500

if __name__ == '__main__':
    ExecutionLogger.log_info("NeuroVista Precision Server Active.")
    app.run(debug=True, port=5000)