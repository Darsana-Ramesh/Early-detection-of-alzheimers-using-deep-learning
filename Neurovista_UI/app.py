import os, torch, joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from scipy import signal
from statsmodels.tsa.api import VAR
from torch_geometric.data import Data
from model_utils import NeuroVistaGNN
from nilearn import plotting

app = Flask(__name__)
UPLOAD_FOLDER = os.path.abspath('uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device("cpu")
TR, NUM_REGIONS, PCA_DIM, AUX_SIZE = 3.0, 116, 32, 1024
DIAG_CLASSES = ['AD', 'LMCI', 'EMCI', 'CN']

# ── Atlas labels ──
ATLAS_LABELS = [f"Region {i}" for i in range(NUM_REGIONS)]
try:
    label_path = os.path.join(app.root_path, 'static', 'atlas_labels.txt')
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines:
                ATLAS_LABELS = lines
except Exception as e:
    print(f"Label load warning: {e}")

# ── AAL-116 MNI coordinates ──
def _load_aal_coords():
    npy_path = os.path.join(app.root_path, 'aal116_coords.npy')
    if os.path.exists(npy_path):
        coords = np.load(npy_path)
        print(f"[Atlas] Loaded exact AAL coords  shape={coords.shape}")
        return coords
    try:
        from nilearn import datasets as _ds
        import nibabel as nib
        aal       = _ds.fetch_atlas_aal(version='SPM12')
        label_img = nib.load(aal.maps)
        data      = label_img.get_fdata()
        affine    = label_img.affine
        coords    = []
        for lv in [int(i) for i in aal.indices]:
            vox = np.argwhere(data == lv)
            if len(vox) == 0:
                coords.append([0., 0., 0.])
            else:
                coords.append(nib.affines.apply_affine(affine, vox.mean(axis=0)).tolist())
        coords = np.array(coords, dtype=float)
        print(f"[Atlas] Fetched AAL coords via nilearn  shape={coords.shape}")
        return coords
    except Exception as e:
        print(f"[Atlas] nilearn fetch failed: {e}  — using approximate coords")
    approx = np.array([
        [-23,62,2],[23,62,2],[-20,55,20],[20,55,20],[-9,55,28],[9,55,28],
        [-35,48,10],[35,48,10],[-28,42,30],[28,42,30],[-10,45,40],[10,45,40],
        [-40,30,30],[40,30,30],[-28,22,50],[28,22,50],[-8,18,60],[8,18,60],
        [-42,10,30],[42,10,30],[-35,5,50],[35,5,50],[-8,8,55],[8,8,55],
        [-52,-5,0],[52,-5,0],[-55,-15,-5],[55,-15,-5],[-50,-25,0],[50,-25,0],
        [-52,-35,5],[52,-35,5],[-55,-45,10],[55,-45,10],[-45,-55,15],[45,-55,15],
        [-28,-2,-25],[28,-2,-25],[-25,-12,-25],[25,-12,-25],
        [-38,-45,55],[38,-45,55],[-35,-52,55],[35,-52,55],
        [-8,-58,58],[8,-58,58],[-25,-60,48],[25,-60,48],
        [-48,-30,45],[48,-30,45],[-52,-20,40],[52,-20,40],
        [-25,-82,5],[25,-82,5],[-15,-90,12],[15,-90,12],
        [-28,-78,18],[28,-78,18],[-18,-70,28],[18,-70,28],
        [-38,-72,8],[38,-72,8],[-32,-88,10],[32,-88,10],
        [-5,8,40],[5,8,40],[-5,-5,42],[5,-5,42],
        [-5,-28,28],[5,-28,28],[-5,-40,28],[5,-40,28],
        [-38,0,4],[38,0,4],[-40,-8,6],[40,-8,6],
        [-14,10,0],[14,10,0],[-20,8,-2],[20,8,-2],
        [-22,-5,6],[22,-5,6],[-18,0,8],[18,0,8],
        [-15,-8,6],[15,-8,6],[-12,2,-2],[12,2,-2],
        [-10,-16,8],[10,-16,8],[-8,-20,5],[8,-20,5],
        [-25,-18,-15],[25,-18,-15],[-28,-28,-12],[28,-28,-12],
        [-22,-35,-8],[22,-35,-8],[-22,-42,-5],[22,-42,-5],
        [-22,0,-18],[22,0,-18],
        [-22,-55,-30],[22,-55,-30],[-35,-60,-28],[35,-60,-28],
        [-15,-65,-30],[15,-65,-30],[-40,-68,-28],[40,-68,-28],
        [-22,-72,-32],[22,-72,-32],[-10,-72,-35],[10,-72,-35],
        [-30,-78,-30],[30,-78,-30],[-15,-78,-32],[15,-78,-32],
        [-8,-72,-30],[8,-72,-30],[-22,-80,-28],[22,-80,-28],
        [-42,-55,8],[42,-55,8],[-40,-65,18],[40,-65,18],
        [-5,28,32],[5,28,32],[-5,38,22],[5,38,22],
        [-8,-62,30],[8,-62,30],[-18,-68,38],[18,-68,38],
        [-42,18,12],[42,18,12],[-48,5,2],[48,5,2]
    ], dtype=float)
    while len(approx) < 116:
        approx = np.vstack([approx, [0., 0., 0.]])
    return approx[:116]

AAL116_COORDS = _load_aal_coords()

# ── Models ──
model_a = NeuroVistaGNN(node_in=PCA_DIM, vector_in=AUX_SIZE, out_channels=4)
model_a.load_state_dict(torch.load('multiclassification.pth', map_location=DEVICE))
model_a.eval()
scaler_a = joblib.load('multiclassification_scaler.pkl')

model_b = NeuroVistaGNN(node_in=PCA_DIM, vector_in=AUX_SIZE, out_channels=2)
model_b.load_state_dict(torch.load('binary_model.pth', map_location=DEVICE))
model_b.eval()
assets_b = joblib.load('binary_model.pkl')
scaler_b = assets_b['scaler']
PROG_CLASSES = ['sMCI', 'pMCI']


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def compute_matrices(ts_data):
    p = np.corrcoef(ts_data.T)
    np.nan_to_num(p, copy=False)
    try:
        var_model = VAR(ts_data)
        g = np.abs(var_model.fit(1).coefs[0])
    except:
        g = np.zeros((NUM_REGIONS, NUM_REGIONS))
    fs = 1.0 / TR
    c = np.eye(NUM_REGIONS)
    for i in range(NUM_REGIONS):
        for j in range(i + 1, NUM_REGIONS):
            f, Cxy = signal.coherence(ts_data[:, i], ts_data[:, j], fs=fs, nperseg=64)
            mask = (f >= 0.01) & (f <= 0.1)
            c[i, j] = c[j, i] = np.mean(Cxy[mask]) if np.any(mask) else 0
    return p, g, c


def get_edge_index(mat):
    t = np.percentile(np.abs(mat), 85)
    return torch.tensor(np.argwhere(np.abs(mat) > t).T, dtype=torch.long)


def compute_node_importance(xai, num_regions=NUM_REGIONS):
    """Aggregate GAT attention per destination node. Returns raw sums."""
    edge_weights = xai['node_attn']
    edge_index   = xai['edge_index']
    node_importance = np.zeros(num_regions)
    for i in range(len(edge_weights)):
        t = int(edge_index[1, i])
        if t < num_regions:
            node_importance[t] += edge_weights[i]
    return node_importance


def importance_for_plot(node_importance):
    """Normalize to [0,1] for marker size/color scaling only."""
    mx = node_importance.max()
    return node_importance / mx if mx > 0 else node_importance


def relative_importance(node_importance, top_indices):
    """How many times more attention vs the mean across all 116 regions."""
    mean_attn = node_importance.mean()
    if mean_attn == 0:
        return np.zeros(len(top_indices))
    return node_importance[top_indices] / mean_attn


def build_brain_heatmap_html(node_importance, prediction_label, top_n=10):
    """
    Renders axial + coronal views (side by side, full width) with numbered
    colored markers. Below: a 5-column region key grid (no table, no sagittal).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import io, base64

    n = len(node_importance)
    coords = AAL116_COORDS[:n]

    top_indices  = np.argsort(node_importance)[-top_n:][::-1]
    top_coords   = coords[top_indices]
    top_norm_top = importance_for_plot(node_importance)[top_indices]
    top_labels   = [
        ATLAS_LABELS[int(i)] if int(i) < len(ATLAS_LABELS) else f"Region {int(i)}"
        for i in top_indices
    ]

    RANK_COLORS = [
        '#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4',
        '#3b82f6', '#a855f7', '#ec4899', '#14b8a6', '#f59e0b',
    ]

    marker_sizes = 60 + top_norm_top * 140

    def make_view(display_mode, title):
        fig, ax = plt.subplots(figsize=(9, 7), facecolor='#0a0a0f')
        ax.set_facecolor('#0a0a0f')
        plotting.plot_markers(
            node_values=top_norm_top,
            node_coords=top_coords,
            node_size=marker_sizes.tolist(),
            node_cmap='hot',
            node_vmin=0,
            node_vmax=1,
            display_mode=display_mode,
            colorbar=False,
            annotate=False,
            figure=fig,
            axes=ax,
            title=None,
            alpha=0.92,
        )
        for slice_ax in fig.get_axes():
            for coll in slice_ax.collections:
                offsets = coll.get_offsets()
                if len(offsets) == 0:
                    continue
                for idx, (ox, oy) in enumerate(offsets):
                    if idx >= top_n:
                        break
                    color = RANK_COLORS[idx % len(RANK_COLORS)]
                    slice_ax.annotate(
                        f" {idx + 1} ",
                        xy=(ox, oy),
                        xytext=(0, 18),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold',
                        color='#ffffff', fontfamily='monospace',
                        clip_on=True, zorder=11,
                        arrowprops=dict(arrowstyle='-', color=color, lw=1.5),
                        bbox=dict(boxstyle='round,pad=0.28', facecolor=color,
                                  edgecolor='none', alpha=1.0),
                    )
        fig.text(0.5, 0.98, title, ha='center', va='top',
                 fontsize=10, color='#94a3b8', fontfamily='monospace')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='#0a0a0f', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    try:
        axial_b64   = make_view('z', 'Axial — Top view')
        coronal_b64 = make_view('y', 'Coronal — Front view')

        # 5-column region key grid
        region_grid = ""
        for i, label in enumerate(top_labels):
            color = RANK_COLORS[i % len(RANK_COLORS)]
            region_grid += f"""
            <div style="display:flex;align-items:center;gap:8px;padding:8px 10px;
                        background:#f8fafc;border-radius:8px;border:1px solid #f1f5f9;">
              <span style="display:inline-flex;align-items:center;justify-content:center;
                           min-width:22px;height:22px;border-radius:50%;background:{color};
                           color:#fff;font-size:0.65rem;font-weight:700;flex-shrink:0;
                           font-family:'JetBrains Mono',monospace;">{i+1}</span>
              <span style="font-size:0.78rem;color:#111827;font-weight:500;
                           line-height:1.3;overflow:hidden;text-overflow:ellipsis;
                           white-space:nowrap;" title="{label}">{label}</span>
            </div>"""

        html = f"""
        <div style="background:#0a0a0f;padding:16px 20px 16px;">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <img src="data:image/png;base64,{axial_b64}"
                 style="width:100%;border-radius:8px;border:1px solid #1e293b;" />
            <img src="data:image/png;base64,{coronal_b64}"
                 style="width:100%;border-radius:8px;border:1px solid #1e293b;" />
          </div>
        </div>
        <div style="background:#ffffff;padding:14px 20px 20px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;
                      letter-spacing:0.1em;text-transform:uppercase;color:#9ca3af;
                      margin-bottom:10px;">Region key — numbers match brain markers</div>
          <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;">
            {region_grid}
          </div>
        </div>"""
        return html

    except Exception as e:
        import traceback
        return f"<p style='color:red;padding:16px'>Brain map error: {e}<br><pre>{traceback.format_exc()}</pre></p>"


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_diag', methods=['POST'])
def predict_diag():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"status": "error", "error": "No file uploaded"}), 400

        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        ts_data = np.load(path)
        p_mat, g_mat, c_mat = compute_matrices(ts_data)
        combined_flat = np.hstack([p_mat.flatten(), g_mat.flatten(), c_mat.flatten()])
        norm_a = scaler_a.transform(combined_flat.reshape(1, -1))

        graph_input = Data(
            x=torch.tensor(p_mat[:, :PCA_DIM], dtype=torch.float),
            edge_index=get_edge_index(p_mat),
            edge_granger=get_edge_index(g_mat),
            edge_wavelet=get_edge_index(c_mat),
            extra_feat=torch.tensor(norm_a[:, :AUX_SIZE], dtype=torch.float),
            batch=torch.zeros(NUM_REGIONS, dtype=torch.long)
        )

        with torch.no_grad():
            logits_a, xai = model_a(graph_input, return_attn=True)
            probs_a = torch.softmax(logits_a, dim=1).numpy()[0]
            prediction = str(DIAG_CLASSES[np.argmax(probs_a)])

        node_importance = compute_node_importance(xai)
        top_indices = np.argsort(node_importance)[-10:][::-1]
        top_regions = [
            ATLAS_LABELS[int(i)] if int(i) < len(ATLAS_LABELS) else f"Region {i}"
            for i in top_indices
        ]

        modality_weights = xai['modality_weights']
        if modality_weights.ndim > 1:
            modality_weights = modality_weights.mean(axis=0)

        brain_html = build_brain_heatmap_html(node_importance, prediction)

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": float(np.max(probs_a)),
            "explainability": {
                "top_regions": top_regions,
                "modality_scores": modality_weights.tolist(),
                "node_importance": importance_for_plot(node_importance).tolist(),
                "brain_heatmap_html": brain_html,
            },
            "file_path": path
        })

    except Exception as e:
        print(f"DIAGNOSTIC CRASH: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/predict_prog', methods=['POST'])
def predict_prog():
    try:
        path = request.form.get('file_path')
        if not path or not os.path.exists(path):
            file = request.files.get('file')
            if not file:
                return jsonify({"status": "error", "error": "Missing data"}), 400
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

        ts_data = np.load(path)
        p_mat, g_mat, c_mat = compute_matrices(ts_data)
        combined_flat = np.hstack([p_mat.flatten(), g_mat.flatten(), c_mat.flatten()])
        norm_b = scaler_b.transform(combined_flat.reshape(1, -1))

        graph_input = Data(
            x=torch.tensor(p_mat[:, :PCA_DIM], dtype=torch.float),
            edge_index=get_edge_index(p_mat),
            edge_granger=get_edge_index(g_mat),
            edge_wavelet=get_edge_index(c_mat),
            extra_feat=torch.tensor(norm_b[:, :AUX_SIZE], dtype=torch.float),
            batch=torch.zeros(NUM_REGIONS, dtype=torch.long)
        )

        with torch.no_grad():
            logits_b, xai_b = model_b(graph_input, return_attn=True)
            probs_b = torch.softmax(logits_b, dim=1).numpy()[0]
            label = PROG_CLASSES[np.argmax(probs_b)]

        node_importance = compute_node_importance(xai_b)
        top_indices = np.argsort(node_importance)[-10:][::-1]
        top_regions = [
            ATLAS_LABELS[int(i)] if int(i) < len(ATLAS_LABELS) else f"Region {i}"
            for i in top_indices
        ]

        m_weights = xai_b['modality_weights']
        if m_weights.ndim > 1:
            m_weights = m_weights.mean(axis=0)

        brain_html = build_brain_heatmap_html(node_importance, label)

        return jsonify({
            "status": "success",
            "label": label,
            "confidence": float(np.max(probs_b)),
            "explainability": {
                "top_regions": top_regions,
                "modality_scores": m_weights.tolist(),
                "node_importance": importance_for_plot(node_importance).tolist(),
                "brain_heatmap_html": brain_html,
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)