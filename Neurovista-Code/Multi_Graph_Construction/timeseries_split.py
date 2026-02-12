import os
import numpy as np
from scipy import signal
from statsmodels.tsa.api import VAR
from pathlib import Path

# ================= CONFIGURATION =================
# 1. INPUT: Your original time series folder
INPUT_DIR = r"C:\Users\DELL\Desktop\4th-year-project\connectivity_graph_construction\Timeseries_Output"

# 2. OUTPUTS: New folders for the augmented graphs
BASE_OUT = r"C:\Users\DELL\Desktop\4th-year-project\connectivity_graph_construction\Augmented_Graphs_MCI"
OUT_P = os.path.join(BASE_OUT, "Pearson")
OUT_C = os.path.join(BASE_OUT, "Coherence")
OUT_G = os.path.join(BASE_OUT, "Granger")

# 3. SLIDING WINDOW PARAMETERS
WINDOW_SIZE = 60  # ~3 minutes of data per window (if TR=3.0)
STRIDE = 20       # Shift by 1 minute to create the next window
TR = 3.0          # Repetition Time
# =================================================

# Create directories
for d in [OUT_P, OUT_C, OUT_G]: os.makedirs(d, exist_ok=True)

print(f"--- Starting Multi-View Sliding Window Augmentation ---")
ts_files = list(Path(INPUT_DIR).glob("*_timeseries.npy"))
print(f"Found {len(ts_files)} subjects. Augmenting now...")

for ts_file in ts_files:
    subject_id = ts_file.name.split('_timeseries')[0]
    ts_data = np.load(ts_file) # Shape: (Time, 116)
    n_timepoints, n_regions = ts_data.shape
    
    window_count = 0
    # Loop through the time series to create windows
    for start in range(0, n_timepoints - WINDOW_SIZE + 1, STRIDE):
        window_count += 1
        end = start + WINDOW_SIZE
        w_ts = ts_data[start:end, :] # The sliced time-series window
        
        window_suffix = f"win{window_count:02d}"
        print(f" Processing: {subject_id} | {window_suffix}...")

        # --- VIEW 1: PEARSON CORRELATION ---
        p_mat = np.nan_to_num(np.corrcoef(w_ts.T))
        np.save(os.path.join(OUT_P, f"{subject_id}_{window_suffix}_pearson.npy"), p_mat)
        
        # --- VIEW 2: SPECTRAL COHERENCE (0.01 - 0.1 Hz) ---
        c_mat = np.zeros((n_regions, n_regions))
        fs = 1.0 / TR
        for i in range(n_regions):
            for j in range(i+1, n_regions):
                f, Cxy = signal.coherence(w_ts[:, i], w_ts[:, j], fs=fs, nperseg=32)
                # Average coherence in the specific resting-state band
                mean_coh = np.mean(Cxy[(f >= 0.01) & (f <= 0.1)])
                c_mat[i, j] = c_mat[j, i] = mean_coh
        np.fill_diagonal(c_mat, 1.0)
        np.save(os.path.join(OUT_C, f"{subject_id}_{window_suffix}_coherence.npy"), c_mat)
        
        # --- VIEW 3: GRANGER CAUSALITY (Flow) ---
        try:
            # Fit a Vector Autoregression model (Lag 1)
            model = VAR(w_ts).fit(1)
            g_mat = np.abs(model.coefs[0]) # Use absolute values for edge weights
        except Exception:
            # If the window is too noisy to fit, we use an empty matrix
            g_mat = np.zeros((n_regions, n_regions))
        np.save(os.path.join(OUT_G, f"{subject_id}_{window_suffix}_granger.npy"), g_mat)

print("\n" + "="*50)
print(f"SUCCESS! Graphs generated for {len(ts_files)} subjects.")
print(f"Files saved in: {BASE_OUT}")
print("="*50)