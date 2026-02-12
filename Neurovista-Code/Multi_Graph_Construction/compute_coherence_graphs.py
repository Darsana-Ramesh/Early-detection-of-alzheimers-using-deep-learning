import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# ================= CONFIGURATION =================
# 1. INPUT: Time Series Matrices
INPUT_DIR = r"C:\Users\DELL\Desktop\4th-year-project\connectivity_graph_construction\Timeseries_Output"

# 2. OUTPUT: Where to save Coherence Graphs
OUTPUT_DIR = r"C:\Users\DELL\Desktop\4th-year-project\connectivity_graph_construction\Graph_Output_Coherence"

# 3. TR (Repetition Time): Needed for frequency analysis
TR = 3.0  # seconds
# =================================================

print("--- Starting Spectral Coherence Graph Construction ---")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ts_files = list(Path(INPUT_DIR).glob("*_timeseries.npy"))
print(f"Found {len(ts_files)} files. Computing coherence (this takes longer than correlation)...")

for ts_file in ts_files:
    subject_id = ts_file.name.split('_timeseries')[0]
    print(f"Processing: {subject_id}...")
    
    # Load Data: Shape is (Timepoints, 116 Regions)
    time_series = np.load(ts_file)
    n_timepoints, n_regions = time_series.shape
    
    # Initialize empty Coherence Matrix (116 x 116)
    coherence_matrix = np.zeros((n_regions, n_regions))
    
    # Sampling frequency (Hz)
    fs = 1.0 / TR
    
    # --- CORE CALCULATION: Magnitude Squared Coherence ---
    # We must loop through every pair of regions (heavy computation)
    # To save time, we calculate only the upper triangle and mirror it
    for i in range(n_regions):
        for j in range(i+1, n_regions):
            
            # Calculate coherence between Region i and Region j
            f, Cxy = signal.coherence(time_series[:, i], time_series[:, j], fs=fs, nperseg=64)
            
            # Average coherence across the band of interest (e.g., 0.01 - 0.1 Hz)
            # This captures "functional connectivity" frequency range
            freq_mask = (f >= 0.01) & (f <= 0.1)
            mean_coherence = np.mean(Cxy[freq_mask])
            
            # Fill matrix (Symmetric)
            coherence_matrix[i, j] = mean_coherence
            coherence_matrix[j, i] = mean_coherence
            
        # Set diagonal to 1 (region perfectly coherent with itself)
        coherence_matrix[i, i] = 1.0

    # Save Matrix
    output_name = os.path.join(OUTPUT_DIR, f"{subject_id}_coherence_matrix.npy")
    np.save(output_name, coherence_matrix)

print("\n" + "="*50)
print(f"DONE! Coherence Graphs saved in: {OUTPUT_DIR}")
print("="*50)

# --- VISUALIZATION ---
print("Visualizing last coherence matrix...")
plt.figure(figsize=(10, 8))
plt.imshow(coherence_matrix, cmap='plasma', vmin=0, vmax=1)
plt.colorbar(label='Spectral Coherence')
plt.title(f"Spectral Coherence Graph (G_H): {subject_id}")
plt.xlabel("Brain Regions")
plt.ylabel("Brain Regions")
plt.show()