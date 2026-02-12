import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================= CONFIGURATION =================
# 1. INPUT: Where your time series .npy files are
INPUT_DIR = r"C:\Users\DELL\Desktop\4th-year-project\connectivity_graph_construction\Timeseries_Output"

# 2. OUTPUT: Where to save the Connectivity Graphs
OUTPUT_DIR = r"C:\Users\DELL\Desktop\4th-year-project\connectivity_graph_construction\Graph_Output"
# =================================================

print("--- Starting Functional Connectivity Graph Construction ---")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Find all Time Series files
ts_files = list(Path(INPUT_DIR).glob("*_timeseries.npy"))
print(f"Found {len(ts_files)} time series files.")

if not ts_files:
    print("❌ No files found! Run the extraction script first.")
    exit()

# 2. Process each subject
for ts_file in ts_files:
    subject_id = ts_file.name.split('_timeseries')[0]
    print(f"Processing: {subject_id}...")
    
    
    # Load Time Series (Time x Regions)
    # e.g., (200, 116)
    time_series = np.load(ts_file)
    
    # --- CORE STEP: Calculate Pearson Correlation ---
    # We transpose (.T) because corrcoef expects rows=variables(regions), cols=observations(time)
    # Result shape will be (116, 116)
    correlation_matrix = np.corrcoef(time_series.T)
    
    # Replace NaN (errors) with 0 just in case
    np.nan_to_num(correlation_matrix, copy=False)
    
    # Save the Matrix
    output_filename = os.path.join(OUTPUT_DIR, f"{subject_id}_corr_matrix.npy")
    np.save(output_filename, correlation_matrix)
    
    print(f"  -> Graph created: {correlation_matrix.shape}")

print("\n" + "="*50)
print(f"DONE! Graphs saved in: {OUTPUT_DIR}")
print("="*50)

# --- VISUALIZATION CHECK (Last Subject) ---
print("Visualizing the last connectivity graph...")
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Strength')
plt.title(f"Functional Connectivity Graph: {subject_id}\n(116 x 116 Nodes)")
plt.xlabel("Brain Regions (ROIs)")
plt.ylabel("Brain Regions (ROIs)")
plt.show()