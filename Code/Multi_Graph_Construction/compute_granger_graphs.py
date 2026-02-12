import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from pathlib import Path

# ================= CONFIGURATION =================
# 1. INPUT: Time Series Matrices
INPUT_DIR = r"C:\Users\DELL\Desktop\4th-year-project\connectivity_graph_construction\Timeseries_Output"

# 2. OUTPUT: Where to save Granger Graphs
OUTPUT_DIR = r"C:\Users\DELL\Desktop\4th-year-project\connectivity_graph_construction\Graph_Output_Granger"

# 3. Model Lag Order (1 is standard for fMRI with TR=3.0s)
# This checks if brain state at time (t-1) predicts state at time (t)
LAG_ORDER = 1
# =================================================

print("--- Starting Granger Causality (Directed Graph) Construction ---")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ts_files = list(Path(INPUT_DIR).glob("*_timeseries.npy"))
print(f"Found {len(ts_files)} files. Fitting VAR models (this determines direction)...")

for ts_file in ts_files:
    subject_id = ts_file.name.split('_timeseries')[0]
    print(f"Processing: {subject_id}...")
    
    # Load Data
    time_series = np.load(ts_file)
    
    try:
        # --- CORE CALCULATION: Vector Autoregression (VAR) ---
        # VAR models the relationship between all regions simultaneously.
        model = VAR(time_series)
        results = model.fit(LAG_ORDER)
        
        # The coefficient matrix IS the directed graph.
        # It tells us how much Region J (at time t-1) influences Region I (at time t).
        # Shape: (116, 116)
        granger_matrix = results.coefs[0]  # Get coefficients for lag 1
        
        # Take absolute value (we care about strength of influence, not sign)
        granger_matrix = np.abs(granger_matrix)

        # Save Matrix
        output_name = os.path.join(OUTPUT_DIR, f"{subject_id}_granger_matrix.npy")
        np.save(output_name, granger_matrix)
        
    except Exception as e:
        print(f"  ❌ Error fitting model for {subject_id}: {e}")

print("\n" + "="*50)
print(f"DONE! Directed Graphs saved in: {OUTPUT_DIR}")
print("="*50)

# --- VISUALIZATION ---
print("Visualizing last Directed Graph (Notice it is NOT symmetric)...")
plt.figure(figsize=(10, 8))
plt.imshow(granger_matrix, cmap='magma', vmin=0, vmax=0.5) # Capped vmax to see details
plt.colorbar(label='Causal Influence Strength')
plt.title(f"Granger Causality Graph (G_C): {subject_id}\n(Directional: Row predicts Column)")
plt.xlabel("Source Region (From)")
plt.ylabel("Target Region (To)")
plt.show()